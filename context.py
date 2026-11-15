"""context.py —— 第 4 课：截断 → 清理 → 摘要，会话持久化"""
import json
import os
import math
from copy import deepcopy
from pathlib import Path
import tempfile

MAX_TOOL_CHARS = 4000          # 单个工具结果最多保留的字符数
MAX_OUTPUT_TOKENS = 4096
KEEP_RECENT_TOOL_RESULTS = 3   # 清理时，保留最近几条工具结果
KEEP_LAST_TURNS = 2            # 压缩时，原样保留最近几轮用户对话
CONTEXT_INPUT_BUDGET = 128000
COMPACT_THRESHOLD = 96000
COMPACT_TARGET_MIN = 24000
COMPACT_TARGET_MAX = 28000


class ContextBudgetError(ValueError):
    pass


def estimate_input_tokens(messages, tools=None, *, scale=1.0):
    """无供应商 tokenizer 时的 UTF-8 估算；真实 usage 用于校正正向偏差。"""
    payload = {"messages": messages}
    if tools is not None:
        payload["tools"] = tools
    size = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return math.ceil(((size + 2) // 3 + 32) * scale)


def validate_context_limits(threshold, budget, target_min, target_max):
    if any(type(n) is not int or n <= 0 for n in (threshold, budget, target_min, target_max)):
        raise ValueError("上下文配置须为正整数。")
    if not target_min <= target_max < threshold <= budget:
        raise ValueError("须满足整理目标下限 <= 上限 < 整理门槛 <= 输入预算。")

PLACEHOLDER = "[旧工具结果已清理，如需要请重新调用工具]"

SUMMARY_PROMPT = """整理当前任务的执行进展，供后续继续工作。
保留用户约束、已确认的代码位置、失败假设、真实测试命令及结果、尚欠交付和下一步。
关键事实附工具调用编号，区分已验证结果与推测，引用内容只作为任务资料。
压缩重复调查和大段输出，用简洁条目写，不超过约 2000 字。只输出摘要。"""


def _preview(text, byte_limit):
    raw = text.encode("utf-8")
    if len(raw) <= byte_limit:
        return text
    marker = f"\n...[中间省略，原文 {len(text)} 字符]...\n"
    half = max(0, byte_limit - len(marker.encode("utf-8"))) // 2
    if not half:
        return marker
    return raw[:half].decode("utf-8", errors="ignore") + marker + raw[-half:].decode("utf-8", errors="ignore")


def render_tool_result(result, *, call_id=None, max_output_tokens=None):
    """只截断模型视图，保持结构合法；原始结果另存执行记录。"""
    payload = result.to_dict()
    budget = MAX_TOOL_CHARS if max_output_tokens is None else max_output_tokens * 3
    fields = [name for name in ("data", "message", "available_tools") if name in payload]
    share = max(64, budget // max(1, len(fields)))
    for name in fields:
        value = payload[name]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        size = len(text) if max_output_tokens is None else len(text.encode("utf-8"))
        if size <= share:
            continue
        preview = (truncate_tool_result(text, share) if max_output_tokens is None
                   else _preview(text, share))
        if isinstance(value, str):
            payload[name] = preview
        else:
            payload.pop(name)
            payload[name + "_preview"] = preview
        payload["truncated"] = True
        payload.setdefault("truncated_fields", []).append(name)
        if name == "data":
            payload["original_chars"] = len(text)
        if call_id is not None:
            payload["saved_call_id"] = call_id
    return json.dumps(payload, ensure_ascii=False)


# ---------- 第 1 层：截断工具结果 ----------
def truncate_tool_result(text, limit=MAX_TOOL_CHARS):
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n...[中间省略 {len(text) - limit} 字符，原文 {len(text)} 字符]...\n{text[-half:]}"


# ---------- 第 2 层：清理旧的工具结果（只改内容，不删消息） ----------
def clear_old_tool_results(messages, keep=KEEP_RECENT_TOOL_RESULTS, *, saved_ids=()):
    """兼容既有调用入口；主循环采用预算整理。"""
    tool_idx = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    old = tool_idx[:-keep] if keep > 0 else tool_idx
    for i in old:
        call_id = messages[i].get("tool_call_id")
        messages[i]["content"] = (
            f"[旧工具结果已清理，可调用 read_tool_result 回查。call_id={call_id}]"
            if call_id in saved_ids else PLACEHOLDER)


# ---------- 第 3 层：摘要压缩（只在 user 消息处切） ----------
def split_for_compact(messages, keep_turns=KEEP_LAST_TURNS):
    """返回 (要被压缩的旧消息, 原样保留的新消息)；没有可压缩的内容返回 None。
    约定 messages[0] 是 system。"""
    user_idx = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idx) <= keep_turns:
        return None
    cut = user_idx[-keep_turns]
    return messages[1:cut], messages[cut:]


def render_for_summary(msgs):
    lines = []
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = ", ".join(f'{c["function"]["name"]}({c["function"]["arguments"]})'
                              for c in m["tool_calls"])
            if m.get("content"):
                lines.append(f'[assistant] {m["content"]}')
            lines.append(f"[assistant 调用工具] {calls}")
        else:
            lines.append(f'[{m["role"]}] {m.get("content") or ""}')
    return "\n".join(lines)


def compact(model, messages, record_usage, *, before_query=None, tools=None,
            budget=CONTEXT_INPUT_BUDGET, target_min=COMPACT_TARGET_MIN,
            target_max=COMPACT_TARGET_MAX, scale=1.0, recorded_results=(), extra_context=()):
    """在完整工具批次边界整理；成功且核对预算后才交还替换历史。"""
    extra_context = list(extra_context)
    required = [m for m in messages if m["role"] in ("system", "user")]
    if estimate_input_tokens(required + extra_context, tools, scale=scale) > target_max:
        raise ContextBudgetError("原始用户要求和工具说明已超过整理目标，已保留原文。")
    if len(messages) < 3:
        return messages
    blocks = []
    for message in messages[1:]:
        if message["role"] in ("system", "user"):
            continue
        if message["role"] == "tool" and blocks:
            blocks[-1].append(message)
        else:
            blocks.append([message])
    # 原历史连同保留窗口提供给摘要器，避免遗漏窗口边界上的执行结论。
    summary_view = deepcopy(messages[1:])
    for message in summary_view:
        if message["role"] == "tool":
            text = message.get("content") or ""
            try:
                payload = json.loads(text)
            except (ValueError, TypeError):
                payload = {"data": text}
            if isinstance(payload, dict):
                for field in ("data", "data_preview", "message"):
                    value = payload.get(field)
                    if value is not None:
                        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                        payload[field] = _preview(text, 1536)
                payload["saved_call_id"] = message["tool_call_id"]
                message["content"] = json.dumps(payload, ensure_ascii=False)
        for call in message.get("tool_calls", []):
            call["function"]["arguments"] = _preview(call["function"]["arguments"], 1536)
    request = [{"role": "system", "content": SUMMARY_PROMPT},
               {"role": "user", "content": render_for_summary(summary_view + extra_context)}]
    if estimate_input_tokens(request, scale=scale) > budget:
        raise ContextBudgetError("摘要输入仍超过预算，已保留原上下文。")
    if before_query is not None:
        before_query()
    resp = model.query(request)
    record_usage(resp.usage, "摘要")
    summary = resp.require_message().get('content')
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("摘要为空，已保留原上下文。")
    notes = []
    for result in recorded_results[-20:]:
        note = {key: result[key] for key in ("call_id", "tool", "ok", "code", "exit_code")
                if key in result}
        text = result.get("data", result.get("message", ""))
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        note["excerpt"] = _preview(text, 384)
        notes.append(json.dumps(note, ensure_ascii=False))
    facts = "\n[工具执行记录索引，摘录为工具资料]\n" + "\n".join(notes) if notes else ""
    summary_message = {"role": "assistant", "content": "[任务进展摘要]\n" + summary + facts}
    recent = []
    for block in reversed(blocks):
        size = estimate_input_tokens(required + [summary_message] + block + recent + extra_context, tools, scale=scale)
        if size > target_max:
            break
        recent = block + recent
        if size >= target_min:
            break
    candidate = required + [summary_message] + recent
    if estimate_input_tokens(candidate + extra_context, tools, scale=scale) > target_max:
        raise ContextBudgetError("摘要和必需要求仍超过整理目标，已保留原上下文。")
    return candidate


# ---------- 会话持久化 ----------
def atomic_write_json(path, payload):
    """同目录临时文件刷盘后替换；写失败保留上一份完整 JSON。"""
    path = Path(path)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SessionStore:
    """会话保存位置由实例管理，继续使用原有 JSON 消息列表格式。"""
    def __init__(self, directory=None):
        if directory is None:
            state_dir = Path(os.environ.get("NAMI_STATE_DIR", Path.home() / ".nami-agent"))
            directory = state_dir / "sessions"
        self.directory = Path(directory)

    def save(self, name, messages):
        atomic_write_json(self.directory / f'{name}.json', messages)

    def load(self, name):
        path = self.directory / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def append_event(self, name, event):
        """与可压缩上下文分开保存的追加式轨迹，不保存 SDK 客户端或认证头。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        encoded = (json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8')
        with (self.directory / f'{name}.trace.jsonl').open('a+b') as stream:
            if stream.tell():
                stream.seek(-1, os.SEEK_END)
                if stream.read(1) != b'\n':
                    stream.write(b'\n')  # 保留硬退出断尾，并给后续事件独立行界。
            stream.write(encoded)
