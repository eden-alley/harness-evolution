"""tools.py —— 第 2 课的注册表 + 第 3 课加固：文件工具只认工作区，shell 固定目录、限时"""
import subprocess
import urllib.request
from urllib.error import HTTPError
from urllib.parse import urljoin
import inspect
import json
import hashlib
from copy import deepcopy
from dataclasses import dataclass
from http.client import IncompleteRead

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry

from context import MAX_TOOL_CHARS, MAX_OUTPUT_TOKENS
from sandbox import run_in_sandbox
from execution_backend import CodexWindowsSandboxBackend
from permissions import fetch_url_denial
from file_edits import MAX_FILE_BYTES, MAX_EDITS, FileEditError, apply_file_edits, write_whole_file
from file_search import (FileSearch, SearchError, SEARCH_TOOLS, DEFAULT_FIND_LIMIT,
                         MAX_FIND_LIMIT, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT)

MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_FETCH_REDIRECTS = 5
MAX_TASK_PROGRESS_BYTES = 3000
TOOLS = {}


@dataclass(frozen=True)
class ToolResult:
    """单次工具结果；legacy 只表示旧函数正常返回，不保证业务操作成功。"""
    ok: bool
    data: object = None
    code: str = ""
    message: str = ""
    available_tools: list | None = None
    exit_code: int | None = None
    result_contract: str = "structured"

    def to_dict(self):
        result = {"ok": self.ok}
        if self.data is not None:
            result["data"] = self.data
        if not self.ok:
            result.update(code=self.code, message=self.message)
        if self.available_tools is not None:
            result["available_tools"] = self.available_tools
        if self.exit_code is not None:
            result["exit_code"] = self.exit_code
        if self.result_contract == "legacy":
            result["result_contract"] = "legacy"
        return result


def _checked_result(result, structured):
    try:
        if not isinstance(result, ToolResult):
            if structured:
                raise ValueError("原生工具必须返回 ToolResult")
            result = ToolResult(True, data=result, result_contract="legacy")
        if type(result.ok) is not bool:
            raise ValueError("ok 必须是布尔值")
        if not isinstance(result.code, str) or not isinstance(result.message, str):
            raise ValueError("code 和 message 必须是字符串")
        if not result.ok and (not result.code.strip() or not result.message.strip()):
            raise ValueError("失败结果必须包含错误码和说明")
        if result.ok and result.code:
            raise ValueError("成功结果不能包含错误码")
        if result.result_contract not in ("structured", "legacy") or (
                structured and result.result_contract != "structured"):
            raise ValueError("返回契约标记不合法")
        if result.exit_code is not None and type(result.exit_code) is not int:
            raise ValueError("exit_code 必须是整数")
        if result.available_tools is not None and (
                not isinstance(result.available_tools, list) or
                any(not isinstance(name, str) for name in result.available_tools)):
            raise ValueError("available_tools 必须是字符串列表")
        json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
    except Exception as error:
        return ToolResult(False, code="invalid_tool_result",
                          message=f"工具已返回，但结果不符合契约：{error}；执行状态需核实，勿盲目重试。")
    return result


def tool(name, description, parameters):
    def decorator(func):
        if name in TOOLS:
            raise ValueError(f"工具重复注册：{name}")
        schema = deepcopy(parameters)
        schema.setdefault("properties", {})["max_output_tokens"] = {
            "type": "integer", "minimum": 1, "maximum": MAX_OUTPUT_TOKENS,
            "description": "可选结果预览预算，按 UTF-8 字节估算；完整结果仍保存以便回查。"}
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError(f"工具 {name} 的 schema 不合法：{error.message}") from error
        TOOLS[name] = {
            "schema": {"type": "function", "function": {
                "name": name, "description": description, "parameters": schema}},
            "func": func,
            "validator": Draft202012Validator(schema, registry=Registry()),
            "structured_result": True,
        }
        return func
    return decorator


def get_schemas(*, task_progress=False, search_available=True):
    return [t["schema"] for name, t in TOOLS.items()
            if (task_progress or name != "update_progress")
            and (search_available or name not in SEARCH_TOOLS)]


def _params(*, required=None, **props):
    return {"type": "object",
            "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props) if required is None else required,
            "additionalProperties": False}


@tool("update_progress", "更新当前任务的简短工作记录；不代表验收通过。四项内容整体替换，最多 3000 UTF-8 字节，结论引用已返回的工具调用编号。",
      {"type": "object", "required": ["deliverables", "findings", "remaining", "next_action"],
       "additionalProperties": False, "properties": {
           "deliverables": {"type": "array", "minItems": 1,
                            "items": {"type": "string", "minLength": 1}},
           "findings": {"type": "array", "items": {
               "type": "object", "required": ["text", "call_ids"], "additionalProperties": False,
               "properties": {"text": {"type": "string", "minLength": 1},
                              "call_ids": {"type": "array", "minItems": 1, "uniqueItems": True,
                                           "items": {"type": "string", "minLength": 1}}}}},
           "remaining": {"type": "array", "items": {"type": "string", "minLength": 1}},
           "next_action": {"type": "string", "minLength": 1},
       }})
def update_progress(tools, deliverables, findings, remaining, next_action, *, recorded_results=()):
    document = {"deliverables": deliverables, "findings": findings,
                "remaining": remaining, "next_action": next_action}
    texts = [*deliverables, *remaining, next_action, *(item["text"] for item in findings)]
    if any(not text.strip() for text in texts):
        return ToolResult(False, code="invalid_arguments", message="进度内容不能只包含空白。")
    if len(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_TASK_PROGRESS_BYTES:
        return ToolResult(False, code="invalid_arguments", message="任务进度超过 3000 UTF-8 字节，请压缩记录，不要放入原始日志。")
    for item in findings:
        for call_id in item["call_ids"]:
            matches = [result for result in recorded_results
                       if result["call_id"] == call_id and result["source"] == "model"
                       and result["tool"] != "update_progress"]
            if len(matches) != 1:
                return ToolResult(False, code="invalid_arguments",
                                  message=f"证据 {call_id!r} 不是本任务唯一对应的已记录工具结果；进度记录不能充当证据。")
    return ToolResult(True, data=document)


@tool("list_dir", "列出工作区内某个目录下的文件和子目录。",
      _params(required=[], path="目录路径，省略时为工作区根目录 ."))
def list_dir(tools, path="."):
    p = tools.workspace.resolve(path)
    if p is None:                      # 双保险：即使绕过了权限检查，这里还会拦
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    data = "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())) or "(空目录)"
    return ToolResult(True, data=data)


def _run_search(tools, action, path, **args):
    if tools.workspace.resolve(path) is None:
        return ToolResult(False, code="policy_denied", message="搜索路径在工作区之外，禁止访问。")
    try:
        return ToolResult(True, data=getattr(tools.file_search, action)(path=path, **args))
    except SearchError as error:
        return ToolResult(False, data=error.partial, code=error.code, message=str(error))


@tool("find_files", "按基础 glob 查找工作区内文件，返回工作区相对路径，若已启用 read_file 可直接交给它读取。支持 *、?、[]、**。默认遵循项目忽略规则、包括隐藏文件、排除 .git 和链接。默认 200 条，最多 2000 条；每次最多 30 秒、64 KiB。输出注明是否完整。",
      {"type": "object", "required": ["pattern"], "additionalProperties": False,
       "properties": {
           "pattern": {"type": "string", "minLength": 1, "description": "相对 path 的基础 glob；无 / 时匹配任意深度文件名。"},
           "path": {"type": "string", "minLength": 1, "default": ".", "description": "搜索目录，省略为工作区根。"},
           "limit": {"type": "integer", "minimum": 1, "maximum": MAX_FIND_LIMIT, "default": DEFAULT_FIND_LIMIT},
           "ignore_case": {"type": "boolean", "default": False},
           "include_ignored": {"type": "boolean", "default": False, "description": "显式包含被项目忽略的文件，仍不遍历 .git 和链接。"},
       }})
def find_files(tools, pattern, path=".", limit=DEFAULT_FIND_LIMIT, ignore_case=False, include_ignored=False):
    return _run_search(tools, "find", path, pattern=pattern, limit=limit,
                       ignore_case=ignore_case, include_ignored=include_ignored)


@tool("search_text", "搜索工作区文本，返回相对工作区的路径、与 read_file 一致的行号及可选上下文。默认字面量、区分大小写、遵循项目忽略规则；可显式开启单行正则。默认 100 个匹配行，最多 1000 行；每次最多 30 秒、64 KiB。limit 不计上下文行；输出注明是否完整，未收集部分需重新搜索。",
      {"type": "object", "required": ["pattern"], "additionalProperties": False,
       "properties": {
           "pattern": {"type": "string", "minLength": 1, "pattern": "^[^\r\n]+$", "description": "普通文本或单行 rg 正则；不支持跨行搜索。"},
           "path": {"type": "string", "minLength": 1, "default": ".", "description": "搜索目录或文件，省略为工作区根。"},
           "glob": {"type": "string", "minLength": 1, "description": "目录搜索的基础 glob，规则与 find_files 一致。"},
           "regex": {"type": "boolean", "default": False},
           "ignore_case": {"type": "boolean", "default": False, "description": "同时忽略文本和 glob 文件名大小写。"},
           "context_lines": {"type": "integer", "minimum": 0, "maximum": 5, "default": 0},
           "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT, "default": DEFAULT_SEARCH_LIMIT},
           "include_ignored": {"type": "boolean", "default": False},
       }})
def search_text(tools, pattern, path=".", glob=None, regex=False, ignore_case=False,
                context_lines=0, limit=DEFAULT_SEARCH_LIMIT, include_ignored=False):
    return _run_search(tools, "search", path, pattern=pattern, glob=glob, regex=regex,
                       ignore_case=ignore_case, context_lines=context_lines, limit=limit,
                       include_ignored=include_ignored)


def _read_size(text, byte_budget):
    return len(text.encode("utf-8")) if byte_budget is not None else len(text)


def _file_page_frame(path, start, last, *, eof, range_done, end_line, byte_budget):
    status = "文件已读完" if eof else ("指定范围已读完，文件尚有内容" if range_done else "输出额度不足，需续读")
    head = f"[行 {start}-{last}；{status}]\n"
    if eof:
        return head, ""
    following = {"path": path, "start_line": last + 1}
    if not range_done and end_line is not None:
        following["end_line"] = end_line
    if byte_budget is not None:
        following["max_output_tokens"] = byte_budget // 3
    return head, "\n[续读：read_file(" + json.dumps(following, ensure_ascii=False) + ")]"


@tool("read_file", "按完整行分页读取 UTF-8 文本，默认从第 1 行开始，返回行号、结束状态及续读参数。只保存本页；未读内容须继续 read_file。超长单行不拆分，可提高 max_output_tokens（上限4096）；若已启用 run_shell，仍不足时可用它分段读取。",
      {"type": "object", "required": ["path"], "additionalProperties": False,
       "properties": {
           "path": {"type": "string", "description": "文件路径"},
           "start_line": {"type": "integer", "minimum": 1, "description": "起始行，从 1 开始，默认第 1 行"},
            "end_line": {"type": "integer", "minimum": 1, "description": "请求范围的结束行，包含该行；省略表示以文件末尾为目标，单次仍受输出额度限制"},
       }})
def read_file(tools, path, *, start_line=None, end_line=None, _output_chars=MAX_TOOL_CHARS,
              _output_bytes=None):
    p = tools.workspace.resolve(path)
    if p is None:
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    start_line = start_line or 1
    if end_line is not None and end_line < start_line:
        return ToolResult(False, code="invalid_arguments", message="结束行不能小于起始行。")

    limit = _output_bytes if _output_bytes is not None else _output_chars
    lines = []
    size = 0
    with p.open(encoding="utf-8") as stream:
        for _ in range(1, start_line):
            if not stream.readline():
                return ToolResult(False, code="invalid_arguments", message="起始行超过文件末尾。")
        line = stream.readline()
        if not line:
            if start_line != 1:
                return ToolResult(False, code="invalid_arguments", message="起始行超过文件末尾。")
            empty = "[空文件；文件已读完]"
            if _read_size(empty, _output_bytes) > limit:
                return ToolResult(False, code="output_limit_exceeded", message="输出额度不足以显示读取状态，请提高 max_output_tokens。")
            return ToolResult(True, data=empty)
        number = start_line
        while True:
            # 仅探查下一字符；不为统计总行数扫描未请求的文件尾部。
            following = stream.read(1)
            eof = following == ""
            range_done = end_line is not None and number >= end_line
            numbered = f"{number}: {line}"
            measure = _read_size(numbered, _output_bytes)
            lines.append(numbered)
            size += measure
            minimal_head, _ = _file_page_frame(path, start_line, number, eof=True, range_done=False,
                                              end_line=end_line, byte_budget=_output_bytes)
            # 先容纳可能完整结束的页；不能因假想的续读说明而提前拒绝。
            if eof or range_done or size + _read_size(minimal_head, _output_bytes) > limit:
                break
            # 预读字符若已是换行，它单独构成一个空行，不能再吞入下一物理行。
            line = following if following == "\n" else following + stream.readline()
            number += 1
    while lines:
        head, tail = _file_page_frame(path, start_line, number, eof=eof, range_done=range_done,
                                       end_line=end_line, byte_budget=_output_bytes)
        if size + _read_size(head + tail, _output_bytes) <= limit:
            return ToolResult(True, data=head + "".join(lines) + tail)
        size -= _read_size(lines.pop(), _output_bytes)
        number -= 1
        eof = range_done = False
    recovery = ("仍不足时用现有 run_shell 分段读取，沿用原权限。" if tools.is_available("run_shell")
                else "本会话未启用 run_shell，不能通过它分段读取；需要调整工具配置或提供较短内容。")
    return ToolResult(False, code="output_limit_exceeded",
                      message=f"第 {start_line} 行（含行号和读取说明）无法在当前输出额度内完整返回。请提高 max_output_tokens（上限4096）；{recovery}")


@tool("read_tool_result", "按调用编号回查本次任务已保存的返回内容，不能找回尚未读取的文件部分或未收集的搜索结果。offset 为从0开始的字符位置；返回实际字段、左闭右开的字符范围和结束状态，未完按提示续读。正文前有说明行，完整JSON需去除说明后解析。",
      {"type": "object", "required": ["call_id"], "additionalProperties": False,
       "properties": {
           "call_id": {"type": "string", "minLength": 1, "description": "原工具调用编号"},
           "offset": {"type": "integer", "minimum": 0, "description": "从第几个字符开始，默认 0"},
           "field": {"type": "string", "enum": ["data", "message", "result"], "default": "data",
                     "description": "回查正文、错误说明或含状态的完整结果对象"},
       }})
def read_tool_result(tools, call_id, offset=0, field="data", *, recorded_results=(), _output_chars=MAX_TOOL_CHARS,
                     _output_bytes=None):
    matches = [item for item in recorded_results if item["call_id"] == call_id
               and item["source"] == "model" and item["tool"] != "read_tool_result"]
    if len(matches) != 1:
        return ToolResult(False, code="result_unavailable", message="本次任务没有唯一对应的已保存结果。")
    original = matches[0]
    text = original.get(field)
    if field == "result" or not isinstance(text, str):
        text = json.dumps(original, ensure_ascii=False)
        field = "result"
    if offset > len(text):
        return ToolResult(False, code="invalid_arguments", message=f"offset={offset} 超过已保存正文长度 {len(text)}。")
    limit = _output_bytes if _output_bytes is not None else _output_chars

    def page(end):
        complete = end == len(text)
        head = f"[字段 {field}；字符 {offset}-{end}；{'结果已读完' if complete else '需续读'}]\n"
        tail = "" if complete else f"\n[请用 offset={end} 续读]"
        return head + text[offset:end] + tail

    # 完整末页不预留续读提示；其余页按字符边界寻找能容纳的最长片段。
    if len(text) - offset <= limit:
        complete = page(len(text))
        if _read_size(complete, _output_bytes) <= limit:
            return ToolResult(True, data=complete)
    low, high = offset, min(len(text) - 1, offset + limit)
    while low < high:
        middle = (low + high + 1) // 2
        if _read_size(page(middle), _output_bytes) <= limit:
            low = middle
        else:
            high = middle - 1
    if low <= offset:
        return ToolResult(False, code="output_limit_exceeded",
                          message="输出额度不足以容纳正文与读取说明，请提高 max_output_tokens。")
    return ToolResult(True, data=page(low))


@tool("edit_file", f"精确修改已有 UTF-8 文件（可带 BOM），上限 {MAX_FILE_BYTES} 字节。先读取原文；各处 oldText 必须在原文件中唯一且不重叠，只兼容换行差异，不模糊匹配。全部检查通过才写入并返回差异。局部修改优先使用本工具；若已启用 write_file，可用它创建或整文件重写。",
      {"type": "object", "required": ["path", "edits"], "additionalProperties": False,
       "properties": {
           "path": {"type": "string", "minLength": 1, "description": "工作区内已有文件的路径"},
           "edits": {"type": "array", "minItems": 1, "maxItems": MAX_EDITS,
                     "description": f"最多 {MAX_EDITS} 处替换，均基于同一份原文件匹配。newText 可为空以删除原文。",
                     "items": {"type": "object", "required": ["oldText", "newText"],
                               "additionalProperties": False, "properties": {
                                   "oldText": {"type": "string", "minLength": 1},
                                   "newText": {"type": "string"}}}},
       }})

def edit_file(tools, path, edits):
    p = tools.workspace.resolve(path)
    if p is None:
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    try:
        return ToolResult(True, data=apply_file_edits(p, edits, workspace_root=tools.workspace.root))
    except FileEditError as error:
        return ToolResult(False, code=error.code, message=str(error), data=error.details or None)


@tool("write_file", f"新建或完整覆盖 UTF-8 文件，上限 {MAX_FILE_BYTES} 字节。精确保留 content 中的 BOM 和换行，不自动继承旧格式；若已启用 edit_file，局部修改优先用它。同目录暂存完整后提交，检测执行期间的文件变化但不保护更早读取的旧内容。返回路径、创建/替换/未变化、字节数及内容指纹；新建安全提交本批仅验收 Windows。", _params(path="文件路径", content="完整内容"))
def write_file(tools, path, content):
    p = tools.workspace.resolve(path)
    if p is None:
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    try:
        return ToolResult(True, data=write_whole_file(p, content, workspace_root=tools.workspace.root))
    except FileEditError as error:
        return ToolResult(False, code=error.code, message=str(error), data=error.details or None)


def _decode_command_output(data):
    # 可替换后端也可使用 subprocess 的 text=True；不再次解码文本。
    if isinstance(data, str):
        return data.replace("\r\n", "\n").replace("\r", "\n")
    # 本机编码适配：先读原始字节，避免 Windows 的读取线程按 GBK 解码 UTF-8。
    for encoding in ("utf-8-sig", "gbk"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    # 保留原先 text=True 对换行符的统一处理。
    return text.replace("\r\n", "\n").replace("\r", "\n")


@tool("run_shell", "通过配置的执行后端运行命令。内置后端为 Windows 原生沙箱，默认 cmd（for 变量 %%），可选 Git Bash。Windows .cmd 脚本使用 cmd。",
      {"type": "object", "required": ["command"], "additionalProperties": False,
       "properties": {"command": {"type": "string", "description": "对应 Shell 语法的命令"},
                      "shell": {"type": "string", "enum": ["cmd", "bash"], "default": "cmd"},
                      "timeout": {"type": "number", "exclusiveMinimum": 0, "maximum": 300,
                                  "description": "秒，默认 30，上限 300"}}})
def run_shell(tools, command, timeout=30, shell="cmd"):
    try:
        if shell == "cmd":
            p = tools.run_command(command, tools.workspace.root, timeout)
        else:
            p = tools.run_command(command, tools.workspace.root, timeout, shell=shell)
    except subprocess.TimeoutExpired as error:
        output = _decode_command_output(error.output or b"") + _decode_command_output(error.stderr or b"")
        return ToolResult(False, code="execution_error",
                          data=output,
                          message=f"[超时] 命令执行超过 {timeout} 秒，已终止。")
    output = f"[exit {p.returncode}]\n{_decode_command_output(p.stdout)}{_decode_command_output(p.stderr)}"
    if p.returncode and shell == "bash" and "NtCreateDirectoryObject" in output and "0xC0000022" in output:
        return ToolResult(False, data=output, exit_code=p.returncode, code="shell_unavailable",
                          message="检测到 Git Bash/MSYS 在 MXC 中初始化失败；执行状态需核实，请勿盲目重试。")
    return ToolResult(p.returncode == 0, data=output, exit_code=p.returncode,
                      code="execution_error" if p.returncode else "",
                      message="命令非零退出" if p.returncode else "")


@tool("fetch_url", "抓取网页文本，url 以 http/https 开头。", _params(url="完整的 URL"))
def fetch_url(tools, url):
    denial = fetch_url_denial(url)
    if denial:
        return ToolResult(False, code="policy_denied", message=denial)
    opener = urllib.request.build_opener(_NoRedirect())
    for followed in range(MAX_FETCH_REDIRECTS + 1):
        denial = fetch_url_denial(url)
        if denial:
            return ToolResult(False, code="policy_denied", message=denial)
        req = urllib.request.Request(url, headers={"User-Agent": "nami-agent"})
        try:
            resp = opener.open(req, timeout=15)
        except HTTPError as error:
            location = (getattr(req, "_agent_redirect_url", None) or
                        error.headers.get("Location") or error.headers.get("URI"))
            redirect = error.code in (301, 302, 303, 307, 308)
            error.close()  # 重定向及错误正文都不应被无限读取。
            if not redirect or not location:
                raise
            if followed == MAX_FETCH_REDIRECTS:
                return ToolResult(False, code="output_limit_exceeded",
                                  message=f"网页重定向超过 {MAX_FETCH_REDIRECTS} 次，已停止。")
            url = urljoin(req.full_url, location)
            continue
        with resp:
            data = resp.read(MAX_FETCH_BYTES + 1)
            remaining = getattr(resp, "length", None)
            if len(data) <= MAX_FETCH_BYTES and remaining not in (None, 0):
                raise IncompleteRead(data, remaining)
        if len(data) > MAX_FETCH_BYTES:
            return ToolResult(False, code="output_limit_exceeded",
                              message=f"网页响应超过 {MAX_FETCH_BYTES} 字节，未返回不完整正文。")
        return ToolResult(True, data=data.decode("utf-8", errors="replace"))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        req._agent_redirect_url = newurl  # 保留标准库已编码、解析好的目标。
        return None


class Toolset:
    """绑定工作区与命令执行器；默认执行器始终是原生沙箱。"""
    def __init__(self, workspace, *, run_command=None, backend=None, mcp=None, task_progress=False,
                 enabled_tools=None):
        if type(task_progress) is not bool:
            raise ValueError("task_progress 必须是布尔值")
        self.task_progress = task_progress
        self.workspace = workspace
        self.file_search = FileSearch(workspace.root)
        if run_command is not None and backend is not None:
            raise ValueError("backend 与 run_command 不能同时指定")
        self._backend = None if run_command is not None else (backend if backend is not None else CodexWindowsSandboxBackend())
        self.run_command = run_command if run_command is not None else self._backend.run
        if mcp is not None and TOOLS.keys() & mcp.tools.keys():
            raise ValueError("MCP 工具名与原生工具冲突")
        self.mcp = mcp
        self._enabled_tools = None
        if enabled_tools is not None:
            if not isinstance(enabled_tools, (list, tuple)) or any(
                    not isinstance(name, str) or not name for name in enabled_tools):
                raise ValueError("enabled_tools 须为非空工具名组成的列表或元组；空列表表示不启用工具")
            if len(enabled_tools) != len(set(enabled_tools)):
                raise ValueError("enabled_tools 不能包含重复工具名")
            known = TOOLS.keys() | (mcp.tools.keys() if mcp is not None else set())
            unknown = set(enabled_tools) - known
            if unknown:
                raise ValueError("未知工具名：" + ", ".join(sorted(unknown)))
            if "update_progress" in enabled_tools and not task_progress:
                raise ValueError("启用 update_progress 需要先开启 AGENT_TASK_PROGRESS（程序调用 task_progress=True）")
            if SEARCH_TOOLS.intersection(enabled_tools) and not self.search_available:
                raise ValueError("显式启用搜索工具需要工作区外已安装的 rg；请检查依赖或调整 enabled_tools")
            self._enabled_tools = frozenset(enabled_tools)

    @property
    def enabled_tools(self):
        return self._enabled_tools

    def _lookup(self, name):
        if self._enabled_tools is not None and name not in self._enabled_tools:
            return None
        if name in TOOLS:
            if name == "update_progress" and not self.task_progress:
                return None
            if name in SEARCH_TOOLS and not self.search_available:
                return None
            return TOOLS[name]
        return self.mcp.tools.get(name) if self.mcp is not None else None

    def is_available(self, name):
        return self._lookup(name) is not None

    @property
    def search_available(self):
        return self.file_search.available

    def _uses_backend(self):
        return self._backend is not None and self.run_command == self._backend.run

    @property
    def execution_binding(self):
        if not self._uses_backend():
            runner = self.run_command
            kind = type(runner)
            module = getattr(runner, "__module__", None) or kind.__module__
            qualname = getattr(runner, "__qualname__", None) or kind.__qualname__
            identity = json.dumps([module, qualname], ensure_ascii=False).encode("utf-8")
            return {"name": "injected_runner", "policy_version": 1,
                    "identity_hash": hashlib.sha256(identity).hexdigest()}
        return deepcopy(self._backend.binding)

    def execution_info(self, name):
        if not self.is_available(name):
            return {"location": "unavailable"}
        if name == "run_shell":
            location = ("unknown" if not self._uses_backend() else
                        "sandbox" if type(self._backend) is CodexWindowsSandboxBackend else "backend")
            return {"location": location, "backend": self.execution_binding}
        if name in TOOLS:
            return {"location": "host"}
        if self.mcp is not None and name in self.mcp.tools:
            binding = getattr(self.mcp, "binding", {})
            if isinstance(binding, dict) and "url" in binding:
                return {"location": "remote_mcp" if binding["url"] is not None else "host_mcp"}
            return {"location": "unknown"}
        return {"location": "unavailable"}

    def get_schemas(self):
        names = [*TOOLS, *(self.mcp.tools if self.mcp is not None else ())]
        return [deepcopy(tool["schema"]) for name in names if (tool := self._lookup(name)) is not None]

    def validate_call(self, name, args):
        tool = self._lookup(name)
        if tool is None:
            available = sorted(item["function"]["name"] for item in self.get_schemas())
            known = name in TOOLS or (self.mcp is not None and name in self.mcp.tools)
            reason = f"本会话未启用工具 '{name}'，或其开关/依赖当前不可用" if known else f"不存在工具 '{name}'"
            return ToolResult(False, code="tool_unavailable",
                              message=reason,
                              available_tools=available)
        if not isinstance(args, dict):
            return ToolResult(False, code="invalid_arguments", message="工具参数必须是 JSON 对象")
        try:
            if "func" in tool:
                inspect.signature(tool["func"]).bind(self, **{
                    key: value for key, value in args.items() if key != "max_output_tokens"})
            json.dumps(args, allow_nan=False)
        except TypeError as error:
            return ToolResult(False, code="invalid_arguments", message=f"TypeError: {error}")
        except ValueError as error:
            return ToolResult(False, code="invalid_arguments", message=f"参数不是合法 JSON：{error}")
        validator = tool.get("validator")
        if validator is not None:
            try:
                validator.validate(args)
            except ValidationError as error:
                field = ".".join(map(str, error.absolute_path)) or "<参数对象>"
                return ToolResult(False, code="invalid_arguments",
                                  message=f"参数 {field} 不符合工具契约：{error.message}")
            except Exception as error:
                return ToolResult(False, code="execution_error",
                                  message=f"工具参数契约无法校验：{error}")
        return None

    def execute(self, name, args, *, recorded_results=()):
        """执行结果统一封装；完整数据留给记录，模型视图由上下文层截断。"""
        invalid = self.validate_call(name, args)
        if invalid is not None:
            return invalid
        try:
            if name not in TOOLS:
                return _checked_result(self.mcp.execute(name, args), structured=True)
            args = dict(args)
            output_tokens = args.pop("max_output_tokens", None)
            if output_tokens is not None and name in ("read_file", "read_tool_result"):
                args["_output_bytes"] = output_tokens * 3
            if name in ("read_tool_result", "update_progress"):
                args = {**args, "recorded_results": recorded_results}
            result = TOOLS[name]["func"](self, **args)
        except Exception as e:
            return ToolResult(False, code="execution_error", message=f"{type(e).__name__}: {e}")
        return _checked_result(result, TOOLS[name].get("structured_result", False))
