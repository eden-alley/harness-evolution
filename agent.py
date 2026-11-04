"""Agent 核心循环：依赖和运行状态均由实例持有。"""
import json

from context import truncate_tool_result, clear_old_tool_results, compact
from permissions import PermissionManager
from prompts import SYSTEM, EXIT_CHECK_SYSTEM
from token_usage import TokenUsage

MODEL = "deepseek-flash"
MAX_STEPS = 30
SOFT_LIMIT = 16000
HARD_LIMIT = 24000


class Agent:
    def __init__(self, client, tools, *, permissions=None, token_usage=None,
                 model=MODEL, max_steps=MAX_STEPS,
                 soft_limit=SOFT_LIMIT, hard_limit=HARD_LIMIT):
        self.client = client
        self.tools = tools
        self.permissions = (permissions if permissions is not None
                            else PermissionManager(workspace=tools.workspace))
        self.token_usage = token_usage if token_usage is not None else TokenUsage()
        self.model = model
        self.max_steps = max_steps
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.last_prompt_tokens = 0

    def handle_call(self, call):
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            return "Error: 参数不是合法的 JSON"
        decision = self.permissions.check(name, args)
        if not decision.allowed:
            return f"[权限拒绝] {decision.reason}"
        return self.tools.execute(name, args)

    def check_exit(self, task):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXIT_CHECK_SYSTEM},
                {"role": "user", "content": json.dumps(
                    {"task": task, "messages": self.messages}, ensure_ascii=False)},
            ],
            extra_body={"thinking": {"type": "disabled"}},
        )
        # 收到的用量先计入累计，即使随后发现判定无效；不改上下文压缩依据。
        self.token_usage.record(resp.usage, "退出检查")
        msg = resp.choices[0].message
        if msg.tool_calls:
            raise ValueError("退出检查器不允许调用工具")
        result = json.loads(msg.content or "")
        if (not isinstance(result, dict)
                or result.get("decision") not in ("completed", "continue", "blocked")
                or not isinstance(result.get("reason"), str)
                or not result["reason"].strip()):
            raise ValueError("检查结果必须包含合法的 decision 和非空字符串 reason")
        return result

    def run(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        for _ in range(self.max_steps):
            # 用上一次调用的真实 token 数，决定这一轮要不要处理
            if self.last_prompt_tokens > self.hard_limit:
                self.messages[:] = compact(
                    self.client, self.model, self.messages, self.token_usage.record)
            elif self.last_prompt_tokens > self.soft_limit:
                clear_old_tool_results(self.messages)

            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=self.tools.get_schemas(),
                # 本机模型适配：关闭默认思考模式，沿用原文的普通工具消息格式。
                extra_body={"thinking": {"type": "disabled"}},
            )
            self.last_prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
            self.token_usage.record(resp.usage, "对话")
            msg = resp.choices[0].message
            self.messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                try:
                    check = self.check_exit(user_input)
                except Exception as error:
                    return f"（退出检查失败，任务未标记完成：{type(error).__name__}: {error}）"
                if check["decision"] == "completed":
                    return msg.content or ""
                if check["decision"] == "blocked":
                    return f"（任务受阻，未完成）{check['reason']}"
                self.messages.append({"role": "user", "content":
                                 f"[退出检查反馈：继续当前任务]\n{check['reason']}"})
                continue   # 继续原来的循环，不重新调用 run()，不重置轮数。
            for call in msg.tool_calls:
                print(f"[工具] {call.function.name}({call.function.arguments[:80]})")
                result = self.handle_call(call)
                self.messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": truncate_tool_result(result)})   # 第 1 层
        return "（达到最大步数，已停止）"


if __name__ == "__main__":
    from cli import main
    main()
