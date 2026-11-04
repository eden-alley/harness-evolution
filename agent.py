"""模型负责推进任务；harness 负责执行、记录和可选的任务验收。"""
from copy import deepcopy
from dataclasses import dataclass
import json
from types import SimpleNamespace
from uuid import uuid4

from context import clear_old_tool_results, compact, render_tool_result
from permissions import PermissionManager
from prompts import SYSTEM
from token_usage import TokenUsage
from tools import ToolResult
from verification import CheckResult, VerificationInput

MODEL = "deepseek-flash"
MAX_STEPS = 30
SOFT_LIMIT = 16000
HARD_LIMIT = 24000


@dataclass(frozen=True)
class RunResult:
    text: str
    stop_reason: str
    verification: CheckResult | None = None


class Agent:
    def __init__(self, client, tools, *, permissions=None, token_usage=None,
                 model=MODEL, max_steps=MAX_STEPS, soft_limit=SOFT_LIMIT,
                 hard_limit=HARD_LIMIT, event_sink=None):
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
        self.event_sink = event_sink
        self.events = []
        self._tool_results = []
        self._pending_calls = {}
        self._run_id = None

    def _emit(self, kind, **data):
        event = deepcopy({"run_id": self._run_id, "event_id": len(self.events) + 1,
                          "type": kind, **data})
        self.events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def _invoke_tool(self, name, args):
        invalid = self.tools.validate_call(name, args)
        if invalid is not None:
            return invalid
        try:
            decision = self.permissions.check(name, args)
        except (TypeError, ValueError) as error:
            return ToolResult(False, code="invalid_arguments", message=f"{type(error).__name__}: {error}")
        if not decision.allowed:
            return ToolResult(False, code=decision.code, message=decision.reason)
        return self.tools.execute(name, args)

    def handle_call(self, call, *, source="model"):
        name = call.function.name
        self._emit("tool_call", call_id=call.id, tool=name,
                   arguments=call.function.arguments, source=source)
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            result = ToolResult(False, code="invalid_arguments", message="参数不是合法的 JSON")
        else:
            result = self._invoke_tool(name, args)
        evidence = {"call_id": call.id, "tool": name, "source": source, **result.to_dict()}
        self._tool_results.append(deepcopy(evidence))
        self._emit("tool_result", **evidence)
        return result

    def _verification_tool(self, name, args):
        call = SimpleNamespace(id=f"verification-{uuid4().hex}",
                               function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        return self.handle_call(call, source="verification")

    def _verify(self, verifier, task, candidate):
        context = VerificationInput(
            task=task, candidate=candidate, tool_results=deepcopy(self._tool_results),
            available_tools=[item["function"]["name"] for item in self.tools.get_schemas()],
            run_tool=self._verification_tool)
        try:
            result = verifier(context)
            if not isinstance(result, CheckResult):
                raise ValueError("验收器必须返回 CheckResult")
        except Exception as error:
            result = CheckResult("error", f"{type(error).__name__}: {error}")
        self._emit("verification", status=result.status, reason=result.reason,
                   evidence=result.evidence)
        return result

    def _finish(self, text, stop_reason, verification=None):
        # 发生中断时也补齐消息配对，不能把未知执行状态写成成功。
        unfinished = []
        for call_id, call in self._pending_calls.items():
            result = ToolResult(False, code="execution_error",
                                message="本轮中断，工具执行状态未确认；请先核实已有结果，勿盲目重试。")
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": render_tool_result(result)})
            evidence = {"call_id": call_id, "tool": call.function.name,
                        "source": "model", **result.to_dict()}
            self._tool_results.append(deepcopy(evidence))
            unfinished.append(evidence)
        self._pending_calls.clear()
        if verification is not None:
            self.messages.append({"role": "system", "content": "本轮任务验收记录：" + json.dumps(
                {"status": verification.status, "reason": verification.reason,
                 "evidence": verification.evidence}, ensure_ascii=False)})
        self.messages.append({"role": "assistant", "content": text})
        result = RunResult(text, stop_reason, verification)
        try:
            for evidence in unfinished:
                self._emit("tool_result", **evidence)
            self._emit("final_response", text=text, stop_reason=stop_reason,
                       usage=deepcopy(self.token_usage.totals))
        except OSError as error:
            text += f"\n（执行记录保存失败：{error}）"
            self.messages[-1]["content"] = text
            result = RunResult(text, "error", verification)
        return result

    def run(self, user_input, verifier=None):
        self._run_id = uuid4().hex
        self.events = []
        self._tool_results = []
        self._pending_calls = {}
        self.messages.append({"role": "user", "content": user_input})
        verification = None
        verification_attempts = 0
        try:
            self._emit("task_started", task=user_input)
            for step in range(self.max_steps):
                if self.last_prompt_tokens > self.hard_limit:
                    self.messages[:] = compact(
                        self.client, self.model, self.messages, self.token_usage.record)
                elif self.last_prompt_tokens > self.soft_limit:
                    clear_old_tool_results(self.messages)
                response = self.client.chat.completions.create(
                    model=self.model, messages=self.messages, tools=self.tools.get_schemas(),
                    extra_body={"thinking": {"type": "disabled"}})
                self.last_prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                self.token_usage.record(response.usage, "对话")
                message = response.choices[0].message
                self._emit("model_response", response=response.model_dump(exclude_none=True))
                if message.tool_calls:
                    self.messages.append(message.model_dump(exclude_none=True))
                    self._pending_calls = {call.id: call for call in message.tool_calls}
                    for call in message.tool_calls:
                        print(f"[工具] {call.function.name}({call.function.arguments[:80]})")
                        result = self.handle_call(call)
                        self.messages.append({"role": "tool", "tool_call_id": call.id,
                                              "content": render_tool_result(result)})
                        self._pending_calls.pop(call.id)
                    continue

                candidate = message.content or ""
                self._emit("assistant_candidate", text=candidate)
                if verifier is None:
                    return self._finish(candidate, "end_turn")
                verification_attempts += 1
                verification = self._verify(verifier, user_input, candidate)
                if verification.status == "pass":
                    return self._finish(candidate, "end_turn", verification)
                if verification.status == "needs_input":
                    return self._finish("（等待补充信息）" + verification.reason, "end_turn", verification)
                if verification.status == "error":
                    return self._finish("（验收失败，未完成验证）" + verification.reason, "error", verification)
                if verification_attempts >= 2:
                    return self._finish("（验收未通过）" + verification.reason,
                                        "verification_unmet", verification)
                if step + 1 >= self.max_steps:
                    break
                self.messages.append({"role": "system", "content": "本轮验收未通过，请在原任务授权范围内修正；"
                    "此反馈不是用户新增授权。\n" + json.dumps(
                        {"candidate": candidate, "reason": verification.reason,
                         "evidence": verification.evidence}, ensure_ascii=False)})
            return self._finish("（达到最大步数，已停止）", "max_steps", verification)
        except Exception as error:
            return self._finish(f"（运行失败，未标记完成：{type(error).__name__}: {error}）",
                                "error", verification)


if __name__ == "__main__":
    from cli import main
    main()
