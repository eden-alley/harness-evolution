"""Single-call execution; scheduling, recovery and finalization stay in Agent.

after_tool is a trusted, synchronous observation callback after the existing
result-save point returns. Without a checkpoint store this means memory only.
Delivery is best-effort / at-most-once within the Agent-managed call lifecycle:
ordinary failures are isolated, there are no retries, and recorded operations
are not notified again on resume. A crash between saving and notifying may lose
the notification; this is not a reliable delivery mechanism for business work.

Callbacks must return quickly, must not modify Agent internals, and must not
reenter the same Agent's run/resume. They run inside the current execution lease
when one is held; no thread, process or security isolation is provided.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from typing import Any, Callable, Protocol, TYPE_CHECKING

from context import render_tool_result
from tools import ToolResult, TOOLS

if TYPE_CHECKING:
    from permissions import PermissionManager
    from tools import Toolset
    from .state import RunState


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolResultSummary:
    """Scalar observation fields only; ok is not independent task acceptance."""
    run_id: str | None
    operation_id: str
    call_id: str
    tool: str
    source: str
    ok: bool
    code: str
    exit_code: int | None
    result_contract: str

    def __post_init__(self):
        # A legal str subclass may carry mutable attributes or runtime references.
        for name in ('operation_id', 'call_id', 'tool', 'source', 'code', 'result_contract'):
            object.__setattr__(self, name, str.__str__(getattr(self, name)))
        if self.run_id is not None:
            object.__setattr__(self, 'run_id', str.__str__(self.run_id))


class ToolRuntime(Protocol):
    """Internal execution view, not a user-hook context or a security boundary."""
    @property
    def tools(self) -> Toolset: ...

    @property
    def permissions(self) -> PermissionManager: ...

    @property
    def run_state(self) -> RunState: ...

    @property
    def messages(self) -> list[dict]: ...

    def _register_call(self, call: dict, source: str) -> str: ...

    def _emit(self, kind: str, **data: Any) -> None: ...

    def _save_checkpoint(self) -> None: ...


class ToolRunner:
    def recorded_model_results(self, runtime: ToolRuntime):
        by_id = {}
        for result in runtime.run_state.tool_results:
            by_id.setdefault(result["call_id"], []).append(result)
        return [record["result"] for record in runtime.run_state.call_records.values()
                if record["source"] == "model" and record["status"] == "recorded"
                and record["result"] in by_id.get(record["call"]["id"], ())]

    def invoke(self, runtime: ToolRuntime, name, args, operation_id=None, *,
               recorded_model_results: Callable | None = None):
        args = deepcopy(args)
        invalid = runtime.tools.validate_call(name, args)
        if invalid is not None:
            return invalid
        try:
            approval_args = deepcopy(args)
            decision = runtime.permissions.check(name, approval_args)
        except (TypeError, ValueError) as error:
            return ToolResult(False, code="invalid_arguments", message=f"{type(error).__name__}: {error}")
        if not decision.allowed:
            return ToolResult(False, code=decision.code, message=decision.reason)
        if approval_args != args:
            return ToolResult(False, code="execution_error", message="权限组件修改了工具参数，未执行操作。")
        if operation_id is not None:
            runtime.run_state.call_records[operation_id]['status'] = 'executing'
            runtime._save_checkpoint()
        if name == "update_progress":
            records = (recorded_model_results() if recorded_model_results is not None
                       else self.recorded_model_results(runtime))
            return runtime.tools.execute(name, args, recorded_results=records)
        if name == "read_tool_result":
            return runtime.tools.execute(name, args, recorded_results=runtime.run_state.tool_results)
        return runtime.tools.execute(name, args)

    def handle_call(self, runtime: ToolRuntime, call, *, source="model", operation_id=None,
                    invoke_tool: Callable,
                    after_tool: Callable[[ToolResultSummary], object] | None = None):
        operation_id = operation_id or runtime._register_call(call, source)
        name = call['function']['name']
        arguments = call['function']['arguments']
        runtime._emit("tool_call", call_id=call['id'], tool=name,
                      arguments=arguments, source=source, execution=runtime.tools.execution_info(name))
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            result = ToolResult(False, code="invalid_arguments", message="参数不是合法的 JSON")
        else:
            result = invoke_tool(name, args, operation_id)
        evidence = {"call_id": call['id'], "tool": name, "source": source, **result.to_dict()}
        runtime.run_state.tool_results.append(deepcopy(evidence))
        record = runtime.run_state.call_records[operation_id]
        record.update(status='recorded', result=deepcopy(evidence))
        if source == 'model' and call['id'] in runtime.run_state.pending_calls:
            try:
                output_tokens = json.loads(arguments or "{}").get("max_output_tokens")
            except (ValueError, AttributeError):
                output_tokens = None
            if name not in TOOLS or type(output_tokens) is not int or not 1 <= output_tokens <= 4096:
                output_tokens = None
            runtime.messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                     'content': render_tool_result(result, call_id=call['id'],
                                                                   max_output_tokens=output_tokens)})
            runtime.run_state.pending_calls.pop(call['id'])
        runtime._emit("tool_result", **evidence)
        runtime._save_checkpoint()
        if after_tool is not None:
            try:
                after_tool(ToolResultSummary(
                    run_id=runtime.run_state.run_id, operation_id=operation_id,
                    call_id=call['id'], tool=name, source=source, ok=result.ok,
                    code=result.code, exit_code=result.exit_code,
                    result_contract=result.result_contract))
            except Exception as error:
                try:
                    logger.warning("after_tool callback failed: operation_id=%s error=%s",
                                   operation_id, type(error).__name__)
                except Exception:
                    pass  # Optional diagnostics must not change the recorded result.
        return result
