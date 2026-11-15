"""模型负责推进任务；harness 负责执行、记录和可选的任务验收。"""
from collections import Counter
from copy import deepcopy
from contextlib import nullcontext
import json
import importlib.util
from pathlib import Path
import sys
from uuid import uuid4

from context import (compact, render_tool_result, estimate_input_tokens, validate_context_limits,
                     ContextBudgetError, CONTEXT_INPUT_BUDGET, COMPACT_THRESHOLD,
                     COMPACT_TARGET_MIN, COMPACT_TARGET_MAX)
from permissions import PermissionManager
from prompts import SYSTEM, TASK_PROGRESS_SYSTEM
from token_usage import TokenUsage
from tools import ToolResult, TOOLS
from verification import CheckResult, VerificationInput

MODEL = "deepseek-flash"
MAX_STEPS = 30
SOFT_LIMIT = COMPACT_THRESHOLD
HARD_LIMIT = CONTEXT_INPUT_BUDGET

# 每份 Agent 源码绑定自己的模型模块；兼容只复制根目录 *.py 的版本加载器。
_model_spec = importlib.util.spec_from_file_location(
    f'{__name__}._models_{uuid4().hex}', Path(__file__).with_name('models.py'),
    submodule_search_locations=[str(Path(__file__).parent)])
models = importlib.util.module_from_spec(_model_spec)
sys.modules[_model_spec.name] = models
_model_spec.loader.exec_module(models)
_state = importlib.import_module(f'{models.__name__}.state')
checkpoints = importlib.import_module(f'{models.__name__}.checkpoint')
limits = importlib.import_module(f'{models.__name__}.limits')
_tool_runner = importlib.import_module(f'{models.__name__}.tool_runner')
RunResult, RunState = _state.RunResult, _state.RunState


class _ToolRuntimeView:
    """Read current Agent dependencies without exposing its model or scheduling API."""
    __slots__ = ('_owner',)

    def __init__(self, owner):
        self._owner = owner

    @property
    def tools(self):
        return self._owner.tools

    @property
    def permissions(self):
        return self._owner.permissions

    @property
    def run_state(self):
        return self._owner.run_state

    @property
    def messages(self):
        return self._owner.messages

    def _register_call(self, call, source):
        return self._owner._register_call(call, source)

    def _emit(self, kind, **data):
        return self._owner._emit(kind, **data)

    def _save_checkpoint(self):
        return self._owner._save_checkpoint()


class Agent:
    def __init__(self, client, tools, *, permissions=None, token_usage=None,
                 model=MODEL, max_steps=MAX_STEPS, soft_limit=SOFT_LIMIT,
                 hard_limit=HARD_LIMIT, event_sink=None, model_adapter=None, checkpoint_store=None,
                 token_limit=0, wall_time_limit_seconds=0,
                 progress_report=None, progress_test=None,
                 compact_target_min=COMPACT_TARGET_MIN, compact_target_max=COMPACT_TARGET_MAX,
                 after_tool=None):
        """after_tool observes immutable summaries after the result-save point.

        It is optional trusted Python code, runs synchronously in the current
        lease, and must return quickly without mutating or reentering this Agent.
        Ordinary failures are isolated; recorded operations are not notified
        again on resume. No checkpoint store means memory recording only.
        """
        limits.validate(token_limit, wall_time_limit_seconds)
        validate_context_limits(soft_limit, hard_limit, compact_target_min, compact_target_max)
        self.client = client
        self._model_adapter = model_adapter
        self.tools = tools
        self.permissions = (permissions if permissions is not None
                            else PermissionManager(workspace=tools.workspace))
        self.token_usage = token_usage if token_usage is not None else TokenUsage()
        self.model = model
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.wall_time_limit_seconds = wall_time_limit_seconds
        self.progress_report = progress_report
        self.progress_test = progress_test
        if progress_report and tools.workspace.resolve(progress_report) is None:
            raise ValueError("进展检查的报告路径须在工作区内。")
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.compact_target_min = compact_target_min
        self.compact_target_max = compact_target_max
        self.context_scale = 1.0
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.last_prompt_tokens = 0
        self.event_sink = event_sink
        self.after_tool = after_tool
        self.run_state = RunState()
        self.checkpoint_store = checkpoint_store
        self._checkpoint_binding = None
        self._checkpoint_session_id = None
        self.tool_runner = _tool_runner.ToolRunner()
        self._tool_runtime = _ToolRuntimeView(self)
        self._usage_owner = uuid4().hex

    @property
    def model_adapter(self):
        adapter = self._model_adapter
        # 旧入口始终使用当前注入的客户端，保留预算代理和运行前替换 client 的行为。
        if adapter is None:
            adapter = models.get_model(models.ModelConfig(model_name=self.model), client=self.client)
        if self.token_limit or self.wall_time_limit_seconds:
            return limits.LimitedModel(adapter, self._before_model_request, self._after_model_request)
        return adapter

    def _before_model_request(self):
        if self.run_state.phase in ('preparing', 'model', 'tools', 'verifying'):
            limits.check(self.run_state, self.token_limit, self.wall_time_limit_seconds)

    def _after_model_request(self, usage):
        if self.run_state.phase in ('preparing', 'model', 'tools', 'verifying'):
            limits.record(self.run_state, usage)

    @property
    def events(self):
        return self.run_state.events

    def _emit(self, kind, **data):
        self.run_state.last_event_id += 1
        event = deepcopy({"run_id": self.run_state.run_id, "event_id": self.run_state.last_event_id,
                          "type": kind, **data})
        self.events.append(event)
        if self.event_sink is not None:
            self.event_sink(event)

    def _save_checkpoint(self):
        if self.checkpoint_store is not None:
            try:
                self.checkpoint_store.save(checkpoints.capture(
                    self, self._checkpoint_session_id, self._checkpoint_binding))
            except OSError as error:
                raise checkpoints.CheckpointError(f'检查点保存失败：{error}') from error

    def _note_interruption(self, reason):
        if self.checkpoint_store is not None:
            try:
                with self.checkpoint_store.lease():
                    saved = self.checkpoint_store.load()
                    if saved and saved['status'] == 'run' and saved['run_state']['run_id'] == self.run_state.run_id:
                        saved['interruption'] = reason
                        self.checkpoint_store.save(saved)
            except (checkpoints.CheckpointError, OSError):
                pass  # 诊断写入失败不覆盖旧快照，也不启动任何操作。

    def _checkpoint_failure(self, error):
        state = self.run_state
        if state.result is not None and self.messages[-1].get('content') == state.result.text:
            self.messages.pop()
        return self._finish(f'（检查点失败，已停止后续操作：{error}）', 'error',
                            state.verification, checkpoint_final=False)

    def _register_call(self, call, source):
        operation_id = uuid4().hex
        self.run_state.call_records[operation_id] = {
            'call': deepcopy(call), 'source': source, 'status': 'pending', 'result': None}
        return operation_id

    def _recorded_model_results(self):
        return self.tool_runner.recorded_model_results(self._tool_runtime)

    def _invoke_tool(self, name, args, operation_id=None):
        return self.tool_runner.invoke(
            self._tool_runtime, name, args, operation_id,
            recorded_model_results=lambda: self._recorded_model_results())

    def handle_call(self, call, *, source="model", operation_id=None):
        return self.tool_runner.handle_call(
            self._tool_runtime, call, source=source, operation_id=operation_id,
            invoke_tool=lambda name, args, operation_id=None: self._invoke_tool(name, args, operation_id),
            after_tool=self.after_tool)

    def _verification_tool(self, name, args):
        call = {'id': f'verification-{uuid4().hex}',
                'function': {'name': name, 'arguments': json.dumps(args)}}
        return self.handle_call(call, source="verification")

    def _verify(self, verifier, task, candidate):
        context = VerificationInput(
            task=task, candidate=candidate, tool_results=deepcopy(self.run_state.tool_results),
            available_tools=[item["function"]["name"] for item in self.tools.get_schemas()],
            run_tool=self._verification_tool)
        try:
            result = verifier(context)
            if not isinstance(result, CheckResult):
                raise ValueError("验收器必须返回 CheckResult")
        except (checkpoints.CheckpointError, limits.LimitReached):
            raise
        except Exception as error:
            result = CheckResult("error", f"{type(error).__name__}: {error}")
        self.run_state.verification = result
        self._emit("verification", status=result.status, reason=result.reason,
                   evidence=result.evidence)
        return result

    def _finish(self, text, stop_reason, verification=None, *, checkpoint_final=True):
        state = self.run_state
        # 发生中断时也补齐消息配对，不能把未知执行状态写成成功。
        unfinished = []
        for call_id, call in state.pending_calls.items():
            result = ToolResult(False, code="execution_error",
                                message="本轮中断，工具执行状态未确认；请先核实已有结果，勿盲目重试。")
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": render_tool_result(result)})
            evidence = {"call_id": call_id, "tool": call["function"]["name"],
                        "source": "model", **result.to_dict()}
            state.tool_results.append(deepcopy(evidence))
            unfinished.append(evidence)
        state.pending_calls.clear()
        if verification is not None:
            self.messages.append({"role": "system", "content": "本轮任务验收记录：" + json.dumps(
                {"status": verification.status, "reason": verification.reason,
                 "evidence": verification.evidence}, ensure_ascii=False)})
        self.messages.append({"role": "assistant", "content": text})
        state.phase = "stopped"
        state.verification = verification
        state.result = RunResult(text, stop_reason, verification)
        try:
            for evidence in unfinished:
                self._emit("tool_result", **evidence)
            self._emit("final_response", text=text, stop_reason=stop_reason,
                       usage=deepcopy(self.token_usage.totals))
        except OSError as error:
            text += f"\n（执行记录保存失败：{error}）"
            self.messages[-1]["content"] = text
            state.result = RunResult(text, "error", verification)
        if checkpoint_final and all(r['status'] == 'recorded' for r in state.call_records.values()):
            state.next_action = 'finished'
            state.request_inflight = None
            self._save_checkpoint()
        else:
            self._note_interruption(text)
        return state.result

    def run(self, user_input, verifier=None):
        try:
            with self.checkpoint_store.lease() if self.checkpoint_store is not None else nullcontext():
                if self.checkpoint_store is not None:
                    saved = self.checkpoint_store.load()
                    if self.checkpoint_store.pending(saved):
                        return RunResult('存在未完成检查点，请先 /resume 或 /new。', 'error')
                    self._checkpoint_session_id = saved['session_id'] if saved else uuid4().hex
                    self._checkpoint_binding = checkpoints.binding(self)
                self.run_state = RunState(run_id=uuid4().hex, user_input=user_input,
                                          phase='preparing', requires_verifier=verifier is not None,
                                          limit_started_at=limits.clock() if self.wall_time_limit_seconds else None)
                self.messages.append({"role": "user", "content": user_input})
                return self._run_loop(verifier)
        except KeyboardInterrupt:
            self.run_state.phase = "interrupted"
            self.run_state.result = None
            self._note_interruption('用户中断，执行现场已保留。')
            raise
        except checkpoints.CheckpointError as error:
            return RunResult(str(error), 'error')

    def inspect_checkpoint(self, snapshot=None):
        try:
            saved = snapshot if snapshot is not None else self.checkpoint_store.load()
            if not saved or saved['status'] == 'idle':
                return {'status': 'none', 'reason': '没有可恢复检查点'}
            checkpoints.validate(saved)
            if saved['binding'] != checkpoints.binding(self):
                return {'status': 'blocked', 'reason': '检查点源码、工作区、工具、模型或额度配置不匹配'}
            state = saved['run_state']
            if any(r['status'] == 'executing' for r in state['call_records'].values()):
                return {'status': 'blocked', 'reason': '工具执行状态未知，禁止重放；请保留现场或 /new'}
            if state['next_action'] == 'finished':
                return {'status': 'finished', 'reason': '任务已停止，只需同步会话'}
            if state['requires_verifier']:
                return {'status': 'blocked', 'reason': '自定义验收任务暂不支持跨进程续跑'}
            return {'status': 'ready', 'reason': '可通过 /resume 显式继续'}
        except (checkpoints.CheckpointError, ValueError, TypeError, KeyError, OSError) as error:
            return {'status': 'blocked', 'reason': str(error)}

    def resume(self, snapshot=None):
        if self.checkpoint_store is None:
            return RunResult('未配置检查点存储', 'error')
        try:
            with self.checkpoint_store.lease():
                saved = self.checkpoint_store.load()
                if snapshot is not None and snapshot != saved:
                    return RunResult('检查点已变化，不能恢复旧副本', 'error')
                decision = self.inspect_checkpoint(saved)
                if decision['status'] not in ('ready', 'finished'):
                    return RunResult(decision['reason'], 'error')
                state = checkpoints.restore_state(saved['run_state'])
                state.last_event_id = max(state.last_event_id,
                    self.checkpoint_store.trace_event_floor(state.run_id))
                self.run_state, self.messages = state, deepcopy(saved['messages'])
                self.last_prompt_tokens = saved['last_prompt_tokens']
                self.context_scale = saved.get('context_scale', 1.0)
                self._checkpoint_session_id = saved['session_id']
                self._checkpoint_binding = saved['binding']
                if decision['status'] == 'finished':
                    return state.result
                if state.request_inflight is not None:
                    state.unknown_usage.append({'source': state.request_inflight,
                                                'steps_used': state.steps_used})
                    if self.token_limit:
                        state.limit_usage_incomplete = True
                    if state.request_inflight == 'model':
                        state.next_action = 'prepare'  # 已消耗的执行轮数不退还。
                    state.request_inflight = None
                state.phase, state.result = 'preparing', None
                return self._run_loop(None, resumed=True)
        except KeyboardInterrupt:
            self.run_state.phase, self.run_state.result = 'interrupted', None
            self._note_interruption('恢复过程被用户中断。')
            raise
        except (checkpoints.CheckpointError, ValueError, TypeError, KeyError, OSError) as error:
            return RunResult(f'检查点恢复失败：{error}', 'error')

    def _task_progress_context(self):
        if not self.tools.is_available("update_progress"):
            return []
        # 进度正文已作为工具结果进入历史，此处只附加固定指导语。
        return [{"role": "system", "content": TASK_PROGRESS_SYSTEM}]

    def _progress_reminder(self):
        if not (self.progress_report or self.progress_test):
            return None
        state = self.run_state
        for event in state.events:
            if event["type"] == "progress_reminder":
                return event["text"] if event["step"] == state.steps_used else None

        records = [record for record in state.call_records.values()
                   if record["source"] == "model" and record["status"] == "recorded"]
        recent = records[-6:]
        read_tools = {"read_file", "list_dir", "read_tool_result"}
        if len(recent) < 6 or any(not r["result"]["ok"] or r["result"]["tool"] not in read_tools for r in recent):
            return None
        reads = Counter((r["result"]["tool"],
                         json.dumps(json.loads(r["call"]["function"]["arguments"] or "{}"), sort_keys=True))
                        for r in recent)
        if max(reads.values()) < 3:
            return None

        outstanding = []
        if self.progress_report:
            path = self.tools.workspace.resolve(self.progress_report)
            if path is not None and path.is_file():
                return None
            outstanding.append(f"交付文件 {self.progress_report} 尚未出现")
        if self.progress_test:
            if any(r["result"].get("exit_code") == 0 and r["result"]["ok"]
                   and r["result"]["tool"] == "run_shell"
                   and json.loads(r["call"]["function"]["arguments"]).get("command") == self.progress_test
                   for r in records):
                return None
            outstanding.append(f"指定测试尚无成功记录（{self.progress_test}）")

        text = "近期有多次相同查阅，且还未看到约定的进展：" + "，".join(outstanding)
        text += "。请检查已有证据，判断接下来需要完成哪项交付或补充哪条信息。"
        self._emit("progress_reminder", step=state.steps_used, text=text)
        return text

    def _run_loop(self, verifier, *, resumed=False):
        state = self.run_state
        try:
            self._emit('task_resumed' if resumed else 'task_started', task=state.user_input)
            self._save_checkpoint()
            model = self.model_adapter
            while True:
                if state.next_action == 'prepare':
                    if self.max_steps is not None and state.steps_used >= self.max_steps:
                        return self._finish('（达到最大步数，已停止）', 'max_steps', state.verification)
                    state.steps_used += 1
                    state.phase = 'preparing'
                    estimated = estimate_input_tokens(self.messages + self._task_progress_context(),
                                                      self.tools.get_schemas(), scale=self.context_scale)
                    if max(estimated, self.last_prompt_tokens) >= self.soft_limit:
                        state.next_action = 'summary'
                    else:
                        state.next_action = 'request'
                    self._save_checkpoint()
                if state.next_action == 'summary':
                    def before_summary():
                        state.request_inflight = 'summary'
                        self._save_checkpoint()
                    progress_context = self._task_progress_context()
                    before = estimate_input_tokens(self.messages + progress_context, self.tools.get_schemas())
                    candidate = compact(model, self.messages, self.token_usage.record,
                                        before_query=before_summary, tools=self.tools.get_schemas(),
                                        budget=self.hard_limit, target_min=self.compact_target_min,
                                        target_max=self.compact_target_max, scale=self.context_scale,
                                        recorded_results=state.tool_results, extra_context=progress_context)
                    self.messages[:] = candidate
                    after = estimate_input_tokens(self.messages + progress_context, self.tools.get_schemas())
                    self._emit("context_compacted", before_tokens=before, after_tokens=after,
                               estimated=True, target_min=self.compact_target_min,
                               target_max=self.compact_target_max)
                    self.last_prompt_tokens = 0
                    state.request_inflight, state.next_action = None, 'request'
                    self._save_checkpoint()
                if state.next_action == 'request':
                    state.phase, state.request_inflight = 'model', 'model'
                    reminder = self._progress_reminder()
                    self._save_checkpoint()
                    messages = self.messages + self._task_progress_context()
                    if reminder is not None:
                        messages = [*messages, {"role": "system", "content": reminder}]
                    input_estimate = estimate_input_tokens(messages, self.tools.get_schemas())
                    if estimate_input_tokens(messages, self.tools.get_schemas(), scale=self.context_scale) > self.hard_limit:
                        raise ContextBudgetError("本次请求超过输入预算，已保留执行现场。")
                    response = model.query(messages, tools=self.tools.get_schemas())
                    self.last_prompt_tokens = (response.usage.prompt_tokens or 0) if response.usage else 0
                    if self.last_prompt_tokens:
                        self.context_scale = max(1.0, self.last_prompt_tokens / input_estimate)
                    self.token_usage.record(response.usage, '对话')
                    state.response = {'message': response.message, 'protocol_error': response.protocol_error}
                    state.request_inflight, state.next_action = None, 'response'
                    self._emit('model_response', response=response.raw)
                    self._save_checkpoint()
                if state.next_action == 'response':
                    if state.response['protocol_error']:
                        return self._finish('（模型响应无效）' + state.response['protocol_error'], 'error')
                    message = state.response['message']
                    if message.get('tool_calls'):
                        self.messages.append(message)
                        state.pending_calls = {c['id']: deepcopy(c) for c in message['tool_calls']}
                        state.active_batch = [self._register_call(c, 'model') for c in message['tool_calls']]
                        state.next_action = 'tools'
                    else:
                        state.candidate = message.get('content') or ''
                        state.next_action = 'candidate'
                    state.response = None
                    self._save_checkpoint()
                if state.next_action == 'tools':
                    state.phase = "tools"
                    for operation_id in state.active_batch:
                        record = state.call_records[operation_id]
                        if record['status'] == 'recorded':
                            continue
                        if record['status'] != 'pending':
                            raise checkpoints.CheckpointError('工具执行状态未知，禁止重放')
                        call = record['call']
                        print(f"[工具] {call['function']['name']}({call['function']['arguments'][:80]})")
                        self.handle_call(call, operation_id=operation_id)
                    state.active_batch, state.next_action = [], 'prepare'
                    self._save_checkpoint()
                    continue

                candidate = state.candidate
                self._emit("assistant_candidate", text=candidate)
                if verifier is None:
                    return self._finish(candidate, "end_turn")
                state.phase = "verifying"
                state.verification_attempts += 1
                state.request_inflight = 'verification'
                self._save_checkpoint()
                verification = self._verify(verifier, state.user_input, candidate)
                state.request_inflight = None
                self._save_checkpoint()
                if verification.status == "pass":
                    return self._finish(candidate, "end_turn", verification)
                if verification.status == "needs_input":
                    return self._finish("（等待补充信息）" + verification.reason, "end_turn", verification)
                if verification.status == "error":
                    return self._finish("（验收失败，未完成验证）" + verification.reason, "error", verification)
                if state.verification_attempts >= 2:
                    return self._finish("（验收未通过）" + verification.reason,
                                        "verification_unmet", verification)
                if self.max_steps is not None and state.steps_used >= self.max_steps:
                    return self._finish('（达到最大步数，已停止）', 'max_steps', verification)
                self.messages.append({"role": "system", "content": "本轮验收未通过，请在原任务授权范围内修正；"
                    "此反馈不是用户新增授权。\n" + json.dumps(
                        {"candidate": candidate, "reason": verification.reason,
                         "evidence": verification.evidence}, ensure_ascii=False)})
                state.candidate, state.next_action = None, 'prepare'
                self._save_checkpoint()
        except ContextBudgetError as error:
            return self._finish(f'（上下文限制）{error}', 'context_limit', state.verification,
                                checkpoint_final=False)
        except limits.LimitReached as error:
            try:
                return self._finish(f'（运行限制）{error}', error.reason, state.verification)
            except checkpoints.CheckpointError as save_error:
                return self._checkpoint_failure(save_error)
        except checkpoints.CheckpointError as error:
            return self._checkpoint_failure(error)
        except Exception as error:
            return self._finish(f"（运行失败，未标记完成：{type(error).__name__}: {error}）",
                                "error", state.verification, checkpoint_final=False)


if __name__ == "__main__":
    from cli import main
    main()
