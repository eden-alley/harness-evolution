"""任务进度：真实循环与本地模型响应，不调用真实 API。"""
from test_file_reads import result_page
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import contextlib
import inspect
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, cli, context, permissions, tools, answer, call, tool_reply
from verification import CheckResult


def document(*, next_action="核对剩余分支，再撰写报告", call_ids=()):
    return {
        "deliverables": ["定位调用链并交付报告"],
        "findings": ([{"text": "已定位入口", "call_ids": list(call_ids)}] if call_ids else []),
        "remaining": ["核对剩余分支", "交付报告"],
        "next_action": next_action,
    }


def current_views(request):
    return [m for m in request["messages"]
            if m["role"] == "user" and (m.get("content") or "").startswith("[当前任务进度]")]


class TaskProgressScenarios(OfflineScenario):
    def enable(self):
        self.assertIn("task_progress", inspect.signature(tools.Toolset).parameters,
                      "Toolset 尚未实现默认关闭的任务进度选项")
        self.tools = tools.Toolset(self.tools.workspace, run_command=self.fixture_command, task_progress=True)
        self.agent.tools = self.tools

    def test_updates_stay_in_tool_history_and_preserve_permissions_usage_and_exit(self):
        self.enable()
        (self.workspace / "source.txt").write_text("entry", encoding="utf-8")
        first = document(call_ids=["read"])
        second = document(next_action="因写入被拒，向用户说明尚欠交付", call_ids=["read", "write"])
        asked = []
        policy = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or "n", "保留文件"))
        self.model([
            (tool_reply(call("read", "read_file", {"path": "source.txt"})), 10),
            (tool_reply(call("p1", "update_progress", first),
                        call("write", "write_file", {"path": "report.md", "content": "report"})), 20),
            (tool_reply(call("p2", "update_progress", second)), 30),
            (answer("写入被拒，报告尚未交付。"), 40),
        ])
        self.run_agent("定位并交付报告", policy)
        self.assertEqual(asked, ["write_file"])
        self.assertFalse((self.workspace / "report.md").exists())
        self.assertEqual(len(self.requests), 4)
        self.assertTrue(all(not current_views(request) for request in self.requests))
        for request, expected in ((self.requests[2], {"p1": first}),
                                  (self.requests[3], {"p1": first, "p2": second})):
            for call_id, value in expected.items():
                replies = [m for m in request["messages"]
                           if m["role"] == "tool" and m.get("tool_call_id") == call_id]
                self.assertEqual(len(replies), 1)
                self.assertEqual(json.loads(replies[0]["content"])["data"], value)
        self.assertFalse(any((m.get("content") or "").startswith("[当前任务进度]")
                             for m in self.agent.messages))
        state = self.agent.run_state
        self.assertEqual(state.user_input, "定位并交付报告")
        self.assertEqual([r["call_id"] for r in state.tool_results], ["read", "p1", "write", "p2"])
        self.assertEqual(state.tool_results[-1]["data"], second)
        self.assertEqual(state.steps_used, 4)
        self.assertEqual(self.last_result.stop_reason, "end_turn")
        self.assertIsNone(self.last_result.verification)
        self.assertEqual(self.agent.token_usage.totals,
                         {"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104})

    def test_failed_updates_keep_previous_record_and_cannot_use_fake_or_progress_evidence(self):
        self.enable()
        boundary = document(next_action="x")
        size = len(json.dumps(boundary, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        boundary["next_action"] += "x" * (3000 - size)
        self.assertTrue(self.tools.execute("update_progress", boundary).ok)
        self.assertFalse(self.tools.execute("update_progress",
                         {**boundary, "next_action": boundary["next_action"] + "x"}).ok)
        valid = document()
        bad_ids = document(call_ids=["other-run"])
        progress_evidence = document(call_ids=["good"])
        oversized = document(next_action="中" * 1100)
        malformed = {**document(), "deliverables": []}
        blank = document(next_action="   ")
        self.model([
            tool_reply(call("good", "update_progress", valid)),
            tool_reply(*(call(str(i), "update_progress", value)
                         for i, value in enumerate([bad_ids, progress_evidence, oversized, malformed, blank]))),
            answer("需要继续"),
        ])
        self.run_agent("task", self.agent.permissions)
        results = self.agent.run_state.tool_results
        self.assertTrue(results[0]["ok"])
        self.assertTrue(all(not r["ok"] and r["code"] == "invalid_arguments" for r in results[1:]))
        self.assertTrue(all(not current_views(request) for request in self.requests))
        good_reply = next(m for m in self.requests[-1]["messages"] if m.get("tool_call_id") == "good")
        self.assertEqual(json.loads(good_reply["content"])["data"], valid)
        self.assertEqual(len(self.requests), 3)

    def test_default_off_rejects_direct_and_model_calls_and_binds_enabled_tools(self):
        self.assertFalse(any(s["function"]["name"] == "update_progress" for s in self.tools.get_schemas()))
        result = self.tools.execute("update_progress", document())
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "tool_unavailable")
        old_binding = agent.checkpoints.binding(self.agent)["tools"]
        self.model([tool_reply(call("hidden", "update_progress", document())), answer("done")])
        self.run_agent("task", self.agent.permissions)
        self.assertEqual(self.agent.run_state.tool_results[0]["code"], "tool_unavailable")
        self.assertTrue(all(not current_views(r) for r in self.requests))
        self.assertTrue(all("update_progress" not in m.get("content", "")
                            for r in self.requests for m in r["messages"] if m["role"] == "system"))
        self.enable()
        self.assertNotEqual(agent.checkpoints.binding(self.agent)["tools"], old_binding)

    def test_compaction_preserves_progress_for_recall_without_extra_card_and_counts_all_usage(self):
        self.enable()
        (self.workspace / "source.txt").write_text("entry", encoding="utf-8")
        self.agent.messages.append(answer("x" * 270000))
        self.model([
            (tool_reply(call("read", "read_file", {"path": "source.txt"})), 90000),
            (tool_reply(call("p", "update_progress", document(call_ids=["read"]))), 96001),
            (answer("已定位入口，仍需核对分支和交付报告。"), 100),
            (tool_reply(call("recall", "read_tool_result", {"call_id": "p", "field": "result"})), 50),
            (answer("done"), 30),
        ])
        self.run_agent("keep exact user requirement", self.agent.permissions, self.agent.messages)
        self.assertEqual(len(self.requests), 5)
        self.assertNotIn("tools", self.requests[2])
        final = self.requests[-1]
        self.assertTrue(all(not current_views(request) for request in self.requests))
        recalled = json.loads(result_page(self.agent.run_state.tool_results[-1]["data"])[1])
        self.assertEqual(recalled["call_id"], "p")
        self.assertEqual(recalled["data"], document(call_ids=["read"]))
        self.assert_complete_tool_pairs(final["messages"])
        self.assertIn({"role": "user", "content": "keep exact user requirement"}, final["messages"])
        self.assertLessEqual(context.estimate_input_tokens(final["messages"], self.tools.get_schemas(),
                                                          scale=self.agent.context_scale), 28000)
        self.assertFalse(any((m.get("content") or "").startswith("[当前任务进度]")
                             for m in self.agent.messages))
        self.assertEqual(self.agent.token_usage.totals["prompt_tokens"], 186181)
        self.assertEqual(len(self.agent.run_state.tool_results), 3)

    def test_compaction_reserves_fixed_context_in_target_and_rejects_excess_without_request(self):
        self.assertIn("extra_context", inspect.signature(context.compact).parameters,
                      "上下文整理尚未预留固定指导语的预算")
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        for i in range(80):
            messages.extend([tool_reply(call(str(i), "read_file", {"path": "source.py"})),
                             {"role": "tool", "tool_call_id": str(i),
                              "content": json.dumps({"ok": True, "data": "x" * 3000})}])
        extra = [{"role": "system", "content": "固定指导语\n" + "中" * 900}]
        self.model([(answer("已定位，继续交付"), 100)])
        result = context.compact(self.agent.model_adapter, messages, self.agent.token_usage.record,
                                 tools=self.tools.get_schemas(), extra_context=extra)
        measured = context.estimate_input_tokens(result + extra, self.tools.get_schemas())
        self.assertGreaterEqual(measured, 24000)
        self.assertLessEqual(measured, 28000)
        self.assertNotIn(extra[0], result)
        self.assert_complete_tool_pairs(result)
        self.assertEqual(len(self.requests), 1)
        with self.assertRaises(context.ContextBudgetError):
            context.compact(self.agent.model_adapter, messages, self.agent.token_usage.record,
                            tools=self.tools.get_schemas(), extra_context=extra * 100)
        self.assertEqual(len(self.requests), 1)

    def test_progress_guidance_is_included_in_hard_budget_check(self):
        self.enable()
        def narrow_budget(event):
            if event["type"] == "tool_result" and event["tool"] == "update_progress":
                bare = context.estimate_input_tokens(self.agent.messages, self.tools.get_schemas(),
                                                      scale=self.agent.context_scale)
                self.agent.hard_limit = bare + 5
                self.agent.soft_limit = bare + 10000
        self.agent.event_sink = narrow_budget
        self.model([(tool_reply(call("p", "update_progress", document())), 10)])
        self.run_agent("task", self.agent.permissions)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.last_result.stop_reason, "context_limit")
        self.assertEqual(self.agent.run_state.tool_results[0]["data"], document())

    def test_resume_recalls_saved_progress_without_card_and_new_session_keeps_usage(self):
        self.enable()
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / "checkpoints")
        self.agent.checkpoint_store = cp
        self.model([(tool_reply(call("p", "update_progress", document())), 10)])
        original = cp.save
        def interrupt(snapshot):
            original(snapshot)
            if snapshot["run_state"]["next_action"] == "prepare" and snapshot["run_state"]["tool_results"]:
                raise KeyboardInterrupt()
        with patch.object(cp, "save", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.run_agent("original task", self.agent.permissions)
        saved = cp.load()
        self.assertEqual(saved["run_state"]["tool_results"][0]["data"], document())
        self.assertNotIn("task_progress", saved["run_state"])
        client = self.agent.client
        self.agent = agent.Agent(client, self.tools, checkpoint_store=cp)
        self.model([(tool_reply(call("recall", "read_tool_result", {"call_id": "p", "field": "result"})), 20),
                    (answer("恢复后回答"), 5)])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(self.agent.run_state.steps_used, 3)
        self.assertTrue(all(not current_views(request) for request in self.requests))
        self.assertEqual(len(self.agent.run_state.tool_results), 2)
        recalled = json.loads(result_page(self.agent.run_state.tool_results[-1]["data"])[1])
        self.assertEqual(recalled["call_id"], "p")
        self.assertEqual(recalled["data"], document())
        with contextlib.redirect_stdout(io.StringIO()):
            cli.start_new_session(self.agent, self.store)
        totals = deepcopy(self.agent.token_usage.totals)
        self.model([(answer("new answer"), 30)])
        self.run_agent("new task", self.agent.permissions, self.agent.messages)
        self.assertFalse(current_views(self.requests[-1]))
        self.assertEqual(self.agent.run_state.tool_results, [])
        self.assertEqual(self.agent.token_usage.totals["prompt_tokens"], totals["prompt_tokens"] + 30)
        self.assertTrue(list(self.store.directory.glob("default-backup-*.json")))

    def test_progress_does_not_reset_step_limit_or_override_failed_verification(self):
        self.enable()
        self.agent.max_steps = 1
        self.model([(tool_reply(call("p", "update_progress", document())), 10)])
        self.run_agent("task", self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, "max_steps")
        self.assertEqual(self.agent.run_state.steps_used, 1)
        self.assertEqual(len(self.requests), 1)
        self.agent.max_steps = 5
        self.model([tool_reply(call("p2", "update_progress", {**document(), "remaining": []})),
                    answer("done"), answer("done again")])
        self.run_agent("another task", self.agent.permissions,
                       verifier=lambda value: CheckResult("fail", "尚欠真实交付"))
        self.assertEqual(self.last_result.stop_reason, "verification_unmet")
        self.assertEqual(self.agent.run_state.verification_attempts, 2)
        self.assertEqual(self.agent.run_state.steps_used, 3)

    def test_recovery_ignores_orphan_progress_and_rejects_orphan_evidence(self):
        self.enable()
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / "checkpoints")
        self.agent.checkpoint_store = cp
        self.model([tool_reply(call("new-update", "update_progress", document(call_ids=["orphan-read"]))),
                    answer("done")])
        self.agent.run_state = agent.RunState(run_id="a" * 32, user_input="task", phase="preparing")
        self.agent.messages.append({"role": "user", "content": "task"})
        snapshot = agent.checkpoints.capture(self.agent, "b" * 32, agent.checkpoints.binding(self.agent))
        snapshot["run_state"]["tool_results"].extend([
            {"call_id": "orphan-update", "tool": "update_progress", "source": "model", "ok": True,
             "data": document(next_action="这条记录没有实际调用")},
            {"call_id": "orphan-read", "tool": "read_file", "source": "model", "ok": True,
             "data": "没有对应调用的伪造读取"},
        ])
        cp.save(snapshot)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertFalse(any(current_views(request) for request in self.requests))
        update = self.agent.run_state.tool_results[-1]
        self.assertEqual(update["call_id"], "new-update")
        self.assertFalse(update["ok"])
        self.assertEqual(update["code"], "invalid_arguments")

    def test_cli_flag_defaults_off_accepts_on_and_rejects_typos_before_credentials(self):
        dummy = SimpleNamespace(client=None)
        for setting, expected in ((None, False), ("off", False), ("on", True)):
            with self.subTest(setting=setting):
                env = {"DEEPSEEK_API_KEY": "offline-test-key"}
                if setting is not None:
                    env["AGENT_TASK_PROGRESS"] = setting
                with patch.dict(os.environ, env, clear=True), patch.object(cli.models, "get_model", return_value=dummy):
                    runtime = cli.create_agent(workspace=self.tools.workspace)
                names = {s["function"]["name"] for s in runtime.tools.get_schemas()}
                self.assertEqual("update_progress" in names, expected)
        with patch.dict(os.environ, {"AGENT_TASK_PROGRESS": "onn"}, clear=True), \
                patch("winreg.OpenKey", side_effect=AssertionError("选项应在读取凭证前校验")):
            with self.assertRaisesRegex(ValueError, "AGENT_TASK_PROGRESS"):
                cli.create_agent(workspace=self.tools.workspace)

    def test_frozen_requests_keep_history_guidance_and_options_without_card_for_on_and_off(self):
        # Assert the serialized SDK boundary, including the default-off payload.
        fixture = json.loads((Path(__file__).parent / "fixtures" /
                              "task_progress_requests.json").read_text(encoding="utf-8"))
        for item in fixture["cases"]:
            for enabled in (False, True):
                with self.subTest(case=item["case_id"], task_progress=enabled):
                    self.requests.clear()
                    self.check_requests.clear()
                    self.all_requests.clear()
                    self.tools = tools.Toolset(self.tools.workspace, run_command=self.fixture_command,
                                               task_progress=enabled)
                    self.agent = agent.Agent(None, self.tools)
                    payload = deepcopy(item["payload"])
                    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")
                    self.assertEqual(sha256(canonical).hexdigest(), item["payload_sha256"])
                    history = payload["messages"][:-2]
                    guidance, original_card = payload["messages"][-2:]
                    self.assertEqual(guidance["role"], "system")
                    self.assertEqual(original_card["role"], "assistant")
                    update = next(c for m in history for c in m.get("tool_calls", [])
                                  if c["function"]["name"] == "update_progress")
                    reply = next(m for m in history if m.get("tool_call_id") == update["id"])
                    recorded = {"call_id": update["id"], "tool": "update_progress", "source": "model",
                                **json.loads(reply["content"])}
                    self.agent.messages = deepcopy(history)
                    self.agent.run_state = agent.RunState(
                        run_id="a" * 32,
                        user_input=next(m["content"] for m in history if m["role"] == "user"),
                        phase="preparing", steps_used=5, next_action="request", tool_results=[recorded])
                    operation = self.agent._register_call(update, "model")
                    self.agent.run_state.call_records[operation].update(status="recorded", result=deepcopy(recorded))
                    self.model([(answer("offline request-shape probe"), 10)])
                    options = {k: deepcopy(v) for k, v in payload.items()
                               if k not in ("model", "messages", "tools")}
                    sdk_create = self.agent.client.chat.completions.create
                    def create(**kwargs):
                        return sdk_create(**{**options, **kwargs})
                    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                    self.agent._model_adapter = agent.models.get_model(
                        agent.models.ModelConfig(provider="openai_chat", model_name=payload["model"]), client=client)
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = self.agent._run_loop(None)
                    self.assertEqual(len(self.requests), 1)
                    expected = deepcopy(payload)
                    expected["messages"] = [*history, guidance] if enabled else history
                    if not enabled:
                        expected["tools"] = [s for s in expected["tools"]
                                             if s["function"]["name"] != "update_progress"]
                    # Keep the frozen request and its checksum unchanged. The current
                    # default tool table adds exactly one reviewed native tool.
                    write_position = next(i for i, schema in enumerate(expected["tools"])
                                          if schema["function"]["name"] == "write_file")
                    expected["tools"].insert(write_position, deepcopy(tools.TOOLS["edit_file"]["schema"]))
                    if self.tools.search_available:
                        position = next(i for i, schema in enumerate(expected["tools"])
                                        if schema["function"]["name"] == "read_file")
                        expected["tools"][position:position] = [deepcopy(tools.TOOLS[name]["schema"])
                                                                for name in ("find_files", "search_text")]
                    # Only the reviewed reader and write schemas changed; frozen history and
                    # every other published schema remain checked against the fixture.
                    for name in ("read_file", "read_tool_result", "write_file"):
                        position = next(i for i, schema in enumerate(expected["tools"])
                                        if schema["function"]["name"] == name)
                        expected["tools"][position] = deepcopy(tools.TOOLS[name]["schema"])
                    expected.update(expected.pop("extra_body"))
                    self.assertEqual(self.requests[0], expected)
                    self.assertEqual(self.agent.messages[:-1], history)
                    self.assertFalse(current_views({"messages": self.agent.messages}))
                    self.assertEqual(self.agent.run_state.tool_results, [recorded])
                    self.assertEqual(result.stop_reason, "end_turn")
                    self.assertEqual(self.agent.run_state.steps_used, 5)
                    self.assertEqual(self.check_requests, [])
                    self.assertEqual(list(self.workspace.iterdir()), [])

    def test_anomalous_final_text_keeps_existing_exit_rounds_and_usage(self):
        # Characterization only: malformed-looking text is still accepted without a verifier.
        # This must not be made green by adding a retry, completion gate or model judge.
        finals = (
            "<budget_tokens>2</budget_tokens>",
            "<system_warning>工具调用与用户消息之间的进度记录不可作为最终答案，最终需给出完整回复",
        )
        for text in finals:
            with self.subTest(text=text):
                self.requests.clear()
                self.check_requests.clear()
                self.all_requests.clear()
                self.enable()
                self.agent = agent.Agent(None, self.tools)
                progress = document(next_action="仍需修复并交付 report.md")
                self.model([
                    (tool_reply(call("p", "update_progress", progress)), 10),
                    (answer(text), 20),
                ])
                self.run_agent("修复问题并交付 report.md", self.agent.permissions)
                self.assertEqual(self.last_result.text, text)
                self.assertEqual(self.last_result.stop_reason, "end_turn")
                self.assertIsNone(self.last_result.verification)
                self.assertEqual(self.agent.run_state.steps_used, 2)
                self.assertEqual(self.agent.run_state.verification_attempts, 0)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.check_requests, [])
                self.assertFalse(current_views(self.requests[-1]))
                self.assertEqual(self.agent.run_state.tool_results[-1]["data"], progress)
                self.assertFalse((self.workspace / "report.md").exists())
                self.assertEqual(self.agent.token_usage.totals,
                                 {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32})
                self.assertEqual(self.agent.messages[-1], answer(text))

