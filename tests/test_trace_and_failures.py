"""结构化错误、原始轨迹、上下文分离及失败后的会话一致性。"""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx

from scenario_support import OfflineScenario, agent, permissions, answer, call, tool_reply


class TraceAndFailureScenarios(OfflineScenario):
    def test_non_object_arguments_never_request_permission_or_execute(self):
        asked = []
        perms = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or "y", ""))
        self.model([tool_reply(
            call("array", "write_file", "[]"),
            call("null", "run_shell", "null"),
            call("scalar", "write_file", '"text"'),
            call("extra", "read_file", {"path": "seed.txt", "unexpected": True}),
            call("missing", "head", {})), answer("已收到参数和工具错误")])
        self.run_agent("验证异常", perms)
        outputs = {m["tool_call_id"]: json.loads(m["content"])
                   for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertEqual(asked, [])
        self.assertEqual(list(self.workspace.iterdir()), [])
        for call_id in ("array", "null", "scalar", "extra"):
            self.assertEqual(outputs[call_id]["code"], "invalid_arguments")
            self.assertFalse(outputs[call_id]["ok"])
        self.assertEqual(outputs["missing"]["code"], "tool_unavailable")
        self.assertEqual(set(outputs["missing"]["available_tools"]),
                         {item["function"]["name"] for item in self.tools.get_schemas()})

    def test_raw_trace_survives_truncation_cleanup_and_new_session(self):
        raw = "HEAD" + "x" * 11992 + "TAIL"
        (self.workspace / "long.txt").write_text(raw, encoding="utf-8")
        self.model([
            (tool_reply(*(call(f"read-{i}", "read_file", {"path": "long.txt"})
                          for i in range(4))), self.agent.soft_limit + 1),
            (answer("长文件任务结束"), 20), (answer("新会话你好"), 30)])
        with patch("builtins.input", side_effect=["读取长文件", "/new", "你好", "exit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        trace = self.store.directory / "default.trace.jsonl"
        content = trace.read_text(encoding="utf-8")
        events = [json.loads(line) for line in content.splitlines()]
        starts = [e for e in events if e["type"] == "task_started"]
        self.assertEqual([e["task"] for e in starts], ["读取长文件", "你好"])
        self.assertEqual(len({e["run_id"] for e in starts}), 2)
        results = [e for e in events if e["type"] == "tool_result"]
        self.assertEqual(len(results), 4)
        self.assertTrue(all(e["data"] == raw and e["ok"] for e in results))
        model_tools = [m for m in self.requests[1]["messages"] if m["role"] == "tool"]
        self.assertIn("旧工具结果已清理", model_tools[0]["content"])
        preview = json.loads(model_tools[-1]["content"])
        self.assertTrue(preview["truncated"])
        self.assertEqual(preview["original_chars"], len(raw))
        self.assertLess(len(preview["data"]), 4200)
        self.assertNotIn("读取长文件", json.dumps(self.requests[2], ensure_ascii=False))
        self.assertEqual(self.store.load("default")[-1]["content"], "新会话你好")
        self.assertNotIn("offline-test-key", content)
        self.assertNotIn("Authorization", content)
        self.assertEqual(self.check_requests, [])
        self.assertEqual(len([e for e in events if e["type"] == "final_response"]), 2)
        for start in starts:
            ids = [e["event_id"] for e in events if e["run_id"] == start["run_id"]]
            self.assertEqual(ids, list(range(1, len(ids) + 1)))

    def test_model_request_failure_is_saved_as_error_and_does_not_invent_usage(self):
        self.model([httpx.Response(503, json={"error": {"message": "offline request failure"}})])
        self.agent.event_sink = lambda event: self.store.append_event("default", event)
        result, messages = self.run_agent("读取数据", self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, "error")
        self.assertIn("未标记完成", result)
        self.assertEqual(messages[-1]["content"], result)
        self.assertEqual(self.agent.events[-1]["stop_reason"], "error")
        self.assertEqual(self.agent.events[-1]["text"], result)
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 0)
        self.assertEqual(len(self.all_requests), 1)

    def test_mid_batch_failure_pairs_pending_calls_and_records_unknown_execution_state(self):
        (self.workspace / "seed.txt").write_text("first-ok", encoding="utf-8")
        def broken_approval(name, args):
            raise RuntimeError("确认组件故障")
        perms = permissions.PermissionManager(workspace=self.tools.workspace, ask=broken_approval)
        self.model([tool_reply(
            call("first", "read_file", {"path": "seed.txt"}),
            call("broken", "write_file", {"path": "never.txt", "content": "x"}),
            call("pending", "list_dir", {"path": "."}))])
        result, messages = self.run_agent("执行三个工具", perms)
        self.assertEqual(self.last_result.stop_reason, "error")
        self.assertEqual(messages[-1]["content"], result)
        outputs = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        self.assertEqual(outputs[0]["data"], "first-ok")
        self.assertTrue(all(o["code"] == "execution_error" for o in outputs[1:]))
        self.assertFalse((self.workspace / "never.txt").exists())
        recorded = [e for e in self.agent.events if e["type"] == "tool_result"]
        self.assertEqual([e["call_id"] for e in recorded], ["first", "broken", "pending"])
        self.assertTrue(all(e["code"] == "execution_error" for e in recorded[1:]))
        self.assert_complete_tool_pairs(messages)

    def test_cli_finally_saves_history_if_unexpected_exception_escapes(self):
        def fail_run(task):
            self.agent.messages.extend([{"role": "user", "content": task},
                                        answer("最后可保存的回答")])
            raise RuntimeError("离线入口故障")
        with patch.object(self.agent, "run", side_effect=fail_run), \
                patch("builtins.input", side_effect=["任务"]), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            self.run_cli()
        self.assertEqual(self.store.load("default"), self.agent.messages)
        self.assertEqual(self.all_requests, [])
