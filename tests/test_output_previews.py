"""工具预览与原始证据分开验证，禁止真实模型请求。"""
import json

from scenario_support import OfflineScenario, context, tools, answer, call, tool_reply
from test_file_reads import file_page, result_page


class OutputPreviewTests(OfflineScenario):
    def test_unicode_pages_keep_character_offsets_and_small_budget_fails_clearly(self):
        original = {"call_id": "long", "source": "model", "tool": "run_shell",
                    "ok": True, "data": "中" * 900 + "FINAL"}
        args = {"call_id": "long", "max_output_tokens": 1}
        self.assertEqual(self.tools.execute("read_tool_result", args,
                         recorded_results=[original]).code, "output_limit_exceeded")
        rebuilt = ""
        offset = 0
        import re
        while offset < len(original["data"]):
            args = {"call_id": "long", "offset": offset, "max_output_tokens": 128}
            result = self.tools.execute("read_tool_result", args, recorded_results=[original])
            self.assertTrue(result.ok)
            rendered = json.loads(context.render_tool_result(result, max_output_tokens=128))
            self.assertNotIn("truncated", rendered)
            head, body, following = result_page(rendered["data"])
            rebuilt += body
            if following is None:
                break
            self.assertGreater(following, offset)
            offset = following
        self.assertEqual(rebuilt, original["data"])

    def test_true_tail_error_and_structured_result_are_bounded_and_recoverable(self):
        raw = "HEAD" + "x" * 15000 + "REAL-TAIL"
        result = tools.ToolResult(False, data=raw, code="execution_error",
                                  message="e" * 12000, exit_code=7)
        view = json.loads(context.render_tool_result(result, call_id="original"))
        self.assertTrue(view["data"].endswith("REAL-TAIL"))
        self.assertEqual((view["exit_code"], view["code"]), (7, "execution_error"))
        self.assertLess(len(view["message"]), 4200)
        self.assertEqual(view["saved_call_id"], "original")
        self.assertEqual(result.data, raw)
        structured = json.loads(context.render_tool_result(
            tools.ToolResult(True, data={"values": ["x" * 15000]}), call_id="json"))
        self.assertNotIn("data", structured)
        self.assertIsInstance(structured["data_preview"], str)
        self.assertTrue(structured["truncated"])

    def test_model_selects_preview_without_changing_original_or_tool_arguments(self):
        raw = "HEAD\n" + "middle\n" * 100 + "TAIL"
        (self.workspace / "large.txt").write_text(raw, encoding="utf-8")
        self.model([tool_reply(call("read", "read_file", {"path": "large.txt", "max_output_tokens": 64})),
                    tool_reply(call("recover", "read_tool_result", {"call_id": "read", "max_output_tokens": 4096})),
                    answer("done")])
        self.run_agent("read", self.agent.permissions)
        preview = json.loads(self.requests[1]["messages"][-1]["content"])
        _, rows, following = file_page(preview["data"])
        self.assertEqual(rows[0], "HEAD")
        self.assertIsNotNone(following)
        self.assertNotIn("TAIL", preview["data"])
        original, recalled = self.agent.run_state.tool_results
        self.assertEqual(result_page(recalled["data"])[1], original["data"])
        self.assertNotIn("TAIL", recalled["data"])

    def test_output_budget_schema_rejects_invalid_values_before_permissions(self):
        for value in (0, -1, True, "10", 4097):
            with self.subTest(value=value):
                result = self.tools.execute("read_file", {"path": "unused", "max_output_tokens": value})
                self.assertEqual(result.code, "invalid_arguments")
        schema = next(s["function"]["parameters"] for s in self.tools.get_schemas()
                      if s["function"]["name"] == "run_shell")
        self.assertEqual(schema["properties"]["max_output_tokens"]["maximum"], 4096)

    def test_truncated_error_message_and_full_result_can_be_recovered(self):
        original = {"call_id": "failure", "source": "model", "tool": "run_shell",
                    "ok": False, "data": "partial output", "code": "execution_error",
                    "message": "x" * 6000 + "ERROR-TAIL"}
        result = tools.ToolResult(False, data=original["data"], code=original["code"],
                                  message=original["message"])
        view = json.loads(context.render_tool_result(result, call_id="failure"))
        self.assertIn("message", view["truncated_fields"])
        message = self.tools.execute("read_tool_result",
            {"call_id": "failure", "field": "message", "offset": 5800}, recorded_results=[original])
        self.assertTrue(message.data.endswith("ERROR-TAIL"))
        full = self.tools.execute("read_tool_result",
            {"call_id": "failure", "field": "result", "offset": 0, "max_output_tokens": 4096},
            recorded_results=[original])
        self.assertEqual(json.loads(result_page(full.data)[1]), original)
