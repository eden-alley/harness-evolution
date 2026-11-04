"""Token 统计的离线场景：正常与摘要累计，以及缺少 usage 和请求失败。"""
import contextlib
import io
import unittest

import httpx
from openai import InternalServerError

from scenario_support import OfflineScenario, agent, context, permissions, answer, call, tool_reply
from test_context import history


class TokenUsageScenarios(OfflineScenario):
    def test_dialogue_and_summary_counts_are_accumulated_once_without_changing_context_usage(self):
        (self.workspace / "seed.txt").write_text("ok", encoding="utf-8")
        self.agent.last_prompt_tokens = self.agent.hard_limit + 1
        self.model([
            (answer("此前任务摘要"), {"prompt_tokens": 1000, "completion_tokens": 50,
                                     "total_tokens": 1050, "prompt_cache_hit_tokens": 400,
                                     "prompt_cache_miss_tokens": 600}),
            (tool_reply(call("read", "read_file", {"path": "seed.txt"})),
             {"prompt_tokens": 200, "completion_tokens": 10, "total_tokens": 210}),
            (answer("完成"), {"prompt_tokens": 220, "completion_tokens": 20, "total_tokens": 240}),
        ])
        messages = history(3)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.agent.messages = messages
            result = self.agent.run("继续")
        self.assertEqual(result, "完成")
        self.assert_complete_tool_pairs(messages)
        text = output.getvalue()
        self.assertEqual(text.count("[摘要 tokens]"), 1)
        self.assertEqual(text.count("[对话 tokens]"), 2)
        self.assertIn("输入=1000 输出=50 合计=1050", text)
        self.assertIn("输入=220 输出=20 合计=240", text)
        self.assertIn("本次运行累计：输入=1420 输出=80 合计=1500", text)
        self.assertEqual(self.agent.last_prompt_tokens, 220)
        self.assertEqual(self.requests[2]["messages"][-1]["content"], "ok")
        self.assertEqual(len(self.requests), 3)

    def test_missing_usage_and_failed_requests_are_not_invented_or_added_to_totals(self):
        self.model([answer("服务端没提供用量")])
        messages = [{"role": "system", "content": agent.SYSTEM}]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.agent.messages = messages
            result = self.agent.run("你好")
        self.assertEqual(result, "服务端没提供用量")
        self.assertIn("未返回 usage，本次未计入累计", output.getvalue())
        self.assertNotIn("输入=0", output.getvalue())

        self.model([httpx.Response(503, json={"error": {"message": "offline failure"}})])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(InternalServerError):
            context.compact(self.agent.client, self.agent.model, history(3), self.agent.token_usage.record)
        self.assertEqual(output.getvalue(), "")

        self.model([(answer("实际返回了零用量"),
                     {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.agent.messages = messages
            result = self.agent.run("再问一次")
        self.assertEqual(result, "实际返回了零用量")
        self.assertIn("本次运行累计：输入=0 输出=0 合计=0", output.getvalue())
        self.assertEqual(len(self.requests), 3)


if __name__ == "__main__":
    unittest.main()
