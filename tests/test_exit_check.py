"""退出检查的五个离线综合场景；检查和执行都经过真实 SDK 的本地模拟传输。"""
import contextlib
import io
import json
import unittest

import httpx

from scenario_support import (OfflineScenario, agent, permissions, token_usage,
                              answer, call, tool_reply)


def verdict(decision, reason):
    return answer(json.dumps({"decision": decision, "reason": reason}, ensure_ascii=False))


class ExitCheckScenarios(OfflineScenario):
    def test_completed_checks_actual_tool_results_and_counts_check_tokens(self):
        """工具阶段不检查；完成检查看到原始任务和结果，用量累计但不污染压缩依据。"""
        (self.workspace / "seed.txt").write_text("真实文件内容", encoding="utf-8")
        task = "读取 seed.txt，告诉我内容"
        self.model([
            (tool_reply(call("read", "read_file", {"path": "seed.txt"})), 10),
            (answer("内容是：真实文件内容"), 20),
        ], check_replies=[(verdict("completed", "已读到文件并回答内容"), 50)])
        output = io.StringIO()
        messages = [{"role": "system", "content": agent.SYSTEM}]
        self.agent.messages = messages
        with contextlib.redirect_stdout(output):
            result = self.agent.run(task)
        self.assertEqual(result, "内容是：真实文件内容")
        self.assertEqual(len(self.check_requests), 1)
        self.assertEqual(len(self.all_requests), 3)
        check = self.check_requests[0]
        self.assertEqual(check["model"], self.requests[0]["model"])
        self.assertNotIn("tools", check)
        self.assertEqual(check["thinking"], {"type": "disabled"})
        reviewed = json.loads(check["messages"][1]["content"])
        self.assertEqual(reviewed["task"], task)
        self.assertEqual(reviewed["messages"], messages)
        self.assertIn({"role": "tool", "tool_call_id": "read", "content": "真实文件内容"}, messages)
        self.assert_complete_tool_pairs(messages)
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 80, "completion_tokens": 3, "total_tokens": 83})
        self.assertEqual(self.agent.last_prompt_tokens, 20)
        self.assertEqual(output.getvalue().count("[退出检查 tokens]"), 1)

    def test_continue_returns_feedback_to_same_loop_and_can_execute_more_tools(self):
        """检查不通过后执行反馈中的工作，再次检查；不把检查对话混进执行历史。"""
        (self.workspace / "seed.txt").write_text("待核实内容", encoding="utf-8")
        self.model([
            (answer("文件内容应该是空的"), 10),
            (tool_reply(call("verify", "read_file", {"path": "seed.txt"})), 20),
            (answer("已读取确认：待核实内容"), 30),
        ], check_replies=[
            (verdict("continue", "还未读取文件，请先读取 seed.txt 核实"), 40),
            (verdict("completed", "已根据读取结果核实"), 50),
        ])
        self.replace(self.agent, "max_steps", 3)
        result, messages = self.run_agent("核实 seed.txt 的内容", permissions.PermissionManager(workspace=self.tools.workspace, ))
        self.assertEqual(result, "已读取确认：待核实内容")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.check_requests), 2)
        self.assertEqual(len(self.all_requests), 5)
        feedback = self.requests[1]["messages"][-1]
        self.assertEqual(feedback["role"], "user")
        self.assertIn("继续当前任务", feedback["content"])
        self.assertIn("还未读取文件，请先读取 seed.txt 核实", feedback["content"])
        self.assertEqual(sum(m.get("content") == "核实 seed.txt 的内容" for m in messages), 1)
        for request in self.check_requests:
            self.assertEqual(json.loads(request["messages"][1]["content"])["task"], "核实 seed.txt 的内容")
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 150, "completion_tokens": 5, "total_tokens": 155})
        self.assertEqual(self.agent.last_prompt_tokens, 30)

    def test_blocked_reports_reason_instead_of_candidate_answer_and_stops(self):
        """受阻不是已完成，也不继续重试已被拒绝的工具。"""
        self.model([
            tool_reply(call("write", "write_file", {"path": "denied.txt", "content": "x"})),
            answer("写入任务已完成"),
        ], check_replies=[verdict("blocked", "写入被用户拒绝，需要用户重新授权")])
        asked = []

        def deny(name, args):
            asked.append(name)
            return "n", "不要写入"

        result, messages = self.run_agent("写入 denied.txt", permissions.PermissionManager(workspace=self.tools.workspace, ask=deny))
        self.assertIn("任务受阻", result)
        self.assertIn("未完成", result)
        self.assertIn("需要用户重新授权", result)
        self.assertNotEqual(result, "写入任务已完成")
        self.assertEqual(asked, ["write_file"])
        self.assertFalse((self.workspace / "denied.txt").exists())
        self.assertIn("权限拒绝", next(m["content"] for m in messages if m["role"] == "tool"))
        self.assertEqual(len(self.all_requests), 3)

    def test_check_failure_stops_without_claiming_completion_and_keeps_received_usage(self):
        """请求失败、非法 JSON 和非法字段均停止；收到的检查 usage 即使判定无效也累计。"""
        invalid = [
            answer("不是 JSON"), answer("[]"), answer("null"), answer(None),
            answer('{"decision":"unknown","reason":"未知决策"}'),
            answer('{"decision":"completed"}'),
            answer('{"decision":"completed","reason":"   "}'),
            answer('{"decision":"completed","reason":3}'),
            tool_reply(call("unexpected", "read_file", {"path": "seed.txt"})),
        ]
        cases = [(response, 7) for response in invalid] + [
            httpx.Response(503, json={"error": {"message": "offline checker failure"}}),
        ]
        for index, response in enumerate(cases):
            with self.subTest(case=index):
                self.all_requests.clear()
                self.check_requests.clear()
                self.agent.token_usage.totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.model([(answer("候选完成回答"), 10)], check_replies=[response])
                result, messages = self.run_agent("验证任务", permissions.PermissionManager(workspace=self.tools.workspace, ))
                self.assertIn("退出检查失败", result)
                self.assertIn("未标记完成", result)
                self.assertNotEqual(result, "候选完成回答")
                self.assertEqual(len(self.all_requests), 2)
                self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
                self.assertEqual(self.agent.token_usage.totals["total_tokens"], 11 + (8 if index < len(invalid) else 0))
                self.assertEqual(self.agent.last_prompt_tokens, 10)

    def test_step_limit_is_shared_by_continuations_and_tools(self):
        """连续要求继续也只能执行原定轮数；末轮工具结果仍完整回传。"""
        self.replace(self.agent, "max_steps", 2)
        for last_is_tool in (False, True):
            with self.subTest(last_is_tool=last_is_tool):
                self.requests.clear()
                self.check_requests.clear()
                self.all_requests.clear()
                last = tool_reply(call("last", "list_dir", {"path": "."})) if last_is_tool else answer("仍未完成")
                self.model([answer("提前结束"), last], check_replies=[
                    verdict("continue", "还需要检查目录"),
                    verdict("continue", "仍有可执行的工作"),
                ])
                result, messages = self.run_agent("检查工作区", permissions.PermissionManager(workspace=self.tools.workspace, ))
                self.assertEqual(result, "（达到最大步数，已停止）")
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(len(self.check_requests), 1 if last_is_tool else 2)
                self.assertEqual(len(self.all_requests), 3 if last_is_tool else 4)
                if last_is_tool:
                    self.assertEqual(messages[-1], {"role": "tool", "tool_call_id": "last", "content": "(空目录)"})


if __name__ == "__main__":
    unittest.main()
