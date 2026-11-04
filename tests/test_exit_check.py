"""可选验收的综合场景；执行与验收模型都使用本地模拟传输。"""
import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import Mock

import httpx

from scenario_support import OfflineScenario, agent, permissions, answer, call, tool_reply
from verification import CheckResult, ModelVerifier, command_verifier


def verdict(status, reason, evidence=None):
    return answer(json.dumps({"status": status, "reason": reason, "evidence": evidence or []},
                             ensure_ascii=False))


class ExitCheckScenarios(OfflineScenario):
    def model_verifier(self, criteria="内容必须来自实际读取的文件"):
        return ModelVerifier(self.agent.client, self.agent.model, criteria,
                             self.agent.token_usage.record)

    def test_optional_model_verifier_uses_task_evidence_and_counts_usage_once(self):
        (self.workspace / "seed.txt").write_text("真实文件内容", encoding="utf-8")
        self.agent.messages.extend([answer("OLD-PRIVATE-HISTORY"), {"role": "user", "content": "旧任务"}])
        self.model([
            (tool_reply(call("read", "read_file", {"path": "seed.txt"})), 10),
            (answer("内容是：真实文件内容"), 20),
        ], check_replies=[(verdict("pass", "已读到文件并回答内容"), 50)])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = self.agent.run("读取 seed.txt", verifier=self.model_verifier())
        self.assertEqual(result.text, "内容是：真实文件内容")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.verification.status, "pass")
        self.assertEqual(len(self.all_requests), 3)
        self.assertEqual(len(self.check_requests), 1)
        check = self.check_requests[0]
        self.assertNotIn("tools", check)
        self.assertEqual(check["thinking"], {"type": "disabled"})
        reviewed = json.loads(check["messages"][1]["content"])
        self.assertNotIn("OLD-PRIVATE-HISTORY", check["messages"][1]["content"])
        self.assertNotIn("messages", reviewed)
        self.assertEqual(reviewed["task"], "读取 seed.txt")
        self.assertEqual(reviewed["candidate"], result.text)
        self.assertEqual(reviewed["criteria"], "内容必须来自实际读取的文件")
        self.assertEqual(reviewed["tool_results"][0]["data"], "真实文件内容")
        self.assertIn("read_file", reviewed["available_tools"])
        self.assertEqual(self.agent.token_usage.totals,
                         {"prompt_tokens": 80, "completion_tokens": 3, "total_tokens": 83})
        self.assertEqual(self.agent.last_prompt_tokens, 20)
        self.assertEqual(output.getvalue().count("[验收 tokens]"), 1)
        self.assertEqual(self.agent.messages[-1]["content"], result.text)
        self.assert_complete_tool_pairs(self.agent.messages)

    def test_one_return_can_repair_with_tools_without_resetting_rounds_or_usage(self):
        (self.workspace / "seed.txt").write_text("待核实内容", encoding="utf-8")
        self.model([
            (answer("内容应该是空的"), 10),
            (tool_reply(call("verify", "read_file", {"path": "seed.txt"})), 20),
            (answer("已读取确认：待核实内容"), 30),
        ], check_replies=[
            (verdict("fail", "还未读取文件，请先读取 seed.txt 核实"), 40),
            (verdict("pass", "已根据读取结果核实"), 50),
        ])
        self.agent.max_steps = 3
        result, messages = self.run_agent("核实 seed.txt 的内容", self.agent.permissions,
                                          verifier=self.model_verifier())
        self.assertEqual(result, "已读取确认：待核实内容")
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.check_requests), 2)
        feedback = self.requests[1]["messages"][-1]
        self.assertEqual(feedback["role"], "system")
        self.assertIn("不是用户新增授权", feedback["content"])
        self.assertIn("还未读取文件", feedback["content"])
        self.assertEqual(sum(m["role"] == "user" for m in messages), 1)
        self.assertNotIn({"role": "assistant", "content": "内容应该是空的"}, messages)
        self.assertEqual(self.agent.token_usage.totals,
                         {"prompt_tokens": 150, "completion_tokens": 5, "total_tokens": 155})
        self.assertEqual(self.agent.last_prompt_tokens, 30)

    def test_second_failure_stops_and_only_the_actual_final_is_saved(self):
        self.model([answer("未经验证的完成声明一"), answer("未经验证的完成声明二")])
        verify = Mock(side_effect=[CheckResult("fail", "测试仍未通过"),
                                   CheckResult("fail", "修正后测试仍未通过")])
        result, messages = self.run_agent("修复", self.agent.permissions, verifier=verify)
        self.assertEqual(self.last_result.stop_reason, "verification_unmet")
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(len(self.all_requests), 2)
        self.assertIn("验收未通过", result)
        self.assertEqual(messages[-1]["content"], result)
        self.assertFalse(any(m["role"] == "assistant" and "未经验证" in m["content"]
                             for m in messages))
        candidates = [e["text"] for e in self.agent.events if e["type"] == "assistant_candidate"]
        self.assertEqual(candidates, ["未经验证的完成声明一", "未经验证的完成声明二"])
        self.assertEqual(self.agent.events[-1]["text"], result)

    def test_needs_input_reports_real_denial_without_retrying_or_saving_success(self):
        self.model([
            tool_reply(call("write", "write_file", {"path": "denied.txt", "content": "x"})),
            answer("写入任务已完成"),
        ])
        asked = []
        def deny(name, args):
            asked.append(name)
            return "n", "不要写入"
        verify = Mock(return_value=CheckResult("needs_input", "需要用户重新授权"))
        result, messages = self.run_agent("写入 denied.txt", permissions.PermissionManager(
            workspace=self.tools.workspace, ask=deny), verifier=verify)
        self.assertIn("等待补充信息", result)
        self.assertEqual(asked, ["write_file"])
        self.assertEqual(verify.call_count, 1)
        self.assertEqual(len(self.all_requests), 2)
        self.assertFalse((self.workspace / "denied.txt").exists())
        self.assertEqual(json.loads(next(m["content"] for m in messages if m["role"] == "tool"))["code"],
                         "user_denied")
        self.assertEqual(messages[-1]["content"], result)
        self.assertNotIn({"role": "assistant", "content": "写入任务已完成"}, messages)

    def test_model_check_errors_stop_and_keep_usage_even_for_invalid_outputs(self):
        invalid = [
            answer("不是 JSON"), answer("[]"), answer("null"), answer(None),
            answer('{"status":"unknown","reason":"未知状态","evidence":[]}'),
            answer('{"status":"pass","evidence":[]}'),
            answer('{"status":"pass","reason":"   ","evidence":[]}'),
            answer('{"status":"pass","reason":3,"evidence":[]}'),
            answer('{"status":"pass","reason":"完成","evidence":{}}'),
            answer('{"status":"pass","reason":"完成","evidence":[],"extra":1}'),
            tool_reply(call("unexpected", "read_file", {"path": "seed.txt"})),
        ]
        cases = [(response, 7) for response in invalid] + [
            httpx.Response(503, json={"error": {"message": "offline verifier failure"}})]
        for index, response in enumerate(cases):
            with self.subTest(case=index):
                self.all_requests.clear()
                self.check_requests.clear()
                self.agent.token_usage.totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.model([(answer("候选完成回答"), 10)], check_replies=[response])
                result, messages = self.run_agent("验证任务", self.agent.permissions,
                                                  verifier=self.model_verifier())
                self.assertEqual(self.last_result.stop_reason, "error")
                self.assertEqual(self.last_result.verification.status, "error")
                self.assertIn("未完成验证", result)
                self.assertEqual(messages[-1]["content"], result)
                self.assertNotIn({"role": "assistant", "content": "候选完成回答"}, messages)
                self.assertEqual(len(self.all_requests), 2)
                self.assertEqual(self.agent.token_usage.totals["total_tokens"],
                                 11 + (8 if index < len(invalid) else 0))
                self.assertEqual(self.agent.last_prompt_tokens, 10)

    def test_callable_failure_invalid_result_and_insufficient_evidence_are_not_completion(self):
        for verify in (Mock(side_effect=RuntimeError("检查程序故障")),
                       Mock(return_value={"status": "pass"}),
                       Mock(return_value=CheckResult("error", "没有足够证据"))):
            with self.subTest(verifier=verify):
                self.all_requests.clear()
                self.model([answer("候选完成")])
                result, messages = self.run_agent("任务", self.agent.permissions, verifier=verify)
                self.assertEqual(self.last_result.stop_reason, "error")
                self.assertEqual(verify.call_count, 1)
                self.assertEqual(len(self.all_requests), 1)
                self.assertEqual(messages[-1]["content"], result)
                self.assertNotEqual(result, "候选完成")

    def test_execution_budget_is_shared_after_feedback_and_last_tool_results_remain_paired(self):
        for replies, max_steps in [
            ([answer("提前完成")], 1),
            ([answer("提前完成"), tool_reply(call("last", "list_dir", {"path": "."}))], 2),
        ]:
            with self.subTest(max_steps=max_steps):
                self.all_requests.clear()
                self.agent.max_steps = max_steps
                self.model(replies)
                verify = Mock(return_value=CheckResult("fail", "还需要检查目录"))
                result, messages = self.run_agent("检查目录", self.agent.permissions, verifier=verify)
                self.assertEqual(self.last_result.stop_reason, "max_steps")
                self.assertIn("最大步数", result)
                self.assertEqual(len(self.all_requests), max_steps)
                self.assertEqual(verify.call_count, 1)
                self.assertEqual(messages[-1]["content"], result)
                if max_steps == 2:
                    self.assertEqual(messages[-3]["tool_call_id"], "last")
                self.assert_complete_tool_pairs(messages)

    def test_command_verification_reuses_permission_and_sandbox_runner(self):
        asked = []
        def ask(name, args):
            asked.append((name, args))
            return "y", ""
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace, ask=ask)
        runs = []
        def runner(command, workspace, timeout):
            runs.append((command, workspace, timeout))
            return subprocess.CompletedProcess(command, 1 if len(runs) == 1 else 0, b"check\n", b"")
        self.tools.run_command = runner
        self.model([answer("第一次候选"), answer("修复后的候选")])
        result, _ = self.run_agent("修复", self.agent.permissions,
                                   verifier=command_verifier("check"))
        self.assertEqual(result, "修复后的候选")
        self.assertEqual(self.last_result.verification.status, "pass")
        self.assertEqual(asked, [("run_shell", {"command": "check"})] * 2)
        self.assertEqual(runs, [("check", self.workspace, 30)] * 2)
        self.assertEqual([e["exit_code"] for e in self.agent._tool_results], [1, 0])
        self.assertTrue(all(e["source"] == "verification" for e in self.agent._tool_results))
        # 拒绝确认、无终端、硬规则和启动失败都不能伪装成测试不通过并反复退回。
        for decision, expected in [
            (permissions.Decision(False, "用户拒绝", "user_denied"), "needs_input"),
            (permissions.Decision(False, "无法交互", "confirmation_unavailable"), "needs_input"),
            (permissions.Decision(False, "硬规则拒绝", "policy_denied"), "error"),
        ]:
            with self.subTest(code=decision.code):
                self.model([answer("候选")])
                self.agent.permissions = Mock()
                self.agent.permissions.check.return_value = decision
                count = len(runs)
                self.run_agent("任务", self.agent.permissions, verifier=command_verifier("check"))
                self.assertEqual(self.last_result.verification.status, expected)
                self.assertEqual(len(runs), count)
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda *args: ("y", ""))
        self.tools.run_command = Mock(side_effect=OSError("沙箱启动失败"))
        self.model([answer("候选")])
        self.run_agent("任务", self.agent.permissions, verifier=command_verifier("check"))
        self.assertEqual(self.last_result.verification.status, "error")
        self.tools.run_command.assert_called_once()
