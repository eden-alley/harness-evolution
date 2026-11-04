"""按任务验收的基础契约；本地模型响应，不访问真实 API。"""
import contextlib
import inspect
import io
import json
from unittest.mock import Mock

from scenario_support import OfflineScenario, agent, answer, call, tool_reply


class TaskVerificationScenarios(OfflineScenario):
    def test_default_ends_without_a_second_model_request(self):
        self.model([(answer("请补充需要处理的文件名。"), 10)])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.run("缺少文件名的任务")
        self.assertEqual(len(self.all_requests), 1, "默认结束不得再调用检查模型")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertIsNone(result.verification)
        self.assertEqual(self.agent.messages[-1]["content"], result.text)

    def test_task_can_configure_verification_without_changing_the_default(self):
        self.assertIn("verifier", inspect.signature(self.agent.run).parameters,
                      "缺少按任务配置的验收接口")
        from verification import CheckResult
        verifier = Mock(return_value=CheckResult("pass", "符合明确的验收条件"))
        self.model([answer("已完成")])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.run("任务", verifier=verifier)
        self.assertEqual(result.verification.status, "pass")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(verifier.call_count, 1)
        self.assertEqual(len(self.all_requests), 1)

    def test_unavailable_tools_do_not_enter_user_confirmation(self):
        asked = []
        self.agent.permissions.ask = lambda name, args: (asked.append(name) or "y", "")
        self.model([tool_reply(call("missing", "head", {})), answer("改用现有工具")])
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent.run("查看首行")
        self.assertEqual(asked, [], "工具不存在不得被表述为用户拒绝")
        result = next(m for m in self.agent.messages if m.get("tool_call_id") == "missing")
        payload = json.loads(result["content"])
        self.assertEqual(payload["code"], "tool_unavailable")
        self.assertIn("read_file", payload["available_tools"])
