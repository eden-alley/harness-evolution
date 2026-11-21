"""可选 Shell 的工具入口、权限与超时部分输出。"""
import subprocess
from unittest.mock import Mock

from scenario_support import OfflineScenario, tools, permissions, answer, call, tool_reply
from shell_schema_fixtures import CURRENT_SHELL_DESCRIPTION


class ShellOptionTests(OfflineScenario):
    def test_model_sees_shell_and_file_tools_share_the_workspace_root(self):
        self.model([answer("done")])
        self.run_agent("验证项目", self.agent.permissions)
        schema = next(tool["function"] for tool in self.requests[0]["tools"]
                      if tool["function"]["name"] == "run_shell")
        self.assertEqual(schema["description"], CURRENT_SHELL_DESCRIPTION)

    def test_known_mxc_bash_failure_is_explicit_and_does_not_retry(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 3221225794, b"",
            b"NtCreateDirectoryObject(\\BaseNamedObjects\\msys): 0xC0000022"))
        self.tools.run_command = runner
        result = self.tools.execute("run_shell", {"command": "printf x", "shell": "bash"})
        self.assertEqual(result.code, "shell_unavailable")
        self.assertIn("MXC", result.message)
        self.assertIn("执行状态需核实", result.message)
        self.assertIn('shell="cmd"', result.message)
        self.assertEqual(result.exit_code, 3221225794)
        self.assertIn("NtCreateDirectoryObject", result.data)
        runner.assert_called_once()

    def test_bash_selection_and_limits_pass_through_permission_and_runner(self):
        runner = Mock(return_value=subprocess.CompletedProcess([], 0, b"beta\n", b""))
        self.tools.run_command = runner
        asked = []
        self.model([tool_reply(call("bash", "run_shell", {"command": "printf 'alpha\\nbeta\\n' | rg beta",
                                                         "shell": "bash", "timeout": 45})),
                    answer("done")])
        self.run_agent("search", permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda n, a: (asked.append(a.copy()) or "y", "")))
        self.assertEqual(self.last_result.text, "done")
        self.assertEqual(asked[0]["shell"], "bash")
        self.assertEqual(runner.call_args.kwargs, {"shell": "bash"})
        self.assertEqual(runner.call_args.args[2], 45)
        self.assertIn("bash", permissions.render_call("run_shell", asked[0]))
        for args in ({"command": "echo x", "shell": "powershell"},
                     {"command": "echo x", "timeout": 301}):
            self.assertEqual(self.tools.execute("run_shell", args).code, "invalid_arguments")

    def test_timeout_keeps_partial_bytes_without_claiming_success(self):
        self.tools.run_command = Mock(side_effect=subprocess.TimeoutExpired(
            "offline", 30, output="测试中\n".encode("utf-8"), stderr="未结束\n".encode("gbk")))
        result = self.tools.execute("run_shell", {"command": "offline"})
        self.assertFalse(result.ok)
        self.assertIn("测试中", result.data)
        self.assertIn("未结束", result.data)
        self.assertIn("超时", result.message)
        self.assertIsNone(result.exit_code)
