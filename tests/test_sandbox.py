"""沙箱执行层的离线测试，不需要实际启动 Codex 或调用模型。"""
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from scenario_support import OfflineScenario, tools, permissions, answer, call, tool_reply


class SandboxScenarios(OfflineScenario):
    def setUp(self):
        super().setUp()
        self.tools.run_command = tools.run_in_sandbox
        self.cli_dir = self.workspace.parent / "trusted-cli"
        self.cli_dir.mkdir()
        self.cli = self.cli_dir / "codex.exe"
        self.cli.touch()
        self.env_override = patch.dict(os.environ, {
            "PATH": str(self.cli_dir),
            "DEEPSEEK_API_KEY": "offline-secret", "API_KEY": "offline-secret",
            "OPENAI_API_KEY": "offline-secret", "UNRELATED_SECRET": "offline-secret",
            "HTTPS_PROXY": "http://offline.invalid:1234",
        })
        self.env_override.start()
        self.addCleanup(self.env_override.stop)

    @staticmethod
    def process(stdout=b"ok\r\n", stderr=b"", code=0):
        process = MagicMock()
        process.__enter__.return_value = process
        process.communicate.return_value = (stdout, stderr)
        process.returncode = code
        process.poll.return_value = code
        process.pid = 43210
        return process

    def test_native_launcher_keeps_command_inside_sandbox_and_drops_secrets(self):
        command = 'echo "quoted value" & echo second > marker.txt'
        process = self.process("正常\n".encode("utf-8"), "错误\n".encode("gbk"), 7)
        scripts = []

        def launch_process(argv, **options):
            scripts.append((self.workspace / argv[-1]).read_text(encoding="utf-8"))
            return process

        with patch.object(subprocess, "Popen", side_effect=launch_process) as launch:
            result = tools.run_shell(self.tools, command)
        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.data, "[exit 7]\n正常\n错误\n")
        self.assertEqual(launch.call_count, 1)
        argv = launch.call_args.args[0]
        self.assertIsInstance(argv, list, "不能在宿主 shell 中拼接沙箱命令")
        self.assertEqual(Path(argv[0]), self.cli)
        self.assertEqual(argv[1], "sandbox")
        self.assertEqual(argv[argv.index("--permission-profile") + 1], ":workspace")
        self.assertIn("--include-managed-config", argv)
        self.assertEqual(Path(argv[argv.index("-C") + 1]), self.workspace)
        self.assertIn("windows.sandbox=mxc", argv)
        self.assertEqual(argv[argv.index("--") + 1:-1], [
            str(Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"), "/d", "/s", "/c"])
        self.assertTrue(argv[-1].startswith(".\\.agent-command-"))
        self.assertNotIn(" ", argv[-1])
        self.assertEqual(scripts, ["@echo off\nchcp 65001 >nul\n" + command + "\n"])
        self.assertFalse((self.workspace / argv[-1]).exists(), "执行完应清理临时脚本")
        options = launch.call_args.kwargs
        self.assertFalse(options.get("shell", False))
        self.assertEqual(options["cwd"], self.workspace)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertFalse(options.get("text", False))
        for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "API_KEY", "UNRELATED_SECRET", "HTTPS_PROXY"):
            self.assertNotIn(name, options["env"])
        self.assertIn("SystemRoot".upper(), {name.upper() for name in options["env"]})
        self.assertFalse((self.workspace / "marker.txt").exists())

    def test_missing_or_workspace_supplied_runtime_fails_closed(self):
        (self.workspace / "codex.exe").touch()
        for search_path in ("", str(self.workspace), "."):
            with self.subTest(search_path=search_path), patch.dict(os.environ, {"PATH": search_path}), \
                    patch.object(subprocess, "Popen") as launch:
                result = self.tools.execute("run_shell", {"command": "echo should-not-run"})
                self.assertEqual(result.code, "execution_error")
                self.assertIn("Error:", result.message)
                self.assertIn("codex.exe", result.message)
                launch.assert_not_called()

    def test_sandbox_startup_error_is_returned_without_host_retry(self):
        for error in (FileNotFoundError("offline: runtime gone"), OSError("offline: sandbox unavailable")):
            with self.subTest(error=type(error).__name__), \
                    patch.object(subprocess, "Popen", side_effect=error) as launch:
                result = self.tools.execute("run_shell", {"command": "echo should-not-run"})
                self.assertEqual(result.code, "execution_error")
                self.assertIn("Error:", result.message)
                self.assertEqual(launch.call_count, 1)

    def test_timeout_and_interrupt_terminate_the_process_tree(self):
        for error in (subprocess.TimeoutExpired("sandbox", 0.1), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                process = self.process()
                process.communicate.side_effect = [error, (b"", b"")]
                process.poll.return_value = None
                with patch.object(subprocess, "Popen", return_value=process) as launch, \
                        patch.object(subprocess, "run") as terminate:
                    if isinstance(error, KeyboardInterrupt):
                        with self.assertRaises(KeyboardInterrupt):
                            tools.run_shell(self.tools, "echo wait", timeout=0.1)
                    else:
                        self.assertIn("[超时]", tools.run_shell(self.tools, "echo wait", timeout=0.1).message)
                self.assertEqual(launch.call_count, 1)
                kill_argv = terminate.call_args.args[0]
                self.assertEqual(Path(kill_argv[0]).name.lower(), "taskkill.exe")
                self.assertEqual(kill_argv[1:], ["/PID", "43210", "/T", "/F"])
                process.kill.assert_called_once()
                self.assertEqual(process.communicate.call_count, 2)
                self.assertEqual(list(self.workspace.glob(".agent-command-*.cmd")), [])

    def test_permission_and_sandbox_failure_are_returned_to_offline_agent(self):
        asked = []

        def ask(name, args):
            asked.append(args["command"])
            return ("n", "不执行") if args["command"] == "echo denied" else ("y", "")

        self.model([tool_reply(
            call("denied", "run_shell", {"command": "echo denied"}),
            call("failed", "run_shell", {"command": "echo failed"})), answer("已收到拒绝和执行错误")])
        with patch.object(subprocess, "Popen", side_effect=OSError("offline sandbox failure")) as launch:
            result, _ = self.run_agent("检查执行结果", permissions.PermissionManager(workspace=self.tools.workspace, ask=ask))
        self.assertEqual(result, "已收到拒绝和执行错误")
        self.assertEqual(asked, ["echo denied", "echo failed"])
        self.assertEqual(launch.call_count, 1)
        outputs = {m["tool_call_id"]: m["content"] for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertIn("user_denied", outputs["denied"])
        self.assertIn("offline sandbox failure", outputs["failed"])


if __name__ == "__main__":
    unittest.main()
