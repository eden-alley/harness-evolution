"""实际 Windows 沙箱的本地边界验收；需要本机 codex.exe，不调用真实模型。"""
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from uuid import uuid4
from unittest.mock import patch

from scenario_support import OfflineScenario, ROOT, answer, call, permissions, tool_reply, tools
from sandbox import _codex_executable, _bash_executable, run_in_sandbox


@unittest.skipUnless(os.name == "nt", "真实沙箱验收只适用于 Windows")
class NativeSandboxScenarios(OfflineScenario):
    def setUp(self):
        super().setUp()
        # :workspace 也允许系统临时目录，越界对照必须放在系统临时目录之外。
        temporary = tempfile.TemporaryDirectory(prefix=".sandbox-test-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "workspace with space"
        self.workspace.mkdir()
        self.outside = self.root / "outside"
        self.outside.mkdir()
        self.tools.workspace = permissions.Workspace(self.workspace)
        self.tools.run_command = tools.run_in_sandbox
        try:
            _codex_executable(self.workspace)
        except RuntimeError as error:
            self.skipTest(str(error))

    def python_command(self, code):
        return subprocess.list2cmdline([sys.executable, "-B", "-c", f"exec({code!r})"])

    def test_real_project_read_only_rule(self):
        protected = self.workspace / "input.txt"
        protected.write_bytes(b"original")
        baseline = tools.Toolset(permissions.Workspace(self.workspace))
        control = baseline.execute("run_shell", {"command": "echo ordinary>input.txt"})
        self.assertTrue(control.ok, control)
        self.assertEqual(protected.read_bytes().strip(), b"ordinary")
        protected.write_bytes(b"original")
        toolset = tools.Toolset(permissions.Workspace(self.workspace, read_only_paths=["input.txt"]))

        for command in ("echo changed>input.txt", "ren input.txt moved.txt", "del /q input.txt"):
            with self.subTest(command=command):
                result = toolset.execute("run_shell", {"command": command})
                self.assertEqual(protected.read_bytes(), b"original", result)
                self.assertFalse((self.workspace / "moved.txt").exists())
        valid = toolset.execute("run_shell", {"command": "type input.txt && echo allowed>proposal.txt"})
        self.assertTrue(valid.ok, valid)
        self.assertIn("original", valid.data)
        self.assertEqual((self.workspace / "proposal.txt").read_bytes().strip(), b"allowed")
        self.assertEqual(list(self.workspace.glob(".agent-command-*.cmd")), [])

    def test_real_project_directory_rule(self):
        private_input = self.workspace / "samples"
        private_input.mkdir()
        protected = private_input / "input.txt"
        protected.write_bytes(b"original")
        toolset = tools.Toolset(permissions.Workspace(self.workspace, read_only_paths=["samples"]))
        write = toolset.execute("run_shell", {"command": "echo changed>samples\\input.txt"})
        self.assertFalse(write.ok, write)
        result = toolset.execute("run_shell", {"command": "ren samples moved"})
        self.assertTrue(protected.is_file(), result)
        self.assertEqual(protected.read_bytes(), b"original")
        valid = toolset.execute("run_shell", {"command": "type samples\\input.txt && echo allowed>proposal.txt"})
        self.assertTrue(valid.ok, valid)
        self.assertIn("original", valid.data)
        self.assertEqual((self.workspace / "proposal.txt").read_bytes().strip(), b"allowed")

    def test_real_project_rule_in_agent_loop(self):
        protected = self.workspace / "input.txt"
        protected.write_bytes(b"original")
        self.tools = tools.Toolset(permissions.Workspace(self.workspace, read_only_paths=["input.txt"]))
        self.agent.tools = self.tools
        policy = permissions.PermissionManager(workspace=self.tools.workspace)
        policy.set_auto_approve(True)
        self.model([
            tool_reply(call("builtin", "write_file", {"path": "input.txt", "content": "changed"}),
                       call("shell", "run_shell", {"command": "echo changed>input.txt"})),
            tool_reply(call("proposal", "run_shell", {"command": "type input.txt > proposal.txt"})),
            answer("提案已另存"),
        ])
        self.run_agent("保护样本并另存", policy)
        results = {row["call_id"]: row for row in self.agent.run_state.tool_results}
        self.assertEqual(results["builtin"]["code"], "policy_denied")
        self.assertFalse(results["shell"]["ok"])
        self.assertTrue(results["proposal"]["ok"])
        self.assertEqual(protected.read_bytes(), b"original")
        self.assertEqual((self.workspace / "proposal.txt").read_bytes(), b"original")

    def test_real_write_boundary_read_scope_and_environment(self):
        code = ("import os; from pathlib import Path; "
                "Path('inside.txt').write_text('sandbox-ok', encoding='utf-8'); "
                "print('ENV_SECRET=' + str(any(name in os.environ for name in "
                "['DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'API_KEY'])))")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-secret",
                                    "OPENAI_API_KEY": "offline-secret", "API_KEY": "offline-secret"}):
            result = tools.run_shell(self.tools, self.python_command(code))
        self.assertTrue(result.ok, result)
        self.assertTrue(result.data.startswith("[exit 0]"), result)
        self.assertIn("ENV_SECRET=False", result.data)
        self.assertEqual((self.workspace / "inside.txt").read_text(encoding="utf-8"), "sandbox-ok")
        fixture = self.outside / "existing.txt"
        fixture.write_text("outside-fixture", encoding="utf-8")
        denied = tools.run_shell(self.tools, 'echo should-not-replace> "..\\outside\\existing.txt"')
        self.assertFalse(denied.ok, denied)
        self.assertFalse(denied.data.startswith("[exit 0]"), denied)
        self.assertEqual(fixture.read_text(encoding="utf-8"), "outside-fixture")
        # 记录内置 workspace 的真实读取范围，不把写隔离误称为全盘读隔离。
        readable = tools.run_shell(self.tools, 'type "..\\outside\\existing.txt"')
        self.assertTrue(readable.ok, readable)
        self.assertTrue(readable.data.startswith("[exit 0]"), readable)
        self.assertIn("outside-fixture", readable.data)
        self.assertEqual(list(self.workspace.glob(".agent-command-*.cmd")), [])
        self.assertEqual(self.all_requests, [])

    def test_real_bash_pipeline_with_rg(self):
        try:
            _bash_executable(self.workspace)
        except RuntimeError as error:
            self.skipTest(str(error))
        if not shutil.which("rg"):
            self.skipTest("rg 不在 PATH 中")
        result = run_in_sandbox("printf 'alpha\\nbeta\\n' | rg beta", self.workspace, 20, shell="bash")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), b"beta")
        self.assertEqual(list(self.workspace.glob(".agent-command-*.sh")), [])
        self.assertEqual(self.all_requests, [])

    def test_real_direct_network_connection_to_local_listener_is_blocked(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            # Positive control: the endpoint must be reachable on the host.
            with socket.socket() as control:
                control.settimeout(1)
                self.assertEqual(control.connect_ex(("127.0.0.1", port)), 0)
            connection, _ = listener.accept()
            connection.close()
            command = self.python_command(
                "import socket; s=socket.socket(); s.settimeout(1); "
                f"print('CONNECT_ERR=' + str(s.connect_ex(('127.0.0.1', {port})))); s.close()")
            result = tools.run_shell(self.tools, command)
        self.assertTrue(result.ok, result)
        self.assertTrue(result.data.startswith("[exit 0]"), result)
        match = re.search(r"CONNECT_ERR=(\d+)", result.data)
        self.assertIsNotNone(match, result)
        self.assertNotEqual(int(match[1]), 0, "当前离线工作区配置不应连接本地监听端口")
        self.assertEqual(self.all_requests, [])

    def test_real_udp_to_local_listener_is_blocked(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.settimeout(0.5)
            address = listener.getsockname()
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control:
                control.sendto(b'host-control', address)
            self.assertEqual(listener.recvfrom(100)[0], b'host-control')
            code = ("import socket\n"
                    "s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
                    "try:\n"
                    f"    s.sendto(b'sandbox-canary', {address!r})\n"
                    "    print('UDP_SEND_ACCEPTED')\n"
                    "except OSError as e:\n"
                    "    print('UDP_BLOCKED=' + str(e.winerror))\n"
                    "finally:\n"
                    "    s.close()\n")
            result = tools.run_shell(self.tools, self.python_command(code))
            self.assertTrue(result.ok, result)
            # A blocked datagram can be silently dropped; delivery is the boundary.
            self.assertRegex(result.data, r'UDP_(BLOCKED=\d+|SEND_ACCEPTED)')
            with self.assertRaises(socket.timeout):
                listener.recvfrom(100)
        self.assertEqual(self.all_requests, [])

    def test_real_registry_credential_canary_is_not_readable(self):
        # Security requirement, not a test that approves an observed leak.
        # Only create/read/delete our unique synthetic value, never real keys.
        import winreg
        name = 'HARNESS_CANARY_' + uuid4().hex
        canary = 'synthetic-' + uuid4().hex
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment', 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, canary)
            try:
                code = ("import winreg\n"
                        "try:\n"
                        "    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:\n"
                        f"        value = winreg.QueryValueEx(key, {name!r})[0]\n"
                        f"    print('REGISTRY_CANARY_READABLE=' + str(value == {canary!r}))\n"
                        "except OSError:\n"
                        "    print('REGISTRY_CANARY_READABLE=False')\n")
                result = tools.run_shell(self.tools, self.python_command(code))
            finally:
                winreg.DeleteValue(key, name)
        self.assertTrue(result.ok, result)
        self.assertIn('REGISTRY_CANARY_READABLE=False', result.data,
                      'MXC can read persistent user credentials; isolation requirement unmet')
        self.assertEqual(self.all_requests, [])

    def test_real_timeout_stops_child_process_before_followup_file_write(self):
        child = ("import time; from pathlib import Path; "
                 "Path('child-ready.txt').write_text('ready'); "
                 "list(iter(lambda: Path('go.txt').exists() or time.sleep(0.05), True)); "
                 "Path('escaped.txt').write_text('child survived')")
        parent = ("import subprocess, sys, time; "
                  f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)")
        result = tools.run_shell(self.tools, self.python_command(parent), timeout=8)
        self.assertFalse(result.ok)
        self.assertIn("[超时]", result.message)
        self.assertTrue((self.workspace / "child-ready.txt").exists(),
                        "必须确认子进程已经启动，才能检验超时后的清理")
        (self.workspace / "go.txt").write_text("go", encoding="utf-8")
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and not (self.workspace / "escaped.txt").exists():
            time.sleep(0.05)
        self.assertFalse((self.workspace / "escaped.txt").exists(), "超时后子进程仍在运行")
        self.assertEqual(list(self.workspace.glob(".agent-command-*.cmd")), [])
        self.assertEqual(self.all_requests, [])


if __name__ == "__main__":
    unittest.main()
