"""实际 Windows 沙箱的本地边界验收；需要本机 codex.exe，不调用真实模型。"""
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scenario_support import OfflineScenario, ROOT, permissions, tools
from sandbox import _codex_executable


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
        return subprocess.list2cmdline([sys.executable, "-c", code])

    def test_real_write_boundary_read_scope_and_environment(self):
        code = ("import os; from pathlib import Path; "
                "Path('inside.txt').write_text('sandbox-ok', encoding='utf-8'); "
                "print('ENV_SECRET=' + str(any(name in os.environ for name in "
                "['DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'API_KEY'])))")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-secret",
                                    "OPENAI_API_KEY": "offline-secret", "API_KEY": "offline-secret"}):
            result = tools.run_shell(self.tools, self.python_command(code))
        self.assertTrue(result.startswith("[exit 0]"), result)
        self.assertIn("ENV_SECRET=False", result)
        self.assertEqual((self.workspace / "inside.txt").read_text(encoding="utf-8"), "sandbox-ok")
        fixture = self.outside / "existing.txt"
        fixture.write_text("outside-fixture", encoding="utf-8")
        denied = tools.run_shell(self.tools, 'echo should-not-replace> "..\\outside\\existing.txt"')
        self.assertFalse(denied.startswith("[exit 0]"), denied)
        self.assertEqual(fixture.read_text(encoding="utf-8"), "outside-fixture")
        # 记录内置 workspace 的真实读取范围，不把写隔离误称为全盘读隔离。
        readable = tools.run_shell(self.tools, 'type "..\\outside\\existing.txt"')
        self.assertTrue(readable.startswith("[exit 0]"), readable)
        self.assertIn("outside-fixture", readable)
        self.assertEqual(list(self.workspace.glob(".agent-command-*.cmd")), [])
        self.assertEqual(self.all_requests, [])

    def test_real_direct_network_connection_to_local_listener_is_blocked(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            command = self.python_command(
                "import socket; s=socket.socket(); s.settimeout(1); "
                f"print('CONNECT_ERR=' + str(s.connect_ex(('127.0.0.1', {port})))); s.close()")
            result = tools.run_shell(self.tools, command)
        self.assertTrue(result.startswith("[exit 0]"), result)
        match = re.search(r"CONNECT_ERR=(\d+)", result)
        self.assertIsNotNone(match, result)
        self.assertNotEqual(int(match[1]), 0, "当前离线工作区配置不应连接本地监听端口")
        self.assertEqual(self.all_requests, [])

    def test_real_timeout_stops_child_process_before_followup_file_write(self):
        child = ("import time; from pathlib import Path; "
                 "Path('child-ready.txt').write_text('ready'); "
                 "list(iter(lambda: Path('go.txt').exists() or time.sleep(0.05), True)); "
                 "Path('escaped.txt').write_text('child survived')")
        parent = ("import subprocess, sys, time; "
                  f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(60)")
        result = tools.run_shell(self.tools, self.python_command(parent), timeout=8)
        self.assertIn("[超时]", result)
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
