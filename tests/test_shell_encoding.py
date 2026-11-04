"""真实本地子进程的编码回归测试；模型响应保持离线模拟。"""
import os
import json
import shlex
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from scenario_support import OfflineScenario, permissions, tools, answer, call, tool_reply


def output_command(stdout, stderr=b"", exit_code=0):
    code = (f"import sys; sys.stdout.buffer.write({stdout!r}); sys.stdout.flush(); "
            f"sys.stderr.buffer.write({stderr!r}); sys.stderr.flush(); sys.exit({exit_code})")
    argv = [sys.executable, "-c", code]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


class ShellEncodingScenarios(OfflineScenario):
    def test_real_process_decodes_both_streams_without_reader_thread_errors(self):
        """同时覆盖两种编码、不同编码的双输出流、退出码、换行、BOM、空输出及坏字节。"""
        cases = [
            ("UTF-8 标准输出、GBK 错误输出",
             "中文 UTF-8 €\r\n".encode("utf-8"), "错误 GBK\r\n".encode("gbk"), 0,
             "[exit 0]\n中文 UTF-8 €\n错误 GBK\n"),
            ("GBK 标准输出、UTF-8 错误输出",
             "中文 GBK\r".encode("gbk"), "错误 UTF-8 €\n".encode("utf-8"), 7,
             "[exit 7]\n中文 GBK\n错误 UTF-8 €\n"),
            ("带 BOM 的 UTF-8", b"\xef\xbb\xbf" + "中文\n".encode("utf-8"), b"", 0,
             "[exit 0]\n中文\n"),
            ("没有输出", b"", b"", 0, "[exit 0]\n"),
            ("两种编码均无法解释的字节", b"stdout-\xff\r\n", b"stderr-\x81\n", 2,
             "[exit 2]\nstdout-\ufffd\nstderr-\ufffd\n"),
        ]
        for label, stdout, stderr, code, expected in cases:
            with self.subTest(case=label):
                failures = []
                with patch.object(threading, "excepthook", side_effect=lambda event: failures.append(event.exc_value)):
                    result = tools.run_shell(self.tools, output_command(stdout, stderr, code))
                self.assertEqual(failures, [], "子进程读取线程不应再发生解码异常")
                self.assertEqual(result.data, expected)
                self.assertEqual(result.exit_code, code)
                self.assertEqual(result.ok, code == 0)
                self.assertNotIn("None", result.data)

    def test_decoded_output_passes_through_permission_execution_and_agent_reply(self):
        """真实运行命令后，退出码和两路中文输出须完整回传给模拟模型。"""
        command = output_command("正常输出 €\n".encode("utf-8"), "错误信息\n".encode("gbk"), 7)
        asked = []

        def ask(name, args):
            asked.append((name, args))
            return "y", ""

        self.model([tool_reply(call("encoded", "run_shell", {"command": command})),
                    answer("命令输出已接收")])
        failures = []
        with patch.object(threading, "excepthook", side_effect=lambda event: failures.append(event.exc_value)):
            result, _ = self.run_agent("运行并读取输出", permissions.PermissionManager(workspace=self.tools.workspace, ask=ask))
        self.assertEqual(result, "命令输出已接收")
        self.assertEqual(failures, [])
        self.assertEqual(asked, [("run_shell", {"command": command})])
        message = self.requests[1]["messages"][-1]
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "encoded")
        self.assertEqual(json.loads(message["content"]), {
            "ok": False, "data": "[exit 7]\n正常输出 €\n错误信息\n",
            "code": "execution_error", "message": "命令非零退出", "exit_code": 7})
        self.assertEqual(len(self.requests), 2)


if __name__ == "__main__":
    unittest.main()
