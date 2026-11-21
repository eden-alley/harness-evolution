"""第三、四章的 Agent 基础场景测试；无需真实 API key 或网络。

运行：py -3 -m unittest discover -s tests -p "test_*.py" -v
"""
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, permissions, tools, answer, call, tool_reply, tool_text
from test_file_reads import file_page


class AgentLoopScenarios(OfflineScenario):
    def test_normal_flow_executes_five_tools_and_preserves_history(self):
        """抓住工具未执行、路径错误、输出未截断、重复确认或历史丢失。"""
        asked = []

        def ask(name, args):
            asked.append(name)
            return ("a" if name == "write_file" else "y"), ""

        argv = [sys.executable, "nested/good.py"]
        command = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
        self.model([
            tool_reply(
                call("list", "list_dir", {"path": "."}),
                call("write", "write_file", {"path": "nested/good.py", "content": 'print("offline-ok")\n'}),
                call("write-long", "write_file", {"path": "long.txt", "content": "x" * 8101}),
                call("read", "read_file", {"path": "nested/good.py"}),
                call("read-long", "read_file", {"path": "long.txt"}),
                call("shell", "run_shell", {"command": command}),
                call("fetch", "fetch_url", {"url": "https://offline.invalid/page"})),
            answer("正常任务完成"), answer("还能看到上次的文件内容"),
        ])
        with patch.object(tools.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = io.BytesIO("网页样本".encode())
            result, messages = self.run_agent("创建、读取、运行并抓网页", permissions.PermissionManager(workspace=self.tools.workspace, ask=ask))
        self.assertEqual(result, "正常任务完成")
        self.assertEqual((self.workspace / "nested/good.py").read_text(encoding="utf-8"), 'print("offline-ok")\n')
        self.assertEqual((self.workspace / "long.txt").stat().st_size, 8101)
        raw_outputs = {m["tool_call_id"]: m["content"] for m in self.requests[1]["messages"] if m["role"] == "tool"}
        outputs = {key: tool_text(value) for key, value in raw_outputs.items()}
        self.assertEqual(outputs["list"], "(空目录)")
        self.assertEqual(file_page(outputs["read"])[1], ['print("offline-ok")'])
        self.assertEqual(json.loads(raw_outputs["read-long"])["code"], "output_limit_exceeded")
        self.assertEqual(outputs["shell"].splitlines(), ["[exit 0]", "offline-ok"])
        self.assertEqual(outputs["fetch"], "网页样本")
        self.assertEqual(asked, ["write_file", "run_shell", "fetch_url"])
        schemas = {t["function"]["name"]: t["function"] for t in self.requests[0]["tools"]}
        expected_names = {"list_dir", "read_file", "read_tool_result", "write_file", "edit_file", "run_shell", "fetch_url"}
        if self.tools.search_available:
            expected_names.update({"find_files", "search_text"})
        self.assertEqual(set(schemas), expected_names)
        self.assertEqual(schemas["write_file"]["parameters"]["required"], ["path", "content"])
        result, _ = self.run_agent("继续上一个任务", permissions.PermissionManager(workspace=self.tools.workspace, ), messages)
        self.assertEqual(result, "还能看到上次的文件内容")
        self.assertIn({"role": "user", "content": "创建、读取、运行并抓网页"}, self.requests[2]["messages"])
        self.assertIn({"role": "tool", "tool_call_id": "shell", "content": raw_outputs["shell"]}, self.requests[2]["messages"])
        self.assertEqual(len(self.requests), 3)

    def test_read_file_range_returns_complete_lines_and_continues(self):
        line = "中" * 1800
        (self.workspace / "code.txt").write_bytes(
            ("开头\r\n" + (line + "\r\n") * 2 + "末尾").encode("utf-8"))
        self.model([
            tool_reply(call("first", "read_file", {"path": "code.txt", "start_line": 2, "end_line": 4})),
            answer("已读取"),
        ])
        result, _ = self.run_agent("只读取代码片段", self.agent.permissions)
        self.assertEqual(result, "已读取")
        output = next(json.loads(m["content"]) for m in self.requests[1]["messages"] if m["role"] == "tool")
        self.assertEqual(file_page(output["data"])[1], [line, line, "末尾"])
        (self.workspace / "long.txt").write_text((line + "\n") * 3, encoding="utf-8")
        first = self.tools.execute("read_file", {"path": "long.txt", "start_line": 1})
        self.assertEqual(file_page(first.data)[1:], ([line, line], {"path": "long.txt", "start_line": 3}))
        remaining = self.tools.execute("read_file", {"path": "long.txt", "start_line": 3})
        self.assertEqual(file_page(remaining.data)[1:], ([line], None))
        for args, expected in [
            ({"path": "code.txt", "start_line": 0}, "invalid_arguments"),
            ({"path": "code.txt", "start_line": 3, "end_line": 2}, "invalid_arguments"),
            ({"path": "../outside.txt", "start_line": 1}, "policy_denied"),
        ]:
            with self.subTest(args=args):
                self.assertEqual(self.tools.execute("read_file", args).code, expected)

    def test_saved_tool_result_recovers_evidence_after_truncation_and_cleanup(self):
        text = "A" * 3100 + "实测 0.285714" + "B" * 3000
        with patch.object(self.agent.tools, "run_command") as command:
            command.return_value = subprocess.CompletedProcess([], 0, text.encode(), b"")
            self.model([
                tool_reply(call("test", "run_shell", {"command": "echo offline-fixture"})),
                (tool_reply(*(call(f"list-{i}", "list_dir", {}) for i in range(3))),
                 16001),
                tool_reply(call("recover", "read_tool_result", {"call_id": "test", "offset": 3000})),
                answer("已核实"),
                tool_reply(call("stale", "read_tool_result", {"call_id": "test"})),
                answer("历史结果不可查"),
            ])
            perms = permissions.PermissionManager(workspace=self.tools.workspace, ask=lambda *args: ("y", ""))
            result, messages = self.run_agent("查测试结果", perms)
            self.assertEqual(result, "已核实")
            command.assert_called_once()
            view = lambda index, key: next(m["content"] for m in self.requests[index]["messages"]
                                           if m.get("tool_call_id") == key)
            clipped = json.loads(view(1, "test"))
            self.assertEqual(clipped["saved_call_id"], "test")
            self.assertNotIn("0.285714", clipped["data"])
            self.assertEqual(json.loads(view(2, "test"))["saved_call_id"], "test")
            self.assertIn("实测 0.285714", tool_text(view(3, "recover")))
            self.run_agent("新任务", perms, messages)
            self.assertEqual(json.loads(view(5, "stale"))["code"], "result_unavailable")

    def test_progress_reminder_requires_repeated_reads_and_missing_milestones(self):
        (self.workspace / "seed.txt").write_text("证据\n" * 7, encoding="utf-8")
        for mode in ("repeat", "different", "report_ready", "test_ready"):
            with self.subTest(mode=mode):
                self.requests.clear()
                self.agent.progress_report = "report.md"
                self.agent.progress_test = "echo offline-test"
                report = self.workspace / "report.md"
                report.unlink(missing_ok=True)
                if mode == "report_ready":
                    report.write_text("已交付", encoding="utf-8")
                replies = []
                if mode == "test_ready":
                    replies.append(tool_reply(call("test", "run_shell", {"command": "echo offline-test"})))
                for i in range(7):
                    args = {"path": "seed.txt"}
                    if mode == "different":
                        args["start_line"] = i + 1
                    replies.append(tool_reply(call(f"read-{i}", "read_file", args)))
                self.model([*replies, answer("完成")])
                perms = permissions.PermissionManager(workspace=self.tools.workspace, ask=lambda *args: ("y", ""))
                with patch.object(self.agent.tools, "run_command",
                                  return_value=subprocess.CompletedProcess([], 0, b"ok", b"")):
                    result, messages = self.run_agent("读代码并交付报告", perms)
                self.assertEqual(result, "完成")
                hints = [m["content"] for request in self.requests for m in request["messages"]
                         if m["role"] == "system" and "近期有多次相同查阅" in m["content"]]
                self.assertEqual(len(hints), int(mode == "repeat"))
                self.assertEqual(len(self.requests), len(replies) + 1)
                if hints:
                    self.assertIn("report.md 尚未出现", hints[0])
                    self.assertIn("指定测试尚无成功记录", hints[0])
                self.assertFalse(any(m.get("content") in hints for m in messages))

    def test_errors_and_denials_are_returned_without_side_effects_or_aborting_batch(self):
        """抓住异常中断、拒绝后仍写入、硬规则绕过或漏回 tool_call_id。"""
        (self.workspace / "seed.txt").write_text("后续工具仍执行", encoding="utf-8")
        asked = []

        def ask(name, args):
            asked.append((name, args))
            return ("n", "保留原文件") if args.get("path") == "denied.txt" else ("y", "")

        self.model([tool_reply(
            call("json", "list_dir", "{"),
            call("missing", "read_file", {"path": "missing.txt"}),
            call("bad-args", "write_file", {"path": "no-content.txt"}),
            call("unknown", "not_a_tool", {}),
            call("denied", "write_file", {"path": "denied.txt", "content": "no"}),
            call("outside", "write_file", {"path": "../outside.txt", "content": "no"}),
            call("danger", "run_shell", {"command": "sudo echo harmless-probe"}),
            call("timeout", "run_shell", {"command": "echo timeout-probe"}),
            call("last", "read_file", {"path": "seed.txt"})), answer("异常处理完成")])
        with patch.object(self.tools, "run_command", side_effect=subprocess.TimeoutExpired("echo timeout-probe", 30)):
            result, _ = self.run_agent("检查异常后继续", permissions.PermissionManager(workspace=self.tools.workspace, ask=ask))
        self.assertEqual(result, "异常处理完成")
        outputs = {m["tool_call_id"]: m["content"] for m in self.requests[1]["messages"] if m["role"] == "tool"}
        for call_id, text in {"json": "合法的 JSON", "missing": "FileNotFoundError", "bad-args": "TypeError",
                              "unknown": "不存在工具", "denied": "保留原文件", "outside": "工作区",
                              "danger": "危险模式", "timeout": "[超时]"}.items():
            with self.subTest(call_id=call_id):
                self.assertIn(text, outputs[call_id])
        self.assertEqual(file_page(tool_text(outputs["last"]))[1], ["后续工具仍执行"])
        self.assertEqual(sorted(p.name for p in self.workspace.iterdir()), ["seed.txt"])
        self.assertFalse((self.workspace.parent / "outside.txt").exists())
        self.assertEqual([name for name, _ in asked], ["write_file", "run_shell"])
        for call_id, code in {"json": "invalid_arguments", "bad-args": "invalid_arguments",
                              "unknown": "tool_unavailable", "denied": "user_denied",
                              "outside": "policy_denied", "danger": "policy_denied",
                              "missing": "file_not_found", "timeout": "execution_error"}.items():
            self.assertEqual(json.loads(outputs[call_id])["code"], code)
        for name, args in [("read_file", {"path": "../outside.txt"}),
                           ("list_dir", {"path": ".."}),
                           ("write_file", {"path": "../outside.txt", "content": "no"})]:
            self.assertIn("工作区之外", self.tools.execute(name, args).message)
        self.assertEqual(len(self.requests), 2)

    def test_loop_stops_on_final_answer_empty_answer_and_step_limit(self):
        """抓住最终回答后继续请求、空回答处理错误、越过循环上限或末轮漏结果。"""
        for text, expected in [("已经完成", "已经完成"), (None, "")]:
            with self.subTest(final_answer=text):
                self.requests.clear()
                self.model([answer(text)])
                result, messages = self.run_agent("无需工具", permissions.PermissionManager(workspace=self.tools.workspace, ))
                self.assertEqual(result, expected)
                self.assertEqual(len(self.requests), 1)
                self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.requests.clear()
        self.replace(self.agent, "max_steps", 3)
        self.model([tool_reply(call(f"read-{i}", "read_file", {"path": "missing.txt"})) for i in range(3)])
        result, messages = self.run_agent("模型持续请求工具", permissions.PermissionManager(workspace=self.tools.workspace, ))
        self.assertIn("达到最大步数", result)
        self.assertEqual(len(self.requests), 3)
        outputs = [m for m in messages if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in outputs], ["read-0", "read-1", "read-2"])
        self.assertTrue(all("FileNotFoundError" in m["content"] for m in outputs))
        self.assertEqual(messages[-2]["role"], "tool")
        self.assertEqual(messages[-1], {"role": "assistant", "content": result})
        self.assertEqual(self.last_result.stop_reason, "max_steps")

    def test_terminal_confirmation_refusal_remembering_and_noninteractive_denial(self):
        """抓住终端不展示真实参数、拒绝原因丢失、记忆权限越界或非交互放行。"""
        perms = permissions.PermissionManager(workspace=self.tools.workspace, )
        output = io.StringIO()
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", side_effect=["invalid", "n", "先别覆盖", "a", "y", "y"]), \
                contextlib.redirect_stdout(output):
            denied = perms.check("write_file", {"path": "keep.txt", "content": "真实预览内容"})
            self.assertFalse(denied.allowed)
            self.assertIn("先别覆盖", denied.reason)
            self.assertTrue(perms.check("write_file", {"path": "a.txt", "content": "a"}).allowed)
            self.assertTrue(perms.check("write_file", {"path": "b.txt", "content": "b"}).allowed)
            self.assertFalse(perms.check("write_file", {"path": "../escape.txt", "content": "x"}).allowed)
            self.assertTrue(perms.check("run_shell", {"command": "echo first"}).allowed)
            self.assertTrue(perms.check("run_shell", {"command": "echo second"}).allowed)
        for text in ("keep.txt", "真实预览内容", "echo first", "echo second"):
            self.assertIn(text, output.getvalue())
        with patch.object(sys.stdin, "isatty", return_value=False):
            fresh = permissions.PermissionManager(workspace=self.tools.workspace, )
            self.assertTrue(fresh.check("read_file", {"path": "a.txt"}).allowed)
            denied = fresh.check("write_file", {"path": "pipe.txt", "content": "x"})
        self.assertFalse(denied.allowed)
        self.assertIn("非交互环境", denied.reason)
        self.assertEqual(denied.code, "confirmation_unavailable")
        self.assertEqual(list(self.workspace.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
