"""自动执行开关：保留硬规则，切换不请求模型，新会话保留、重启关闭。"""
import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

from scenario_support import (OfflineScenario, agent, context, permissions, tools,
                              token_usage, answer, call, tool_reply)
from test_exit_check import verdict


class AutoPermissionScenarios(OfflineScenario):
    def test_auto_skips_known_confirmations_but_preserves_hard_limits_and_manual_mode(self):
        asked = []

        def ask(name, args):
            asked.append(name)
            return ("a", "") if name == "write_file" else ("n", "离线拒绝")

        perms = permissions.PermissionManager(workspace=self.tools.workspace, ask=ask)
        self.assertTrue(callable(getattr(perms, "set_auto_approve", None)),
                        "权限管理器缺少自动执行开关")
        self.assertFalse(perms.auto_approve)
        self.assertFalse(perms.check("run_shell", {"command": "echo manual"}).allowed)
        self.assertTrue(perms.check("write_file", {"path": "a.txt", "content": "a"}).allowed)
        self.assertIn("write_file", perms.always_allowed)

        perms.set_auto_approve(True)
        self.assertEqual(perms.always_allowed, set())
        asked.clear()
        with patch.object(sys.stdin, "isatty", return_value=False):
            for name, args in [
                ("read_file", {"path": "a.txt"}), ("list_dir", {"path": "."}),
                ("write_file", {"path": "b.txt", "content": "b"}),
                ("run_shell", {"command": "echo one"}),
                ("run_shell", {"command": "echo two"}),
                ("fetch_url", {"url": "https://offline.invalid"}),
            ]:
                with self.subTest(tool=name, args=args):
                    self.assertTrue(perms.check(name, args).allowed)
        self.assertEqual(asked, [])
        for name, args in [
            ("read_file", {"path": "../outside.txt"}),
            ("list_dir", {"path": ".."}),
            ("write_file", {"path": "../outside.txt", "content": "no"}),
            ("run_shell", {"command": "sudo echo denied"}),
        ]:
            with self.subTest(hard_limit=name):
                self.assertFalse(perms.check(name, args).allowed)
        with patch.dict(permissions.TOOL_RISK, {"write_file": permissions.Risk.DENY}):
            self.assertFalse(perms.check("write_file", {"path": "a.txt", "content": "no"}).allowed)
        self.assertEqual(asked, [])
        self.assertFalse(perms.check("unknown_tool", {}).allowed)
        self.assertEqual(asked, ["unknown_tool"])

        # 关闭自动执行后，旧的「一直允许」不能让写入继续免确认。
        perms.always_allowed.add("write_file")
        perms.set_auto_approve(False)
        self.assertFalse(perms.auto_approve)
        self.assertEqual(perms.always_allowed, set())
        self.assertTrue(perms.check("write_file", {"path": "c.txt", "content": "c"}).allowed)
        self.assertFalse(perms.check("run_shell", {"command": "echo manual again"}).allowed)
        self.assertEqual(asked, ["unknown_tool", "write_file", "run_shell"])

    def test_commands_show_status_and_reject_bad_syntax_without_model_or_session_changes(self):
        old = [{"role": "system", "content": agent.SYSTEM},
               {"role": "user", "content": "原会话"}, answer("原回答")]
        self.store.save("default", old)
        before = (self.store.directory / "default.json").read_bytes()
        self.agent.token_usage.totals.update(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        self.agent.last_prompt_tokens = 123
        # 未实现开关时也能完成循环，最终通过行为断言暴露问题。
        self.model([answer("控制命令不应发给模型")] * 7)
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/auto", "/auto on", "/auto",
                   "/auto invalid", "/auto on extra", "/auto off", "/auto", "exit"]), \
                contextlib.redirect_stdout(output):
            self.run_cli()
        self.assertEqual(len(self.all_requests), 0, "权限开关不得触发模型请求")
        self.assertEqual((self.store.directory / "default.json").read_bytes(), before)
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})
        self.assertEqual(self.agent.last_prompt_tokens, 123)
        self.assertIn("已开启", output.getvalue())
        self.assertIn("已关闭", output.getvalue())
        self.assertEqual(output.getvalue().count("用法：/auto"), 2)

        # 开关只在进程内生效，下一次创建权限管理器恢复关闭。
        with patch("builtins.input", side_effect=["/auto on", "quit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        restarted = io.StringIO()
        with patch("builtins.input", side_effect=["/auto", "quit"]), \
                contextlib.redirect_stdout(restarted):
            self.run_cli()
        self.assertIn("已关闭", restarted.getvalue())
        self.assertEqual(self.all_requests, [])

    def test_auto_survives_new_session_executes_tools_and_turns_confirmation_back_on(self):
        old = [{"role": "system", "content": agent.SYSTEM},
               {"role": "user", "content": "旧任务标记"}, answer("旧回答")]
        self.store.save("default", old)
        self.agent.token_usage.totals.update(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        self.model([
            (tool_reply(
                call("write", "write_file", {"path": "note.txt", "content": "自动写入"}),
                call("shell", "run_shell", {"command": "echo auto-shell-ok"}),
                call("fetch", "fetch_url", {"url": "https://offline.invalid/page"}),
                call("escape", "write_file", {"path": "../outside.txt", "content": "no"}),
                call("danger", "run_shell", {"command": "sudo echo denied"}),
                call("missing", "read_file", {"path": "missing.txt"})), 10),
            (answer("自动执行完成，已收到拒绝和错误"), 20),
            (tool_reply(call("manual", "write_file", {"path": "note.txt", "content": "覆盖"})), 30),
            (answer("已保留原文件"), 40),
        ], check_replies=[(verdict("completed", "已核实工具结果"), 5),
                          (verdict("completed", "已尊重用户拒绝"), 7)])
        output = io.StringIO()
        inputs = iter(["/auto on", "/new", "运行自动任务", "/auto off",
                       "试着覆盖文件", "n", "保留原文件", "quit"])

        def terminal_input(prompt):
            response = next(inputs, None)
            self.assertIsNotNone(response, "出现了预期外的终端输入")
            if prompt.startswith("[y]"):
                self.assertEqual(response, "n", "自动执行期间不应要求确认")
            return response

        with patch("builtins.input", side_effect=terminal_input), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(tools.urllib.request, "urlopen", return_value=io.BytesIO("本地网页样本".encode())), \
                patch.object(self.tools, "run_command", wraps=self.tools.run_command) as sandbox_run, \
                contextlib.redirect_stdout(output):
            self.run_cli()

        self.assertEqual((self.workspace / "note.txt").read_text(encoding="utf-8"), "自动写入")
        self.assertFalse((self.workspace.parent / "outside.txt").exists())
        sandbox_run.assert_called_once_with("echo auto-shell-ok", self.workspace, 30)
        outputs = {m["tool_call_id"]: m["content"] for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertIn("已写入", outputs["write"])
        self.assertEqual(outputs["shell"].splitlines(), ["[exit 0]", "auto-shell-ok"])
        self.assertEqual(outputs["fetch"], "本地网页样本")
        self.assertIn("权限拒绝", outputs["escape"])
        self.assertIn("危险模式", outputs["danger"])
        self.assertIn("FileNotFoundError", outputs["missing"])
        saved = self.store.load("default")
        self.assert_complete_tool_pairs(saved)
        self.assertIn("保留原文件", next(m["content"] for m in saved if m.get("tool_call_id") == "manual"))
        self.assertEqual(output.getvalue().count("[权限确认]"), 1)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(self.check_requests), 2)
        self.assertNotIn("/auto", json.dumps(saved, ensure_ascii=False))
        self.assertNotIn("旧任务标记", json.dumps(self.all_requests, ensure_ascii=False))
        backups = list(self.store.directory.glob("default-backup-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8")), old)
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 212, "completion_tokens": 16, "total_tokens": 228})


if __name__ == "__main__":
    unittest.main()
