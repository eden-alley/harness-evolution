"""/new 的离线综合场景：会话隔离、备份、用量保留与写入失败。"""
import contextlib
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

from scenario_support import (OfflineScenario, agent, context, token_usage,
                              answer, call, tool_reply)
from test_context import history


class NewSessionScenarios(OfflineScenario):
    def test_new_backs_up_history_resets_context_and_survives_restart_without_requests(self):
        """完整保留旧工具记录；只清压缩依据；连续新建不覆盖备份，也不调用模型。"""
        old = history(3)
        self.store.save("default", old)
        self.agent.last_prompt_tokens = self.agent.hard_limit + 1
        self.agent.token_usage.totals.update(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        original_totals = self.agent.token_usage.totals
        expected_totals = dict(original_totals)
        # 未实现 /new 时能走到明确的断言失败，而不是响应脚本耗尽。
        self.model([answer("旧任务摘要"), answer("/new 应该由程序处理")])
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/new", "exit"]), contextlib.redirect_stdout(output):
            self.run_cli()

        fresh = [{"role": "system", "content": agent.SYSTEM}]
        self.assertEqual(self.store.load("default"), fresh)
        self.assertEqual(self.agent.last_prompt_tokens, 0)
        self.assertIs(self.agent.token_usage.totals, original_totals)
        self.assertEqual(self.agent.token_usage.totals, expected_totals)
        self.assertEqual(self.all_requests, [])
        backups = [p for p in self.store.directory.glob("*.json") if p.name != "default.json"]
        self.assertEqual(len(backups), 1)
        first_backup = backups[0]
        self.assertEqual(json.loads(first_backup.read_text(encoding="utf-8")), old)
        self.assert_complete_tool_pairs(self.store.load(first_backup.stem))
        self.assertIn(first_backup.name, output.getvalue())

        # 再次新建也要单独备份；随后重启、直接退出仍保持干净的 default。
        self.model([])
        with patch("builtins.input", side_effect=["/new", "quit"]), contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        backups = [p for p in self.store.directory.glob("*.json") if p.name != "default.json"]
        self.assertEqual(len(backups), 2)
        self.assertEqual(json.loads(first_backup.read_text(encoding="utf-8")), old)
        second_backup = next(p for p in backups if p != first_backup)
        self.assertEqual(json.loads(second_backup.read_text(encoding="utf-8")), fresh)
        with patch("builtins.input", side_effect=["exit"]):
            self.run_cli()
        self.assertEqual(self.store.load("default"), fresh)
        self.assertEqual(self.agent.token_usage.totals, expected_totals)
        self.assertEqual(self.all_requests, [])
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_new_task_and_followups_keep_tools_tokens_and_restart_history(self):
        """默认无额外验收；新任务和追问保留工具记录与累计，不夹带旧任务。"""
        old = [{"role": "system", "content": agent.SYSTEM},
               {"role": "user", "content": "旧任务专属标记 OLD-TASK"}, answer("旧任务已完成")]
        self.store.save("default", old)
        (self.workspace / "seed.txt").write_text("新任务文件内容", encoding="utf-8")
        self.agent.token_usage.totals.update(prompt_tokens=100, completion_tokens=10, total_tokens=110)
        self.model([
            (tool_reply(call("read-new", "read_file", {"path": "seed.txt"})), 10),
            (answer("已读取新文件"), 20),
            (answer("仍记得新文件内容"), 30),
            (answer("重启后仍是新任务"), 40),
        ])
        task = "读取 seed.txt，开始新任务"
        followup = "刚才读取的内容是什么？"
        with patch("builtins.input", side_effect=["/new", task, followup, "exit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        self.assertEqual(self.requests[0]["messages"], [
            {"role": "system", "content": agent.SYSTEM}, {"role": "user", "content": task}])
        self.assertIn({"role": "user", "content": task}, self.requests[2]["messages"])
        self.assertIn({"role": "tool", "tool_call_id": "read-new", "content": json.dumps(
            {"ok": True, "data": "[行 1-1；文件已读完]\n1: 新任务文件内容"}, ensure_ascii=False)},
                      self.requests[2]["messages"])
        self.assertEqual(self.requests[2]["messages"][-1], {"role": "user", "content": followup})
        saved = self.store.load("default")
        self.assert_complete_tool_pairs(saved)
        self.assertEqual(saved[-1]["content"], "仍记得新文件内容")
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 160, "completion_tokens": 13, "total_tokens": 173})
        for request in self.all_requests:
            self.assertNotIn("OLD-TASK", json.dumps(request, ensure_ascii=False))
            self.assertNotIn('"content": "/new"', json.dumps(request, ensure_ascii=False))
        self.assertEqual(self.check_requests, [])

        with patch("builtins.input", side_effect=["继续这项新任务", "quit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        self.assertEqual(self.requests[3]["messages"][:-1], saved)
        self.assertEqual(self.agent.token_usage.totals, {"prompt_tokens": 200, "completion_tokens": 14, "total_tokens": 214})
        self.assertEqual(self.agent.last_prompt_tokens, 40)
        self.assertEqual(len(self.all_requests), 4)
        self.assertNotIn("OLD-TASK", json.dumps(self.all_requests[-2:], ensure_ascii=False))

    def test_write_failure_does_not_reset_usage_or_discard_old_session(self):
        """备份失败时不覆盖 default；新 default 保存失败时已有完整备份，且不请求模型。"""
        old = history(2)
        real_save = self.store.save
        self.model([])
        for failed_step in ("backup", "default"):
            with self.subTest(failed_step=failed_step):
                self.store.save("default", old)
                before = (self.store.directory / "default.json").read_bytes()
                existing = set(self.store.directory.glob("*.json"))
                self.agent.last_prompt_tokens = self.agent.hard_limit + 1
                self.agent.token_usage.totals.update(prompt_tokens=100, completion_tokens=10, total_tokens=110)
                expected_totals = deepcopy(self.agent.token_usage.totals)

                def fail_save(name, messages):
                    if (name == "default") == (failed_step == "default"):
                        raise PermissionError("离线模拟写入失败")
                    return real_save(name, messages)

                with patch.object(self.store, "save", side_effect=fail_save), \
                        patch("builtins.input", side_effect=["/new"]), \
                        contextlib.redirect_stdout(io.StringIO()), self.assertRaises(PermissionError):
                    self.run_cli()
                self.assertEqual((self.store.directory / "default.json").read_bytes(), before)
                self.assertEqual(self.agent.last_prompt_tokens, self.agent.hard_limit + 1)
                self.assertEqual(self.agent.token_usage.totals, expected_totals)
                created = set(self.store.directory.glob("*.json")) - existing
                self.assertEqual(len(created), 1 if failed_step == "default" else 0)
                for path in created:
                    self.assertEqual(json.loads(path.read_text(encoding="utf-8")), old)
                self.assertEqual(self.all_requests, [])


if __name__ == "__main__":
    unittest.main()
