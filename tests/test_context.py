"""第四章的四个离线综合场景：上下文结构、阈值处理、摘要和会话恢复。"""
import contextlib
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import httpx
from openai import InternalServerError

from scenario_support import (OfflineScenario, agent, context, permissions, tools,
                              answer, call, tool_reply, tool_text)


def history(turns=5):
    messages = [{"role": "system", "content": "系统提示"}]
    for i in range(turns):
        request = tool_reply(call(f"read-{i}", "read_file", {"path": f"file{i}.txt"}),
                             call(f"list-{i}", "list_dir", {"path": "."}))
        request["content"] = f"第 {i} 轮先查看文件"
        messages.extend([
            {"role": "user", "content": f"任务 {i}"}, request,
            {"role": "tool", "tool_call_id": f"read-{i}", "content": f"文件内容 {i}"},
            {"role": "tool", "tool_call_id": f"list-{i}", "content": f"file{i}.txt"},
            answer(f"完成 {i}"),
        ])
    return messages


class ContextScenarios(OfflineScenario):
    def test_truncation_cleanup_and_turn_split_preserve_boundaries_and_pairs(self):
        """抓住丢失头尾、删除消息、误改近期结果，以及从工具调用中间切分。"""
        for text in ("", "短文本", "x" * 4000):
            self.assertEqual(context.truncate_tool_result(text), text)
        truncated = context.truncate_tool_result("H" * 2000 + "M" * 2000 + "T" * 2000)
        self.assertTrue(truncated.startswith("H" * 2000 + "\n..."))
        self.assertTrue(truncated.endswith("T" * 2000))
        self.assertIn("省略 2000 字符", truncated)
        self.assertLess(len(truncated), 4200)

        original = history()
        for keep in (3, 0, 20):
            with self.subTest(keep_results=keep):
                messages = deepcopy(original)
                context.clear_old_tool_results(messages, keep=keep)
                self.assertEqual(len(messages), len(original))
                remaining = keep
                # 从尾向前，只有超过保留数量的 tool.content 可以改变。
                for before, after in zip(reversed(original), reversed(messages)):
                    expected = deepcopy(before)
                    if before["role"] == "tool":
                        if remaining <= 0:
                            expected["content"] = "[旧工具结果已清理，如需要请重新调用工具]"
                        remaining -= 1
                    self.assertEqual(after, expected)
                self.assert_complete_tool_pairs(messages)

        old, recent = context.split_for_compact(original)
        self.assertEqual(recent[0], {"role": "user", "content": "任务 3"})
        self.assertEqual(old + recent, original[1:])
        self.assert_complete_tool_pairs([original[0]] + old)
        self.assert_complete_tool_pairs([original[0]] + recent)
        self.assertIsNone(context.split_for_compact(history(2)))
        short = history(1)
        self.assertIs(context.compact(object(), "unused", short, self.agent.token_usage.record), short)

    def test_soft_limit_uses_reported_usage_and_cleans_only_old_tool_contents(self):
        """抓住阈值边界错误、未记录 usage、入历史前未截断或清理损坏工具配对。"""
        (self.workspace / "large.txt").write_text("HEAD" + "x" * 5992 + "TAIL", encoding="utf-8")
        messages = history()
        before = deepcopy(messages)
        self.agent.last_prompt_tokens = self.agent.soft_limit  # 相等时不应清理。
        self.model([
            (tool_reply(call("large", "read_file", {"path": "large.txt"})), self.agent.soft_limit + 1),
            (answer("清理后完成"), 100),
        ])
        result, _ = self.run_agent("读取长文件", permissions.PermissionManager(workspace=self.tools.workspace, ), messages)
        self.assertEqual(result, "清理后完成")
        self.assertEqual(self.requests[0]["messages"][:-1], before)
        second = self.requests[1]["messages"]
        self.assertEqual(len(second), len(self.requests[0]["messages"]) + 2)
        outputs = [m for m in second if m["role"] == "tool"]
        self.assertTrue(all(m["content"].startswith("[旧工具结果已清理") for m in outputs[:-3]))
        self.assertEqual(outputs[-3:-1], [m for m in before if m["role"] == "tool"][-2:])
        self.assertTrue(tool_text(outputs[-1]["content"]).startswith("HEAD"))
        self.assertTrue(tool_text(outputs[-1]["content"]).endswith("TAIL"))
        self.assertIn("中间省略", outputs[-1]["content"])
        self.assertLess(len(outputs[-1]["content"]), 4200)
        self.assertEqual(self.agent.last_prompt_tokens, 100)
        self.assertEqual(len(self.requests), 2)  # 清理不应额外调用模型。

    def test_hard_limit_summarizes_old_turns_and_handles_short_history_or_api_failure(self):
        """抓住摘要覆盖近期消息、替换列表导致调用方不同步、单轮误摘要或失败丢历史。"""
        (self.workspace / "seed.txt").write_text("当前文件", encoding="utf-8")
        messages = history(4)
        original_reference = messages
        recent_start = next(i for i, m in enumerate(messages) if m == {"role": "user", "content": "任务 3"})
        expected_recent = deepcopy(messages[recent_start:])
        self.agent.last_prompt_tokens = self.agent.hard_limit + 1
        self.model([
            answer("旧任务已完成，保留关键路径 file0.txt。"),
            (tool_reply(call("current", "read_file", {"path": "seed.txt"})), 50),
            (answer("摘要后继续完成"), 60),
        ])
        result, _ = self.run_agent("继续读取当前文件", permissions.PermissionManager(workspace=self.tools.workspace, ), messages)
        self.assertEqual(result, "摘要后继续完成")
        self.assertIs(messages, original_reference)
        summary_request = self.requests[0]
        self.assertNotIn("tools", summary_request)
        rendered = summary_request["messages"][1]["content"]
        for text in ("任务 0", "任务 2", "read_file", "list_dir", "file0.txt", "第 0 轮先查看文件"):
            self.assertIn(text, rendered)
        self.assertNotIn("任务 3", rendered)
        self.assertEqual(messages[0], {"role": "system", "content": "系统提示"})
        self.assertIn("旧任务已完成，保留关键路径 file0.txt。", messages[1]["content"])
        self.assertEqual([m["role"] for m in messages[:3]], ["system", "user", "assistant"])
        self.assertEqual(messages[3:3 + len(expected_recent)], expected_recent)
        self.assertEqual(tool_text(self.requests[2]["messages"][-1]["content"]), "当前文件")
        self.assertEqual(self.agent.last_prompt_tokens, 60)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(summary_request["thinking"], {"type": "disabled"})

        self.requests.clear()
        self.agent.last_prompt_tokens = self.agent.hard_limit + 1
        self.model([answer("单个用户回合，无需摘要")])
        result, _ = self.run_agent("只有一个任务", permissions.PermissionManager(workspace=self.tools.workspace, ))
        self.assertEqual(result, "单个用户回合，无需摘要")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.agent.last_prompt_tokens, 0)  # 缺少 usage 时沿用原文回退值。

        self.requests.clear()
        self.model([httpx.Response(503, json={"error": {"message": "offline summary failure"}})])
        intact = history(4)
        snapshot = deepcopy(intact)
        with self.assertRaises(InternalServerError):
            context.compact(self.agent.client, self.agent.model, intact, self.agent.token_usage.record)
        self.assertEqual(intact, snapshot)
        self.assertEqual(len(self.requests), 1)

    def test_main_saves_and_restores_sessions_outside_workspace_without_model_access(self):
        """抓住没自动保存、重启没恢复、中文/结构丢失，或状态文件落到工具工作区。"""
        self.assertIsNone(self.store.load("default"))
        self.model([(answer("记住这是第一次聊天"), 10), (answer("继续第一次聊天"), 20)])
        with patch("builtins.input", side_effect=["你好，记住这个任务", "exit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        path = self.store.directory / "default.json"
        first = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(first[-2:], [{"role": "user", "content": "你好，记住这个任务"},
                                     {"role": "assistant", "content": "记住这是第一次聊天"}])
        self.assertFalse(self.store.directory.is_relative_to(self.workspace))
        decision = permissions.PermissionManager(workspace=self.tools.workspace, ).check("read_file", {"path": str(path)})
        self.assertFalse(decision.allowed)
        self.assertIn("工作区之外", tools.read_file(self.tools, str(path)).message)

        self.agent.last_prompt_tokens = 0  # 模拟重启后的进程初始用量状态。
        with patch("builtins.input", side_effect=["继续", "quit"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.run_cli()
        self.assertEqual(self.requests[1]["messages"][:-1], first)
        restored = self.store.load("default")
        self.assertEqual(restored[-2:], [{"role": "user", "content": "继续"},
                                        {"role": "assistant", "content": "继续第一次聊天"}])
        self.assertEqual(list(self.workspace.iterdir()), [])
        self.assertEqual(len(self.requests), 2)

        rich = history(3)
        self.store.save("含中文和工具调用", rich)
        self.assertEqual(self.store.load("含中文和工具调用"), rich)
        self.assert_complete_tool_pairs(self.store.load("含中文和工具调用"))
        (self.store.directory / "broken.json").write_text("{", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.store.load("broken")
        self.assertEqual(len(self.requests), 2)  # 会话读写不应请求模型。


if __name__ == "__main__":
    unittest.main()
