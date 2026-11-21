"""输入预算、单任务整理和失败现场，均使用离线模型。"""
from copy import deepcopy
import json
import httpx

from scenario_support import OfflineScenario, agent, context, answer, call, tool_reply


class ContextBudgetTests(OfflineScenario):
    def test_compaction_preserves_order_of_user_turns_and_complete_tool_batches(self):
        first = {"role": "user", "content": "先记录原规则"}
        correction = {"role": "user", "content": "更正规则，报告暂不落盘"}
        task = {"role": "user", "content": "现在生成报告"}
        pair = [
            tool_reply(call("read", "read_file", {"path": "sample.txt"})),
            {"role": "tool", "tool_call_id": "read", "content": '{"ok":true,"data":"source"}'},
        ]
        past_reply = answer("本轮已确认，报告暂不落盘")
        messages = [{"role": "system", "content": "system"}, first, answer("已记录"),
                    correction, *pair, past_reply, task]
        self.model([(answer("已核对资料，需按最新请求继续。"), 100)])
        candidate = context.compact(self.agent.model_adapter, messages, self.agent.token_usage.record)
        retained = [message for message in candidate if message in messages]
        self.assertEqual(retained, messages)
        self.assertEqual(candidate[-1], task)
        self.assert_complete_tool_pairs(candidate)

    def test_compaction_retains_enough_complete_batches_to_reach_target(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "task"}]
        for i in range(80):
            messages.extend([tool_reply(call(str(i), "read_file", {"path": "source.py"})),
                             {"role": "tool", "tool_call_id": str(i),
                              "content": json.dumps({"ok": True, "data": "x" * 3000})}])
        self.model([(answer("定位 source.py，继续交付。"), 100)])
        candidate = context.compact(self.agent.model_adapter, messages, self.agent.token_usage.record,
                                    tools=self.tools.get_schemas())
        measured = context.estimate_input_tokens(candidate, self.tools.get_schemas())
        self.assertGreaterEqual(measured, 24000)
        self.assertLessEqual(measured, 28000)
        self.assert_complete_tool_pairs(candidate)

    def test_defaults_and_16k_do_not_erase_old_evidence(self):
        self.assertEqual((self.agent.soft_limit, self.agent.hard_limit), (96000, 128000))
        (self.workspace / "seed.txt").write_text("valuable evidence", encoding="utf-8")
        self.model([(tool_reply(*(call(str(i), "read_file", {"path": "seed.txt"}) for i in range(5))), 16001),
                    (answer("done"), 20)])
        self.run_agent("task", self.agent.permissions)
        views = [m["content"] for m in self.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(sum("valuable evidence" in v for v in views), 5)
        self.assertEqual(len(self.requests), 2)

    def test_single_task_compacts_without_losing_original_evidence_or_usage(self):
        (self.workspace / "seed.txt").write_text("HEAD\n" + "x" * 100 + "\n" + "middle\n" * 2000 + "TAIL", encoding="utf-8")
        self.agent.messages.append(answer("x" * 270000))
        self.model([(tool_reply(call("read", "read_file", {"path": "seed.txt"})), 96001),
                    (answer("已定位 seed.txt；原结果编号 read；尚待交付。"), 100),
                    (answer("done"), 50)])
        self.run_agent("keep my exact task", self.agent.permissions, self.agent.messages)
        self.assertEqual(len(self.requests), 3)
        self.assertNotIn("tools", self.requests[1])
        self.assertIn("HEAD", self.requests[1]["messages"][1]["content"])
        self.assertIn({"role": "user", "content": "keep my exact task"}, self.requests[2]["messages"])
        self.assertLessEqual(context.estimate_input_tokens(self.requests[2]["messages"],
                                                          self.tools.get_schemas()), 28000)
        self.assertLessEqual(len(self.agent.run_state.tool_results[0]["data"]), 4000)
        self.assertNotIn("TAIL", self.agent.run_state.tool_results[0]["data"])
        self.assertEqual(self.agent.token_usage.totals["prompt_tokens"], 96151)
        self.assertTrue(any(e["type"] == "context_compacted" for e in self.agent.events))
        self.assertTrue(any("工具执行记录索引" in m.get("content", "")
                            for m in self.requests[2]["messages"]))

    def test_new_history_is_checked_before_request_and_big_user_input_stays_intact(self):
        task = "uncompressible user requirement " * 18000
        self.model([])
        self.run_agent(task, self.agent.permissions)
        self.assertEqual(self.requests, [])
        self.assertIn({"role": "user", "content": task}, self.agent.messages)
        self.assertEqual(self.last_result.stop_reason, "context_limit")

    def test_summary_failure_keeps_history_and_counts_attempt_without_retry(self):
        for reply in ((answer(""), 100), httpx.Response(503, json={"error": {"message": "offline"}})):
            with self.subTest(reply=type(reply).__name__):
                self.requests.clear()
                self.agent.messages = [{"role": "system", "content": "system"},
                                       {"role": "user", "content": "old task"},
                                       answer("old evidence")]
                before = deepcopy(self.agent.messages)
                self.agent.last_prompt_tokens = 96001
                self.model([reply])
                self.run_agent("new task", self.agent.permissions, self.agent.messages)
                self.assertEqual(self.agent.messages[:len(before)], before)
                self.assertEqual(len(self.requests), 1)
                self.assertEqual(self.last_result.stop_reason, "error")

    def test_compaction_discards_generated_tool_transcript_and_continues(self):
        (self.workspace / "seed.txt").write_text("real content", encoding="utf-8")
        self.agent.messages.append(answer("x" * 270000))
        fabricated = '[assistant 调用工具] write_file({"path":"never.txt","content":"done"})\n[tool] invented receipt'
        self.model([
            (tool_reply(call("real-read", "read_file", {"path": "seed.txt"})), 96001),
            (answer(fabricated), 100),
            (answer("continued"), 20),
        ])
        self.run_agent("preserve my rule", self.agent.permissions, self.agent.messages)
        following = self.requests[2]["messages"]
        self.assertNotIn("invented receipt", json.dumps(following, ensure_ascii=False))
        self.assertIn({"role": "user", "content": "preserve my rule"}, following)
        self.assertTrue(any("real-read" in m.get("content", "") for m in following))
        self.assertFalse((self.workspace / "never.txt").exists())
        self.assertEqual(self.last_result.stop_reason, "end_turn")

    def test_large_tool_arguments_are_bounded_for_summary_and_complete_pairs_survive(self):
        messages = [{"role": "system", "content": "system"}, {"role": "user", "content": "old task"},
                    tool_reply(call("write", "write_file", {"path": "large.txt", "content": "x" * 400000})),
                    {"role": "tool", "tool_call_id": "write", "content": '{"ok":true,"data":"written"}'}]
        self.model([(answer("large.txt 已写入，后续核对。"), 100), (answer("done"), 10)])
        self.run_agent("continue", self.agent.permissions, messages)
        self.assertEqual(len(self.requests), 2)
        self.assertLessEqual(context.estimate_input_tokens(self.requests[0]["messages"]), 128000)
        self.assert_complete_tool_pairs(self.requests[1]["messages"])
        self.assertLessEqual(context.estimate_input_tokens(self.requests[1]["messages"],
                                                          self.tools.get_schemas()), 28000)

    def test_context_configuration_is_validated_and_checkpoint_bound(self):
        for kwargs in ({"soft_limit": 128001}, {"hard_limit": 0},
                       {"compact_target_max": 96000}, {"compact_target_min": 30000}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                agent.Agent(None, self.tools, **kwargs)
        binding = agent.checkpoints.binding(self.agent)
        self.assertEqual(binding["limits"]["compact_target_max"], 28000)
