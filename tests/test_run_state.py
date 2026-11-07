"""RunState 的运行事实、共享轮数、中断和生命周期；仅使用本地响应。"""
import contextlib
import io
import json
import httpx

from scenario_support import OfflineScenario, agent, cli, permissions, answer, call, tool_reply
from verification import CheckResult


class RunStateScenarios(OfflineScenario):
    def test_interrupt_during_error_finalization_does_not_leave_a_returned_result(self):
        self.model([httpx.Response(503, json={"error": {"message": "offline error"}})])

        def interrupt_finalization(event):
            if event["type"] == "final_response":
                raise KeyboardInterrupt()

        self.agent.event_sink = interrupt_finalization
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.agent.run("请求失败后中断收尾")
        self.assertEqual(self.agent.run_state.phase, "interrupted")
        self.assertIsNone(self.agent.run_state.result)
        self.assertEqual(self.agent.run_state.steps_used, 1)
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 0)

    def test_verification_fact_is_visible_before_its_event_even_if_logging_fails(self):
        for trace_failure in (False, True):
            with self.subTest(trace_failure=trace_failure):
                seen = []
                check = CheckResult("pass", "本地检查通过", [{"value": "已核对"}])

                def observe(event):
                    if event["type"] == "verification":
                        state = self.agent.run_state
                        seen.append((state.phase, state.verification_attempts, state.verification))
                        if trace_failure:
                            raise OSError("offline verification trace failure")

                self.agent.event_sink = observe
                self.model([answer("候选结果")])
                self.run_agent("按条件检查", self.agent.permissions, verifier=lambda context: check)
                state = self.agent.run_state
                self.assertEqual(seen, [("verifying", 1, check)])
                self.assertIs(state.verification, check)
                self.assertIs(state.result.verification, check)
                self.assertEqual(state.phase, "stopped")
                self.assertEqual(state.result.stop_reason, "error" if trace_failure else "end_turn")
                self.assertEqual(self.agent.messages[-1]["content"], state.result.text)

    def test_batch_progress_is_observable_and_matches_the_final_result(self):
        self.assertTrue(hasattr(self.agent, "run_state"), "缺少可观察的 RunState")
        self.assertEqual(self.agent.run_state.phase, "idle")
        (self.workspace / "seed.txt").write_text("seed", encoding="utf-8")
        observations = []

        def observe(event):
            state = self.agent.run_state
            observations.append((event["type"], state.phase, state.steps_used,
                                 list(state.pending_calls), state.result))

        self.agent.event_sink = observe
        self.model([
            (tool_reply(call("read", "read_file", {"path": "seed.txt"}),
                        call("list", "list_dir", {"path": "."})), 10),
            (answer("内容是 seed。"), 20),
        ])
        result, messages = self.run_agent("读取并列出文件", self.agent.permissions)
        state = self.agent.run_state
        self.assertEqual((state.user_input, state.phase, state.steps_used),
                         ("读取并列出文件", "stopped", 2))
        self.assertEqual(result, "内容是 seed。")
        self.assertIs(state.result, self.last_result)
        self.assertEqual(state.result.text, messages[-1]["content"])
        self.assertEqual(state.result.stop_reason, "end_turn")
        self.assertIsNone(state.verification)
        self.assertEqual(state.verification_attempts, 0)
        self.assertEqual(state.pending_calls, {})
        self.assertEqual([r["call_id"] for r in state.tool_results], ["read", "list"])
        self.assertTrue(all(r["ok"] for r in state.tool_results))
        self.assertEqual(len(self.all_requests), 2)
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 32)
        self.assertIs(self.agent.events, state.events)
        self.assertEqual({event["run_id"] for event in self.agent.events}, {state.run_id})
        self.assertEqual([event["event_id"] for event in state.events],
                         list(range(1, len(state.events) + 1)))
        self.assertEqual(observations[0][:4], ("task_started", "preparing", 0, []))
        self.assertEqual([item[:4] for item in observations if item[0] == "tool_call"],
                         [("tool_call", "tools", 1, ["read", "list"]),
                          ("tool_call", "tools", 1, ["list"])])
        self.assertEqual([item[1:3] for item in observations if item[0] == "model_response"],
                         [("model", 1), ("model", 2)])
        self.assertEqual(observations[-1][0:3], ("final_response", "stopped", 2))
        self.assertIs(observations[-1][-1], self.last_result)

    def test_verification_feedback_keeps_one_execution_counter(self):
        self.assertTrue(hasattr(self.agent, "run_state"), "缺少运行轮数状态")
        (self.workspace / "seed.txt").write_text("seed", encoding="utf-8")
        for limit, replies, stop_reason, attempts, total in [
            (3, [(answer("初稿"), 10),
                 (tool_reply(call("read", "read_file", {"path": "seed.txt"})), 20),
                 (answer("修正"), 30)], "end_turn", 2, 63),
            (2, [(answer("初稿"), 10),
                 (tool_reply(call("read", "read_file", {"path": "seed.txt"})), 20)],
             "max_steps", 1, 32),
            (0, [], "max_steps", 0, 0),
        ]:
            with self.subTest(limit=limit):
                self.requests.clear()
                self.all_requests.clear()
                self.agent.token_usage.totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.agent.max_steps = limit
                seen = []

                def verify(context):
                    state = self.agent.run_state
                    seen.append((state.phase, state.steps_used, state.verification_attempts, state.run_id))
                    return CheckResult("fail" if len(seen) == 1 else "pass", "检查依据")

                self.model(replies)
                self.run_agent("检查文件", self.agent.permissions, verifier=verify)
                state = self.agent.run_state
                self.assertEqual((state.phase, state.steps_used, state.verification_attempts),
                                 ("stopped", limit, attempts))
                self.assertEqual(state.result.stop_reason, stop_reason)
                self.assertEqual(len(self.all_requests), limit)
                self.assertEqual(self.agent.token_usage.totals["total_tokens"], total)
                self.assertEqual([item[:3] for item in seen],
                                 ([("verifying", 1, 1), ("verifying", 3, 2)] if limit == 3
                                  else [("verifying", 1, 1)] if limit == 2 else []))
                self.assertTrue(all(item[3] == state.run_id for item in seen))
                self.assertEqual(state.verification, self.last_result.verification)
                self.assert_complete_tool_pairs(self.agent.messages)

    def test_errors_finalize_but_keyboard_interrupt_preserves_unknown_calls(self):
        self.assertTrue(hasattr(self.agent, "run_state"), "缺少异常阶段状态")
        self.model([httpx.Response(503, json={"error": {"message": "offline error"}})])
        self.run_agent("请求异常", self.agent.permissions)
        state = self.agent.run_state
        self.assertEqual((state.phase, state.steps_used, state.result.stop_reason),
                         ("stopped", 1, "error"))
        self.assertIs(state.result, self.last_result)
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 0)

        (self.workspace / "seed.txt").write_text("seed", encoding="utf-8")

        def interrupt_confirmation(name, args):
            raise KeyboardInterrupt()

        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=interrupt_confirmation)
        self.model([tool_reply(
            call("done", "read_file", {"path": "seed.txt"}),
            call("unknown", "write_file", {"path": "never.txt", "content": "x"}),
            call("pending", "list_dir", {"path": "."}))])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.agent.run("中断工具批次")
        state = self.agent.run_state
        self.assertEqual((state.phase, state.steps_used), ("interrupted", 1))
        self.assertIsNone(state.result)
        self.assertEqual(list(state.pending_calls), ["unknown", "pending"])
        self.assertEqual(state.pending_calls["unknown"]["function"]["name"], "write_file")
        json.dumps(state.pending_calls)  # 待处理调用保存为普通数据，不保留 SDK 对象。
        self.assertEqual([r["call_id"] for r in state.tool_results], ["done"])
        self.assertFalse((self.workspace / "never.txt").exists())
        self.assertFalse(any(e["type"] == "final_response" for e in state.events))

        def unavailable_trace(event):
            raise OSError("offline trace unavailable")

        self.agent.event_sink = unavailable_trace
        self.model([])
        self.run_agent("日志故障", self.agent.permissions)
        state = self.agent.run_state
        self.assertEqual((state.phase, state.steps_used, state.result.stop_reason),
                         ("stopped", 0, "error"))
        self.assertIs(state.result, self.last_result)
        self.assertIn("执行记录保存失败", state.result.text)
        self.assertEqual(self.agent.messages[-1]["content"], state.result.text)

    def test_new_runs_replace_run_facts_without_resetting_session_or_process_state(self):
        self.assertTrue(hasattr(self.agent, "run_state"), "缺少独立运行生命周期")
        other = agent.Agent(object(), self.tools)
        initial = other.run_state
        totals = self.agent.token_usage.totals
        self.agent.permissions.set_auto_approve(True)
        (self.workspace / "seed.txt").write_text("seed", encoding="utf-8")
        self.model([
            (tool_reply(call("read", "read_file", {"path": "seed.txt"})), 10),
            (answer("第一轮结果"), 20), (answer("第二轮结果"), 30), (answer("新会话结果"), 40),
        ])
        with contextlib.redirect_stdout(io.StringIO()):
            first_result = self.agent.run("第一问")
            first = self.agent.run_state
            self.agent.run("第二问")
            second = self.agent.run_state
            cli.start_new_session(self.agent, self.store)
            self.assertEqual(self.agent.messages, [{"role": "system", "content": agent.SYSTEM}])
            self.assertEqual(self.agent.last_prompt_tokens, 0)
            self.assertTrue(self.agent.permissions.auto_approve)
            self.assertIs(self.agent.token_usage.totals, totals)
            self.assertEqual(totals["total_tokens"], 63)
            self.agent.run("全新的问题")
        third = self.agent.run_state
        self.assertEqual(len({first.run_id, second.run_id, third.run_id}), 3)
        self.assertEqual((first.steps_used, second.steps_used, third.steps_used), (2, 1, 1))
        self.assertEqual([r["call_id"] for r in first.tool_results], ["read"])
        self.assertEqual(second.tool_results, [])
        self.assertEqual(third.tool_results, [])
        self.assertIs(first.result, first_result)
        self.assertEqual(first.events[-1]["text"], "第一轮结果")
        self.assertIn({"role": "user", "content": "第一问"}, self.requests[2]["messages"])
        self.assertEqual(self.requests[3]["messages"],
                         [{"role": "system", "content": agent.SYSTEM},
                          {"role": "user", "content": "全新的问题"}])
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 104)
        self.assertEqual((initial.phase, initial.steps_used, initial.events, initial.pending_calls),
                         ("idle", 0, [], {}))
        self.assertIs(other.run_state, initial)

