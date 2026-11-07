"""可选限制：0 关闭、正数开启；仅本地响应和假时钟，不消费真实 API。"""
from copy import deepcopy
import contextlib
import io
import os
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, cli, permissions, answer, call, tool_reply
from test_context import history
from verification import ModelVerifier


class LimitScenarios(OfflineScenario):
    def configure(self, *, max_steps=3, token_limit=0, wall_time_limit_seconds=0, cp=None):
        self.assertTrue(hasattr(agent, 'limits'), '缺少可选运行限制')
        self.agent = agent.Agent(self.agent.client, self.tools, max_steps=max_steps,
                                 token_limit=token_limit, wall_time_limit_seconds=wall_time_limit_seconds,
                                 checkpoint_store=cp)

    def run_quietly(self, task='限制测试', verifier=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.agent.run(task, verifier=verifier)

    def pause_after(self, cp, predicate):
        save, fired = cp.save, []

        def observe(snapshot):
            save(snapshot)
            if not fired and predicate(snapshot['run_state']):
                fired.append(True)
                raise KeyboardInterrupt()

        self.replace(cp, 'save', observe)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.agent.run('中断后续跑')

    def test_zero_caps_leave_behavior_unchanged_and_none_disables_step_limit(self):
        self.model([tool_reply(call(str(i), 'list_dir', {})) for i in range(4)] + [answer('完成')])
        self.configure(max_steps=None)
        result = self.run_quietly()
        self.assertEqual((result.stop_reason, result.text), ('end_turn', '完成'))
        self.assertEqual(len(self.all_requests), 5)
        self.assertEqual(self.agent.run_state.steps_used, 5)
        self.model([])
        self.configure(max_steps=0)
        self.assertEqual(self.run_quietly().stop_reason, 'max_steps')

    def test_token_cap_counts_summary_execution_and_model_verification_once(self):
        for cap, expected in ((15, 2), (25, 3)):
            with self.subTest(cap=cap):
                self.all_requests.clear()
                self.model([(answer('此前摘要'), 9), (answer('候选'), 9)], check_replies=[
                    (answer('{"status":"fail","reason":"需要修正","evidence":[]}'), 9)])
                self.configure(token_limit=cap)
                self.agent.messages = history(3)
                self.agent.last_prompt_tokens = self.agent.hard_limit + 1
                verify = ModelVerifier(self.agent.model_adapter, '明确条件', self.agent.token_usage.record)
                result = self.run_quietly(verifier=verify)
                self.assertEqual(result.stop_reason, 'token_limit')
                self.assertEqual(len(self.all_requests), expected)
                self.assertEqual(self.agent.run_state.limit_tokens_used, expected * 10)
                self.assertEqual(self.agent.token_usage.totals['total_tokens'], expected * 10)
                self.assertEqual(self.agent.messages[-1]['content'], result.text)

    def test_missing_usage_only_blocks_next_query_when_token_cap_is_enabled(self):
        for cap, stop, count in ((0, 'end_turn', 2), (100, 'usage_unknown', 1)):
            with self.subTest(cap=cap):
                self.all_requests.clear()
                self.model([tool_reply(call('list', 'list_dir', {})), answer('完成')])
                self.configure(token_limit=cap)
                result = self.run_quietly()
                self.assertEqual(result.stop_reason, stop)
                self.assertEqual(len(self.all_requests), count)
                self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)

    def test_wall_time_is_optional_and_checked_before_next_request_without_sleep(self):
        self.assertTrue(hasattr(agent, 'limits'), '缺少可选运行限制')
        for seconds, expected, count in ((0, 'end_turn', 2), (5, 'time_limit', 1)):
            with self.subTest(seconds=seconds):
                self.all_requests.clear()
                self.model([(tool_reply(call('list', 'list_dir', {})), 1), answer('完成')])
                self.configure(wall_time_limit_seconds=seconds)
                now = [1000.0]

                def observe(event):
                    if event['type'] == 'tool_result':
                        now[0] = 1006.0

                self.agent.event_sink = observe
                with patch.object(agent.limits, 'clock', side_effect=lambda: now[0]):
                    result = self.run_quietly()
                self.assertEqual(result.stop_reason, expected)
                self.assertEqual(len(self.all_requests), count)
                self.assertEqual(self.agent.run_state.tool_results[0]['ok'], True)

    def test_resume_reuses_cached_reply_and_keeps_token_allowance_separate_from_process_totals(self):
        self.model([(tool_reply(call('list', 'list_dir', {})), 10)])
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'cap')
        self.configure(token_limit=11, cp=cp)
        self.pause_after(cp, lambda state: state['next_action'] == 'tools')
        old_id = self.agent.run_state.run_id
        self.configure(token_limit=11, cp=cp)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, 'token_limit')
        self.assertEqual(self.agent.run_state.run_id, old_id)
        self.assertEqual(self.agent.run_state.limit_tokens_used, 11)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(cp.load()['run_state']['next_action'], 'finished')

    def test_lost_reply_is_unknown_instead_of_renewing_a_token_allowance(self):
        self.model([(answer('未保存回复'), 10)])
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'lost')
        self.configure(token_limit=100, cp=cp)

        def interrupt(event):
            if event['type'] == 'model_response':
                raise KeyboardInterrupt()

        self.agent.event_sink = interrupt
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.agent.run('响应落盘前中断')
        self.configure(token_limit=100, cp=cp)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, 'usage_unknown')
        self.assertEqual(len(self.all_requests), 1)

    def test_restart_does_not_reset_original_wall_clock_start(self):
        self.assertTrue(hasattr(agent, 'limits'), '缺少可选运行限制')
        self.model([(tool_reply(call('list', 'list_dir', {})), 10)])
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'clock')
        self.configure(wall_time_limit_seconds=5, cp=cp)
        with patch.object(agent.limits, 'clock', return_value=1000.0):
            self.pause_after(cp, lambda state: state['next_action'] == 'tools')
        self.configure(wall_time_limit_seconds=5, cp=cp)
        with patch.object(agent.limits, 'clock', return_value=1100.0), contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, 'time_limit')
        self.assertEqual(self.agent.run_state.limit_started_at, 1000.0)
        self.assertEqual(len(self.all_requests), 1)

    def test_invalid_limits_fail_before_model_or_credential_access(self):
        self.assertTrue(hasattr(agent, 'limits'), '缺少可选运行限制')
        for values in ({'token_limit': -1}, {'token_limit': True}, {'token_limit': 1.5},
                       {'wall_time_limit_seconds': -1}, {'wall_time_limit_seconds': float('inf')},
                       {'wall_time_limit_seconds': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                agent.Agent(None, self.tools, **values)
        for name, value in (('AGENT_STEP_LIMIT', '-1'), ('AGENT_TOKEN_LIMIT', '1.5'),
                            ('AGENT_TIME_LIMIT_SECONDS', 'nan')):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}, clear=True), \
                 patch('openai.OpenAI', side_effect=AssertionError('不应创建客户端')), \
                 patch('winreg.OpenKey', side_effect=AssertionError('不应读取密钥')):
                with self.assertRaises(ValueError):
                    cli.create_agent()

    def test_cli_switches_disable_steps_and_enable_token_or_time_caps(self):
        self.model([tool_reply(call(str(i), 'list_dir', {})) for i in range(31)] + [answer('完成')])
        client = self.agent.client
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline', 'NAMI_WORKSPACE': str(self.workspace),
                                    'AGENT_STEP_LIMIT': '0', 'AGENT_TOKEN_LIMIT': '0',
                                    'AGENT_TIME_LIMIT_SECONDS': '0'}, clear=True), \
             patch('openai.OpenAI', return_value=client):
            configured = cli.create_agent()
            with contextlib.redirect_stdout(io.StringIO()):
                result = configured.run('关闭轮数上限')
        self.assertEqual(result.text, '完成')
        self.assertEqual(configured.run_state.steps_used, 32)
        self.assertEqual(len(self.all_requests), 32)
        for setting, expected in (('AGENT_TOKEN_LIMIT', 'token_limit'),
                                  ('AGENT_TIME_LIMIT_SECONDS', 'time_limit')):
            with self.subTest(setting=setting):
                self.all_requests.clear()
                self.model([(tool_reply(call('list', 'list_dir', {})), 10)])
                now = [1000.0]
                environment = {'DEEPSEEK_API_KEY': 'offline', 'NAMI_WORKSPACE': str(self.workspace),
                               setting: '11' if setting == 'AGENT_TOKEN_LIMIT' else '5'}
                with patch.dict(os.environ, environment, clear=True), \
                     patch('openai.OpenAI', return_value=self.agent.client), \
                     patch.object(agent.limits, 'clock', side_effect=lambda: now[0]):
                    configured = cli.create_agent()
                    configured.event_sink = lambda event: (now.__setitem__(0, 1006.0)
                                                          if event['type'] == 'tool_result' else None)
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = configured.run('开启限制')
                self.assertEqual(result.stop_reason, expected)
                self.assertEqual(len(self.all_requests), 1)

    def test_limit_exit_write_failure_keeps_returned_error_and_public_state_consistent(self):
        self.model([(tool_reply(call('list', 'list_dir', {})), 10)])
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'fault')
        self.configure(token_limit=11, cp=cp)
        save = cp.save

        def fail_final(snapshot):
            if snapshot['run_state']['next_action'] == 'finished':
                raise OSError('limit final snapshot failed')
            return save(snapshot)

        with patch.object(cp, 'save', side_effect=fail_final):
            result = self.run_quietly()
        self.assertEqual(result.stop_reason, 'error')
        self.assertIs(self.agent.run_state.result, result)
        self.assertEqual(self.agent.messages[-1]['content'], result.text)
        self.assertIn('检查点', result.text)
        self.assertEqual(len(self.all_requests), 1)

