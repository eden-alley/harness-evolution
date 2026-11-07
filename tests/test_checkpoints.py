"""检查点与恢复：真实文件、真实硬退出、假模型响应，绝不访问真实 API。"""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, cli, context, permissions, answer, call, tool_reply
from verification import CheckResult
from test_context import history


class CheckpointScenarios(OfflineScenario):
    def enable(self, *, max_steps=3):
        self.assertTrue(hasattr(agent, 'checkpoints'), '缺少检查点模块')
        self.cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'checkpoints')
        self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp,
                                 max_steps=max_steps)
        return self.agent

    def restart(self, *, max_steps=3):
        self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp,
                                 max_steps=max_steps)
        return self.agent

    def interrupt_after_save(self, predicate):
        save = self.cp.save
        fired = []

        def observe(snapshot):
            save(snapshot)
            if not fired and predicate(snapshot['run_state']):
                fired.append(True)
                raise KeyboardInterrupt()

        self.replace(self.cp, 'save', observe)

    def interrupted_run(self, task='恢复任务', verifier=None):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            self.agent.run(task, verifier=verifier)

    def resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.agent.resume()

    def child(self, mode, *, limit=3):
        result = subprocess.run([
            sys.executable, '-B', '-X', 'utf8', str(Path(__file__).with_name('checkpoint_child.py')),
            str(Path(agent.__file__).parent), str(self.workspace), str(self.cp.directory), mode, str(limit),
        ], capture_output=True, text=True, timeout=20)
        return result

    def test_pending_batch_resumes_with_new_permission_and_never_replays_recorded_write(self):
        self.model([(tool_reply(call('a', 'write_file', {'path': 'a.txt', 'content': 'A'}),
                               call('b', 'write_file', {'path': 'b.txt', 'content': 'B'})), 10),
                    (answer('完成'), 20)])
        self.enable()
        self.agent.permissions.set_auto_approve(True)
        self.interrupt_after_save(lambda state: [r['status'] for r in state['call_records'].values()]
                                  == ['recorded', 'pending'])
        self.interrupted_run()
        snapshot = self.cp.load()
        run_id = snapshot['run_state']['run_id']
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 11)
        (self.workspace / 'a.txt').write_text('外部修改，禁止重写', encoding='utf-8')
        self.restart()
        asked = []
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append((name, deepcopy(args))) or 'y', ''))
        result = self.resume()
        self.assertEqual((result.text, result.stop_reason), ('完成', 'end_turn'))
        self.assertEqual((self.workspace / 'a.txt').read_text(encoding='utf-8'), '外部修改，禁止重写')
        self.assertEqual((self.workspace / 'b.txt').read_text(), 'B')
        self.assertEqual(asked, [('write_file', {'path': 'b.txt', 'content': 'B'})])
        self.assertEqual(self.agent.run_state.run_id, run_id)
        self.assertEqual(self.agent.run_state.steps_used, 2)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 21)
        self.assertEqual(len(self.all_requests), 2)
        self.assert_complete_tool_pairs(self.agent.messages)
        self.assertEqual(self.cp.load()['run_state']['result']['text'], result.text)

    def test_remaining_tools_at_last_round_finish_without_resetting_or_querying_again(self):
        self.model([tool_reply(call('a', 'list_dir', {}), call('b', 'write_file',
                                                                 {'path': 'last.txt', 'content': 'ok'}))])
        self.enable(max_steps=1)
        self.interrupt_after_save(lambda state: state['next_action'] == 'tools')
        self.interrupted_run()
        self.restart(max_steps=1)
        self.agent.permissions.set_auto_approve(True)
        result = self.resume()
        self.assertEqual(result.stop_reason, 'max_steps')
        self.assertEqual(self.agent.run_state.steps_used, 1)
        self.assertTrue((self.workspace / 'last.txt').exists())
        self.assertEqual(len(self.all_requests), 1)

    def test_hard_exit_after_effect_blocks_replay_and_all_later_work(self):
        self.model([])
        self.enable()
        child = self.child('effect')
        self.assertEqual(child.returncode, 47, child.stderr)
        self.assertEqual((self.workspace / 'effect.txt').read_text(), 'once')
        before = self.cp.path.read_bytes()
        self.restart()
        result = self.resume()
        self.assertEqual(result.stop_reason, 'error')
        self.assertIn('未知', result.text)
        self.assertEqual(self.cp.path.read_bytes(), before)
        self.assertEqual((self.workspace / 'model-count.txt').read_text().splitlines(), ['request'])
        self.assertEqual(self.all_requests, [])

    def test_hard_exit_after_saved_reply_uses_it_without_requery_or_old_token_recount(self):
        self.model([(answer('续跑完成'), 20)])
        self.enable()
        child = self.child('reply')
        self.assertEqual(child.returncode, 41, child.stderr)
        self.restart()
        self.agent.permissions.set_auto_approve(True)
        result = self.resume()
        self.assertEqual(result.text, '续跑完成')
        self.assertEqual((self.workspace / 'effect.txt').read_text(), 'once')
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 21)
        self.assertEqual(self.agent.run_state.steps_used, 2)
        self.assertEqual(self.agent.run_state.usage_history[0]['totals']['total_tokens'], 11)

    def test_lost_reply_consumes_new_round_and_marks_usage_unknown(self):
        self.model([(answer('重新请求完成'), 5)])
        self.enable()
        child = self.child('lost')
        self.assertEqual(child.returncode, 43, child.stderr)
        self.restart()
        result = self.resume()
        self.assertEqual(result.stop_reason, 'end_turn')
        self.assertEqual(self.agent.run_state.steps_used, 2)
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 6)
        self.assertEqual(self.agent.run_state.unknown_usage[0]['source'], 'model')

    def test_lost_reply_at_round_limit_never_issues_another_request(self):
        self.model([])
        self.enable(max_steps=1)
        child = self.child('lost', limit=1)
        self.assertEqual(child.returncode, 43, child.stderr)
        self.restart(max_steps=1)
        self.assertEqual(self.resume().stop_reason, 'max_steps')
        self.assertEqual(self.all_requests, [])

    def test_snapshot_write_failure_stops_before_or_after_handler_without_false_completion(self):
        self.assertTrue(hasattr(agent, 'checkpoints'), '缺少检查点模块')
        for fail_status in ('executing', 'recorded'):
            with self.subTest(fail_status=fail_status):
                path = self.workspace / 'fault.txt'
                path.unlink(missing_ok=True)
                self.model([tool_reply(call('write', 'write_file', {'path': 'fault.txt', 'content': 'effect'}))])
                self.cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / fail_status)
                self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp, max_steps=3)
                self.agent.permissions.set_auto_approve(True)
                save = self.cp.save

                def fail(snapshot):
                    if any(r['status'] == fail_status for r in snapshot['run_state']['call_records'].values()):
                        raise OSError('injected disk failure')
                    return save(snapshot)

                with patch.object(self.cp, 'save', side_effect=fail), contextlib.redirect_stdout(io.StringIO()):
                    result = self.agent.run('故障注入')
                self.assertEqual(result.stop_reason, 'error')
                self.assertIn('检查点', result.text)
                self.assertEqual(path.exists(), fail_status == 'recorded')
                self.assertFalse(self.cp.load()['synced'])
                statuses = [r['status'] for r in self.cp.load()['run_state']['call_records'].values()]
                self.assertEqual(statuses, ['pending'] if fail_status == 'executing' else ['executing'])
                self.assertEqual(self.cp.load()['run_state']['next_action'], 'tools')

    def test_atomic_write_failure_preserves_snapshot_and_legacy_json(self):
        self.model([answer('候选')])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
        self.interrupted_run()
        before = self.cp.path.read_bytes()
        self.store.save('default', [answer('旧会话')])
        with patch.object(context.os, 'replace', side_effect=OSError('replace failed')):
            with self.assertRaises(agent.checkpoints.CheckpointError):
                self.cp.save(self.cp.load())
            with self.assertRaises(OSError):
                self.store.save('default', [answer('新会话')])
        self.assertEqual(self.cp.path.read_bytes(), before)
        self.assertEqual(self.store.load('default'), [answer('旧会话')])

    def test_invalid_snapshot_or_binding_never_changes_runtime_or_calls_model(self):
        self.model([answer('候选')])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
        self.interrupted_run()
        saved = self.cp.load()
        original = self.cp.path.read_bytes()
        variants = [b'{', json.dumps({**saved, 'schema_version': 999}).encode()]
        for key in ('workspace', 'source', 'tools', 'model', 'limits'):
            damaged = deepcopy(saved)
            damaged['binding'][key] = 'changed'
            variants.append(json.dumps(damaged).encode())
        for data in variants:
            with self.subTest(data=data[:80]):
                self.cp.path.write_bytes(data)
                self.restart()
                before = deepcopy(self.agent.messages)
                result = self.resume()
                self.assertEqual(result.stop_reason, 'error')
                self.assertEqual(self.agent.messages, before)
        self.cp.path.write_bytes(original)
        self.assertEqual(len(self.all_requests), 1)

    def test_incomplete_custom_verifier_is_saved_but_cannot_be_silently_removed(self):
        self.model([answer('未验收候选')])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
        self.interrupted_run(verifier=lambda value: CheckResult('pass', '本地验收'))
        saved = self.cp.path.read_bytes()
        self.restart()
        result = self.resume()
        self.assertEqual(result.stop_reason, 'error')
        self.assertIn('验收', result.text)
        self.assertEqual(self.cp.path.read_bytes(), saved)
        self.assertEqual(len(self.all_requests), 1)

    def test_finished_unsynced_task_repairs_session_without_model_or_tool_replay(self):
        self.model([(answer('已经完成'), 10)])
        self.enable()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.run('完成任务')
        self.assertEqual(result.text, '已经完成')
        self.assertFalse(self.cp.load()['synced'])
        self.restart()
        with patch('builtins.input', side_effect=['/resume', 'exit']), contextlib.redirect_stdout(io.StringIO()):
            cli.main(self.agent, self.store)
        self.assertEqual(self.store.load('default')[-1]['content'], '已经完成')
        self.assertTrue(self.cp.load()['synced'])
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(len(list((self.cp.directory / 'archive').glob('*.json'))), 1)

    def test_cli_gates_new_input_preserves_auto_and_new_backups_without_old_usage_recount(self):
        self.model([(answer('恢复的回答'), 10), (answer('新任务回答'), 20)])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
        self.interrupted_run()
        self.restart()
        with patch('builtins.input', side_effect=['普通新任务', '/auto on', '/resume', '新任务', '/new', 'exit']), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main(self.agent, self.store)
        self.assertEqual(len(self.all_requests), 2)
        self.assertNotIn('普通新任务', json.dumps(self.all_requests, ensure_ascii=False))
        self.assertTrue(self.agent.permissions.auto_approve)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 21)
        self.assertEqual(self.agent.messages, [{'role': 'system', 'content': agent.SYSTEM}])
        self.assertEqual(len(list(self.store.directory.glob('default-backup-*.json'))), 1)
        self.assertEqual(len(list((self.cp.directory / 'archive').glob('*.json'))), 2)

    def test_process_lock_blocks_second_executor_and_is_released_after_owner_exit(self):
        self.model([])
        self.enable()
        with self.cp.lease():
            result = self.child('lock')
            self.assertEqual(result.returncode, 23, result.stderr)
        result = self.child('lock')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_archive_failure_never_acknowledges_completion_or_allows_overwriting_run(self):
        self.model([answer('已完成')])
        self.enable()
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent.run('归档故障')
        write = agent.checkpoints.atomic_write_json

        def fail_archive(path, data):
            if Path(path).parent.name == 'archive':
                raise OSError('archive disk failure')
            return write(path, data)

        with patch.object(agent.checkpoints, 'atomic_write_json', side_effect=fail_archive):
            with self.assertRaises(agent.checkpoints.CheckpointError):
                self.cp.acknowledge()
        self.assertFalse(self.cp.load()['synced'])
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.run('不得覆盖')
        self.assertEqual(result.stop_reason, 'error')
        self.assertEqual(len(self.all_requests), 1)

    def test_new_archives_unreadable_checkpoint_and_keeps_conversation_backup(self):
        self.model([])
        self.enable()
        self.cp.directory.mkdir(parents=True, exist_ok=True)
        damaged = b'{incomplete checkpoint'
        self.cp.path.write_bytes(damaged)
        previous = [{'role': 'system', 'content': agent.SYSTEM}, answer('旧记录')]
        self.agent.messages = deepcopy(previous)
        with contextlib.redirect_stdout(io.StringIO()):
            cli.start_new_session(self.agent, self.store)
        backup = next(self.store.directory.glob('default-backup-*.json'))
        self.assertEqual(json.loads(backup.read_text(encoding='utf-8')), previous)
        archive = next((self.cp.directory / 'archive').glob('*.json'))
        archived = json.loads(archive.read_text(encoding='utf-8'))
        self.assertEqual(bytes.fromhex(archived['unreadable_bytes_hex']), damaged)
        self.assertEqual(self.cp.load()['status'], 'idle')
        self.assertEqual(self.all_requests, [])

    def test_interruption_diagnostic_holds_exclusive_lease_during_read_and_update(self):
        self.model([answer('候选')])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
        self.interrupted_run()
        load = self.cp.load
        probes = []

        def observe_load():
            value = load()
            probes.append(self.child('lock').returncode)
            return value

        with patch.object(self.cp, 'load', side_effect=observe_load):
            self.agent._note_interruption('诊断更新')
        self.assertEqual(probes, [23])

    def test_broken_cursor_or_call_ledger_is_rejected_before_restoring_runtime(self):
        self.model([tool_reply(call('pending', 'write_file', {'path': 'never.txt', 'content': 'x'}))])
        self.enable()
        self.interrupt_after_save(lambda state: state['next_action'] == 'tools')
        self.interrupted_run()
        saved = self.cp.load()
        variants = []
        for change in ('empty_batch', 'missing_pending', 'fake_recorded', 'bad_usage', 'bad_response',
                       'synced_unfinished', 'false_idle', 'bad_cached_message'):
            damaged = deepcopy(saved)
            state = damaged['run_state']
            if change == 'empty_batch':
                state['active_batch'] = []
            elif change == 'missing_pending':
                state['pending_calls'] = {}
            elif change == 'fake_recorded':
                record = next(iter(state['call_records'].values()))
                record.update(status='recorded', result={'ok': True})
            elif change == 'bad_usage':
                state['usage_history'] = [0]
            elif change == 'bad_response':
                state['next_action'], state['response'] = 'response', None
            elif change == 'synced_unfinished':
                damaged['synced'] = True
            elif change == 'false_idle':
                damaged['status'] = 'idle'
            else:
                state['active_batch'], state['pending_calls'], state['call_records'] = [], {}, {}
                damaged['messages'] = damaged['messages'][:-1]
                state['next_action'] = 'response'
                state['response'] = {'message': {'role': 'user', 'content': '错误角色'}, 'protocol_error': None}
            variants.append(damaged)
        for data in variants:
            with self.subTest(data=data['run_state']['next_action']):
                self.cp.path.write_text(json.dumps(data), encoding='utf-8')
                self.restart()
                before = deepcopy(self.agent.messages)
                self.assertEqual(self.resume().stop_reason, 'error')
                self.assertEqual(self.agent.messages, before)
                self.assertFalse((self.workspace / 'never.txt').exists())
        self.assertEqual(len(self.all_requests), 1)

    def test_summary_interruption_keeps_reserved_round_and_same_instance_tokens(self):
        self.model([(answer('旧对话摘要'), 1000), (answer('继续完成'), 20)])
        self.enable()
        self.agent.messages = history(3)
        self.agent.last_prompt_tokens = self.agent.hard_limit + 1
        self.interrupt_after_save(lambda state: state['request_inflight'] == 'summary')
        self.interrupted_run()
        self.assertEqual(self.all_requests, [])
        result = self.resume()
        self.assertEqual(result.stop_reason, 'end_turn')
        self.assertEqual(self.agent.run_state.steps_used, 1)
        self.assertEqual(len(self.all_requests), 2)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 1022)
        before = deepcopy(self.agent.token_usage.totals)
        self.assertEqual(self.resume().text, '继续完成')
        self.assertEqual(self.agent.token_usage.totals, before)
        self.assertEqual(len(self.all_requests), 2)

    def test_in_process_verifier_keeps_feedback_limit_and_finished_verification_can_be_read(self):
        self.model([(answer('初稿'), 10), (answer('修正'), 20)])
        self.enable()
        attempts = []

        def check(value):
            attempts.append(value.candidate)
            return CheckResult('fail' if len(attempts) == 1 else 'pass', '本地检查依据')

        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.run('验收任务', verifier=check)
        self.assertEqual((result.stop_reason, result.verification.status), ('end_turn', 'pass'))
        self.assertEqual(attempts, ['初稿', '修正'])
        self.assertEqual(self.agent.run_state.verification_attempts, 2)
        self.assertEqual(self.agent.run_state.steps_used, 2)
        self.restart()
        restored = self.resume()
        self.assertEqual(restored.verification.status, 'pass')
        self.assertEqual(self.agent.run_state.verification_attempts, 2)
        self.assertEqual(len(self.all_requests), 2)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)

    def test_snapshot_revisions_advance_and_endpoint_credentials_are_not_persisted(self):
        self.model([answer('完成')])
        self.agent.client.base_url = 'https://offline.invalid/v1?api_key=secret-sentinel'
        self.enable()
        save, revisions = self.cp.save, []

        def observe(snapshot):
            save(snapshot)
            revisions.append(self.cp.load()['revision'])

        with patch.object(self.cp, 'save', side_effect=observe), contextlib.redirect_stdout(io.StringIO()):
            self.agent.run('检查元数据')
        self.assertTrue(all(b > a for a, b in zip(revisions, revisions[1:])), revisions)
        self.assertNotIn('secret-sentinel', self.cp.path.read_text(encoding='utf-8'))

    def test_trace_tails_never_block_cached_recovery_or_merge_the_resumed_event(self):
        for variant in ('partial_utf8', 'no_newline', 'non_object'):
            with self.subTest(variant=variant):
                self.all_requests.clear()
                self.model([(answer('缓存回答'), 10)])
                self.cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / variant)
                self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp, max_steps=3)
                self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
                self.interrupted_run()
                run_id = self.cp.load()['run_state']['run_id']
                complete = json.dumps({'run_id': run_id, 'event_id': 50, 'type': 'audit-only'}).encode()
                tail = {'partial_utf8': b'\xe4\xb8', 'no_newline': complete,
                        'non_object': b'0\n' + complete}[variant]
                self.store.directory.mkdir(parents=True, exist_ok=True)
                trace_name = 'damaged-' + variant
                trace = self.store.directory / (trace_name + '.trace.jsonl')
                trace.write_bytes(tail)
                self.restart()
                self.cp.trace_path = trace
                self.agent.event_sink = lambda event: self.store.append_event(trace_name, event)
                result = self.resume()
                self.assertEqual(result.text, '缓存回答')
                self.assertEqual(len(self.all_requests), 1)
                self.assertTrue(trace.read_bytes().startswith(tail))
                events = []
                for line in trace.read_bytes().splitlines():
                    try:
                        value = json.loads(line.decode('utf-8'))
                    except (UnicodeError, ValueError):
                        continue
                    if isinstance(value, dict):
                        events.append(value)
                resumed = [e for e in events if e.get('type') == 'task_resumed']
                self.assertEqual(len(resumed), 1)
                if variant != 'partial_utf8':
                    self.assertGreater(resumed[0]['event_id'], 50)
                self.assertEqual(events[-1]['type'], 'final_response')

    def test_first_new_or_restored_commit_failure_finalizes_the_public_state(self):
        for mode in ('new', 'resume'):
            with self.subTest(mode=mode):
                self.model([answer('候选')])
                self.cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / ('first-' + mode))
                self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp, max_steps=3)
                if mode == 'resume':
                    self.interrupt_after_save(lambda state: state['next_action'] == 'candidate')
                    self.interrupted_run()
                    self.restart()
                before = len(self.all_requests)
                with patch.object(self.cp, 'save', side_effect=OSError('first commit failure')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = self.agent.run('尚未开始') if mode == 'new' else self.agent.resume()
                self.assertEqual(result.stop_reason, 'error')
                self.assertEqual(self.agent.run_state.phase, 'stopped')
                self.assertIs(self.agent.run_state.result, result)
                self.assertEqual(self.agent.messages[-1]['content'], result.text)
                self.assertEqual(len(self.all_requests), before)

    def test_terminal_default_enables_checkpoint_but_injected_agent_stays_opt_in(self):
        self.model([(answer('终端完成'), 10)])
        runtime = self.agent
        directory = self.store.directory.parent / 'checkpoints'
        self.assertIsNone(runtime.checkpoint_store)
        self.assertFalse(directory.exists())
        # 只替换需要真实凭据的启动工厂，终端/Agent/文件保存均使用正式实现。
        with patch.object(cli, 'create_agent', return_value=runtime), \
             patch('builtins.input', side_effect=['终端任务', 'exit']), \
             contextlib.redirect_stdout(io.StringIO()):
            cli.main(store=self.store)
        self.assertEqual(runtime.checkpoint_store.directory, directory)
        self.assertTrue(runtime.checkpoint_store.load()['synced'])
        self.assertEqual(self.store.load('default')[-1]['content'], '终端完成')
        self.assertEqual(runtime.token_usage.totals['total_tokens'], 11)
        self.assertEqual(len(self.all_requests), 1)
