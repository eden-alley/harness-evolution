"""after_tool disk ordering and Agent-managed, at-most-once recovery boundaries."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from checkpoint_backend import CheckpointFixtureBackend
from scenario_support import OfflineScenario, agent, tools, answer, call, tool_reply


class AfterToolRecoveryScenarios(OfflineScenario):
    def enable(self, after_tool):
        self.cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / 'checkpoints')
        self.tools = tools.Toolset(self.tools.workspace, backend=CheckpointFixtureBackend())
        self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp,
                                 after_tool=after_tool, max_steps=3)
        self.agent.permissions.set_auto_approve(True)
        self.effects = []
        execute = self.tools.execute

        def execute_observed(name, args, **kwargs):
            result = execute(name, args, **kwargs)
            if name == 'write_file' and result.ok:
                self.effects.append(deepcopy(args))
            return result

        self.replace(self.tools, 'execute', execute_observed)

    def run_quietly(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.agent.run('one durable tool result')

    def restart_and_resume(self, after_tool):
        self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=self.cp,
                                 after_tool=after_tool, max_steps=3)
        self.agent.permissions.set_auto_approve(True)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.agent.resume()

    def test_callback_observes_real_recorded_disk_result_message_and_emitted_event(self):
        # Catches notification before result checkpoint/message/event processing.
        self.model([tool_reply(call('write', 'write_file', {'path': 'effect.txt', 'content': 'once'})),
                    answer('done')])
        observed, emitted = [], []

        def observe(summary):
            observed.append((summary, self.cp.load(), deepcopy(emitted)))

        self.enable(observe)
        self.agent.event_sink = lambda event: emitted.append(deepcopy(event))
        result = self.run_quietly()
        self.assertEqual(result.stop_reason, 'end_turn')
        self.assertEqual((len(self.all_requests), len(self.effects), len(observed)), (2, 1, 1))
        summary, snapshot, events = observed[0]
        state = snapshot['run_state']
        self.assertEqual(summary.run_id, state['run_id'])
        record = state['call_records'][summary.operation_id]
        self.assertEqual(record['status'], 'recorded')
        self.assertEqual((summary.call_id, summary.tool, summary.source, summary.ok),
                         ('write', 'write_file', 'model', True))
        self.assertEqual(state['tool_results'], [record['result']])
        self.assertEqual(state['pending_calls'], {})
        tool_messages = [m for m in snapshot['messages'] if m['role'] == 'tool']
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]['tool_call_id'], 'write')
        self.assertTrue(json.loads(tool_messages[0]['content'])['ok'])
        self.assertEqual([e['type'] for e in events].count('tool_result'), 1)
        self.assertEqual(events[-1]['type'], 'tool_result')
        self.assertEqual((self.workspace / 'effect.txt').read_text(encoding='utf-8'), 'once')
        self.assert_complete_tool_pairs(snapshot['messages'])

    def test_executing_checkpoint_failure_has_no_side_effect_or_notification(self):
        # Catches execution/notification despite failure of the pre-execution save.
        self.model([tool_reply(call('write', 'write_file', {'path': 'effect.txt', 'content': 'once'}))])
        notified = []
        self.enable(notified.append)
        save = self.cp.save

        def fail(snapshot):
            if any(r['status'] == 'executing' for r in snapshot['run_state']['call_records'].values()):
                raise OSError('injected pre-execution disk failure')
            return save(snapshot)

        with patch.object(self.cp, 'save', side_effect=fail):
            result = self.run_quietly()
        self.assertEqual(result.stop_reason, 'error')
        self.assertEqual((len(self.all_requests), len(self.effects), len(notified)), (1, 0, 0))
        self.assertFalse((self.workspace / 'effect.txt').exists())
        state = self.cp.load()['run_state']
        self.assertEqual([r['status'] for r in state['call_records'].values()], ['pending'])
        self.assertFalse(any(e['type'] == 'tool_result' and e['ok'] for e in state['events']))

    def test_result_checkpoint_failure_suppresses_notification_and_blocks_replay(self):
        # Catches a notification for uncommitted results and replay of an unknown effect.
        self.model([tool_reply(call('write', 'write_file', {'path': 'effect.txt', 'content': 'once'}))])
        notified = []
        self.enable(notified.append)
        save = self.cp.save

        def fail(snapshot):
            if any(r['status'] == 'recorded' for r in snapshot['run_state']['call_records'].values()):
                raise OSError('injected result disk failure')
            return save(snapshot)

        with patch.object(self.cp, 'save', side_effect=fail):
            result = self.run_quietly()
        self.assertEqual(result.stop_reason, 'error')
        self.assertEqual((len(self.all_requests), len(self.effects), len(notified)), (1, 1, 0))
        self.assertEqual((self.workspace / 'effect.txt').read_text(encoding='utf-8'), 'once')
        saved = self.cp.path.read_bytes()
        self.assertEqual([r['status'] for r in self.cp.load()['run_state']['call_records'].values()], ['executing'])
        (self.workspace / 'effect.txt').write_text('external edit', encoding='utf-8')
        resumed = self.restart_and_resume(notified.append)
        self.assertEqual(resumed.stop_reason, 'error')
        self.assertIn('未知', resumed.text)
        self.assertEqual(self.cp.path.read_bytes(), saved)
        self.assertEqual((len(self.all_requests), len(self.effects), len(notified)), (1, 1, 0))
        self.assertEqual((self.workspace / 'effect.txt').read_text(encoding='utf-8'), 'external edit')

    def child(self, mode):
        directory = self.store.directory.parent / 'hard-exit'
        result = subprocess.run([sys.executable, '-B', '-X', 'utf8',
            str(Path(__file__).with_name('after_tool_child.py')), str(Path(agent.__file__).parent),
            str(directory), mode], capture_output=True, text=True, timeout=20)
        return directory, result

    @staticmethod
    def observations(directory):
        return [json.loads(line) for line in (directory / 'observations.jsonl').read_text(encoding='utf-8').splitlines()]

    def assert_child_recovery(self, crash_mode, exit_code, expected_notifications):
        directory, child = self.child(crash_mode)
        self.assertEqual(child.returncode, exit_code, child.stdout + child.stderr)
        observations = self.observations(directory)
        self.assertEqual(sum(e['kind'] == 'request' for e in observations), 1)
        self.assertEqual(sum(e['kind'] == 'effect' for e in observations), 1)
        self.assertEqual(sum(e['kind'] == 'notification' for e in observations), expected_notifications)
        cp_path = directory / 'checkpoints' / 'default.json'
        before = json.loads(cp_path.read_text(encoding='utf-8'))
        self.assertEqual([r['status'] for r in before['run_state']['call_records'].values()], ['recorded'])
        self.assertEqual(before['run_state']['pending_calls'], {})
        self.assert_complete_tool_pairs(before['messages'])
        path = directory / 'workspace' / 'effect.txt'
        self.assertEqual(path.read_text(encoding='utf-8'), 'once')
        path.write_text('external edit survives resume', encoding='utf-8')
        _, resumed = self.child('resume')
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        observations = self.observations(directory)
        self.assertEqual(sum(e['kind'] == 'request' for e in observations), 2)
        self.assertEqual(sum(e['kind'] == 'effect' for e in observations), 1)
        self.assertEqual(sum(e['kind'] == 'notification' for e in observations), expected_notifications)
        recovery = [e for e in observations if e['kind'] == 'resume']
        self.assertEqual(len(recovery), 1)
        self.assertEqual(recovery[0]['decision']['status'], 'ready')
        self.assertEqual((recovery[0]['stop_reason'], recovery[0]['text']), ('end_turn', 'resumed'))
        self.assertTrue(recovery[0]['binding_unchanged'])
        self.assertEqual(before['binding'], recovery[0]['snapshot']['binding'])
        self.assertEqual(before['run_state']['call_records'], recovery[0]['snapshot']['run_state']['call_records'])
        self.assertEqual(path.read_text(encoding='utf-8'), 'external edit survives resume')
        self.assert_complete_tool_pairs(recovery[0]['snapshot']['messages'])
        # The second process continues from the durable tool message, not a replay.
        resume_requests = [e for e in observations if e['kind'] == 'request' and e['mode'] == 'resume']
        self.assertEqual(len(resume_requests), 1)
        self.assertEqual([m['tool_call_id'] for m in resume_requests[0]['messages'] if m['role'] == 'tool'], ['write'])

    def test_hard_exit_after_result_save_before_callback_resumes_without_replay_or_notification(self):
        # Catches replay/backfill notifications when the process dies after commit.
        self.assert_child_recovery('before_notify', 51, 0)

    def test_hard_exit_in_callback_keeps_observation_without_redelivery_on_resume(self):
        # Catches notification redelivery when a callback completed an external effect then died.
        self.assert_child_recovery('in_notify', 52, 1)
