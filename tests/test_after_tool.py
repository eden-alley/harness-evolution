"""After-tool notifications at the recorded-result boundary, entirely offline."""
from dataclasses import FrozenInstanceError, fields
import json
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, answer, call, permissions, tool_reply, tools


class AfterToolContract(OfflineScenario):
    def configured(self, callback):
        self.agent = agent.Agent(None, self.tools, after_tool=callback)
        return self.agent

    def test_summary_is_minimal_immutable_and_after_record_event_and_save(self):
        observed = []
        def callback(summary):
            observed.append((summary, list(order), self.agent.run_state.call_records[summary.operation_id]['status']))
        self.configured(callback)
        order = []
        self.agent.event_sink = lambda event: order.append(event['type'])
        self.agent._save_checkpoint = lambda: order.append('save')
        result = self.agent.handle_call(call('id1', 'list_dir', {'path': '.'}))
        summary, at_callback, status = observed[0]
        self.assertTrue(result.ok)
        self.assertEqual(at_callback[-2:], ['tool_result', 'save'])
        self.assertEqual(status, 'recorded')
        self.assertEqual([f.name for f in fields(summary)], ['run_id', 'operation_id', 'call_id', 'tool', 'source', 'ok', 'code', 'exit_code', 'result_contract'])
        self.assertIsInstance(summary, agent._tool_runner.ToolResultSummary)
        self.assertEqual((summary.run_id, summary.call_id, summary.tool, summary.source, summary.ok, summary.code, summary.exit_code, summary.result_contract),
                         (None, 'id1', 'list_dir', 'model', True, '', None, 'structured'))
        self.assertFalse(hasattr(summary, '__dict__'))
        for sensitive in ('arguments', 'data', 'message', 'available_tools'):
            self.assertFalse(hasattr(summary, sensitive))
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            summary.ok = False

    def test_summary_normalizes_string_subclasses_without_mutating_result(self):
        class Code(str):
            pass
        code = Code('custom_error')
        code.payload = ['private']
        contract = Code('structured')
        contract.payload = ['private-contract']
        original = tools.ToolResult(False, code=code, message='failure', result_contract=contract)
        received = []
        self.configured(received.append)
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: ('y', ''))
        with patch.dict(tools.TOOLS):
            @tools.tool('subclass_probe', 'local probe', {'type': 'object', 'additionalProperties': False})
            def probe(toolset):
                return original
            result = self.agent.handle_call(call('subclass', 'subclass_probe', {}))
        self.assertIs(result, original)
        self.assertIs(result.code, code)
        self.assertEqual(code.payload, ['private'])
        self.assertIs(result.result_contract, contract)
        self.assertEqual(contract.payload, ['private-contract'])
        self.assertEqual(len(received), 1)
        self.assertIs(type(received[0].code), str)
        self.assertEqual(received[0].code, 'custom_error')
        self.assertFalse(hasattr(received[0].code, 'payload'))
        self.assertIs(type(received[0].result_contract), str)
        self.assertEqual(received[0].result_contract, 'structured')
        self.assertFalse(hasattr(received[0].result_contract, 'payload'))

    def test_denied_malformed_and_invalid_schema_notify_without_execution(self):
        received = []
        self.configured(received.append)
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace, ask=lambda name, args: ('n', ''))
        denied = self.agent.handle_call(call('deny', 'write_file', {'path': 'x.txt', 'content': 'secret'}))
        malformed = self.agent.handle_call(call('json', 'list_dir', '{'))
        invalid = self.agent.handle_call(call('schema', 'write_file', {'path': 'y.txt', 'content': 123}))
        self.assertFalse(denied.ok)
        self.assertEqual(malformed.code, 'invalid_arguments')
        self.assertEqual(invalid.code, 'invalid_arguments')
        self.assertEqual([(s.call_id, s.ok, s.code) for s in received],
                         [('deny', False, denied.code), ('json', False, 'invalid_arguments'), ('schema', False, 'invalid_arguments')])
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_verification_and_legacy_preserve_message_and_contract(self):
        received = []
        self.configured(received.append)
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: ('y', ''))
        with patch.dict(tools.TOOLS):
            tools.TOOLS['legacy_probe'] = {'schema': {'type': 'function', 'function': {
                'name': 'legacy_probe', 'description': 'legacy', 'parameters': {'type': 'object'}}},
                'func': lambda toolset: {'error': 'opaque'}}
            model_call = call('model', 'legacy_probe', {})
            self.agent.run_state.pending_calls['model'] = model_call
            self.agent.handle_call(model_call)
            before = list(self.agent.messages)
            self.agent.handle_call(call('verify', 'list_dir', {}), source='verification')
        self.assertEqual(self.agent.messages, before)
        self.assertEqual(json.loads(before[-1]['content'])['result_contract'], 'legacy')
        self.assertEqual([(s.source, s.result_contract, s.ok) for s in received],
                         [('model', 'legacy', True), ('verification', 'structured', True)])

    def test_callback_exception_and_warning_exception_do_not_change_loop(self):
        callbacks = []
        executions = []
        def broken(summary):
            callbacks.append(summary)
            raise RuntimeError('callback failed')
        self.configured(broken)
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: ('y', ''))
        execute = self.tools.execute
        def counted_execute(*args, **kwargs):
            result = execute(*args, **kwargs)
            executions.append((args, result))
            return result
        self.model([tool_reply(call('write', 'write_file', {'path': 'once.txt', 'content': 'once'})), answer('done')])
        with patch.object(self.tools, 'execute', side_effect=counted_execute), \
             patch.object(agent._tool_runner.logger, 'warning', side_effect=RuntimeError('logger failed')) as warning:
            text, messages = self.run_agent('write', self.agent.permissions)
        self.assertEqual(text, 'done')
        self.assertEqual(len(self.all_requests), 2)
        self.assertEqual(warning.call_count, 1)
        self.assertEqual(len(callbacks), 1)
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0][0][0], 'write_file')
        self.assertIsInstance(executions[0][1], tools.ToolResult)
        self.assertTrue(executions[0][1].ok)
        self.assertEqual(json.loads(next(m['content'] for m in messages if m.get('tool_call_id') == 'write'))['ok'], True)
        self.assertEqual((self.workspace / 'once.txt').read_text(encoding='utf-8'), 'once')
        self.assertEqual(len(self.agent.run_state.tool_results), 1)

    def test_callback_exception_is_logged_and_return_value_has_no_effect(self):
        received = []
        def callback(summary):
            received.append(summary)
            if len(received) == 1:
                raise ValueError('broken')
            return {'retry': True}
        self.configured(callback)
        self.model([tool_reply(call('a', 'list_dir', {}), call('b', 'list_dir', {})), answer('done')])
        with self.assertLogs(agent._tool_runner.logger, level='WARNING') as log:
            text, _ = self.run_agent('list', self.agent.permissions)
        self.assertEqual(text, 'done')
        self.assertEqual(len(received), 2)
        self.assertEqual(len(self.all_requests), 2)
        self.assertEqual(len(self.agent.run_state.tool_results), 2)
        self.assertTrue(log.output)

    def test_base_exceptions_propagate_after_result_recorded(self):
        for exception in (KeyboardInterrupt, SystemExit):
            with self.subTest(exception=exception):
                def abort(summary):
                    raise exception()
                self.configured(abort)
                with self.assertRaises(exception):
                    self.agent.handle_call(call('id', 'list_dir', {}))
                self.assertEqual(len(self.agent.run_state.tool_results), 1)
                self.assertEqual(self.agent.events[-1]['type'], 'tool_result')

    def test_direct_invoke_and_finish_unknown_do_not_notify(self):
        received = []
        self.configured(received.append)
        self.assertTrue(self.agent._invoke_tool('list_dir', {}).ok)
        pending = call('unfinished', 'list_dir', {})
        self.agent.run_state.pending_calls['unfinished'] = pending
        self.agent._finish('stopped', 'error')
        self.assertEqual(received, [])
        self.assertEqual(self.agent.run_state.tool_results[-1]['code'], 'execution_error')

    def test_dynamic_run_state_and_no_notification_on_failed_save(self):
        received = []
        self.configured(received.append)
        self.agent.run_state = agent.RunState(run_id='first')
        self.agent.handle_call(call('a', 'list_dir', {}))
        self.agent.run_state = agent.RunState(run_id='second')
        self.agent.handle_call(call('b', 'list_dir', {}))
        self.assertEqual([(s.run_id, s.call_id) for s in received], [('first', 'a'), ('second', 'b')])
        self.agent._save_checkpoint = lambda: (_ for _ in ()).throw(RuntimeError('save failed'))
        with self.assertRaisesRegex(RuntimeError, 'save failed'):
            self.agent.handle_call(call('c', 'list_dir', '{'))
        self.assertEqual(len(received), 2)

    def test_no_callback_does_not_construct_summary_or_log(self):
        with patch.object(agent._tool_runner, 'ToolResultSummary', side_effect=AssertionError('constructed')), \
             patch.object(agent._tool_runner.logger, 'warning', side_effect=AssertionError('warned')):
            self.assertTrue(self.agent.handle_call(call('id', 'list_dir', {})).ok)
