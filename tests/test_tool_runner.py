"""Tool runner extraction: public behavior and the narrow runtime boundary."""
from copy import deepcopy
import importlib
from pathlib import Path

from scenario_support import OfflineScenario, agent, call, permissions, tools


class RunnerContract(OfflineScenario):
    def test_runtime_boundary_is_dynamic_and_direct_invoke_has_no_record_or_save(self):
        module = importlib.import_module(f'{agent.models.__name__}.tool_runner')
        self.assertTrue(hasattr(module, 'ToolRuntime'))
        self.assertIsInstance(self.agent.tool_runner, module.ToolRunner)
        original = self.agent.tool_runner
        captured = []

        class Capture:
            def invoke(inner, runtime, name, args, operation_id=None, **kwargs):
                captured.append(runtime)
                return original.invoke(runtime, name, args, operation_id, **kwargs)
            def handle_call(inner, *args, **kwargs):
                return original.handle_call(*args, **kwargs)

        self.agent.tool_runner = Capture()
        saves = []
        self.agent._save_checkpoint = lambda: saves.append(True)
        args = {'path': '.'}
        self.assertTrue(self.agent._invoke_tool('list_dir', args).ok)
        self.assertEqual(args, {'path': '.'})
        self.assertEqual(self.agent.run_state.call_records, {})
        self.assertEqual(saves, [])
        runtime = captured[0]
        for name in ('client', 'model_adapter', 'max_steps', 'checkpoint_store'):
            self.assertFalse(hasattr(runtime, name), name)
        old = tuple(getattr(runtime, name) for name in ('tools', 'permissions', 'run_state', 'messages'))
        self.agent.tools = tools.Toolset(permissions.Workspace(self.workspace))
        self.agent.permissions = permissions.PermissionManager(workspace=self.agent.tools.workspace)
        self.agent.run_state = agent.RunState()
        self.agent.messages = []
        self.assertTrue(self.agent._invoke_tool('list_dir', {'path': '.'}).ok)
        for index, name in enumerate(('tools', 'permissions', 'run_state', 'messages')):
            self.assertIs(getattr(runtime, name), getattr(self.agent, name))
            self.assertIsNot(getattr(runtime, name), old[index])
            with self.assertRaises((AttributeError, TypeError)):
                setattr(runtime, name, None)

    def test_existing_operation_and_dynamic_override_during_event(self):
        c = call('a', 'list_dir', {'path': '.'})
        op = self.agent._register_call(c, 'model')
        seen = []
        def replaced(name, args, operation_id=None):
            seen.append((name, args, operation_id))
            return tools.ToolResult(True, data='override')
        def on_event(event):
            if event['type'] == 'tool_call':
                self.agent._invoke_tool = replaced
        self.agent.event_sink = on_event
        result = self.agent.handle_call(c, operation_id=op)
        self.assertTrue(result.ok)
        self.assertEqual(result.data, 'override')
        self.assertEqual(seen, [('list_dir', {'path': '.'}, op)])
        self.assertEqual(list(self.agent.run_state.call_records), [op])
        self.assertEqual(self.agent.run_state.call_records[op]['status'], 'recorded')
        self.assertEqual([e['type'] for e in self.agent.events], ['tool_call', 'tool_result'])

    def test_subclass_override_and_evidence_filter_verification(self):
        class CustomAgent(agent.Agent):
            def _invoke_tool(inner, name, args, operation_id=None):
                inner.intercepted = (name, args, operation_id)
                return super()._invoke_tool(name, args, operation_id)
        self.agent = CustomAgent(None, self.tools)
        self.agent.run_state.pending_calls['model-call'] = call('model-call', 'list_dir', {'path': '.'})
        model_call = call('model-call', 'list_dir', {'path': '.'})
        self.assertTrue(self.agent.handle_call(model_call).ok)
        self.assertEqual(self.agent.intercepted[:2], ('list_dir', {'path': '.'}))
        messages = deepcopy(self.agent.messages)
        verification_call = call('verify-call', 'list_dir', {'path': '.'})
        self.agent.run_state.pending_calls['verify-call'] = verification_call
        self.assertTrue(self.agent.handle_call(verification_call, source='verification').ok)
        self.assertEqual(self.agent.messages, messages)
        self.assertIn('verify-call', self.agent.run_state.pending_calls)
        evidence = self.agent._recorded_model_results()
        self.assertEqual([r['call_id'] for r in evidence], ['model-call'])
        op = next(k for k, v in self.agent.run_state.call_records.items() if v['source'] == 'model')
        self.agent.run_state.tool_results.clear()
        self.assertEqual(self.agent._recorded_model_results(), [])
        self.agent.run_state.tool_results.append(deepcopy(self.agent.run_state.call_records[op]['result']))
        self.agent.run_state.call_records[op]['status'] = 'executing'
        self.assertEqual(self.agent._recorded_model_results(), [])

    def test_malformed_json_emits_before_parse_and_mutating_permissions_do_not_execute(self):
        result = self.agent.handle_call(call('bad', 'list_dir', '{'))
        self.assertEqual(result.code, 'invalid_arguments')
        self.assertEqual([e['type'] for e in self.agent.events], ['tool_call', 'tool_result'])
        class MutatingPermission:
            def check(inner, name, args):
                args['path'] = 'changed.txt'
                return type('Decision', (), {'allowed': True})()
        self.agent.permissions = MutatingPermission()
        original = {'path': 'original.txt', 'content': 'sample'}
        result = self.agent._invoke_tool('write_file', original)
        self.assertEqual(result.code, 'execution_error')
        self.assertEqual(original, {'path': 'original.txt', 'content': 'sample'})
        self.assertEqual(list(self.workspace.iterdir()), [])
    def test_permission_exception_propagates_without_result_or_save(self):
        class BrokenPermission:
            def check(inner, name, args):
                raise RuntimeError('permission failed')
        self.agent.permissions = BrokenPermission()
        saves = []
        self.agent._save_checkpoint = lambda: saves.append(True)
        with self.assertRaisesRegex(RuntimeError, 'permission failed'):
            self.agent.handle_call(call('err', 'list_dir', {'path': '.'}))
        self.assertEqual([e['type'] for e in self.agent.events], ['tool_call'])
        self.assertEqual(self.agent.run_state.tool_results, [])
        self.assertEqual(saves, [])
        self.assertEqual(next(iter(self.agent.run_state.call_records.values()))['status'], 'pending')

