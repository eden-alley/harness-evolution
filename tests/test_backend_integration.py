import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agent
from permissions import Workspace
from tools import Toolset
import tools


class FakeBackend:
    def __init__(self, name='fake'):
        self.binding = {'name': name, 'policy_version': 1}
        self.calls = []

    def run(self, command, workspace, timeout, *, shell='cmd'):
        self.calls.append((command, workspace, timeout, shell))
        return subprocess.CompletedProcess(command, 0, b'fake output', b'')


class BackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Workspace(directory.name)

    def test_alternative_backend_executes_through_toolset(self):
        backend = FakeBackend()
        toolset = Toolset(self.workspace, backend=backend)
        result = toolset.execute('run_shell', {'command': 'demo', 'shell': 'bash', 'timeout': 5})
        self.assertTrue(result.ok)
        self.assertEqual(result.data, '[exit 0]\nfake output')
        self.assertEqual(backend.calls, [('demo', self.workspace.root, 5, 'bash')])
        self.assertEqual(toolset.execution_binding, {'name': 'fake', 'policy_version': 1})

    def test_text_completed_process_preserves_exit_status_and_newlines(self):
        backend = FakeBackend()
        backend.run = lambda command, workspace, timeout: subprocess.CompletedProcess(
            command, 7, '中文输出\r\n', '错误说明\r\n')
        result = Toolset(self.workspace, backend=backend).execute('run_shell', {'command': 'demo'})
        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.data, '[exit 7]\n中文输出\n错误说明\n')

    def test_runner_injection_and_later_assignment_do_not_claim_mxc(self):
        def runner(command, workspace, timeout):
            return subprocess.CompletedProcess(command, 0, b'runner output', b'')
        toolset = Toolset(self.workspace, run_command=runner)
        self.assertEqual(toolset.execute('run_shell', {'command': 'demo'}).data,
                         '[exit 0]\nrunner output')
        self.assertEqual(toolset.execution_binding['name'], 'injected_runner')
        toolset = Toolset(self.workspace)
        toolset.run_command = runner
        self.assertEqual(toolset.execute('run_shell', {'command': 'demo'}).data,
                         '[exit 0]\nrunner output')
        self.assertEqual(toolset.execution_binding['name'], 'injected_runner')
        with self.assertRaises(ValueError):
            Toolset(self.workspace, backend=FakeBackend(), run_command=runner)

    def test_patching_runner_restores_backend_identity(self):
        toolset = Toolset(self.workspace)
        original = toolset.execution_binding
        def replacement(command, workspace, timeout):
            return subprocess.CompletedProcess(command, 0, b'fixture', b'')
        with patch.object(toolset, 'run_command', replacement):
            self.assertEqual(toolset.execution_binding['name'], 'injected_runner')
            self.assertEqual(toolset.execution_info('run_shell')['location'], 'unknown')
        self.assertEqual(toolset.execution_binding, original)
        self.assertEqual(toolset.execution_info('run_shell')['location'], 'sandbox')

    def test_distinct_named_runners_have_distinct_checkpoint_bindings(self):
        def first(command, workspace, timeout):
            return subprocess.CompletedProcess(command, 0, b'', b'')
        def second(command, workspace, timeout):
            return subprocess.CompletedProcess(command, 0, b'', b'')
        toolset = Toolset(self.workspace, run_command=first)
        first_binding = toolset.execution_binding
        toolset.run_command = second
        self.assertNotEqual(first_binding, toolset.execution_binding)
        self.assertNotIn('first', json.dumps(first_binding))

    def test_tool_locations_and_event_do_not_expose_mcp_configuration(self):
        mcp = SimpleNamespace(tools={'mcp__notes__read': {}},
                              binding={'url': 'https://secret.example/path', 'args': ['secret']})
        toolset = Toolset(self.workspace, backend=FakeBackend(), mcp=mcp)
        self.assertEqual(toolset.execution_info('run_shell'),
                         {'location': 'backend', 'backend': {'name': 'fake', 'policy_version': 1}})
        for name in ('list_dir', 'read_file', 'write_file', 'read_tool_result', 'fetch_url'):
            self.assertEqual(toolset.execution_info(name), {'location': 'host'})
        self.assertEqual(toolset.execution_info('mcp__notes__read'), {'location': 'remote_mcp'})
        mcp.binding['url'] = None
        self.assertEqual(toolset.execution_info('mcp__notes__read'), {'location': 'host_mcp'})
        self.assertEqual(toolset.execution_info('missing'), {'location': 'unavailable'})
        with patch.dict(tools.TOOLS, {'local_extension': {}}):
            self.assertEqual(toolset.execution_info('local_extension'), {'location': 'host'})
        runtime = agent.Agent(None, toolset)
        runtime.handle_call({'id': '1', 'function': {'name': 'read_file', 'arguments': '{"path":"missing"}'}})
        event = runtime.events[0]
        self.assertEqual(event['execution'], {'location': 'host'})
        self.assertNotIn('execution', runtime.run_state.tool_results[0])
        self.assertNotIn('secret', json.dumps(event))

    def test_checkpoint_binding_changes_when_backend_changes(self):
        runtime = agent.Agent(None, Toolset(self.workspace, backend=FakeBackend('first')))
        first = agent.checkpoints.binding(runtime)
        runtime.tools = Toolset(self.workspace, backend=FakeBackend('second'))
        second = agent.checkpoints.binding(runtime)
        self.assertEqual(first['execution_backend'], {'name': 'first', 'policy_version': 1})
        self.assertNotEqual(first, second)
        json.dumps(second, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
