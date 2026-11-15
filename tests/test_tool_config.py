"""工具名单的离线契约。"""
from scenario_support import OfflineScenario, tools, permissions


class ToolConfigScenarios(OfflineScenario):
    def configured(self, names, **kwargs):
        return tools.Toolset(permissions.Workspace(self.workspace), run_command=self.fixture_command,
                             enabled_tools=names, **kwargs)

    def test_selection_is_copied_and_preserves_registry_order(self):
        default_names = [s['function']['name'] for s in self.tools.get_schemas()]
        selected = list(reversed(default_names))
        configured = self.configured(selected)
        selected.clear()
        self.assertEqual([s['function']['name'] for s in configured.get_schemas()], default_names)
        self.assertEqual(configured.enabled_tools, frozenset(default_names))
        self.assertEqual([s['function']['name'] for s in self.tools.get_schemas()], default_names)

    def test_disabled_tools_are_unavailable_everywhere_before_permission(self):
        configured = self.configured(['list_dir'])
        self.agent.tools = configured
        asked = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=configured.workspace, ask=lambda *args: asked.append(args))
        for name, args in [('read_file', {'path': 'x'}), ('write_file', {'path': 'x', 'content': 'x'}),
                           ('run_shell', {'command': 'echo x'})]:
            with self.subTest(name=name):
                self.assertFalse(configured.is_available(name))
                self.assertEqual(configured.execution_info(name)['location'], 'unavailable')
                self.assertEqual(configured.validate_call(name, args).code, 'tool_unavailable')
                self.assertEqual(configured.execute(name, args).code, 'tool_unavailable')
                self.assertEqual(self.agent._invoke_tool(name, args).code, 'tool_unavailable')
        for args in (None, [], 'bad'):
            with self.subTest(name='read_file', args=args):
                self.assertEqual(configured.validate_call('read_file', args).code, 'tool_unavailable')
                self.assertEqual(configured.execute('read_file', args).code, 'tool_unavailable')
                self.assertEqual(self.agent._invoke_tool('read_file', args).code, 'tool_unavailable')
        self.assertEqual(configured.validate_call('list_dir', None).code, 'invalid_arguments')
        self.assertEqual(asked, [])
        self.assertEqual([s['function']['name'] for s in configured.get_schemas()], ['list_dir'])

    def test_invalid_names_and_progress_configuration_are_rejected(self):
        for names in ('read_file', [None], [''], ['read_file', 'read_file'], ['unknown'], ['*'],
                      ['+read_file'], ['update_progress']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.configured(names)
        self.assertTrue(self.configured(('update_progress',), task_progress=True).is_available('update_progress'))
import contextlib
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scenario_support import agent, answer, cli
from verification import command_verifier
import test_mcp_toolset as mcp_tests


class ToolConfigIntegration(OfflineScenario):
    configured = ToolConfigScenarios.configured

    def test_empty_catalog_request_and_answer(self):
        self.agent.tools = self.configured([])
        self.model([(answer('完成'), 5)])
        self.run_agent('完成任务', self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, 'end_turn')
        self.assertNotIn('tools', self.requests[0])
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 6)

    def test_progress_prompt_only_when_tool_selected(self):
        self.agent.tools = self.configured([], task_progress=True)
        self.assertEqual(self.agent._task_progress_context(), [])
        self.agent.tools = self.configured(['update_progress'], task_progress=True)
        self.assertTrue(self.agent._task_progress_context())

    def test_command_verifier_cannot_bypass_disabled_shell(self):
        self.agent.tools = self.configured([])
        context = type('Context', (), {'run_tool': lambda _, name, args: self.agent._invoke_tool(name, args)})()
        result = command_verifier('echo x')(context)
        self.assertEqual(result.status, 'error')
        self.assertEqual(result.evidence[0]['code'], 'tool_unavailable')

    def test_binding_ignores_selection_order_but_detects_different_tools(self):
        def binding(names):
            runtime = agent.Agent(None, self.configured(names))
            return agent.checkpoints.binding(runtime)['tools']
        self.assertEqual(binding(['read_file', 'list_dir']), binding(['list_dir', 'read_file']))
        self.assertNotEqual(binding(['read_file', 'list_dir']), binding(['list_dir']))

    def test_explicit_search_without_rg_fails_at_start(self):
        with patch('file_search.discover_rg', return_value=None):
            for name in tools.SEARCH_TOOLS:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.configured([name])

    def test_search_disappears_after_valid_explicit_selection(self):
        # 只借用工作区外的现有文件模拟已发现的 rg，断言整个过程不执行它。
        with patch('file_search.discover_rg', return_value=Path(sys.executable)):
            local = self.configured(['find_files', 'read_file'])
        self.assertTrue(local.is_available('find_files'))
        self.assertIn('find_files', [s['function']['name'] for s in local.get_schemas()])
        local.file_search.rg_path = None
        with patch.object(tools, '_run_search', side_effect=AssertionError('搜索不应执行')):
            result = local.execute('find_files', {'pattern': '*'})
        self.assertFalse(local.is_available('find_files'))
        self.assertEqual([s['function']['name'] for s in local.get_schemas()], ['read_file'])
        self.assertEqual(local.execution_info('find_files'), {'location': 'unavailable'})
        self.assertEqual(result.code, 'tool_unavailable')
        self.assertEqual(result.available_tools, ['read_file'])

    def test_long_line_recovery_respects_disabled_shell_and_preserves_file(self):
        local = self.configured(['read_file'])
        original = '原始任务正文' + '中' * 5000
        path = self.workspace / 'long.txt'
        path.write_text(original, encoding='utf-8')
        description = local.get_schemas()[0]['function']['description']
        self.assertIn('若已启用 run_shell', description)
        with patch.object(tools, 'run_shell', side_effect=AssertionError('shell 不应执行')):
            result = local.execute('read_file', {'path': 'long.txt', 'max_output_tokens': 4096})
        self.assertEqual(result.code, 'output_limit_exceeded')
        self.assertIn('未启用 run_shell', result.message)
        self.assertNotIn('用现有 run_shell 分段读取', result.message)
        self.assertEqual(path.read_text(encoding='utf-8'), original)

    def test_create_agent_receives_explicit_environment_selection(self):
        fake = SimpleNamespace(client=object())
        with patch.dict(os.environ, {'AGENT_TOOLS': '["read_file", "list_dir"]',
                                     'DEEPSEEK_API_KEY': 'offline-dummy'}, clear=True), \
             patch.object(cli.models, 'get_model', return_value=fake):
            runtime = cli.create_agent(workspace=self.tools.workspace)
        self.assertEqual(runtime.tools.enabled_tools, frozenset({'read_file', 'list_dir'}))
        self.assertEqual({s['function']['name'] for s in runtime.tools.get_schemas()},
                         {'read_file', 'list_dir'})

    def test_environment_json_and_early_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(cli.tool_setting())
        for value in ('', 'null', '{}', '"read_file"', 'oops'):
            with self.subTest(value=value), patch.dict(os.environ, {'AGENT_TOOLS': value}, clear=True):
                with self.assertRaises(ValueError):
                    cli.tool_setting()
                with patch.object(cli.models, 'get_model', side_effect=AssertionError('created adapter')):
                    with self.assertRaises(ValueError):
                        cli.create_agent(workspace=self.tools.workspace)
        with patch.dict(os.environ, {'AGENT_TOOLS': '[]'}, clear=True):
            self.assertEqual(cli.tool_setting(), [])
        with patch.dict(os.environ, {'AGENT_TOOLS': '["unknown"]'}, clear=True):
            with patch.object(cli.models, 'get_model', side_effect=AssertionError('created adapter')):
                with self.assertRaises(ValueError):
                    cli.create_agent(workspace=self.tools.workspace)

    def test_empty_list_does_not_skip_configured_mcp_connection(self):
        events = []
        @contextlib.contextmanager
        def connection(path, *, workspace):
            events.append(('opened', path))
            yield None
            events.append(('closed', path))
        with patch.dict(os.environ, {'AGENT_TOOLS': '[]', 'NAMI_MCP_CONFIG': 'stub.json',
                                     'NAMI_WORKSPACE': str(self.workspace)}, clear=True), \
             patch.object(cli, 'open_mcp', side_effect=connection), \
             patch.object(cli, 'create_agent', return_value=self.agent), \
             patch.object(cli, 'run_session'):
            cli.main()
        self.assertEqual(events, [('opened', 'stub.json'), ('closed', 'stub.json')])

    def test_tools_command_is_local_and_extra_args_show_usage(self):
        self.agent.tools = self.configured(['list_dir'])
        before = list(self.agent.messages)
        with patch('builtins.input', side_effect=['/tools', '/tools extra', 'exit']), \
             patch.object(self.agent, 'run', side_effect=AssertionError('model called')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            cli.interact(self.agent, self.store)
        self.assertIn('list_dir', output.getvalue())
        self.assertEqual(self.agent.tools.execution_info('list_dir')['location'], 'host')
        self.assertIn('宿主机', output.getvalue())
        self.assertIn('/tools', output.getvalue())
        self.assertEqual(self.agent.messages, before)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)


class MCPToolConfigScenarios(OfflineScenario):
    def setUp(self):
        super().setUp()
        self.config = mcp_tests.MCPConfig("notes", "python.exe", ("notes.py",), self.workspace,
                                          ("lookup.note", "manual"), frozenset({"lookup.note"}))

    adapter = mcp_tests.MCPToolsetScenarios.adapter
    def test_selection_intersects_mcp_catalog_and_cannot_enlarge_allow_list(self):
        calls = []
        adapter = self.adapter(lambda *args: calls.append(args), listed=[mcp_tests.advertised('hidden'),
                               mcp_tests.advertised('manual'), mcp_tests.advertised('lookup.note')])
        selected = 'mcp__notes__lookup_note'
        local = tools.Toolset(self.tools.workspace, mcp=adapter, enabled_tools=[selected])
        self.assertEqual([s['function']['name'] for s in local.get_schemas()], [selected])
        self.assertEqual(local.execute('mcp__notes__manual', {'note_id': 'x'}).code, 'tool_unavailable')
        self.assertEqual(local.execution_info('mcp__notes__manual')['location'], 'unavailable')
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            tools.Toolset(self.tools.workspace, mcp=adapter, enabled_tools=['mcp__notes__hidden'])
