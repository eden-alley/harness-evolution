"""模型层离线契约：真实 SDK 编解码，本地响应，不读取真实密钥。"""
from test_file_reads import file_page
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI, InternalServerError

from scenario_support import OfflineScenario, agent, cli, context, permissions, answer, call, tool_reply
from test_context import history
from verification import ModelVerifier


def completion(message=None, *, finish='stop', usage=None, choices=None):
    body = {'id': 'offline', 'object': 'chat.completion', 'created': 0, 'model': 'fixture',
            'choices': choices if choices is not None else [
                {'index': 0, 'message': message if message is not None else answer('完成'),
                 'finish_reason': finish}]}
    if usage is not None:
        body['usage'] = usage
    return httpx.Response(200, json=body)


class ModelScenarios(OfflineScenario):
    def layer(self):
        self.assertTrue(hasattr(agent, 'models'), '缺少独立模型适配层')
        return agent.models

    def use_provider(self, provider):
        layer = self.layer()
        adapter = layer.get_model(layer.ModelConfig(provider=provider, model_name='fixture-model'),
                                  client=self.agent.client)
        self.agent = agent.Agent(self.agent.client, self.tools, model='fixture-model',
                                 model_adapter=adapter)
        return adapter

    def test_provider_parameters_cover_execution_summary_and_optional_verification(self):
        # 若某个消费者仍直连 SDK，通用协议就会收到 DeepSeek 参数。
        for provider in ('deepseek', 'openai_chat'):
            with self.subTest(provider=provider):
                self.requests.clear()
                self.check_requests.clear()
                self.all_requests.clear()
                self.model([(answer('此前摘要'), 10), (answer('已核实'), 20)],
                           check_replies=[(answer('{"status":"pass","reason":"已核对","evidence":[]}'), 30)])
                adapter = self.use_provider(provider)
                self.agent.messages = history(3)
                self.agent.last_prompt_tokens = self.agent.hard_limit + 1
                check = ModelVerifier(adapter, '按本地证据核对', self.agent.token_usage.record)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = self.agent.run('继续', verifier=check)
                self.assertEqual((result.stop_reason, result.verification.status), ('end_turn', 'pass'))
                self.assertEqual(len(self.all_requests), 3)
                for request in self.all_requests:
                    self.assertEqual(request['model'], 'fixture-model')
                    if provider == 'deepseek':
                        self.assertEqual(request['thinking'], {'type': 'disabled'})
                    else:
                        self.assertNotIn('thinking', request)
                self.assertNotIn('tools', self.all_requests[0])
                self.assertIn('tools', self.all_requests[1])
                self.assertNotIn('tools', self.all_requests[2])
                self.assertEqual(self.agent.token_usage.totals,
                                 {'prompt_tokens': 60, 'completion_tokens': 3, 'total_tokens': 63})
                self.assertEqual(self.agent.last_prompt_tokens, 20)

    def test_truncated_filtered_and_malformed_tool_batches_never_execute_partial_actions(self):
        valid = call('first', 'write_file', {'path': 'must-not-exist.txt', 'content': 'bad'})
        invalid = deepcopy(valid)
        invalid['id'] = ''
        variants = [
            completion(tool_reply(valid), finish='length', usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion(tool_reply(valid), finish='content_filter', usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion(tool_reply(valid, invalid), finish='tool_calls', usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion(tool_reply(valid, valid), finish='tool_calls', usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion(answer('截断文本'), finish='unexpected', usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion(choices=[], usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
            completion({'role': 'user', 'content': '非法角色'}, usage={'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9}),
        ]
        for response in variants:
            with self.subTest(response=response.json()):
                (self.workspace / 'must-not-exist.txt').unlink(missing_ok=True)
                self.all_requests.clear()
                self.agent.token_usage.totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                self.model([response])
                perms = permissions.PermissionManager(workspace=self.tools.workspace)
                perms.set_auto_approve(True)
                self.run_agent('验证响应', perms)
                self.assertFalse((self.workspace / 'must-not-exist.txt').exists())
                self.assertEqual(self.last_result.stop_reason, 'error')
                self.assertEqual(self.agent.run_state.tool_results, [])
                self.assertEqual(self.agent.run_state.pending_calls, {})
                self.assertEqual(len(self.all_requests), 1)
                self.assertEqual(self.agent.token_usage.totals,
                                 {'prompt_tokens': 7, 'completion_tokens': 2, 'total_tokens': 9})
                raw = [e for e in self.agent.events if e['type'] == 'model_response']
                self.assertEqual(len(raw), 1)
                self.assertEqual(raw[0]['response']['usage']['total_tokens'], 9)

    def test_partial_and_missing_usage_stays_unknown_and_cumulative_deficit_remains_visible(self):
        self.model([
            completion(usage={'prompt_tokens': 12}),
            completion(usage={'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6}),
            completion(),
            completion(usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}),
        ])
        outputs = []
        for _ in range(4):
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = self.agent.run('统计')
            self.assertEqual(result.stop_reason, 'end_turn')
            outputs.append(stream.getvalue())
        self.assertIn('输入=12 输出=未知 合计=未知', outputs[0])
        self.assertIn('累计含缺失', outputs[1])
        self.assertIn('未返回 usage', outputs[2])
        self.assertIn('输入=0 输出=0 合计=0', outputs[3])
        self.assertIn('累计含缺失', outputs[3])
        self.assertEqual(self.agent.token_usage.totals,
                         {'prompt_tokens': 16, 'completion_tokens': 2, 'total_tokens': 6})

    def test_invalid_usage_fields_are_not_coerced_or_subtracted(self):
        # 统计层直接收到的字段；SDK 可能已转换过服务端原始类型。
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            self.agent.token_usage.record(SimpleNamespace(
                prompt_tokens=-1, completion_tokens=True, total_tokens='8'), '对话')
        self.assertEqual(self.agent.token_usage.totals,
                         {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        self.assertIn('输入=未知 输出=未知 合计=未知', stream.getvalue())

    def test_bad_summary_preserves_history_and_bad_verification_keeps_usage(self):
        self.layer()
        for kind in ('summary', 'verification'):
            with self.subTest(kind=kind):
                bad = completion(answer('不完整'), finish='length',
                                 usage={'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12})
                self.model([bad] if kind == 'summary' else [(answer('候选'), 3)],
                           check_replies=None if kind == 'summary' else [bad])
                self.agent.token_usage.totals.update(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                if kind == 'summary':
                    original = history(3)
                    before = deepcopy(original)
                    with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                        context.compact(self.agent.model_adapter, original, self.agent.token_usage.record)
                    self.assertEqual(original, before)
                    self.assertEqual(self.agent.token_usage.totals['total_tokens'], 12)
                else:
                    check = ModelVerifier(self.agent.model_adapter, '明确条件', self.agent.token_usage.record)
                    self.run_agent('检查', self.agent.permissions, verifier=check)
                    self.assertEqual(self.last_result.stop_reason, 'error')
                    self.assertEqual(self.last_result.verification.status, 'error')
                    self.assertEqual(self.agent.token_usage.totals['total_tokens'], 16)

    def test_wire_metadata_does_not_enter_canonical_history_and_bad_arguments_still_reach_tools(self):
        layer = self.layer()
        message = tool_reply(call('bad-json', 'read_file', '{'),
                             call('good', 'read_file', {'path': 'seed.txt'}))
        message.update(reasoning_content='private reasoning', vendor_field='vendor')
        (self.workspace / 'seed.txt').write_text('seed', encoding='utf-8')
        self.model([completion(message, finish='tool_calls'), answer('完成')])
        self.use_provider('openai_chat')
        self.run_agent('读文件', self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, 'end_turn')
        evidence = self.agent.run_state.tool_results
        self.assertEqual(evidence[0]['code'], 'invalid_arguments')
        self.assertTrue(evidence[1]['ok'])
        self.assertEqual(file_page(evidence[1]['data'])[1], ['seed'])
        history_wire = self.requests[-1]['messages']
        self.assertFalse(any('reasoning_content' in m or 'vendor_field' in m for m in history_wire))
        raw = next(e['response'] for e in self.agent.events if e['type'] == 'model_response')
        self.assertEqual(raw['choices'][0]['message']['reasoning_content'], 'private reasoning')

    def test_legacy_client_replacement_and_budget_wrapper_are_never_bypassed(self):
        layer = self.layer()
        self.model([(answer('第一次'), 2)])
        first = self.agent.client
        self.run_agent('一次', self.agent.permissions)
        self.model([(answer('第二次'), 3)])
        second = self.agent.client
        self.assertIsNot(first, second)

        class BudgetClient:
            def __init__(self, wrapped):
                self.wrapped = wrapped
                self.calls = 0
                self.chat = type('Chat', (), {'completions': self})()

            def create(self, **kwargs):
                self.calls += 1
                kwargs['max_tokens'] = 9
                return self.wrapped.chat.completions.create(**kwargs)

        budget = BudgetClient(second)
        self.agent.client = budget
        with patch('openai.OpenAI', side_effect=AssertionError('额外创建客户端')):
            self.run_agent('二次', self.agent.permissions)
        self.assertEqual(self.last_result.text, '第二次')
        self.assertIs(self.agent.client, budget)
        self.assertEqual(budget.calls, 1)
        self.assertEqual(self.requests[-1]['max_tokens'], 9)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 7)

    def test_empty_advertised_tools_rejects_unsolicited_calls_without_retry(self):
        self.replace(self.tools, 'get_schemas', lambda: [])
        self.model([tool_reply(call('missing', 'not_installed', {}))])
        self.run_agent('没有可用工具', self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, 'error')
        self.assertNotIn('tools', self.requests[0])
        self.assertEqual(self.agent.run_state.tool_results, [])
        self.assertEqual(len(self.all_requests), 1)

    def test_unknown_tool_with_nonempty_catalog_returns_feedback(self):
        self.model([tool_reply(call('missing', 'not_installed', {})), answer('缺少工具')])
        self.run_agent('尝试工具', self.agent.permissions)
        self.assertEqual(self.last_result.stop_reason, 'end_turn')
        self.assertEqual(self.agent.run_state.tool_results[0]['code'], 'tool_unavailable')
        self.assertEqual(len(self.all_requests), 2)

    def test_request_failures_are_not_retried_or_reclassified_by_adapter(self):
        layer = self.layer()
        self.model([httpx.Response(503, json={'error': {'message': 'offline'}})])
        adapter = self.agent.model_adapter
        with self.assertRaises(InternalServerError):
            adapter.query([answer('请求')])
        self.assertEqual(len(self.all_requests), 1)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 0)

    def test_model_modules_are_isolated_when_loading_another_source_version(self):
        layer = self.layer()
        root = Path(agent.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory)
            for source in root.glob('*.py'):
                shutil.copyfile(source, candidate / source.name)
            spec = importlib.util.spec_from_file_location('agent_other_model_version', candidate / 'agent.py')
            other = importlib.util.module_from_spec(spec)
            self.addCleanup(sys.modules.pop, spec.name, None)
            sys.modules[spec.name] = other
            spec.loader.exec_module(other)
            self.assertIsNot(layer.ModelConfig, other.models.ModelConfig)
            self.model([answer('保留本版本')])
            adapter = layer.get_model(layer.ModelConfig(), client=self.agent.client)
            original = self.agent.client
            other_adapter = other.models.get_model(other.models.ModelConfig(), client=object())
            self.assertIsNot(type(adapter), type(other_adapter))
            self.assertIs(adapter.client, original)
            self.assertTrue(Path(other.models.__file__).resolve().is_relative_to(candidate.resolve()))
            self.assertEqual(adapter.query([answer('继续')]).require_message()['content'], '保留本版本')

    def test_same_module_name_keeps_each_source_versions_provider_behavior(self):
        # 若仅按模块名缓存子模块，第二份会错用第一份 provider 参数。
        self.layer()
        self.model([answer('第一版'), answer('第二版')])
        alias = 'same_agent_model_fixture'
        self.addCleanup(sys.modules.pop, alias, None)
        loaded = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                candidate = Path(directory) / str(index)
                candidate.mkdir()
                for source in Path(agent.__file__).parent.glob('*.py'):
                    shutil.copyfile(source, candidate / source.name)
                if index:
                    (candidate / 'model_deepseek.py').write_text(
                        "from .model_openai import OpenAIChatModel\n"
                        "class DeepSeekModel(OpenAIChatModel):\n"
                        "    def _request_options(self):\n"
                        "        return {'extra_body': {'thinking': {'type': 'fixture-second'}}}\n",
                        encoding='utf-8')
                spec = importlib.util.spec_from_file_location(alias, candidate / 'agent.py')
                module = importlib.util.module_from_spec(spec)
                sys.modules[alias] = module
                spec.loader.exec_module(module)
                loaded.append(module)
            for module in loaded:
                adapter = module.models.get_model(client=self.agent.client)
                adapter.query([answer('查询')])
        self.assertEqual([request['thinking']['type'] for request in self.requests],
                         ['disabled', 'fixture-second'])
        self.assertIsNot(loaded[0].models.ModelConfig, loaded[1].models.ModelConfig)


class StartupConfiguration(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('离线配置测试禁止联网'))
        guard.start()
        self.addCleanup(guard.stop)

    def test_default_deepseek_and_explicit_generic_provider_use_only_selected_fake_credentials(self):
        # 真实 SDK + 假 transport，拦截的只是客户端创建边界，不访问真实环境值。
        configs = [
            ({'DEEPSEEK_API_KEY': 'fake-deepseek'}, 'deepseek-flash', 'fake-deepseek', True, False,
             'https://api.deepseek.com/chat/completions'),
            ({'API_KEY': 'fake-default'}, 'deepseek-flash', 'fake-default', True, False,
             'https://api.deepseek.com/chat/completions'),
            ({}, 'deepseek-flash', 'fake-registry', True, True,
             'https://api.deepseek.com/chat/completions'),
            ({'MODEL_PROVIDER': 'openai_chat', 'MODEL': 'my-model', 'BASE_URL': 'https://other.invalid/v1',
              'MODEL_API_KEY_ENV': 'TEST_PROVIDER_KEY', 'TEST_PROVIDER_KEY': 'fake-generic',
              'DEEPSEEK_API_KEY': 'must-not-use'}, 'my-model', 'fake-generic', False, False,
             'https://other.invalid/v1/chat/completions'),
            ({'MODEL_PROVIDER': 'deepseek', 'MODEL': 'deepseek-test', 'BASE_URL': 'https://custom.invalid/v1',
              'MODEL_API_KEY_ENV': 'TEST_PROVIDER_KEY', 'TEST_PROVIDER_KEY': 'fake-custom',
              'DEEPSEEK_API_KEY': 'must-not-use'}, 'deepseek-test', 'fake-custom', True, False,
             'https://custom.invalid/v1/chat/completions'),
        ]
        for environment, model, key, thinking, registry, url in configs:
            with self.subTest(provider=environment.get('MODEL_PROVIDER', 'default')), tempfile.TemporaryDirectory() as directory:
                requests = []
                clients = []

                def transport(request):
                    requests.append(request)
                    return completion(answer('配置正常'))

                def construct(**kwargs):
                    client = OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(transport)))
                    clients.append(client)
                    return client

                def open_registry(*args):
                    if not registry:
                        raise AssertionError('不应读取注册表')
                    return contextlib.nullcontext(object())

                with patch.dict(os.environ, {**environment, 'NAMI_WORKSPACE': directory}, clear=True), \
                     patch('openai.OpenAI', side_effect=construct), \
                     patch('winreg.OpenKey', side_effect=open_registry), \
                     patch('winreg.QueryValueEx', return_value=(key, 1)):
                    configured = cli.create_agent()
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = configured.run('回答')
                for client in clients:
                    client.close()
                self.assertEqual(result.text, '配置正常')
                self.assertEqual(len(requests), 1)
                self.assertEqual(str(requests[0].url), url)
                self.assertEqual(requests[0].headers['Authorization'], 'Bearer ' + key)
                payload = json.loads(requests[0].content)
                self.assertEqual(payload['model'], model)
                self.assertEqual('thinking' in payload, thinking)
                self.assertEqual(configured.model, model)
                self.assertIs(configured.client, clients[0])

    def test_unknown_or_incomplete_provider_config_fails_without_client_or_key_fallback(self):
        variants = [
            {'MODEL_PROVIDER': 'unknown'},
            {'MODEL_PROVIDER': 'openai_chat'},
            {'MODEL_PROVIDER': 'openai_chat', 'MODEL': 'x', 'BASE_URL': 'https://other.invalid',
             'DEEPSEEK_API_KEY': 'must-not-use'},
            {'MODEL_PROVIDER': 'deepseek', 'BASE_URL': 'https://other.invalid', 'DEEPSEEK_API_KEY': 'must-not-use'},
        ]
        for environment in variants:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                 patch('openai.OpenAI', side_effect=AssertionError('无效配置创建客户端')), \
                 patch('winreg.OpenKey', side_effect=AssertionError('隐式读取旧密钥')):
                with self.assertRaises(ValueError):
                    cli.create_agent()

