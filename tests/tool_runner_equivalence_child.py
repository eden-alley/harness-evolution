"""One isolated version per process; deterministic offline observations, no production shim."""
from copy import deepcopy
from dataclasses import asdict
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from uuid import UUID

source, directory, scenario, stage = sys.argv[1:]
source, directory = Path(source).resolve(), Path(directory).resolve()
sys.path.insert(0, str(source))
import agent
import context
import permissions
import tools
import verification
from shell_schema_fixtures import CURRENT_SHELL_DESCRIPTION, LEGACY_SHELL_DESCRIPTION

for module in (agent, context, permissions, tools, verification, agent.models, agent.checkpoints):
    assert Path(module.__file__).resolve().parent == source, module.__file__
assert agent.TOOLS is tools.TOOLS
runner_module = getattr(agent, '_tool_runner', None)
if (source / 'tool_runner.py').exists():
    assert runner_module is not None, 'candidate did not load its ToolRunner module'
    assert Path(runner_module.__file__).resolve() == source / 'tool_runner.py'
    assert runner_module.__name__ == f'{agent.models.__name__}.tool_runner'
    assert sys.modules[runner_module.__name__] is runner_module
else:
    assert runner_module is None, 'baseline unexpectedly imported a candidate ToolRunner'

# Compare execution under the same textual tool schema. The actual new schema
# is asserted separately in test_shell_options and the frozen request test.
shell = tools.TOOLS['run_shell']['schema']['function']
expected = CURRENT_SHELL_DESCRIPTION if runner_module else LEGACY_SHELL_DESCRIPTION
assert shell['description'] == expected
shell['description'] = LEGACY_SHELL_DESCRIPTION

def no_network(*args, **kwargs):
    raise AssertionError('equivalence fixtures forbid network')

socket.socket.connect = no_network
socket.create_connection = no_network
directory.mkdir(parents=True, exist_ok=True)
workspace = directory / 'workspace'
workspace.mkdir(exist_ok=True)
trace = directory / 'observations.jsonl'
counter = 1000 if stage == 'resume' else 0

def uuid4():
    global counter
    counter += 1
    return UUID(int=counter)

agent.uuid4 = agent.checkpoints.uuid4 = uuid4
runtime = None

def log(kind, **data):
    with trace.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'kind': kind, **data}, ensure_ascii=False, allow_nan=False) + '\n')

def observed_state():
    state = runtime.run_state
    return deepcopy({'phase': state.phase, 'next_action': state.next_action,
                     'pending_calls': state.pending_calls, 'active_batch': state.active_batch,
                     'call_records': state.call_records, 'tool_results': state.tool_results,
                     'messages': runtime.messages})

class Backend:
    binding = {'name': 'equivalence_fixture', 'policy_version': 1}

    def run(self, command, workspace, timeout, *, shell='cmd'):
        log('backend', command=command, timeout=timeout, shell=shell)
        return subprocess.CompletedProcess(command, 7 if command == 'fail' else 0,
                                           'HEAD' + '中' * 1600 + 'TAIL', '')

class MemoryMCP:
    binding = {'name': 'equivalence_memory_transport', 'url': None}
    tools = {'remote_echo': {'schema': {'type': 'function', 'function': {
        'name': 'remote_echo', 'description': 'Offline remote fixture',
        'parameters': {'type': 'object', 'properties': {'max_output_tokens': {'type': 'integer'}},
                       'required': ['max_output_tokens'], 'additionalProperties': False}}}}}

    def execute(self, name, args):
        log('mcp_execute', name=name, args=deepcopy(args))
        return tools.ToolResult(True, data='HEAD' + '中' * 1600 + 'TAIL')

toolset = tools.Toolset(permissions.Workspace(workspace), backend=Backend(), mcp=MemoryMCP(),
                       task_progress=True, enabled_tools=['read_file', 'write_file', 'run_shell',
                       'read_tool_result', 'update_progress', 'remote_echo'])
original_validate, original_execute = toolset.validate_call, toolset.execute

def validate(name, args):
    result = original_validate(name, args)
    log('validate', name=name, args=deepcopy(args), result=None if result is None else result.to_dict())
    return result

def execute(name, args, *, recorded_results=()):
    log('execute', name=name, args=deepcopy(args), recorded_results=deepcopy(recorded_results))
    result = original_execute(name, args, recorded_results=recorded_results)
    if name == 'write_file' and result.ok:
        count = directory / 'effect-count.txt'
        count.write_text(str(int(count.read_text()) + 1 if count.exists() else 1), encoding='utf-8')
        log('effect', path=args['path'], content=(workspace / args['path']).read_text(encoding='utf-8'))
        if scenario == 'hard_exit' and stage == 'start':
            os._exit(47)
    return result

toolset.validate_call, toolset.execute = validate, execute

class Policy:
    def check(self, name, args):
        log('permission', name=name, args=deepcopy(args))
        if args.get('path') == 'denied.txt':
            return permissions.Decision(False, 'fixture denied', 'user_denied')
        if args.get('path') == 'mutate.txt':
            args['content'] = 'changed by policy'
        if args.get('path') == 'policy-error.txt':
            raise ValueError('fixture policy error')
        return permissions.Decision(True)

def call(identifier, name, args):
    return {'id': identifier, 'type': 'function', 'function': {
        'name': name, 'arguments': args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)}}

def batch(*calls):
    return {'role': 'assistant', 'tool_calls': list(calls)}

answer = {'role': 'assistant', 'content': '完成；依据工具真实结果。'}
write_a = call('a', 'write_file', {'path': 'a.txt', 'content': 'A'})
write_b = call('b', 'write_file', {'path': 'b.txt', 'content': 'B'})
if scenario == 'mixed':
    script = [batch(write_a, call('read', 'read_file', {'path': 'a.txt'}),
                    call('denied', 'write_file', {'path': 'denied.txt', 'content': 'NO'}),
                    call('missing', 'unknown_tool', {}), call('json', 'read_file', '{bad'),
                    call('array', 'read_file', []),
                    call('mutate', 'write_file', {'path': 'mutate.txt', 'content': 'original'}),
                    call('policy-error', 'write_file', {'path': 'policy-error.txt', 'content': 'NO'}),
                    call('native', 'run_shell', {'command': 'fail', 'max_output_tokens': 128}),
                    call('remote', 'remote_echo', {'max_output_tokens': 128})), answer]
elif scenario == 'verification':
    script = [batch(write_a), answer]
elif scenario in ('save_executing', 'save_recorded', 'hard_exit', 'event_exception'):
    script = [batch(write_a), answer]
elif scenario == 'pending_resume':
    script = [answer] if stage == 'resume' else [batch(write_a, write_b), answer]
elif scenario == 'evidence_filter':
    script = [answer]
else:
    raise AssertionError(scenario)

class Model:
    config = agent.models.ModelConfig(base_url='https://offline.invalid')
    client = None

    def query(self, messages, *, tools=None):
        log('request', messages=deepcopy(messages), tools=deepcopy(tools))
        assert script, 'unexpected extra model request'
        message = script.pop(0)
        return agent.models.ModelResponse(deepcopy(message), agent.models.ModelUsage(10, 2, 12),
                                          'tool_calls' if message.get('tool_calls') else 'stop',
                                          {'fixture_response': deepcopy(message)})

store = agent.checkpoints.CheckpointStore(directory / 'checkpoints')
save = store.save
fault_fired = False

def save_observed(snapshot):
    global fault_fired
    log('checkpoint_save', snapshot=deepcopy(snapshot))
    statuses = [r['status'] for r in snapshot['run_state']['call_records'].values()]
    target = scenario.removeprefix('save_')
    if scenario.startswith('save_') and target in statuses:
        log('checkpoint_save_error', status=target)
        raise OSError('fixture disk failure at ' + target)
    save(snapshot)
    if scenario == 'pending_resume' and stage == 'start' and not fault_fired and statuses == ['recorded', 'pending']:
        fault_fired = True
        raise KeyboardInterrupt()

store.save = save_observed

def sink(event):
    log('event', event=deepcopy(event), observed=observed_state())
    if scenario == 'event_exception' and event['type'] == 'tool_result':
        raise RuntimeError('fixture event callback failure')

runtime = agent.Agent(None, toolset, permissions=Policy(), model_adapter=Model(),
                      checkpoint_store=store, event_sink=sink, max_steps=3)

# Import provenance is a separate structural assertion and artifact. The added
# candidate module must not become a behavior-comparison input or be normalized.
runner_instance = getattr(runtime, 'tool_runner', None)
if runner_module is not None:
    assert runner_instance is not None
    assert type(runner_instance) is runner_module.ToolRunner
    assert type(runner_instance).__module__ == runner_module.__name__
    assert sys.modules[type(runner_instance).__module__] is runner_module
else:
    assert runner_instance is None
imports = {
    'stage': stage, 'source_root': str(source), 'scenario_root': str(directory),
    'root_python_files': sorted(path.name for path in source.glob('*.py')),
    'modules': {name: {'name': module.__name__, 'file': str(Path(module.__file__).resolve())}
                for name, module in (('agent', agent), ('tools', tools), ('context', context),
                    ('permissions', permissions), ('verification', verification),
                    ('models', agent.models), ('checkpoint', agent.checkpoints))},
    'runner_module': None if runner_module is None else {
        'name': runner_module.__name__, 'file': str(Path(runner_module.__file__).resolve())},
    'runner_instance': None if runner_instance is None else {
        'class_name': type(runner_instance).__name__, 'class_module': type(runner_instance).__module__},
}
(directory / f'{stage}-imports.json').write_text(
    json.dumps(imports, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def verifier(value):
    log('verification_input', task=value.task, candidate=value.candidate,
        tool_results=value.tool_results, available_tools=value.available_tools)
    result = value.run_tool('read_file', {'path': 'a.txt'})
    return verification.CheckResult('pass', 'verified using actual read', [result.to_dict()])

if scenario == 'evidence_filter':
    # Same intentionally inconsistent in-memory evidence on both versions. Only
    # current, recorded, model-origin ledger entries may authorize progress.
    runtime.run_state.run_id = uuid4().hex
    runtime.run_state.user_input = 'fixture task'
    runtime.messages.append({'role': 'user', 'content': 'fixture task'})
    runtime._checkpoint_session_id = uuid4().hex
    runtime._checkpoint_binding = agent.checkpoints.binding(runtime)
    runtime.handle_call(call('verified', 'run_shell', {'command': 'ok'}), source='verification')
    orphan = {'call_id': 'orphan', 'tool': 'read_file', 'source': 'model', 'ok': True, 'data': 'unregistered'}
    runtime.run_state.tool_results.append(orphan)
    def model_call(value):
        # Reproduce the model-batch registration boundary before direct exercise
        # so every intermediate checkpoint still passes the real validator.
        state = runtime.run_state
        runtime.messages.append(batch(value))
        state.pending_calls = {value['id']: deepcopy(value)}
        operation_id = runtime._register_call(value, 'model')
        state.active_batch, state.next_action = [operation_id], 'tools'
        result = runtime.handle_call(value, operation_id=operation_id)
        state.active_batch, state.next_action = [], 'prepare'
        return result
    model_call(call('recorded', 'run_shell', {'command': 'ok'}))
    for identifier in ('verified', 'orphan', 'recorded'):
        model_call(call('progress-' + identifier, 'update_progress', {
            'deliverables': ['report'], 'findings': [{'text': 'claim', 'call_ids': [identifier]}],
            'remaining': ['report'], 'next_action': 'write report'}))
    model_call(call('recall', 'read_tool_result', {'call_id': 'orphan', 'field': 'result', 'max_output_tokens': 4096}))
    result = None
else:
    try:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = runtime.resume() if stage == 'resume' else runtime.run(
                'fixture task', verifier=verifier if scenario == 'verification' else None)
        log('stdout', text=output.getvalue())
    except KeyboardInterrupt:
        log('interrupted', observed=observed_state())
        result = None

if scenario == 'pending_resume' and stage == 'start':
    (workspace / 'a.txt').write_text('external modification; never replay', encoding='utf-8')
log('terminal', result=None if result is None else asdict(result), state=asdict(runtime.run_state),
    messages=runtime.messages, usage=runtime.token_usage.totals, usage_incomplete=runtime.token_usage.incomplete,
    checkpoint=store.load(), effects={p.relative_to(workspace).as_posix(): p.read_text(encoding='utf-8')
                                    for p in sorted(workspace.rglob('*')) if p.is_file()},
    effect_count=int((directory / 'effect-count.txt').read_text()) if (directory / 'effect-count.txt').exists() else 0)
