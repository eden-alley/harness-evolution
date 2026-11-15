"""Real hard-exit/resume fixture; network is forbidden and all state is temporary."""
from copy import deepcopy
import json
import os
from pathlib import Path
import socket
import sys


def no_network(*args, **kwargs):
    raise AssertionError('after_tool recovery fixtures forbid network')


def main():
    source, directory, mode = sys.argv[1:]
    source, directory = Path(source).resolve(), Path(directory).resolve()
    sys.path.insert(0, str(source))
    import agent
    import permissions
    import tools
    from checkpoint_backend import CheckpointFixtureBackend

    # No normalization or replacement of source/execution bindings is allowed.
    for module in (agent, permissions, tools, agent.models, agent.checkpoints, agent._tool_runner):
        assert Path(module.__file__).resolve().parent == source, module.__file__
    socket.socket.connect = no_network
    socket.create_connection = no_network
    directory.mkdir(parents=True, exist_ok=True)
    workspace = directory / 'workspace'
    workspace.mkdir(exist_ok=True)
    observations = directory / 'observations.jsonl'

    def log(kind, **data):
        with observations.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'kind': kind, **data}, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    class Model:
        config = agent.models.ModelConfig(base_url='https://offline.invalid')
        client = None

        def query(self, messages, *, tools=None):
            log('request', mode=mode, messages=deepcopy(messages))
            if mode == 'resume':
                message = {'role': 'assistant', 'content': 'resumed'}
            else:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': 'write', 'type': 'function', 'function': {
                        'name': 'write_file',
                        'arguments': '{"path":"effect.txt","content":"once"}'}}]}
            return agent.models.ModelResponse(message, agent.models.ModelUsage(10, 1, 11),
                'tool_calls' if message.get('tool_calls') else 'stop', {'fixture': True})

    toolset = tools.Toolset(permissions.Workspace(workspace), backend=CheckpointFixtureBackend())
    execute = toolset.execute

    def execute_observed(name, args, **kwargs):
        result = execute(name, args, **kwargs)
        if name == 'write_file' and result.ok:
            log('effect', path=args['path'], content=(workspace / args['path']).read_text(encoding='utf-8'))
        return result

    toolset.execute = execute_observed
    store = agent.checkpoints.CheckpointStore(directory / 'checkpoints')
    save = store.save

    def save_observed(snapshot):
        save(snapshot)  # Exercise the real validator, lease and atomic disk write first.
        records = list(snapshot['run_state']['call_records'].values())
        if mode == 'before_notify' and records and all(r['status'] == 'recorded' for r in records):
            log('crash_boundary', boundary='saved_result_before_notification', snapshot=store.load())
            os._exit(51)

    store.save = save_observed

    def after_tool(summary):
        log('notification', mode=mode, summary={name: getattr(summary, name) for name in (
            'run_id', 'operation_id', 'call_id', 'tool', 'source', 'ok', 'code',
            'exit_code', 'result_contract')}, snapshot=store.load())
        if mode == 'in_notify':
            os._exit(52)

    runtime = agent.Agent(None, toolset, model_adapter=Model(), checkpoint_store=store,
                          after_tool=after_tool, max_steps=3)
    runtime.permissions.set_auto_approve(True)
    if mode == 'resume':
        before = store.load()
        decision = runtime.inspect_checkpoint(before)
        result = runtime.resume()
        log('resume', decision=decision, stop_reason=result.stop_reason, text=result.text,
            binding_unchanged=before['binding'] == store.load()['binding'], snapshot=store.load())
        return 0
    runtime.run('hard exit after tool')
    raise AssertionError('fixture did not reach requested hard-exit boundary')


if __name__ == '__main__':
    raise SystemExit(main())
