"""硬退出/跨进程锁的离线辅助程序；只读写传入的临时目录。"""
from pathlib import Path
import os
import sys

source, workspace, state_dir, mode, limit = sys.argv[1:]
sys.path.insert(0, source)
import agent
from permissions import Workspace
from tools import Toolset
from checkpoint_backend import CheckpointFixtureBackend
import tools

store = agent.checkpoints.CheckpointStore(state_dir)
if mode == 'lock':
    try:
        with store.lease():
            pass
    except agent.checkpoints.CheckpointError:
        raise SystemExit(23)
    raise SystemExit(0)


class FixtureModel:
    config = agent.models.ModelConfig(base_url='https://offline.invalid')
    client = None

    def query(self, messages, *, tools=None):
        counter = Path(workspace) / 'model-count.txt'
        counter.write_text(counter.read_text() + 'request\n' if counter.exists() else 'request\n')
        if mode == 'lost':
            os._exit(43)
        message = {'role': 'assistant', 'tool_calls': [{'id': 'write', 'type': 'function',
                   'function': {'name': 'write_file',
                                'arguments': '{"path":"effect.txt","content":"once"}'}}]}
        return agent.models.ModelResponse(message, agent.models.ModelUsage(10, 1, 11),
                                          'tool_calls', {'fixture': True})


if mode == 'effect':
    original = tools.TOOLS['write_file']['func']

    def crash_after_effect(*args, **kwargs):
        result = original(*args, **kwargs)
        os._exit(47)

    tools.TOOLS['write_file']['func'] = crash_after_effect

if mode == 'reply':
    save = store.save

    def crash_after_reply(snapshot):
        save(snapshot)
        if snapshot['run_state']['next_action'] == 'response':
            os._exit(41)

    store.save = crash_after_reply

runtime = agent.Agent(None, Toolset(Workspace(workspace), backend=CheckpointFixtureBackend()), model_adapter=FixtureModel(),
                      checkpoint_store=store, max_steps=int(limit))
runtime.permissions.set_auto_approve(True)
runtime.run('硬退出测试')
raise SystemExit('fixture did not exit at requested boundary')
