"""精确编辑硬退出测试；只操作传入的测试工作区，模型为本地固定响应。"""
import os
from pathlib import Path
import sys

source, workspace, state_dir = sys.argv[1:]
sys.path.insert(0, source)
import agent
import tools
from permissions import Workspace
from checkpoint_backend import CheckpointFixtureBackend


class FixtureModel:
    config = agent.models.ModelConfig(base_url="https://offline.invalid")
    client = None

    def query(self, messages, *, tools=None):
        message = {"role": "assistant", "tool_calls": [{"id": "edit", "type": "function",
            "function": {"name": "edit_file", "arguments": '{"path":"sample.txt","edits":[{"oldText":"before","newText":"after"}]}'}}]}
        return agent.models.ModelResponse(message, agent.models.ModelUsage(10, 1, 11), "tool_calls", {"fixture": True})


original = tools.TOOLS["edit_file"]["func"]
def crash_after_edit(*args, **kwargs):
    result = original(*args, **kwargs)
    if not result.ok:
        raise AssertionError(result)
    os._exit(47)
tools.TOOLS["edit_file"]["func"] = crash_after_edit
runtime = agent.Agent(None, tools.Toolset(Workspace(workspace), backend=CheckpointFixtureBackend()),
    model_adapter=FixtureModel(), checkpoint_store=agent.checkpoints.CheckpointStore(state_dir), max_steps=3)
runtime.permissions.set_auto_approve(True)
runtime.run("编辑硬退出测试")
raise SystemExit("fixture did not exit after edit")
