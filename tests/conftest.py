"""仅在临时工作区中测试，不调用真实模型。"""
import os
import sys
import tempfile
from pathlib import Path

_previous_dir = Path.cwd()
_workspace = tempfile.TemporaryDirectory(prefix="nami-ws-")
_state = tempfile.TemporaryDirectory(prefix="nami-state-")
os.environ["NAMI_WORKSPACE"] = _workspace.name
os.environ["NAMI_STATE_DIR"] = _state.name
os.environ["DEEPSEEK_API_KEY"] = "test"
os.chdir(_workspace.name)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_sessionfinish(session, exitstatus):
    os.chdir(_previous_dir)
    _workspace.cleanup()
    _state.cleanup()
