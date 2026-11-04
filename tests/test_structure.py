"""结构边界：导入无副作用、实例状态隔离、工作区共享、执行器注入。"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StructureTests(unittest.TestCase):
    def test_import_does_not_read_credentials_or_construct_client(self):
        code = """from unittest.mock import patch
import os
with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "", "API_KEY": ""}), patch("openai.OpenAI", side_effect=AssertionError("import constructed client")), patch("winreg.OpenKey", side_effect=AssertionError("import read credentials")):
    import agent, cli
    assert hasattr(agent, "Agent"), "Agent instance is missing"
"""
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_instances_share_workspace_but_not_runtime_state(self):
        import agent
        self.assertTrue(hasattr(agent, "Agent"), "运行状态尚未收进 Agent 实例")
        from permissions import Workspace
        from tools import Toolset
        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            toolset = Toolset(workspace, run_command=lambda *args: None)
            first = agent.Agent(object(), toolset)
            second = agent.Agent(object(), toolset)
            self.assertIs(first.permissions.workspace, workspace)
            self.assertIs(first.tools.workspace, workspace)
            first.messages.append({"role": "user", "content": "仅属于第一个实例"})
            first.last_prompt_tokens = 20000
            first.permissions.set_auto_approve(True)
            first.token_usage.totals["total_tokens"] = 10
            self.assertEqual(len(second.messages), 1)
            self.assertEqual(second.last_prompt_tokens, 0)
            self.assertFalse(second.permissions.auto_approve)
            self.assertEqual(second.token_usage.totals["total_tokens"], 0)

    def test_injected_runner_preserves_output_and_file_boundary(self):
        import tools
        self.assertTrue(hasattr(tools, "Toolset"), "工具仍依赖模块级工作区")
        from permissions import Workspace
        calls = []

        def runner(command, workspace, timeout):
            calls.append((command, workspace, timeout))
            return subprocess.CompletedProcess(command, 0, "测试输出".encode(), b"")

        with tempfile.TemporaryDirectory() as directory:
            workspace = Workspace(directory)
            toolset = tools.Toolset(workspace, run_command=runner)
            self.assertEqual(toolset.execute("run_shell", {"command": "fixture"}).data,
                             "[exit 0]\n测试输出")
            self.assertEqual(calls, [("fixture", workspace.root, 30)])
            self.assertIn("工作区之外", toolset.execute(
                "write_file", {"path": "../escape", "content": "no"}).message)


if __name__ == "__main__":
    unittest.main()
