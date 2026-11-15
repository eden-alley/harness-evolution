"""精确编辑接入现有权限、SDK、state 和 checkpoint；全部离线。"""
from test_file_reads import result_page
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from scenario_support import OfflineScenario, agent, permissions, tools, answer, call, tool_reply
from checkpoint_backend import CheckpointFixtureBackend


class EditFileScenarios(OfflineScenario):
    def args(self, old="before", new="after", **extra):
        return {"path": "sample.txt", "edits": [{"oldText": old, "newText": new}], **extra}

    def test_manual_grants_are_per_tool_and_reset_with_auto_mode(self):
        path = self.workspace / "sample.txt"
        path.write_bytes(b"before")
        asked = []
        perms = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append((name, deepcopy(args))) or "n", "暂不修改"))
        self.agent.permissions = perms
        perms.always_allowed.add("write_file")
        result = self.agent._invoke_tool("edit_file", self.args())
        self.assertEqual(result.code, "user_denied")
        self.assertEqual(path.read_bytes(), b"before")
        self.assertEqual([name for name, _ in asked], ["edit_file"])
        perms.ask = lambda name, args: (asked.append((name, deepcopy(args))) or "a", "")
        self.assertTrue(self.agent._invoke_tool("edit_file", self.args()).ok)
        self.assertTrue(self.agent._invoke_tool("edit_file", self.args("after", "done")).ok)
        self.assertEqual([name for name, _ in asked], ["edit_file", "edit_file"])
        self.assertIn("edit_file", perms.always_allowed)
        self.assertEqual(path.read_bytes(), b"done")
        perms.set_auto_approve(False)
        self.assertEqual(perms.always_allowed, set())

    def test_auto_mode_keeps_hard_boundary_and_deny_policy(self):
        path = self.workspace / "sample.txt"
        path.write_bytes(b"before")
        outside = self.workspace.parent / "outside.txt"
        outside.write_bytes(b"before")
        asked = []
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or "n", ""))
        self.agent.permissions.set_auto_approve(True)
        self.assertTrue(self.agent._invoke_tool("edit_file", self.args()).ok)
        outside_args = self.args()
        outside_args["path"] = "../outside.txt"
        self.assertEqual(self.agent._invoke_tool("edit_file", outside_args).code, "policy_denied")
        self.assertEqual(self.tools.execute("edit_file", outside_args).code, "policy_denied")
        with patch.dict(permissions.TOOL_RISK, {"edit_file": permissions.Risk.DENY}):
            self.assertEqual(self.agent._invoke_tool("edit_file", self.args("after", "bad")).code, "policy_denied")
        self.assertEqual(path.read_bytes(), b"after")
        self.assertEqual(outside.read_bytes(), b"before")
        self.assertEqual(asked, [])

    def test_nested_parameters_are_checked_before_permission_and_twenty_edits_work(self):
        path = self.workspace / "sample.txt"
        path.write_bytes(b"before")
        asked = []
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or "y", ""))
        invalid = [[], [{"oldText": "", "newText": "x"}], [{"oldText": "before"}],
                   [{"oldText": "before", "newText": False}],
                   [{"oldText": "before", "newText": "x", "extra": 1}],
                   [{"oldText": "before", "newText": "x"}] * 21]
        for edits in invalid:
            result = self.agent._invoke_tool("edit_file", {"path": "sample.txt", "edits": edits})
            self.assertEqual(result.code, "invalid_arguments")
        self.assertEqual(asked, [])
        self.assertEqual(path.read_bytes(), b"before")
        path.write_text("\n".join(f"line{i:02d}=old" for i in range(20)), encoding="utf-8")
        edits = [{"oldText": f"line{i:02d}=old", "newText": f"line{i:02d}=new"} for i in range(20)]
        self.assertTrue(self.agent._invoke_tool("edit_file", {"path": "sample.txt", "edits": edits}).ok)
        self.assertEqual(path.read_text(encoding="utf-8"), "\n".join(f"line{i:02d}=new" for i in range(20)))
        self.assertEqual(asked, ["edit_file"])

    def test_approval_preview_is_bounded_and_cannot_rewrite_nested_edits(self):
        arguments = self.args("before" * 1000, "after" * 1000)
        original = deepcopy(arguments)
        preview = permissions.render_call("edit_file", arguments, max_chars=200)
        self.assertLess(len(preview), 400)
        self.assertIn("sample.txt", preview)
        self.assertEqual(arguments, original)
        (self.workspace / "sample.txt").write_bytes(b"before")
        def rewrite(name, args):
            args["edits"][0]["newText"] = "rewritten"
            return "a", ""
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace, ask=rewrite)
        arguments = self.args()
        original = deepcopy(arguments)
        result = self.agent._invoke_tool("edit_file", arguments)
        self.assertEqual(result.code, "execution_error")
        self.assertEqual(arguments, original)
        self.assertEqual((self.workspace / "sample.txt").read_bytes(), b"before")
        self.assertEqual(self.agent.permissions.always_allowed, set())

    def test_real_loop_saves_full_diff_and_existing_result_tool_can_recover_it(self):
        path = self.workspace / "sample.txt"
        path.write_bytes(b"before\r\nend")
        replacement = "start\n" + "detail\n" * 100 + "after"
        events = []
        self.agent.event_sink = events.append
        self.agent.permissions.set_auto_approve(True)
        self.model([
            (tool_reply(call("edit", "edit_file", self.args("before", replacement, max_output_tokens=32))), 10),
            (tool_reply(call("recall", "read_tool_result", {"call_id": "edit", "field": "result", "max_output_tokens": 4096})), 20),
            (answer("已修改并核对"), 30),
        ])
        self.run_agent("只修改指定内容", self.agent.permissions)
        self.assertEqual(path.read_bytes(), (replacement.replace("\n", "\r\n") + "\r\nend").encode())
        preview = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertTrue(preview["truncated"])
        self.assertEqual(preview["saved_call_id"], "edit")
        raw, recalled = self.agent.run_state.tool_results
        self.assertTrue(raw["ok"])
        self.assertIn("+after", raw["data"]["diff"])
        self.assertEqual(json.loads(result_page(recalled["data"])[1]), raw)
        self.assertEqual(next(record for record in self.agent.run_state.call_records.values() if record["call"]["id"] == "edit")["status"], "recorded")
        self.assertEqual((self.last_result.stop_reason, self.agent.run_state.steps_used), ("end_turn", 3))
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 63)
        tool_events = [event for event in events if event["type"] == "tool_call"]
        self.assertEqual(tool_events[0]["execution"], {"location": "host"})
        self.assertEqual(len(self.all_requests), 3)

    def test_hard_exit_after_edit_does_not_replay_an_unknown_write(self):
        (self.workspace / "sample.txt").write_bytes(b"before")
        self.model([])
        self.tools = tools.Toolset(self.tools.workspace, backend=CheckpointFixtureBackend())
        cp = agent.checkpoints.CheckpointStore(self.store.directory.parent / "checkpoints")
        self.agent = agent.Agent(self.agent.client, self.tools, checkpoint_store=cp, max_steps=3)
        child = subprocess.run([sys.executable, "-B", "-X", "utf8",
            str(Path(__file__).with_name("edit_checkpoint_child.py")),
            str(Path(agent.__file__).parent), str(self.workspace), str(cp.directory)],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(child.returncode, 47, child.stderr)
        self.assertEqual((self.workspace / "sample.txt").read_bytes(), b"after")
        before = cp.path.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.agent.resume()
        self.assertEqual(result.stop_reason, "error")
        self.assertIn("未知", result.text)
        self.assertEqual(cp.path.read_bytes(), before)
        self.assertEqual((self.workspace / "sample.txt").read_bytes(), b"after")
        self.assertEqual(self.all_requests, [])
        self.assertEqual(next(record for record in cp.load()["run_state"]["call_records"].values() if record["call"]["id"] == "edit")["status"], "executing")

    def test_root_python_snapshot_loads_its_own_edit_module_in_a_fresh_process(self):
        source = Path(agent.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            for file in source.glob("*.py"):
                shutil.copyfile(file, Path(directory) / file.name)
            code = '''from pathlib import Path
import agent, file_edits
from permissions import Workspace
from tools import Toolset
assert Path(file_edits.__file__).parent == Path.cwd()
workspace = Path('workspace'); workspace.mkdir()
(workspace/'x.txt').write_bytes(b'before')
result = Toolset(Workspace(workspace), run_command=lambda *args: None).execute('edit_file', {'path':'x.txt','edits':[{'oldText':'before','newText':'after'}]})
assert result.ok, result
assert (workspace/'x.txt').read_bytes() == b'after'
'''
            child = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=directory,
                                   capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)


if __name__ == "__main__":
    import unittest
    unittest.main()
