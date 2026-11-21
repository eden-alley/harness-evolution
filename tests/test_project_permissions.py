import os
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scenario_support import OfflineScenario, agent, answer, call, cli, permissions, tool_reply, tools


class ProjectPermissionTests(OfflineScenario):
    def setUp(self):
        super().setUp()
        self.input_file = self.workspace / "input.txt"
        self.input_file.write_bytes(b"original")

    def protected_workspace(self):
        return permissions.Workspace(self.workspace, read_only_paths=["input.txt"])

    def test_builtin_tools_preserve_input_and_allow_output(self):
        self.tools = tools.Toolset(self.protected_workspace())
        self.agent.tools = self.tools
        policy = permissions.PermissionManager(workspace=self.tools.workspace, ask=Mock())
        policy.set_auto_approve(True)
        self.model([
            tool_reply(call("read", "read_file", {"path": "input.txt"}),
                       call("write", "write_file", {"path": "input.txt", "content": "changed"}),
                       call("edit", "edit_file", {"path": "INPUT.txt" if os.name == "nt" else "input.txt",
                                                 "edits": [{"oldText": "original", "newText": "changed"}]})),
            tool_reply(call("save", "write_file", {"path": "proposal.txt", "content": "changed"})),
            answer("另存已完成"),
        ])
        self.run_agent("核对输入并另存提案", policy)
        results = {row["call_id"]: row for row in self.agent.run_state.tool_results}
        self.assertTrue(results["read"]["ok"])
        self.assertEqual(results["write"]["code"], "policy_denied")
        self.assertEqual(results["edit"]["code"], "policy_denied")
        self.assertTrue(results["save"]["ok"])
        self.assertEqual(self.input_file.read_bytes(), b"original")
        self.assertEqual((self.workspace / "proposal.txt").read_bytes(), b"changed")
        policy.ask.assert_not_called()

    def test_direct_tool_invocation_also_checks_rule(self):
        toolset = tools.Toolset(self.protected_workspace())
        write = toolset.execute("write_file", {"path": "./input.txt", "content": "changed"})
        edit = toolset.execute("edit_file", {"path": "./input.txt", "edits": [
            {"oldText": "original", "newText": "changed"}]})
        self.assertEqual((write.code, edit.code), ("policy_denied", "policy_denied"))
        self.assertEqual(self.input_file.read_bytes(), b"original")

    def test_protected_backend_passes_same_paths_and_binds_checkpoint_identity(self):
        protected = tools.Toolset(self.protected_workspace())
        ordinary = tools.Toolset(permissions.Workspace(self.workspace))
        self.assertEqual(protected.execution_binding["read_only_paths"], [str(self.input_file.resolve())])
        self.assertNotEqual(agent.checkpoints.binding(agent.Agent(None, protected)),
                            agent.checkpoints.binding(agent.Agent(None, ordinary)))
        with patch("execution_backend.sandbox.run_in_sandbox", return_value=subprocess.CompletedProcess([], 0, b"", b"")) as run:
            self.assertTrue(protected.execute("run_shell", {"command": "type input.txt"}).ok)
            run.assert_called_once_with("type input.txt", self.workspace.resolve(), 30,
                                        shell="cmd", read_only_paths=(self.input_file.resolve(),))

    def test_protected_workspace_rejects_incompatible_backends(self):
        workspace = self.protected_workspace()
        for options in ({"run_command": Mock()}, {"backend": Mock()}, {"mcp": SimpleNamespace(tools={})}):
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                tools.Toolset(workspace, **options)
        toolset = tools.Toolset(workspace)
        runner = Mock()
        toolset.run_command = runner
        self.assertEqual(toolset.execute("run_shell", {"command": "echo hi"}).code, "policy_denied")
        runner.assert_not_called()
        ordinary = tools.Toolset(permissions.Workspace(self.workspace))
        ordinary.workspace = workspace
        self.assertEqual(ordinary.execute("run_shell", {"command": "echo hi"}).code, "policy_denied")

    def test_cli_parses_explicit_file_rules(self):
        with patch.dict(os.environ, {"NAMI_WORKSPACE": str(self.workspace),
                                     "AGENT_READ_ONLY_PATHS": '["input.txt"]'}):
            setup = cli.configured_workspace()
        self.assertEqual(setup.read_only_paths, (self.input_file.resolve(),))
        nested = self.workspace / "samples"
        nested.mkdir()
        (nested / "other.txt").write_text("test", encoding="utf-8")
        for raw in ('"input.txt"', '["."]', '["missing.txt"]', '["../elsewhere"]', '[""]', '["samples/other.txt"]'):
            with self.subTest(raw=raw), patch.dict(os.environ, {"NAMI_WORKSPACE": str(self.workspace),
                                                                 "AGENT_READ_ONLY_PATHS": raw}), self.assertRaises(ValueError):
                cli.configured_workspace()

    def test_explicit_directory_rule_covers_existing_and_new_descendants(self):
        (self.workspace / "samples").mkdir()
        (self.workspace / "samples/input.txt").write_bytes(b"original")
        workspace = permissions.Workspace(self.workspace, read_only_paths=["samples"])
        toolset = tools.Toolset(workspace)
        for path in ("samples/input.txt", "samples/new.txt"):
            write = toolset.execute("write_file", {"path": path, "content": "changed"})
            self.assertEqual(write.code, "policy_denied")
        self.assertEqual((self.workspace / "samples/input.txt").read_bytes(), b"original")
        self.assertTrue(toolset.execute("write_file", {"path": "proposal.txt", "content": "changed"}).ok)

    def test_cli_rejects_mcp_before_opening_it_in_protected_mode(self):
        with patch.dict(os.environ, {"NAMI_WORKSPACE": str(self.workspace), "AGENT_READ_ONLY_PATHS": '["input.txt"]',
                                     "NAMI_MCP_CONFIG": "unused.json"}), patch.object(cli, "open_mcp") as connect, \
                patch.object(cli, "create_agent", side_effect=AssertionError("must stop before model setup")):
            with self.assertRaises(SystemExit):
                cli.main()
        connect.assert_not_called()

    def test_cli_creates_agent_with_the_same_rules_for_tools_and_approval(self):
        with patch.dict(os.environ, {"NAMI_WORKSPACE": str(self.workspace), "AGENT_READ_ONLY_PATHS": '["input.txt"]',
                                     "DEEPSEEK_API_KEY": "offline"}, clear=True), \
                patch.object(cli.models, "get_model", return_value=SimpleNamespace(client=None)):
            runtime = cli.create_agent()
        self.assertIs(runtime.permissions.workspace, runtime.tools.workspace)
        self.assertEqual(runtime.tools.workspace.read_only_paths, (self.input_file.resolve(),))
