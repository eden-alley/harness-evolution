"""真实 rg + SDK MockTransport 验证新增搜索接入，不调用真实模型。"""
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

from scenario_support import OfflineScenario, agent, cli, permissions, tools, answer, call, tool_reply
import file_search


class SearchToolScenarios(OfflineScenario):
    def test_locate_read_edit_chain_preserves_permissions_history_usage_and_final_bytes(self):
        path = self.workspace / "src/config.py"
        path.parent.mkdir()
        path.write_bytes(b"# header\r\nTIMEOUT = 30\r\n")
        asked = []
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append((name, deepcopy(args))) or "y", ""))
        self.model([
            (tool_reply(call("find", "find_files", {"pattern": "*config*.py"})), 10),
            (tool_reply(call("search", "search_text", {"pattern": "TIMEOUT", "path": "src", "glob": "*.py", "context_lines": 1})), 20),
            (tool_reply(call("read", "read_file", {"path": "src/config.py", "start_line": 2, "end_line": 2})), 30),
            (tool_reply(call("edit", "edit_file", {"path": "src/config.py", "edits": [{"oldText": "TIMEOUT = 30", "newText": "TIMEOUT = 60"}]})), 40),
            (answer("已修改指定值"), 50),
        ])
        self.run_agent("找到配置并只修改 TIMEOUT", self.agent.permissions)
        self.assertEqual(path.read_bytes(), b"# header\r\nTIMEOUT = 60\r\n")
        self.assertEqual([name for name, _ in asked], ["edit_file"])
        self.assertEqual(self.agent.run_state.steps_used, 5)
        self.assertEqual(self.last_result.stop_reason, "end_turn")
        self.assertEqual(self.agent.token_usage.totals["total_tokens"], 155)
        results = self.agent.run_state.tool_results
        self.assertEqual([item["call_id"] for item in results], ["find", "search", "read", "edit"])
        self.assertTrue(all(item["ok"] for item in results))
        self.assertIn("src/config.py:2: TIMEOUT = 30", results[1]["data"])
        self.assertTrue(all(record["status"] == "recorded" for record in self.agent.run_state.call_records.values()))
        self.assertEqual(len(self.all_requests), 5)
        for event in self.agent.events:
            if event["type"] == "tool_call" and event["tool"] in ("find_files", "search_text"):
                self.assertEqual(event["execution"], {"location": "host"})

    def test_invalid_arguments_and_escape_never_ask_or_start_a_process(self):
        asked = []
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or "y", ""))
        cases = [("find_files", {"pattern": "*", "limit": 0}),
                 ("find_files", {"pattern": "*", "limit": True}),
                 ("find_files", {"pattern": "*", "limit": 2001}),
                 ("find_files", {"pattern": "*", "command": "echo nope"}),
                 ("search_text", {"pattern": "x", "regex": "false"}),
                 ("search_text", {"pattern": "x", "context_lines": 6}),
                 ("search_text", {"pattern": "x", "limit": 1001}),
                 ("search_text", {"pattern": ""}),
                 ("search_text", {"pattern": "x\ny"})]
        with patch.object(file_search.subprocess, "Popen", side_effect=AssertionError("invalid search started process")):
            for name, args in cases:
                with self.subTest(name=name, args=args):
                    self.assertEqual(self.agent._invoke_tool(name, args).code, "invalid_arguments")
            self.assertEqual(self.agent._invoke_tool("search_text", {"pattern": "x", "path": "../outside"}).code, "policy_denied")
        self.assertEqual(asked, [])

    def test_search_preview_can_recover_only_the_saved_full_result(self):
        raw = "\n".join(f"needle-{i:02d} " + "x" * 120 for i in range(40))
        (self.workspace / "data.txt").write_text(raw, encoding="utf-8")
        self.model([
            tool_reply(call("search", "search_text", {"pattern": "needle", "max_output_tokens": 32})),
            tool_reply(call("recall", "read_tool_result", {"call_id": "search", "max_output_tokens": 4096})),
            answer("已回查"),
        ])
        self.run_agent("查找并回查", self.agent.permissions)
        preview = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertTrue(preview["truncated"])
        self.assertEqual(preview["saved_call_id"], "search")
        self.assertIn("complete=true", preview["data"])
        original, recalled = self.agent.run_state.tool_results
        self.assertEqual(original["data"], result_page(recalled["data"])[1])
        self.assertIn("data.txt:40: needle-39", original["data"])
        self.assertNotEqual(preview["data"], original["data"])
        self.assertEqual((self.workspace / "data.txt").read_text(encoding="utf-8"), raw)

    def test_search_at_round_limit_finishes_tools_without_extra_model_request(self):
        (self.workspace / "data.txt").write_text("needle", encoding="utf-8")
        self.agent.max_steps = 1
        self.model([tool_reply(call("search", "search_text", {"pattern": "needle"}))])
        self.run_agent("查找", self.agent.permissions)
        self.assertTrue(self.agent.run_state.tool_results[0]["ok"])
        self.assertEqual((self.last_result.stop_reason, self.agent.run_state.steps_used), ("max_steps", 1))
        self.assertEqual(len(self.all_requests), 1)

    def test_missing_rg_reports_status_without_breaking_cli_or_read_tool(self):
        with patch.object(file_search, "discover_rg", return_value=None):
            self.tools = tools.Toolset(self.tools.workspace, run_command=self.fixture_command)
        self.agent.tools = self.tools
        stream = io.StringIO()
        with patch("builtins.input", return_value="exit"), contextlib.redirect_stdout(stream):
            cli.run_session(self.agent, self.store, terminal=True)
        self.assertIn("搜索工具未启用", stream.getvalue())
        self.assertEqual(self.all_requests, [])
        self.assertEqual(self.agent._invoke_tool("search_text", {"pattern": "x"}).code, "tool_unavailable")

    def test_root_source_snapshot_includes_search_module_and_runs_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as directory:
            for source in Path(agent.__file__).parent.glob("*.py"):
                shutil.copyfile(source, Path(directory) / source.name)
            code = '''from pathlib import Path
import file_search
from tools import Toolset
from permissions import Workspace
assert Path(file_search.__file__).parent == Path.cwd()
root=Path('workspace');root.mkdir();(root/'test.py').write_text('needle',encoding='utf-8')
t=Toolset(Workspace(root),run_command=lambda *a:None)
r=t.execute('search_text',{'pattern':'needle'})
assert r.ok,r
assert 'test.py:1: needle' in r.data,r
'''
            child = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], cwd=directory,
                                   capture_output=True, text=True, timeout=20)
            self.assertEqual(child.returncode, 0, child.stderr)

    def test_native_search_ignores_rg_environment_configuration(self):
        (self.workspace / "data.py").write_text("needle", encoding="utf-8")
        config = self.store.directory.parent / "rg.config"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("--glob\n*.not-present\n--pre\nnot-a-real-program\n", encoding="utf-8")
        with patch.dict(file_search.os.environ, {"RIPGREP_CONFIG_PATH": str(config), "DEEPSEEK_API_KEY": "FAKE-ONLY"}):
            result = self.agent._invoke_tool("search_text", {"pattern": "needle"})
        self.assertTrue(result.ok, result)
        self.assertIn("data.py:1: needle", result.data)


if __name__ == "__main__":
    import unittest
    unittest.main()
