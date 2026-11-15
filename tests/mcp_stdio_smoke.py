"""Real local stdio, with model replies supplied by OfflineScenario."""

import contextlib
from datetime import timedelta
import io
import json
import os
import socket
import sys
from unittest.mock import patch

from scenario_support import OfflineScenario, ROOT, answer, call, cli, tool_reply


ALIAS = "mcp__notes__lookup_note"
_CONNECT = socket.socket.connect


def loopback_connect(sock, address):
    # Windows asyncio uses a loopback socketpair to wake the portal thread.
    if address[0] not in ("127.0.0.1", "::1"):
        raise AssertionError("MCP 离线测试禁止外部网络")
    return _CONNECT(sock, address)


class MCPStdioScenarios(OfflineScenario):
    def run_trial(self, commands, *, read_only, script=ROOT / "mcp_notes_server.py"):
        from mcp import ClientSession
        from mcp.client import stdio

        config = self.workspace.parent / "mcp.json"
        config.write_text(json.dumps({
            "server": "notes",
            "command": os.environ.get("AGENT_TEST_MCP_PYTHON", "python"),
            "args": ["-B", str(script)],
            "allow_tools": ["lookup_note"],
            "read_only_tools": ["lookup_note"] if read_only else [],
        }), encoding="utf-8")
        model_config = cli.models.ModelConfig()
        adapter = cli.models.get_model(model_config, client=self.agent.client)
        env = {
            "NAMI_MCP_CONFIG": str(config), "NAMI_WORKSPACE": str(self.workspace),
            "MODEL_PROVIDER": "deepseek", "MODEL": "deepseek-flash",
            "BASE_URL": "https://api.deepseek.com", "MODEL_API_KEY_ENV": "",
            "DEEPSEEK_API_KEY": "offline-test-key", "AGENT_STEP_LIMIT": "4",
            "AGENT_TOKEN_LIMIT": "0", "AGENT_TIME_LIMIT_SECONDS": "0",
            "NAMI_MCP_TEST_SECRET": "fixture-only-value",
        }
        commands = iter(commands)
        asked = []
        processes = []
        self.remote_calls = []
        start_process = stdio._create_platform_compatible_process
        call_tool = ClientSession.call_tool

        async def tracked_start(*args, **kwargs):
            process = await start_process(*args, **kwargs)
            processes.append(process)
            return process

        async def tracked_call(session, name, args, **kwargs):
            self.remote_calls.append((name, args))
            return await call_tool(session, name, args, **kwargs)

        def terminal_input(prompt):
            if prompt.startswith("[y]"):
                asked.append(prompt)
                return "n"
            if prompt.startswith("拒绝原因"):
                return "离线拒绝"
            return next(commands)

        with patch.dict(os.environ, env), patch.object(cli.models, "get_model", return_value=adapter), \
                patch("builtins.input", side_effect=terminal_input), \
                patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(socket.socket, "connect", loopback_connect), \
                patch.object(stdio, "_create_platform_compatible_process", tracked_start), \
                patch.object(ClientSession, "call_tool", tracked_call), \
                contextlib.redirect_stdout(io.StringIO()):
            cli.main(store=self.store)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode, "CLI 退出后 stdio 服务须已结束")
        return asked

    def test_cli_discovers_and_routes_a_read_only_tool(self):
        self.model([
            tool_reply(call("note", ALIAS, {"note_id": "pilot"}),
                       call("bad", ALIAS, {"note_id": 1}),
                       call("missing", ALIAS, {"note_id": "missing"}),
                       call("native", "list_dir", {}),
                       call("unlisted", "mcp__notes__unlisted", {})),
            answer("已读取 MCP 示例笔记"),
        ])
        self.assertEqual(self.run_trial(["读笔记", "exit"], read_only=True), [])
        schemas = {item["function"]["name"]: item["function"] for item in self.requests[0]["tools"]}
        self.assertIn(ALIAS, schemas)
        self.assertIn("read_file", schemas)
        self.assertEqual(schemas[ALIAS]["parameters"]["required"], ["note_id"])
        self.assertNotIn("mcp__notes__unlisted", schemas)
        outputs = {m["tool_call_id"]: json.loads(m["content"])
                   for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertTrue(outputs["note"]["ok"])
        self.assertEqual(outputs["note"]["data"]["note_id"], "pilot")
        self.assertEqual(outputs["note"]["data"]["text"], "本地 MCP 只读示例笔记。")
        self.assertEqual(outputs["bad"]["code"], "invalid_arguments")
        self.assertEqual(outputs["missing"]["code"], "mcp_tool_error")
        self.assertEqual(outputs["unlisted"]["code"], "tool_unavailable")
        self.assertTrue(outputs["native"]["ok"])
        self.assertEqual(self.remote_calls, [("lookup_note", {"note_id": "pilot"}),
                                             ("lookup_note", {"note_id": "missing"})])
        trace = self.store.directory / "default.trace.jsonl"
        events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["type"] == "tool_result" and e["tool"] == ALIAS and
                            e["call_id"] == "note" and e["ok"] for e in events))
        self.assertEqual(self.store.load("default")[-1]["content"], "已读取 MCP 示例笔记")

    def test_auto_mode_does_not_approve_untrusted_mcp_call(self):
        self.model([tool_reply(call("denied", ALIAS, {"note_id": "pilot"})), answer("已收到拒绝")])
        asked = self.run_trial(["/auto on", "读笔记", "exit"], read_only=False)
        result = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(result["code"], "user_denied")
        self.assertEqual(len(asked), 1)
        self.assertEqual(self.remote_calls, [])

    def test_stdio_timeout_reports_unknown_and_filters_environment(self):
        script = self.workspace.parent / "controlled_server.py"
        script.write_text(
            """import asyncio
import os
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("test-notes", log_level="WARNING")

@mcp.tool(structured_output=True)
async def lookup_note(note_id: str) -> dict[str, bool]:
    if note_id == "slow":
        await asyncio.sleep(30)
    return {"sentinel_present": "NAMI_MCP_TEST_SECRET" in os.environ}

mcp.run(transport="stdio")
""", encoding="utf-8")
        self.model([
            tool_reply(call("env", ALIAS, {"note_id": "env"}),
                       call("slow", ALIAS, {"note_id": "slow"})),
            answer("已收到 MCP 超时结果"),
        ])
        with patch("mcp_adapter.CALL_TIMEOUT", timedelta(seconds=1)):
            self.assertEqual(self.run_trial(["检查服务", "exit"], read_only=True, script=script), [])
        outputs = {m["tool_call_id"]: json.loads(m["content"])
                   for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertEqual(outputs["env"]["data"], {"sentinel_present": False})
        self.assertEqual(outputs["slow"]["code"], "mcp_timeout")
        self.assertIn("执行状态待核实", outputs["slow"]["message"])
        self.assertEqual(self.remote_calls, [("lookup_note", {"note_id": "env"}),
                                             ("lookup_note", {"note_id": "slow"})])

    def test_timeout_also_covers_stall_before_the_sdk_response_timer(self):
        from anyio import sleep
        from mcp import ClientSession
        from mcp.types import CallToolResult

        async def stalled_call(*args, **kwargs):
            await sleep(0.5)
            return CallToolResult(content=[], structuredContent={"late": True})

        self.model([tool_reply(call("stall", ALIAS, {"note_id": "pilot"})), answer("已收到结果")])
        with patch.object(ClientSession, "call_tool", stalled_call), \
                patch("mcp_adapter.CALL_TIMEOUT", timedelta(seconds=0.1)):
            self.run_trial(["检查连接", "exit"], read_only=True)
        result = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(result.get("code"), "mcp_timeout")
        self.assertEqual(self.remote_calls, [("lookup_note", {"note_id": "pilot"})])
