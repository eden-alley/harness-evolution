"""Streamable HTTP client with an in-memory HTTP transport and fake model."""

import json
import socket
from unittest.mock import patch

import httpx

from mcp_adapter import open_mcp
from mcp_stdio_smoke import loopback_connect
from scenario_support import OfflineScenario, answer, call, permissions, tool_reply, tools


URL = "https://docs.invalid/mcp"
NAME = "mcp__docs__lookup"


class MCPHTTPScenarios(OfflineScenario):
    def test_remote_catalog_and_policy_through_the_agent_loop(self):
        config = self.workspace.parent / "remote.json"
        config.write_text(json.dumps({
            "server": "docs", "url": URL, "allow_tools": ["lookup"],
        }), encoding="utf-8")
        requests = []
        confirmed = []
        definition = {
            "name": "lookup",
            "description": "Read a public repository's documentation",
            "inputSchema": {
                "type": "object", "properties": {"repoName": {"type": "string"}},
                "required": ["repoName"], "additionalProperties": False,
            },
        }

        async def http_handler(request):
            self.assertEqual(str(request.url), URL)
            self.assertEqual(request.method, "POST")
            payload = json.loads(request.content)
            method = payload["method"]
            requests.append(payload)
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "initialize":
                result = {
                    "protocolVersion": payload["params"]["protocolVersion"],
                    "capabilities": {"tools": {}}, "serverInfo": {"name": "docs", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [definition, dict(definition, name="unlisted")]}
            elif method == "tools/call":
                self.assertEqual(payload["params"]["name"], "lookup")
                result = {"content": [{"type": "text", "text": "Documentation topics"}],
                          "isError": False}
            else:
                self.fail(f"Unexpected MCP method {method}")
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

        client_class = httpx.AsyncClient

        def http_client(**kwargs):
            self.assertIs(kwargs.get("trust_env"), False)
            return client_class(transport=httpx.MockTransport(http_handler), **kwargs)

        def ask(name, args):
            confirmed.append((name, args))
            return ("y", "") if args["repoName"] == "org/public" else ("n", "fixture denial")

        self.model([
            tool_reply(call("bad", NAME, {"repoName": 7}),
                       call("denied", NAME, {"repoName": "org/denied"}),
                       call("ok", NAME, {"repoName": "org/public"})),
            answer("已取得目录"),
        ])
        with patch.object(socket.socket, "connect", loopback_connect), \
                patch.object(httpx, "AsyncClient", side_effect=http_client):
            with open_mcp(config, workspace=self.tools.workspace.root) as mcp:
                self.agent.tools = tools.Toolset(self.tools.workspace, mcp=mcp)
                perms = permissions.PermissionManager(workspace=self.tools.workspace, ask=ask,
                                                      mcp_read_only=mcp.read_only_tools)
                perms.set_auto_approve(True)
                self.run_agent("查询公开仓库目录", perms)
        names = {item["function"]["name"] for item in self.requests[0]["tools"]}
        self.assertIn(NAME, names)
        self.assertNotIn("mcp__docs__unlisted", names)
        outputs = {m["tool_call_id"]: json.loads(m["content"])
                   for m in self.requests[1]["messages"] if m["role"] == "tool"}
        self.assertEqual(outputs["bad"]["code"], "invalid_arguments")
        self.assertEqual(outputs["denied"]["code"], "user_denied")
        self.assertEqual(outputs["ok"], {"ok": True, "data": "Documentation topics"})
        self.assertEqual(confirmed, [(NAME, {"repoName": "org/denied"}), (NAME, {"repoName": "org/public"})])
        self.assertEqual([p["params"]["arguments"] for p in requests if p["method"] == "tools/call"],
                         [{"repoName": "org/public"}])
