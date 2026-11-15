"""MCP routing and recovery contracts without the optional SDK."""

from dataclasses import replace
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from mcp_adapter import MCPAdapter, MCPConfig, _build_catalog
from scenario_support import OfflineScenario, agent, cli, permissions, tools


INPUT_SCHEMA = {
    "type": "object", "properties": {"note_id": {"type": "string"}},
    "required": ["note_id"], "additionalProperties": False,
}


def advertised(name, *, output_schema=None):
    return SimpleNamespace(name=name, description="读取笔记", inputSchema=INPUT_SCHEMA,
                           outputSchema=output_schema, annotations=SimpleNamespace(readOnlyHint=True))


class MCPToolsetScenarios(OfflineScenario):
    def setUp(self):
        super().setUp()
        self.config = MCPConfig("notes", "python.exe", ("notes.py",), self.workspace,
                                ("lookup.note", "manual"), frozenset({"lookup.note"}))

    def adapter(self, request, *, config=None, listed=None):
        config = config or self.config
        catalog = _build_catalog(config, listed or [advertised(name) for name in config.allow_tools])
        return MCPAdapter(config, catalog, request, {"name": "notes", "version": "1", "protocol": "2025-11-25"})

    def test_catalog_and_permissions_belong_to_the_selected_instance(self):
        called = []
        asked = []

        def request(name, args):
            called.append((name, args))
            return SimpleNamespace(structuredContent={"note_id": args["note_id"]}, content=[], isError=False)

        def ask(name, args):
            asked.append((name, args))
            return "n", "离线测试"

        adapter = self.adapter(request, listed=[advertised("hidden"), advertised("manual"),
                                                advertised("lookup.note")])
        local = tools.Toolset(self.tools.workspace, mcp=adapter)
        perms = permissions.PermissionManager(workspace=local.workspace, ask=ask,
                                              mcp_read_only=adapter.read_only_tools)
        perms.set_auto_approve(True)
        runtime = agent.Agent(None, local, permissions=perms)
        valid = {"note_id": "pilot"}
        selected = "mcp__notes__lookup_note"
        manual = "mcp__notes__manual"
        unlisted = runtime._invoke_tool("mcp__notes__hidden", valid)
        self.assertEqual(unlisted.code, "tool_unavailable")
        self.assertIn(selected, unlisted.available_tools)
        self.assertNotIn("mcp__notes__hidden", unlisted.available_tools)
        self.assertEqual(runtime._invoke_tool(manual, {"note_id": 7}).code, "invalid_arguments")
        self.assertEqual(runtime._invoke_tool(manual, valid).code, "user_denied")
        self.assertEqual(runtime._invoke_tool(selected, valid).data, {"note_id": "pilot"})
        self.assertEqual(asked, [(manual, valid)])
        self.assertEqual(called, [("lookup.note", valid)])
        self.assertEqual(self.tools.validate_call(selected, valid).code, "tool_unavailable")

    def test_timeout_and_disconnect_report_unconfirmed_effect_with_one_request(self):
        for error, code in [(TimeoutError(), "mcp_timeout"), (BrokenPipeError(), "mcp_call_unconfirmed")]:
            with self.subTest(code=code):
                requests = []

                def request(name, args):
                    requests.append((name, args))
                    raise error

                local = tools.Toolset(self.tools.workspace, mcp=self.adapter(request))
                args = {"note_id": "pilot"}
                result = local.execute("mcp__notes__lookup_note", args)
                self.assertFalse(result.ok)
                self.assertEqual(result.code, code)
                self.assertIn("执行状态待核实", result.message)
                self.assertEqual(requests, [("lookup.note", args)])

    def test_checkpoint_binding_covers_the_remote_name_behind_an_alias(self):
        self.model([])
        first = replace(self.config, allow_tools=("lookup.note",), read_only_tools=frozenset())
        second = replace(first, allow_tools=("lookup_note",))
        fingerprints = []
        catalogs = []
        for config in (first, second):
            adapter = self.adapter(lambda *args: self.fail("binding must not call tools"), config=config)
            local = tools.Toolset(self.tools.workspace, mcp=adapter)
            runtime = agent.Agent(self.agent.client, local)
            catalogs.append(local.get_schemas())
            fingerprints.append(agent.checkpoints.binding(runtime)["tools"])
        self.assertEqual(catalogs[0], catalogs[1])
        self.assertNotEqual(fingerprints[0], fingerprints[1])

    def test_discovery_rejects_bad_allow_lists_and_external_schema_references(self):
        with self.assertRaisesRegex(ValueError, "缺少白名单工具"):
            _build_catalog(self.config, [advertised("lookup.note")])
        with self.assertRaisesRegex(ValueError, "内部引用"):
            _build_catalog(self.config, [
                advertised("lookup.note", output_schema={"$ref": "https://offline.invalid/schema.json"}),
                advertised("manual"),
            ])
        config = replace(self.config, allow_tools=("lookup.note", "lookup_note"))
        with self.assertRaisesRegex(ValueError, "冲突"):
            _build_catalog(config, [advertised("lookup.note"), advertised("lookup_note")])

    def test_cli_rejects_mcp_config_in_the_agent_writable_workspace(self):
        config = self.workspace / "mcp.json"
        config.write_text(json.dumps({
            "server": "notes", "command": "python", "allow_tools": ["lookup_note"],
        }), encoding="utf-8")
        env = {"NAMI_MCP_CONFIG": str(config), "NAMI_WORKSPACE": str(self.workspace)}
        with patch.dict(os.environ, env), \
                patch("anyio.from_thread.start_blocking_portal", side_effect=AssertionError("server started")), \
                patch.object(cli, "create_agent", side_effect=AssertionError("model created")):
            with self.assertRaisesRegex(SystemExit, "配置.*工作区之外"):
                cli.main(store=self.store)

    def test_remote_endpoint_is_explicit_and_binds_to_checkpoint(self):
        config_path = self.workspace.parent / "remote.json"
        settings = {"server": "notes", "url": "https://docs.invalid/mcp",
                    "allow_tools": ["lookup.note"]}
        config_path.write_text(json.dumps(settings), encoding="utf-8")
        first = MCPConfig.load(config_path, self.workspace)
        self.assertEqual(first.url, settings["url"])
        self.model([])
        fingerprints = []
        for url in (first.url, "https://other.invalid/mcp"):
            adapter = self.adapter(lambda *args: self.fail("binding must not call tools"),
                                   config=replace(first, url=url))
            local = tools.Toolset(self.tools.workspace, mcp=adapter)
            fingerprints.append(agent.checkpoints.binding(agent.Agent(self.agent.client, local))["tools"])
        self.assertNotEqual(fingerprints[0], fingerprints[1])
        for extra in [{"url": "http://docs.invalid/mcp"},
                      {"url": "https://user:pass@docs.invalid/mcp"},
                      {"url": "https://docs.invalid/mcp?token=fixture"},
                      {"url": "https://docs.invalid/mcp#fragment"},
                      {"command": "python"}, {"args": []}]:
            with self.subTest(extra=extra):
                config_path.write_text(json.dumps(settings | extra), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MCPConfig.load(config_path, self.workspace)
