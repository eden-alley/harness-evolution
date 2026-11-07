"""原生工具契约、旧接入兼容和权限参数一致性；没有真实模型或网络。"""
from copy import deepcopy
import json
import socket
from unittest.mock import patch

from scenario_support import OfflineScenario, permissions, tools, answer, call, tool_reply


class ToolContractScenarios(OfflineScenario):
    def test_native_bad_types_and_unpublished_arguments_never_request_permission(self):
        asked = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: (asked.append(name) or "y", ""))
        cases = [
            ("write_file", {"path": "bad.txt", "content": 123}),
            ("read_file", {"path": False}),
            ("run_shell", {"command": "echo harmless", "timeout": 3600}),
            ("list_dir", {"path": ".", "extra": True}),
            ("fetch_url", {"url": ["https://offline.invalid"]}),
        ]
        for name, args in cases:
            with self.subTest(name=name):
                result = self.agent._invoke_tool(name, args)
                self.assertFalse(result.ok)
                self.assertEqual(result.code, "invalid_arguments")
                self.assertTrue(result.message)
        self.assertEqual(asked, [])
        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_registered_schemas_enforce_nested_rules_without_network_or_coercion(self):
        performed = []
        schema = {"type": "object", "properties": {
            "count": {"type": "integer", "minimum": 0},
            "options": {"type": "object", "properties": {"tag": {"enum": ["a", "b"]}},
                        "required": ["tag"], "additionalProperties": False}},
            "required": ["count", "options"], "additionalProperties": False}
        with patch.dict(tools.TOOLS):
            @tools.tool("contract_probe", "本地样本", schema)
            def probe(toolset, count, options):
                performed.append((count, options))
                return tools.ToolResult(True, data={"count": count, "options": options})

            for args in [
                {"count": True, "options": {"tag": "a"}},
                {"count": -1, "options": {"tag": "a"}},
                {"count": "1", "options": {"tag": "a"}},
                {"count": 1, "options": {"tag": "c"}},
                {"count": 1, "options": {}},
                {"count": float("nan"), "options": {"tag": "a"}},
            ]:
                with self.subTest(args=args):
                    result = self.tools.execute("contract_probe", args)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.code, "invalid_arguments")
            self.assertEqual(performed, [])
            self.assertTrue(self.tools.execute("contract_probe", {"count": 1, "options": {"tag": "a"}}).ok)
            self.assertEqual(performed, [(1, {"tag": "a"})])
            with self.assertRaises(ValueError):
                tools.tool("contract_probe", "重复名称", schema)(probe)
            with self.assertRaises(ValueError):
                tools.tool("bad_schema_probe", "非法 schema", {"type": "not-a-json-type"})(probe)

            @tools.tool("remote_schema_probe", "不应联网", {"$ref": "https://offline.invalid/schema.json"})
            def remote(toolset, **kwargs):
                performed.append("remote")
                return tools.ToolResult(True)

            attempted = []
            with patch.object(socket.socket, "connect",
                              side_effect=lambda address: attempted.append(address)):
                result = self.tools.execute("remote_schema_probe", {})
            self.assertFalse(result.ok)
            self.assertEqual(attempted, [])
            self.assertEqual(performed, [(1, {"tag": "a"})])

    def test_result_contract_rejects_invalid_native_data_but_preserves_opaque_legacy_values(self):
        with patch.dict(tools.TOOLS):
            values = ["unstructured", tools.ToolResult(1), tools.ToolResult(False),
                      tools.ToolResult(True, data={1, 2}), tools.ToolResult(True, code="error"),
                      tools.ToolResult(True, data=float("nan"))]
            executed = []
            for index, value in enumerate(values):
                name = f"result_probe_{index}"

                @tools.tool(name, "返回值样本", {"type": "object", "additionalProperties": False})
                def invalid_result(toolset, value=value, index=index):
                    executed.append(index)
                    return value

                result = self.tools.execute(name, {})
                with self.subTest(value=index):
                    self.assertFalse(result.ok)
                    self.assertEqual(result.code, "invalid_tool_result")
                    self.assertIn("状态", result.message)
            self.assertEqual(executed, list(range(len(values))))

            @tools.tool("business_failure_probe", "明确失败", {"type": "object"})
            def explicit_failure(toolset):
                return tools.ToolResult(False, code="business_error", message="样本操作失败")

            result = self.tools.execute("business_failure_probe", {})
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "business_error")

            def legacy(toolset, value="default"):
                return {"error": value}

            tools.TOOLS["legacy_probe"] = {"schema": {"type": "function", "function": {
                "name": "legacy_probe", "description": "旧接口",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}}}},
                "func": legacy}
            result = self.tools.execute("legacy_probe", {"value": "opaque"})
            self.assertTrue(result.ok, "旧接口保持“函数返回”含义，不能猜测业务状态")
            self.assertEqual(result.data, {"error": "opaque"})
            self.assertEqual(result.to_dict().get("result_contract"), "legacy")
            json.dumps(result.to_dict(), allow_nan=False)
            self.assertEqual(self.tools.execute("legacy_probe", {"unexpected": 1}).code,
                             "invalid_arguments")

    def test_approval_cannot_change_the_arguments_that_are_executed(self):
        original = {"path": "original.txt", "content": "  原文\n"}
        before = deepcopy(original)

        def rewrite(name, args):
            args.update(path="changed.txt", content="被改写")
            return "a", ""

        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace, ask=rewrite)
        result = self.agent._invoke_tool("write_file", original)
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "execution_error")
        self.assertIn("参数", result.message)
        self.assertEqual(original, before)
        self.assertEqual(list(self.workspace.iterdir()), [])
        self.assertEqual(self.agent.permissions.always_allowed, set(),
                         "改写参数的确认不能留下会话授权")

        approved = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace, ask=lambda name, args: (approved.append(deepcopy(args)) or "y", ""))
        result = self.agent._invoke_tool("write_file", original)
        self.assertTrue(result.ok)
        self.assertEqual(approved, [before])
        self.assertEqual((self.workspace / "original.txt").read_text(encoding="utf-8"), "  原文\n")
        self.assertFalse((self.workspace / "changed.txt").exists())

    def test_omitted_directory_path_and_empty_file_content_work_through_the_real_loop(self):
        schema = next(row["function"]["parameters"] for row in self.tools.get_schemas()
                      if row["function"]["name"] == "list_dir")
        self.assertNotIn("path", schema.get("required", []))
        self.agent.permissions.set_auto_approve(True)
        self.model([
            tool_reply(call("list", "list_dir", {}),
                       call("write", "write_file", {"path": "empty.txt", "content": ""}),
                       call("read", "read_file", {"path": "empty.txt"})),
            answer("已创建空文件"),
        ])
        self.run_agent("列出目录并创建空文件", self.agent.permissions)
        self.assertEqual((self.workspace / "empty.txt").read_bytes(), b"")
        self.assertEqual(self.agent.run_state.steps_used, 2)
        outputs = [json.loads(m["content"]) for m in self.requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual([item["ok"] for item in outputs], [True, True, True])
        self.assertEqual([item["data"] for item in outputs][::2], ["(空目录)", ""])
        self.assertTrue(all("result_contract" not in item for item in outputs))
        direct = tools.read_file(self.tools, "empty.txt")
        self.assertIsInstance(direct, tools.ToolResult)
        self.assertEqual(direct.data, "")

