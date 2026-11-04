"""场景测试共用的临时目录、真实 SDK 和本地响应传输。"""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent
import cli
import context
import permissions
import tools
import token_usage
from verification import VERIFICATION_SYSTEM


def call(call_id, name, args):
    arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments}}


def tool_reply(*calls):
    return {"role": "assistant", "content": None, "tool_calls": list(calls)}


def answer(text):
    return {"role": "assistant", "content": text}


def tool_text(content):
    """旧场景仍断言实际输出；结构与错误码另有专门的契约断言。"""
    payload = json.loads(content)
    return payload.get("data", payload.get("message", ""))


class OfflineScenario(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="agent-loop-test-")
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name) / "workspace"
        self.workspace.mkdir()
        elsewhere = Path(temporary.name) / "elsewhere"
        elsewhere.mkdir()
        workspace = permissions.Workspace(self.workspace)
        self.tools = tools.Toolset(workspace, run_command=self.fixture_command)
        self.agent = agent.Agent(None, self.tools)
        self.store = context.SessionStore(Path(temporary.name) / "state" / "sessions")
        self.replace(socket.socket, "connect", AssertionError("离线测试禁止联网"), side_effect=True)
        previous = Path.cwd()
        os.chdir(elsewhere)  # 验证工具依据工作区工作，而非意外使用进程当前目录。
        self.addCleanup(os.chdir, previous)
        self.requests = []
        self.check_requests = []
        self.all_requests = []

    def replace(self, owner, name, value, *, side_effect=False):
        override = patch.object(owner, name, side_effect=value) if side_effect else patch.object(owner, name, value)
        override.start()
        self.addCleanup(override.stop)

    def assert_complete_tool_pairs(self, messages):
        pending = set()
        for message in messages:
            if message["role"] == "tool":
                self.assertIn(message["tool_call_id"], pending)
                pending.remove(message["tool_call_id"])
            else:
                self.assertFalse(pending, "上一轮工具调用缺少结果")
                pending = {c["id"] for c in message.get("tool_calls", [])}
        self.assertFalse(pending)

    def model(self, replies, *, check_replies=None):
        replies = iter(replies)
        checks = iter(check_replies) if check_replies is not None else None

        def transport(request):
            payload = json.loads(request.content)
            self.assert_complete_tool_pairs(payload["messages"])
            self.all_requests.append(payload)
            is_check = payload["messages"][0]["content"] == VERIFICATION_SYSTEM
            (self.check_requests if is_check else self.requests).append(payload)
            try:
                if is_check:
                    # 不提供兜底通过：未显式配置的额外模型验收属于错误请求。
                    if checks is None:
                        raise AssertionError("未配置模型验收响应")
                    message = next(checks)
                else:
                    message = next(replies)
            except StopIteration:
                raise AssertionError("Agent 发出了预期之外的模型请求") from None
            if isinstance(message, httpx.Response):
                return message
            prompt_tokens = None
            if isinstance(message, tuple):
                message, prompt_tokens = message
            body = {
                "id": "offline-response", "object": "chat.completion", "created": 0,
                "model": "offline-model", "choices": [{"index": 0, "message": message,
                    "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
            }
            if prompt_tokens is not None:
                body["usage"] = prompt_tokens if isinstance(prompt_tokens, dict) else {
                    "prompt_tokens": prompt_tokens, "completion_tokens": 1,
                    "total_tokens": prompt_tokens + 1}
            return httpx.Response(200, json=body)

        client = OpenAI(api_key="offline-test-key", base_url="https://offline.invalid",
                        http_client=httpx.Client(transport=httpx.MockTransport(transport)), max_retries=0)
        self.addCleanup(client.close)
        self.replace(self.agent, "client", client)

    def run_agent(self, task, perms, messages=None, *, verifier=None):
        if messages is None:
            messages = [{"role": "system", "content": agent.SYSTEM}]
        with contextlib.redirect_stdout(io.StringIO()):
            self.agent.messages = messages
            self.agent.permissions = perms
            result = self.agent.run(task, verifier=verifier)
        self.last_result = result
        self.assert_complete_tool_pairs(messages)
        return result.text, messages



    def run_cli(self):
        # 模拟再次启动入口；同一测试进程的累计用量仍由同一个实例持有。
        self.agent.permissions = permissions.PermissionManager(workspace=self.tools.workspace)
        cli.main(self.agent, self.store)

    @staticmethod
    def fixture_command(command, workspace, timeout):
        """仅用于已知测试样本的执行器；正式启动入口从不选择它。"""
        from sandbox import _ENV_NAMES
        environment = {name: value for name, value in os.environ.items()
                       if name.upper() in _ENV_NAMES}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\r\n",
                                         dir=workspace, suffix=".cmd", delete=False) as script:
            script.write("@echo off\nchcp 65001 >nul\n" + command + "\n")
            script_path = Path(script.name)
        try:
            cmd = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
            return subprocess.run([str(cmd), "/d", "/s", "/c", ".\\" + script_path.name],
                                  cwd=workspace, env=environment, stdin=subprocess.DEVNULL,
                                  capture_output=True, timeout=timeout)
        finally:
            script_path.unlink(missing_ok=True)
