"""终端入口：本机初始化、命令交互与会话保存。"""
import os
from uuid import uuid4

from agent import Agent, MODEL
from context import SessionStore
from permissions import Workspace
from prompts import SYSTEM
from tools import Toolset


def create_agent():
    # 只有启动真实程序时读取现有密钥并创建 SDK 客户端。
    from openai import OpenAI
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env:
            api_key = winreg.QueryValueEx(env, "DEEPSEEK_API_KEY")[0]
    client = OpenAI(
        api_key=api_key, base_url=os.environ.get("BASE_URL", "https://api.deepseek.com"))
    workspace = Workspace(os.environ.get("NAMI_WORKSPACE", os.getcwd()))
    return Agent(client, Toolset(workspace), model=os.environ.get("MODEL", MODEL))


def handle_auto(args, perms):
    """处理 /auto 的状态查询、开关参数和用法提示。"""
    if not args:
        print(f"自动执行：{'已开启' if perms.auto_approve else '已关闭'}。\n")
    elif len(args) == 1 and args[0] in ("on", "off"):
        perms.set_auto_approve(args[0] == "on")
        if perms.auto_approve:
            print("自动执行已开启：现有工具免确认；路径硬规则和命令沙箱仍生效，fetch_url 在宿主执行。\n")
        else:
            print("自动执行已关闭：需确认的工具恢复询问，已清除本次记住的工具授权。\n")
    else:
        print("用法：/auto on、/auto off，或 /auto 查看状态。\n")


def start_new_session(agent, store):
    """先备份、保存，再切换上下文；保留 token 累计与权限状态。"""
    backup_name = f"default-backup-{uuid4().hex}"
    fresh = [{"role": "system", "content": SYSTEM}]
    store.save(backup_name, agent.messages)
    store.save("default", fresh)
    agent.messages = fresh
    agent.last_prompt_tokens = 0   # 只重置压缩依据，不清零 token 累计。
    print(f"已开始新会话；旧会话已备份为 {backup_name}.json；token 累计保留。\n")


def main(agent=None, store=None):
    agent = agent if agent is not None else create_agent()
    store = store if store is not None else SessionStore()
    agent.messages = store.load("default") or [{"role": "system", "content": SYSTEM}]
    if agent.event_sink is None:
        agent.event_sink = lambda event: store.append_event("default", event)
    print("自动执行默认关闭；/auto on 开启，/auto off 关闭，/auto 查看状态。")
    while True:
        q = input("你> ").strip()
        if q in ("exit", "quit"):
            break
        command = q.split()
        if command and command[0] == "/auto":
            handle_auto(command[1:], agent.permissions)
            continue
        if q == "/new":
            start_new_session(agent, store)
            continue
        try:
            result = agent.run(q)
            print("\n" + result.text + "\n")
        finally:
            store.save("default", agent.messages)


