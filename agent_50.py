"""最小 Agent：真实 DeepSeek API；工具实现留空。"""

import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError

class Agent:
    def __init__(self):
        # 以前这里是局部变量，现在变成了类的属性，只要 Agent 实例不销毁，它就在
        self.messages = [
            {
                "role": "system",
                "content": "按需调用工具。依据实际结果回答，失败时如实说明。",
            }
        ]
    def run(self, task):
    # 1. 把用户最新的输入加进历史
        self.messages.append({"role": "user", "content": task})

        for _ in range(MAX_ROUNDS):
            if len(self.messages) > 12:
                recent_messages = self.messages[-11:]
                self.messages = [self.messages[0]] + recent_messages

            payload = {
                "model": MODEL,
                "messages": self.messages,
                "tools": TOOLS,
                "max_tokens": 1500,
            }
            request = Request(
                "https://api.deepseek.com/chat/completions",
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + API_KEY,
                },
            )

            with urlopen(request, timeout=90) as response:
                message = json.load(response)["choices"][0]["message"]

            # 只回传协议允许的标准字段，避免把 reasoning_content 等非标准字段塞回请求
            assistant_message = {"role": "assistant", "content": message.get("content")}
            if message.get("tool_calls"):
                assistant_message["tool_calls"] = message["tool_calls"]
            self.messages.append(assistant_message)

            if not message.get("tool_calls"):
                return message.get("content") or "模型未返回回答。"

            for call in message["tool_calls"]:
                print(
                    "[模型工具调用]",
                    json.dumps(call, ensure_ascii=False, indent=2),
                )
                name = call["function"]["name"]
                handler = HANDLERS.get(name)
                if handler is None:
                    result = {"error": f"未知工具：{name}"}
                else:
                    try:
                        args = json.loads(call["function"]["arguments"] or "{}")
                        result = handler(**args)
                    except json.JSONDecodeError as error:
                        result = {"error": f"参数不是合法 JSON：{error}"}
                    except TypeError as error:
                        result = {"error": f"参数不匹配：{error}"}
                        

                print("[工具结果]", json.dumps(result, ensure_ascii=False))
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return "已达到最大请求次数，停止。"

# 1. 基础配置
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_ROUNDS = 6


# 2. 工具描述与参数：发送给模型，供模型选择工具
SPECS = [
    (
        "read_file",
        "读取指定 UTF-8 文本文件，返回文件内容。",
        {"path": "文件完整路径"},
    ),
    (
        "web_search",
        "检索公开网页，返回标题、链接和摘要。",
        {"query": "搜索关键词"},
    ),
    (
        "edit_file",
        "创建或修改 UTF-8 文件；创建时 old_text 为空且拒绝覆盖已有文件；修改时只替换唯一匹配原文。",
        {
            "path": "文件完整路径",
            "old_text": "精确原文；创建时为空",
            "new_text": "新内容",
        },
    ),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in params.items()
                },
                "required": list(params),
                "additionalProperties": False,
            },
        },
    }
    for name, desc, params in SPECS
]


# 3. 工具实现
from pathlib import Path



def read_file(path):
    if not isinstance(path, str):
        return {"error": "path 必须是字符串"}

    try:
        file = Path(path)
        content = file.read_text(encoding="utf-8")
        return {"content": content}
    except (OSError, UnicodeError) as error:
        return {"error": str(error)}


def web_search(query):
    if not isinstance(query, str) or not query.strip():
        return {"error": "query 必须是非空字符串"}

    # 放在函数内导入，缺少依赖时也能返回清楚的工具错误。
    try:
        from ddgs import DDGS
        from ddgs.exceptions import DDGSException
    except ImportError:
        return {"error": "缺少搜索依赖，请在项目虚拟环境中运行 python -m pip install -r requirements.txt"}

    try:
        # 明确指定 DuckDuckGo，不需要搜索 API 密钥。
        items = DDGS(timeout=15).text(
            query.strip(), backend="duckduckgo", max_results=5
        )
        if not items:
            return {"error": "DuckDuckGo 未返回搜索结果，请稍后重试或更换关键词"}

        results = []
        for item in items[:5]:
            results.append({
                "title": item["title"],
                "url": item["href"],
                "content": item["body"],
            })
        return {"results": results}
    except (DDGSException, OSError, ValueError, KeyError, TypeError) as error:
        return {"error": f"搜索失败（{type(error).__name__}）：{error}"}


def edit_file(path, old_text, new_text):
    for value in (path, old_text, new_text):
        if not isinstance(value, str):
            return {"error": "三个参数都必须是字符串"}

    try:
        file = Path(path)

        # 原文为空：创建文件，已有同名文件就报错
        if old_text == "":
            with file.open("x", encoding="utf-8") as handle:
                handle.write(new_text)
            return {"status": "created", "path": str(file)}

        # 原文非空：读取文件，检查是否恰好匹配一处
        content = file.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count != 1:
            return {"error": f"原文匹配了 {count} 处，要求恰好 1 处"}

        updated = content.replace(old_text, new_text, 1)
        file.write_text(updated, encoding="utf-8")
        return {"status": "edited", "path": str(file)}
    except (OSError, UnicodeError) as error:
        return {"error": str(error)}


HANDLERS = {
    "read_file": read_file,
    "web_search": web_search,
    "edit_file": edit_file,
}


# 4. Agent 循环：请求模型 → 执行工具 → 回传结果



# 5. 命令行入口
if __name__ == "__main__":
    agent = Agent()
    while True:
        try:
            print(
                agent.run(input("请输入你的任务："))
                if API_KEY
                else "请先配置 DEEPSEEK_API_KEY。"
            )
        except (URLError, TimeoutError, KeyError, ValueError, TypeError) as error:
            print("请求或工具调用失败：", type(error).__name__,str(error))

