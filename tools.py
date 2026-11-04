"""tools.py —— 第 2 课的注册表 + 第 3 课加固：文件工具只认工作区，shell 固定目录、限时"""
import subprocess
import urllib.request

from sandbox import run_in_sandbox

MAX_OUTPUT = 8000
TOOLS = {}


def tool(name, description, parameters):
    def decorator(func):
        TOOLS[name] = {
            "schema": {"type": "function", "function": {
                "name": name, "description": description, "parameters": parameters}},
            "func": func,
        }
        return func
    return decorator


def get_schemas():
    return [t["schema"] for t in TOOLS.values()]


def _params(**props):
    return {"type": "object",
            "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
            "required": list(props)}


@tool("list_dir", "列出工作区内某个目录下的文件和子目录。", _params(path="目录路径，工作区根目录写 ."))
def list_dir(tools, path="."):
    p = tools.workspace.resolve(path)
    if p is None:                      # 双保险：即使绕过了权限检查，这里还会拦
        return "[错误] 路径在工作区之外"
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())) or "(空目录)"


@tool("read_file", "读取一个文本文件的内容。", _params(path="文件路径"))
def read_file(tools, path):
    p = tools.workspace.resolve(path)
    if p is None:
        return "[错误] 路径在工作区之外"
    return p.read_text(encoding="utf-8")


@tool("write_file", "把内容写入文件，已存在则覆盖。", _params(path="文件路径", content="完整内容"))
def write_file(tools, path, content):
    p = tools.workspace.resolve(path)
    if p is None:
        return "[错误] 路径在工作区之外"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"已写入 {p.relative_to(tools.workspace.root)}（{len(content)} 字符）"


def _decode_command_output(data):
    # 本机编码适配：先读原始字节，避免 Windows 的读取线程按 GBK 解码 UTF-8。
    for encoding in ("utf-8-sig", "gbk"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    # 保留原先 text=True 对换行符的统一处理。
    return text.replace("\r\n", "\n").replace("\r", "\n")


@tool("run_shell", "在 Windows 原生沙箱中执行 cmd 批处理命令，for 循环变量使用 %%。",
      _params(command="要执行的命令"))
def run_shell(tools, command, timeout=30):
    try:
        p = tools.run_command(command, tools.workspace.root, timeout)
    except subprocess.TimeoutExpired:
        return f"[超时] 命令执行超过 {timeout} 秒，已终止。"
    return f"[exit {p.returncode}]\n{_decode_command_output(p.stdout)}{_decode_command_output(p.stderr)}"


@tool("fetch_url", "抓取网页文本，url 以 http/https 开头。", _params(url="完整的 URL"))
def fetch_url(tools, url):
    req = urllib.request.Request(url, headers={"User-Agent": "nami-agent"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="replace")


class Toolset:
    """绑定工作区与命令执行器；默认执行器始终是原生沙箱。"""
    def __init__(self, workspace, *, run_command=None):
        self.workspace = workspace
        self.run_command = run_command if run_command is not None else run_in_sandbox

    def get_schemas(self):
        return get_schemas()

    def execute(self, name, args):
        """执行工具，永远返回字符串，永远不抛异常"""
        if name not in TOOLS:
            return f"Error: 不存在工具 '{name}'，可用工具：{', '.join(TOOLS)}"
        try:
            result = str(TOOLS[name]["func"](self, **args))
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        if len(result) > MAX_OUTPUT:
            result = result[:MAX_OUTPUT] + f"\n...（已截断，共 {len(result)} 字符）"
        return result
