"""tools.py —— 第 2 课的注册表 + 第 3 课加固：文件工具只认工作区，shell 固定目录、限时"""
import subprocess
import urllib.request
import inspect
from dataclasses import dataclass

from sandbox import run_in_sandbox

MAX_OUTPUT = 8000
TOOLS = {}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: object = None
    code: str = ""
    message: str = ""
    available_tools: list | None = None
    exit_code: int | None = None

    def to_dict(self):
        result = {"ok": self.ok}
        if self.data is not None:
            result["data"] = self.data
        if not self.ok:
            result.update(code=self.code, message=self.message)
        if self.available_tools is not None:
            result["available_tools"] = self.available_tools
        if self.exit_code is not None:
            result["exit_code"] = self.exit_code
        return result


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
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())) or "(空目录)"


@tool("read_file", "读取一个文本文件的内容。", _params(path="文件路径"))
def read_file(tools, path):
    p = tools.workspace.resolve(path)
    if p is None:
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
    return p.read_text(encoding="utf-8")


@tool("write_file", "把内容写入文件，已存在则覆盖。", _params(path="文件路径", content="完整内容"))
def write_file(tools, path, content):
    p = tools.workspace.resolve(path)
    if p is None:
        return ToolResult(False, code="policy_denied", message="[错误] 路径在工作区之外")
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
        return ToolResult(False, code="execution_error",
                          message=f"[超时] 命令执行超过 {timeout} 秒，已终止。")
    output = f"[exit {p.returncode}]\n{_decode_command_output(p.stdout)}{_decode_command_output(p.stderr)}"
    return ToolResult(p.returncode == 0, data=output, exit_code=p.returncode,
                      code="execution_error" if p.returncode else "",
                      message="命令非零退出" if p.returncode else "")


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

    def validate_call(self, name, args):
        if not isinstance(args, dict):
            return ToolResult(False, code="invalid_arguments", message="工具参数必须是 JSON 对象")
        if name not in TOOLS:
            return ToolResult(False, code="tool_unavailable",
                              message=f"不存在工具 '{name}'",
                              available_tools=sorted(TOOLS))
        try:
            inspect.signature(TOOLS[name]["func"]).bind(self, **args)
        except TypeError as error:
            return ToolResult(False, code="invalid_arguments", message=f"TypeError: {error}")
        return None

    def execute(self, name, args):
        """执行结果统一封装；完整数据留给记录，模型视图由上下文层截断。"""
        invalid = self.validate_call(name, args)
        if invalid is not None:
            return invalid
        try:
            result = TOOLS[name]["func"](self, **args)
        except Exception as e:
            return ToolResult(False, code="execution_error", message=f"{type(e).__name__}: {e}")
        return result if isinstance(result, ToolResult) else ToolResult(True, data=result)
