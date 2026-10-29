# Harness Evolution

从 `agent_50.py` 的 231 行基础版本开始，记录一个 Agent harness 如何逐步完善。
每次改进用独立的 Git 提交记录：遇到了什么问题、做了什么调整，以及怎样验证。

## 起点

初始提交原样保留已有的 `agent_50.py`，没有重构或修改其运行行为。

当前代码包含：

- 用 Python 标准库调用 DeepSeek 的聊天与工具接口。
- `Agent` 实例内的多轮消息历史，以及简单的历史截断。
- `read_file`、`web_search`、`edit_file` 三个工具。
- 工具调用分发、JSON 参数解析，以及部分错误处理。
- 每个任务最多 6 次模型请求，每次最多输出 1500 tokens。

这是一份学习和演进的基础版本。历史目前按消息条数截断；文件工具按传入路径直接读写。
后续可以通过具体失败案例和验证结果，判断哪些地方需要改进。

## 运行

使用 Python 3.10 或更新版本。在 PowerShell 中进入仓库目录：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

运行前，在当前终端设置自己的 `DEEPSEEK_API_KEY` 环境变量；可用
`DEEPSEEK_MODEL` 选择模型，默认值为 `deepseek-chat`。密钥由代码从环境变量读取。

```powershell
.\.venv\Scripts\python.exe agent_50.py
```

输入任务后，控制台会显示模型工具调用、工具结果和最终回答。按 `Ctrl+C` 退出。
`web_search` 使用 DuckDuckGo 后端，不需要搜索 API 密钥，但需要可用网络。
完整 Agent 会调用真实 DeepSeek API，费用由所用账户计费；请求次数限制不是金额上限。

## 版本记录

`baseline` 标签指向这份基础版本。后续可用下面的命令查看演进：

```powershell
git log --oneline
git diff baseline -- agent_50.py
```
