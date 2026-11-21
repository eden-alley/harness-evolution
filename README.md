# Harness Evolution

一个面向真实任务、持续演进的 Python Agent harness。项目关注模型调用之外的执行工程：
工具契约、权限边界、运行状态、检查点、上下文、资源限制与可复查的执行证据。

目标是构建一个小而完整、容易扩展、行为可验证的基础内核。每次改动需要说明解决了什么问题，
通过可复现测试和版本对照留下依据。当前为单机、同步、Windows 优先的实现，可靠性边界仍在逐项完善。

当前候选版在正式二十题的 19 道有效题上通过 14 道，编程题独立测试通过 6/8。发布版本仍为 v0.41。[五项改动、方法来源与回归清单](docs/validation/2026-11-20-five-area-candidate.md)

## 已实现的能力

- **执行循环**：模型请求、批量工具调用、结果回写和继续推理；每次运行有独立的状态与停止原因。
- **工具系统**：本地函数注册、JSON Schema 参数校验、结构化结果、完整调用证据。
- **权限与命令隔离**：工作区路径校验、人工确认、自动执行开关，以及 Windows 原生命令沙箱。
- **模型适配**：执行、摘要和可选模型验收共用适配接口；支持 DeepSeek 和显式配置的 Chat Completions 兼容服务。
- **上下文与会话**：工具结果截断、旧结果清理、历史摘要、会话保存和新会话备份。
- **检查点与恢复**：保存执行位置、模型回复和工具进度，显式续跑；未知执行状态拒绝重放。
- **资源限制**：执行轮数、任务 token 和总时长分别配置；记录服务实际返回的已知用量。
- **可选任务验收**：调用方配置检查程序或模型验收，最多退回一次；默认不额外调用验收模型。
- **离线验证**：本地模拟模型传输、真实文件操作、故障注入及独立的 Windows 沙箱测试。

工具失败会作为结构化结果交回主模型继续判断；通用的失败恢复策略与重复失败干预尚未实现。

## 架构与职责

| 模块 | 职责 |
| --- | --- |
| `agent.py`、`state.py` | 编排执行、管理单次运行事实、记录停止原因；不把本轮结束等同于任务完成 |
| `models.py`、`model_base.py`、`model_openai.py`、`model_deepseek.py` | 创建适配器、处理模型协议与响应，不作工具执行或业务判断 |
| `tools.py`、`file_edits.py`、`file_search.py` | 工具注册、参数校验、结构化结果；单文件精确编辑与受限搜索 |
| `permissions.py`、`execution_backend.py`、`sandbox.py` | 授权、可替换命令后端和 Windows 原生执行边界 |
| `context.py`、`checkpoint.py` | 管理可压缩聊天历史、会话保存和可恢复的执行现场 |
| `verification.py` | 按调用方给定条件验收，不自行增加任务要求 |
| `limits.py`、`token_usage.py` | 请求前限制检查、单任务额度及进程用量统计 |
| `cli.py`、`prompts.py` | 终端交互、本机配置和执行模型指引 |

通常的执行路径为：输入 → 请求模型 → 校验并授权工具调用 → 执行 → 回写结果 → 再次请求模型。
模型返回普通回答时形成候选；未配置验收器则以 `end_turn` 结束，已配置时按给定条件检查。

模块导入不读取密钥、不创建真实客户端。程序接口可注入模型客户端、工具集、事件接收器和检查点存储；
模型和运行状态模块按源码版本隔离加载，以兼容多版本评测。

## 安装与运行

运行环境为 Windows、Python，以及支持 Windows MXC 的原生 `codex.exe`。
Python 依赖固定在 [requirements.txt](requirements.txt)。本项目只使用 Codex CLI 的本地沙箱子命令，
模型请求由所配置的服务处理。

~~~powershell
git clone https://github.com/eden-alley/harness-evolution.git
Set-Location harness-evolution
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

默认模型为 `deepseek-flash`，地址为 `https://api.deepseek.com`。启动前配置
`DEEPSEEK_API_KEY` 或 `API_KEY`；Windows 进程未继承时，也会尝试读取用户环境注册表中的
`DEEPSEEK_API_KEY`。密钥通过环境配置提供，不写入仓库。

为任务选择独立工作区，然后从仓库目录启动：

~~~powershell
$env:NAMI_WORKSPACE = Join-Path $env:USERPROFILE 'harness-workspace'
New-Item -ItemType Directory -Force -Path $env:NAMI_WORKSPACE | Out-Null
.\.venv\Scripts\python.exe agent.py
~~~

启动后输入任务。实际模型请求会消耗所选服务的 API 额度，工具会真实操作工作区。

| 终端输入 | 行为 |
| --- | --- |
| 普通文本 | 开始新一轮执行，沿用当前会话上下文 |
| `/new` | 先归档检查点、备份旧会话，再清空聊天上下文；保留当前进程 token 累计与权限模式 |
| `/resume` | 显式恢复兼容的未完成检查点，或同步已结束但尚未保存的结果 |
| `/auto` | 查看自动执行状态 |
| `/auto on`、`/auto off` | 开启或关闭已登记工具的免确认模式，默认关闭 |
| `exit`、`quit` | 退出终端 |

`NAMI_WORKSPACE` 默认是启动时的当前目录。未完成检查点存在时，先使用 `/resume` 或 `/new`，
程序不会在启动时自行继续执行。

## 项目只读路径

启动前可在可信终端指定工作区一级的现有文件或目录。以下规则允许读取 `settings.json` 和
`reference` 中的资料，写入应选工作区内其他路径。

~~~powershell
$env:AGENT_READ_ONLY_PATHS = '["settings.json","reference"]'
~~~

文件工具和默认 Windows 原生 Shell 使用同一份规则。`/auto` 仍遵守只读限制，恢复检查点时
会校验规则是否相同。启用规则的会话使用内置工具和默认 Shell，`NAMI_MCP_CONFIG` 应留空。

## 本地 MCP 试点

在项目目录为 MCP 创建独立 Python 环境，再用示例配置启动终端。

~~~powershell
py -3 -m venv .venv-mcp
.\.venv-mcp\Scripts\python.exe -m pip install -r requirements-mcp.txt
$env:NAMI_WORKSPACE = Join-Path $env:USERPROFILE 'harness-mcp-workspace'
New-Item -ItemType Directory -Force -Path $env:NAMI_WORKSPACE | Out-Null
$env:NAMI_MCP_CONFIG = (Resolve-Path .\mcp.example.json).Path
.\.venv-mcp\Scripts\python.exe agent.py
~~~

看到 `mcp__notes__lookup_note` 后，可输入“用 MCP 读取编号 pilot 的示例笔记”。
启动时读取一份本地 stdio 配置。`command: "python"` 指当前解释器，
也可填写本机 .exe 的绝对路径。`args` 中的相对路径以配置文件目录为基准。

`allow_tools` 控制模型可见的工具。`read_only_tools` 是你明确认可可免确认的白名单子集。
其他 MCP 工具逐次确认，`/auto` 开启时也如此。stdio 子进程以宿主账户权限运行，
配置、可执行文件和服务启动脚本应放在任务工作区之外。CLI 会校验配置和可执行文件的位置。
示例只读取固定笔记。

离线验收命令使用真实 stdio 子进程和假模型。

~~~powershell
.\.venv-mcp\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p mcp_stdio_smoke.py -v
~~~

## 远程 MCP

在同一个 `NAMI_MCP_CONFIG` 入口选择远程配置。配置文件用 `url` 指定 HTTPS MCP 端点，
`allow_tools` 决定哪些远程工具进入模型工具表。参数定义从服务发现，调用经过本地校验和权限确认。
更换端点会改变检查点的工具绑定。

~~~powershell
$env:NAMI_WORKSPACE = (Resolve-Path .\playground).Path
$env:NAMI_STATE_DIR = Join-Path $PWD.Path '.mcp-remote-state'
$env:NAMI_MCP_CONFIG = (Resolve-Path .\mcp.remote.example.json).Path
.\.venv-mcp\Scripts\python.exe agent.py
~~~

示例连接公开的 DeepWiki 服务，只开放 `read_wiki_structure`，用于读取公开 GitHub 仓库的文档目录。
输入“用 DeepWiki 查 modelcontextprotocol/python-sdk 的文档目录”，终端会展示发给服务的参数并请求确认。
这一版远程配置适用于公开免鉴权的 Streamable HTTP 端点。远程请求会将已确认的参数发往配置中的服务。

~~~powershell
.\.venv-mcp\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p mcp_http_smoke.py -v
~~~

## 模型配置

`MODEL_PROVIDER=deepseek` 是默认配置。使用其他 Chat Completions 兼容服务时，明确设置：

~~~powershell
$env:MODEL_PROVIDER = 'openai_chat'
$env:MODEL = '<服务支持的模型名>'
$env:BASE_URL = '<服务的 API 基础地址>'
$env:MODEL_API_KEY_ENV = '<存放该服务密钥的环境变量名称>'
.\.venv\Scripts\python.exe agent.py
~~~

`MODEL_API_KEY_ENV` 填环境变量名，密钥放在该变量中。自定义地址不会自动沿用默认 DeepSeek 密钥。
适配层不猜供应商、不自动切换模型、不额外添加重试；保留 SDK 重试及调用方注入的客户端。
兼容服务的工具调用能力需要逐个验证。

当前支持同步文本和原生函数工具调用。流式输出、多模态及其他模型协议不在当前实现范围内。

## 工具、权限与沙箱

保留 `list_dir`、`read_file`、`read_tool_result`、`write_file`、`edit_file`、`run_shell`、`fetch_url` 七个工具。检测到工作区外已安装的原生 `rg` 时，增加 `find_files` 和 `search_text`；开启任务进度后增加 `update_progress`。

不设置 `AGENT_TOOLS` 时沿用上述默认集合、顺序与权限。需要限制本次会话时，可在启动窗口配置精确工具名的 JSON 数组，例如：

~~~powershell
$env:AGENT_TOOLS = '["list_dir","find_files","search_text","read_file","read_tool_result"]'
~~~

`[]` 表示无工具模式；移除该环境变量后重启可恢复默认。启动时显示有效工具，`/tools` 可随时查看当前名称及执行位置，
不请求模型、不改变历史、用量或配置。更改环境变量须重新创建 Agent 或重启，运行中不自动加载。

程序入口使用 `Toolset(workspace, enabled_tools=["read_file", "list_dir"])`，也接受元组；`None` 表示默认，
空列表表示无工具。输入会复制为不可变选择集合，原列表后续修改不影响实例；发布仍按注册顺序，配置排列不影响检查点绑定。
格式错误、重复、空/非字符串名称、未知名称及通配符不被接受，不会悄悄忽略或补齐工具。

显式选择搜索工具但缺少 `rg` 会在创建 Agent 时失败；默认模式仍只隐藏搜索工具。
运行前继续检查已发现的 `rg` 是否可用。`update_progress` 仍需 `AGENT_TASK_PROGRESS=on` 或程序的 `task_progress=True`：
显式选中它但未开进度会报错；进度已开却未选中时，不提供进度工具，也不附加其使用指导。

名单同时覆盖原生工具和已加载的 MCP 完整别名，例如 `mcp__notes__lookup_note`。
它只能缩小现有 MCP `allow_tools` 范围，不能加载白名单外的工具，也不改变 MCP 的权限声明。
保留连接策略：即使名单为空，已配置的 MCP 服务仍会按原流程连接、校验和关闭；“已加载”与“当前可调用”可能不同。

模型说明、参数校验、实际执行、执行位置、可用工具列表和验收调用共用同一查询规则。
关闭工具返回 `tool_unavailable`，不会触发处理函数或人工确认；不是 `policy_denied` 或用户拒绝。
空名单的模型请求省略 `tools` 字段；若响应仍含工具调用，保留 usage 后按协议异常结束，不执行或自动续问。
不同有效工具集合继续拒绝检查点恢复；不删除旧的待处理调用或自动放行。禁用 `run_shell` 也会使命令验收返回不可用错误。
源码版本变化仍遵守原绑定规则，不能据此承诺旧版本检查点可直接恢复。

工具名单控制可调用入口，不替代权限或沙箱。保留 `run_shell` 时仍可能通过命令修改文件；
关闭 `read_tool_result` 也会失去长结果回查能力。工具范围缩小不保证任务仍能完成。

| 工具或组件 | 实际运行位置 | 边界 |
| --- | --- | --- |
| `list_dir`、`read_file`、`write_file`、`edit_file` | 宿主机 Python | 由工作区路径校验和权限组件限制；没有进入命令沙箱 |
| `find_files`、`search_text` | 宿主 Python 与固定的原生 `rg` 进程 | 工作区内只读，固定参数；不经过 shell，也不改变命令沙箱 |
| `read_tool_result` | 宿主机 Python | 只回查本次任务保存的结果 |
| `update_progress`（开启时） | 宿主机 Python | 更新本任务内部进度，免确认；不授予其他工具权限，也不表示验收通过 |
| `run_shell` | 默认 Codex Windows MXC | 由 Codex 的文件、网络策略和本项目的超时清理约束 |
| `fetch_url` | 宿主机 Python | 独立的网页抓取权限及大小、重定向、超时限制 |
| 本地 stdio MCP 服务 | 宿主账户下的独立进程 | 当前没有套用命令沙箱，只能配置可信服务 |
| 远程 MCP 工具 | 外部服务 | 由服务方执行；本机命令沙箱不约束它 |
| 模型请求、会话与检查点 | 宿主机 Python | 模型凭证留在可信调用侧，不能据此推断沙箱不可读取其他凭证副本 |

`tool_call` 事件的 `execution` 字段记录配置的执行位置，不代表权限已通过或工具已经执行。
自定义后端标为 `backend`，旧 `run_command` 注入标为 `unknown`，不推断它们具有沙箱能力。

`read_file` 默认从第 1 行按完整行分页，返回行号、实际范围、结束状态及可直接使用的续读参数。
可指定从 1 起算的 `start_line` 和 `end_line`，两端都包含；到达请求的结束行只表示范围读完，
不等于文件读完。额度不足时续读参数保留原 `end_line`；范围读完而文件未完时，下一次从后续行开始。
空文件明确标记，起始行超过文件末尾报参数错误。正文、状态和续读提示共用输出额度，末页不预留假想的下一页。
读取 UTF-8（保留 BOM），CR/LF/CRLF 按现有逻辑行处理。只探查后续字符判断文件末尾，不为统计总行数读完整个文件。
超长单行仍不拆分：先提高 `max_output_tokens`，达到上限仍不足时可由模型选择现有 `run_shell` 分段读取，
沿用原权限，不自动执行命令、不增加 Bash。跨页读取当前文件，不提供跨调用文件快照。

`read_tool_result` 用调用编号回查本次任务保存的返回内容，按从 0 起算的字符位置 `offset` 续读，
状态行给出左闭右开的字符范围。位置等于正文长度表示结果已读完，超过长度报参数错误。
可用 `field="message"` 回查错误说明，或 `field="result"` 回查含状态的完整对象。
保留原字段回退规则：请求字段不是字符串时返回完整记录的 JSON，并明确标记为 `result`。
返回正文前有一行读取说明；解析完整 JSON 前须去掉说明行，分页时还要去掉续读提示后拼接正文。
每次 `read_file` 保存的是本页，未读取文件内容不进入 state/checkpoint；回查也不能恢复尚未收集的搜索结果。
内置工具均接受可选的 `max_output_tokens`（1–4096），用于调整预览大小。
显式预算按 UTF-8 字节估算，默认预览沿用约 4000 字符；这是内容显示估算，不是包含 JSON 包装的精确 token 上限。
读取工具返回连续页；其他长结果的预览保留真实头尾、状态和回查编号，结构化长结果明确标记为 `data_preview`。
调用顺序固定为参数校验 → 权限检查 → 执行；不会替模型转换参数类型或补造缺失值。

`find_files` 按文件名或相对搜索目录的基础 glob 查找；返回路径始终相对工作区，可直接用于读取和编辑。
`*.py` 匹配任意深度文件名，`src/*.py` 限定该层，`src/**/*.py` 也包含更深层；支持 `*`、`?`、`[]`、`**`。
不支持绝对模式、`..`、排除式和扩展 glob。默认区分大小写，可用 `ignore_case=true` 忽略大小写。

~~~json
{"pattern":"*checkpoint*.py","path":".","limit":200}
~~~

`search_text` 默认搜索字面量，`regex=true` 开启 rg 单行正则。可指定目录和 `glob`，或直接指定文件；
文件模式不再接受 `glob`。结果返回路径、与 `read_file` 一致的逻辑行号和匹配行；`context_lines` 可取 0–5。
支持 UTF-8/BOM 与 CRLF、LF、CR 行号定位；正则锚点采用 rg 的 LF/CRLF 单行语义，不自动转码匹配行；`ignore_case` 同时控制文字和文件 glob。

~~~json
{"pattern":"/resume","path":".","glob":"*.py","context_lines":2}
~~~

两个工具共享候选文件枚举和路径校验。默认由 rg 应用项目及上级目录的 `.gitignore`、`.ignore`、`.rgignore` 等规则，
包含隐藏文件、排除 `.git` 内部；`include_ignored=true` 显式关闭忽略过滤，但仍跳过 `.git`、符号链接、目录联接和多硬链接文件。
先枚举和校验可读候选，再对文件名进行 glob 过滤，避免 include glob 暗中覆盖项目忽略规则；显式文件也遵循相同忽略规则。
内容匹配前分块预检 NUL，以免二进制命中干扰条数上限；这是额外一次顺序读取，共用同一截止时间，不新增文件大小过滤。
禁用用户 rg 配置和全局 Git ignore；只调用工作区外固定的原生程序，不接受命令、预处理器或任意参数。
缺少 rg 时隐藏并拒绝这两个新工具，CLI 给出提示，原有工具仍可使用；不自动下载或切换引擎。

集中上限在 `file_search.py`：文件查找默认 200 条、最多 2000 条；文字搜索默认 100 个匹配逻辑行、最多 1000 行，
上下文行不计入匹配数。单次共用 30 秒截止时间，正文最多 64 KiB，错误说明过长会明确省略；结果按路径和行号稳定排序。
`complete=true` 只表示当前范围及过滤条件内已搜完，不代表整个项目或被忽略文件没有匹配。
确认第 N+1 条才报告条数上限；超时、无法解码、超长行和进程错误明确报告，已有部分结果会保留。
`max_output_tokens` 只控制显示预览，不改变这些搜索上限。已保存的返回正文可用 `read_tool_result` 找回；
未收集的内容不能回查，需要缩小目录、模式、glob 或上下文后重新搜索。

`edit_file` 用于已有文件的局部修改，调用示例：

~~~json
{"path":"src/config.py","edits":[{"oldText":"TIMEOUT = 30","newText":"TIMEOUT = 60"}]}
~~~

各处原文均在同一份原文件中匹配，必须唯一且互不重叠；任意一处失败，整个调用不写入。
只兼容 CRLF/LF/CR 换行，不对空格、标点、大小写进行模糊匹配。支持 UTF-8（含 BOM），
保留未修改片段的字节、换行和末尾状态；新内容采用命中位置的换行风格。
`newText` 可为空以删除内容，`oldText` 不能为空；没有实际变化时返回 `no_changes`，不触碰文件。

当前集中上限在 `file_edits.py`：修改前后均最多 20 MiB，单次最多 20 处；超限明确失败，不截断写入。
差异和摘要在写入前准备，经过同目录临时文件、flush/fsync 和提交前内容核验后替换。
检测到源文件或临时内容变化时返回 `content_changed`，供模型重新读取合并；不自动重试。
这些检查不能消除检查与替换之间的外部竞争，不提供跨文件事务或跨进程恰好一次保证。

第一版支持普通单链接文本文件。只读文件、特殊 Windows 属性、附加数据流，或所有者/权限与同目录临时文件不一致时
会明确失败，避免丢失附加内容、绕过权限或改变硬链接语义；不承诺自定义审计规则等特殊元数据的完整保留。文件工具仍在宿主运行。
权限与 `write_file` 同等级、分别记住会话授权；`/auto` 保留路径和策略硬限制。
完整差异进入已有 state/checkpoint，预览截断可用 `read_tool_result` 回查；已有恢复规则不变。
工具注册与使用说明借鉴 Pi，换行映射和测试案例的来源及许可证见 [Pi 许可说明](docs/third-party/pi-MIT.txt)。

`write_file` 用于新建和完整重写，仍接收 `path`、`content`，沿用现有写入确认和 `/auto` 策略。
输入先精确编码成 UTF-8，再检查 20 MiB 上限；BOM 仅在 `content` 明确包含时写入，
保留输入中的 LF/CRLF/CR，不继承原文件格式，也不经过 Windows 文本换行转换。
覆盖需要能读取并写入已有普通单链接文件；目标和结果均最多 20 MiB，沿用 `edit_file` 的只读、
特殊属性、附加数据流、所有者与访问权限检查，无法满足则拒绝，不改变权限来促成写入。

最终内容先写入同目录临时文件，经完整写入、flush/fsync、目标与临时文件核验后发布。
覆盖使用替换；新建在本机 Windows 使用拒绝覆盖的重命名，目标在调用期间被他人创建则返回
`content_changed`，不会临时改成覆盖。新建安全路径在非 Windows 明确返回 `unsupported_platform`，
本批未扩展其他运行环境。检测范围仅从工具调用开始到提交前，不保护更早的“读取→外部修改→再调用写入”，
也不能消除最后一次检查与文件替换之间的竞争窗口，不提供强并发版本校验或断电恢复保证。

成功 `data` 返回 `path`、`operation`（`created`/`replaced`/`unchanged`）、`bytes` 和 `sha256`。
字节指纹描述本次发布内容，不保证随后未被外部修改。内容字节相同时不替换、不改修改时间。
编码、容量等检查在创建父目录前完成；仍自动创建缺失父目录，后续失败可能留下空目录。
普通异常会尝试清理本次临时文件，清理失败或进程被强制终止时可能残留；不删除其他文件或回滚父目录。
回执在发布前准备；写入与 checkpoint 保存不是同一个事务，写后记录失败沿用未知执行状态且禁止自动重放，
不把已提交文件说成未写入。写入完成也不代表文件功能已经验收。

工具返回 `ToolResult`，区分参数错误、工具不可用、用户拒绝、策略拒绝和执行错误。
结果包含成功状态、数据、错误说明，以及适用时的退出码和可用工具列表。
`ok=True` 只表示工具定义的操作成功，整个任务是否完成还需要相应证据。
工具返回非法结果时标记 `invalid_tool_result`；操作可能已经发生，应先核实状态。

- 工作区内目录浏览与文件读取自动允许；写入、命令和网页抓取默认需要确认。
- 写入可以选择本进程内一直允许；拒绝原因会反馈给模型。
- 路径越界、命中危险规则或被策略禁用的工具不会通过人工放行。
- `/auto on` 仅改变确认步骤，硬规则和命令沙箱继续生效；未知风险工具仍需确认。
- 权限检查使用参数快照，回调修改参数会导致拒绝执行。

默认 `run_shell` 经过 `CodexWindowsSandboxBackend`，调用 `codex.exe sandbox` 的 Windows MXC 执行层，使用 cmd 批处理语法。
本机已通过 MXC 内的 cmd 管道验证。Git Bash 仍可显式请求，兼容性待验收。
本机对照确认同一 Bash 管道在普通进程可用，在 MXC 中因 MSYS 初始化失败无法启动。
遇到该错误会返回 `shell_unavailable` 和原始输出。等价命令可以由调用方明确选择 `shell="cmd"`，仍经过 MXC。
原生执行器必须位于工作区外；缺失、启动失败或不支持时明确失败，不回退宿主 shell。
单条命令默认 30 秒超时，可用 `timeout` 在 0–300 秒内配置（大于 0）。
中断与超时会清理进程树，超时尽量保留已有输出，编码兼容 UTF-8 / GBK。

**隔离尚未完成：** 本机测试确认，工作区外写入被阻止，但默认 `:workspace` 仍允许读取工作区外的模拟文件。
子进程环境中的模拟 API key 已被过滤；同一进程却能读取 `HKCU\Environment` 中的模拟凭证。
因此当前 MXC 配置不能保证 API key 等敏感凭证隔离，不适合作为运行恶意、不可信代码的完整安全边界。
上游也记录了同类[注册表读取问题](https://github.com/microsoft/mxc/issues/1670)。
对只需文件操作的任务，可用 `AGENT_TOOLS` 启用 `list_dir`、`read_file`、`write_file` 和 `edit_file`。
这会从工具目录移除 `run_shell`，并阻断命令验收调用。注册表隔离仍是启用不可信命令执行的独立验收项。

`fetch_url` 在宿主进程执行，允许本机和内网可达地址；
它受 URL 协议、重定向次数、I/O 超时和 2 MiB 响应大小限制，不属于命令沙箱的网络隔离。

### 可替换执行后端

`ExecutionBackend` 仅约定 `run(command, workspace, timeout, *, shell="cmd")` 和无凭据、可 JSON 序列化的 `binding`。
通过 `Toolset(workspace, backend=...)` 注入；不增加后端注册框架，也不自动切换执行环境。
旧 `run_command=...` 及事后赋值继续兼容，与 `backend=...` 不能同时指定。
后端身份进入检查点绑定，切换后端后拒绝恢复旧现场。本批新增绑定字段，已有检查点也会不匹配，旧文件不会被自动迁移或删除。
注入函数以模块名和限定名的哈希标识；不能区分同名闭包或同一可调用类的不同实例。
需要可恢复的自定义执行环境时，应使用明确的后端和策略版本身份；提供方负责更新身份，不把密钥放进绑定。

保留 Codex Windows Sandbox，不自行开发沙箱。独立账户的 Codex `elevated` 方案待单独审核，当前未切换、未安装。
WSL/Linux 及 Docker Backend 暂缓；需要统一 Linux 执行、多任务并发和可复现评测时再评估。

## 运行状态、会话与恢复

`Agent.run(task, verifier=None)` 返回 `RunResult(text, stop_reason, verification)`。
`RunState` 保存单次运行编号、阶段、执行轮数、调用进度、工具证据、用量与最终结果；
`Agent.messages` 保存聊天上下文。`stopped` 表示运行停止，`end_turn` 表示本轮结束。

默认状态目录为 `~/.nami-agent`，通过 `NAMI_STATE_DIR` 可修改：

- `sessions/default.json`：可截断、清理和摘要的聊天历史。
- `sessions/default.trace.jsonl`：追加的原始执行记录，包括模型响应、完整工具结果、验收和停止原因。
- `checkpoints/default.json`：当前执行快照；归档保留在检查点目录下。

会话和工具结果可能包含用户数据。状态目录应放在任务工作区之外，也不应提交到仓库。

每次请求前对系统提示、工具说明、历史和新增结果一起估算。
默认输入预算 128000 Token，达到 96000 时整理，目标保留约 24000–28000 Token。
估算使用 UTF-8 字节数与服务实际 usage 校正，属于近似值。
整理保留原始用户与系统要求、近期完整工具批次、任务进展摘要及最近 20 条结果的简短事实索引。
较短历史不补齐目标下限。原始要求过大、无法满足整理预算时以 `context_limit` 停止并保留现场。
单个长任务也可以整理。摘要计入 token，摘要失败保留原历史。
原始执行证据继续保存在运行状态、轨迹及检查点中，工具回查仍限定当前任务。

可配置 `AGENT_CONTEXT_INPUT_TOKENS`、`AGENT_COMPACT_THRESHOLD_TOKENS`、
`AGENT_COMPACT_TARGET_MIN_TOKENS`、`AGENT_COMPACT_TARGET_MAX_TOKENS`。
须满足目标下限 ≤ 目标上限 < 整理门槛 ≤ 输入预算，配置进入检查点绑定。

终端默认启用检查点；直接构造 `Agent` 默认不读写检查点。检查点在模型请求及回复保存、
工具执行前后和最终收尾处更新，采用同目录原子替换与 OS 文件锁。保存失败停止后续操作。

恢复遵循以下规则：

- 已保存模型回复和已记录工具结果直接复用，不重复调用。
- 未开始的工具重新检查当前参数与权限；已进入执行窗口但缺结果的工具属于未知状态，禁止重放。
- 轮数、验收次数、任务 token 和首次开始时间不因恢复重置；丢失响应的用量标为未知。
- 源码、工作区、工具、执行后端、模型或限制配置不匹配时拒绝恢复，不自动迁移。
- 未完成的自定义验收任务暂不支持续跑；当前进程内验收保持可用。
- 检查点恢复程序进度，文件和外部服务不会回滚到旧状态。

`/new` 会保留旧会话备份并归档现场。恢复请求可能产生新的模型用量；未知执行状态需要先核实，
当前版本不自动核实或填造结果。

## 资源限制与 token 统计

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `AGENT_STEP_LIMIT` | `30` | 单次任务执行轮数；`0` 不限 |
| `AGENT_TOKEN_LIMIT` | `0` | 单次任务累计已知 `total_tokens` 上限；`0` 关闭 |
| `AGENT_TIME_LIMIT_SECONDS` | `0` | 自首次开始计算的总时长，可用小数；`0` 关闭 |

在启动窗口设置，重启进程后读取：

~~~powershell
$env:AGENT_STEP_LIMIT = '30'
$env:AGENT_TOKEN_LIMIT = '50000'
$env:AGENT_TIME_LIMIT_SECONDS = '300'
~~~

限制在下一次模型请求前检查，已开始的请求和当前工具批次可以完成，单次调用可能越过阈值。
总时长包括暂停和重启间隔。启用 token 限制而用量缺失或丢失时，以 `usage_unknown` 停止下一请求；
已知用量或时长达到上限时分别为 `token_limit`、`time_limit`。

程序接口用 `max_steps=None` 关闭轮数限制；旧接口 `max_steps=0` 仍表示零轮立即停止。
任务额度与显示的进程累计分开；后者重启归零，`/new` 保留。
对话、摘要和经 `agent.model_adapter` 发出的模型验收各统计一次；缺失字段标为未知，
缓存输入不重复相加，不计算价格。自定义代码直接使用外部客户端的调用不在此统计接口范围内。

### 任务进度记录（可选）

默认关闭。启动前设置：

~~~powershell
$env:AGENT_TASK_PROGRESS = 'on'
~~~

设置为 `off` 关闭；其他值会在读取模型凭证前报错。程序接口使用
`Toolset(workspace, task_progress=True)`。关闭时不提供、也不执行 `update_progress`。

开启后，主模型可在多步骤任务开始、关键结论变化或阶段验证后调用 `update_progress`。
记录整体替换，只包含 `deliverables`（交付理解）、`findings`（结论与 `call_ids`）、
`remaining`（尚欠工作）和 `next_action`（下一步），紧凑 JSON 不超过 3,000 UTF-8 字节。
引用必须是本次任务唯一对应的已记录模型工具结果，不能引用进度更新来证明结论。
非法或过长更新返回工具错误，不改写之前保存的有效记录。

进度直接保存在现有 `RunState.tool_results` 和 checkpoint 中，不另建文件或状态存储。
进度更新通过正常工具调用和结果进入聊天历史；请求保留固定指导语，不再把最新进度复制成末尾的用户消息。
整理上下文和恢复仍使用现有机制，不保证最新笔记每轮原样展示；原始记录可按调用编号用 `read_tool_result` 回查。
新任务的运行记录仍按原规则重建；源码更新不自动迁移旧版未完成检查点，恢复须通过原有版本绑定校验。
进度正文属于模型工作便笺，不能覆盖用户要求、扩展授权或充当验收通过。
仅更新这份内部记录免确认；文件写入、命令和 MCP 权限仍按原规则执行。

没有额外规划模型或固定轮次更新。更新仍使用正常工具调用、轮数和 Token 统计；
记录及其说明计入输入预算和 24–28K 整理目标。主模型可能漏更新或误读证据，
离线测试只能验证保存和执行机制，能否减少反复调查仍需固定条件的真实任务对照。

可选的进展提醒适合有明确产物的任务。启动前设置 `AGENT_PROGRESS_REPORT` 为工作区内的报告路径，
或设置 `AGENT_PROGRESS_TEST` 为任务要求成功执行的测试命令。程序接口也可在创建 Agent 时传入
`progress_report`、`progress_test`。近期连续查阅且同一项查阅反复出现、约定进展都尚未出现时，
本次任务会收到一次提示。它检查报告是否存在和指定命令是否成功退出，实际交付仍由任务验收判断。

## 按任务验收

终端输入 `/check python -B -m unittest discover -s tests`，随后输入任务。
提交时会运行这条检查命令，失败结果会交回主模型修正一次，再运行同一条检查。
该命令绑定下一次任务，执行仍经过 Shell 权限与沙箱。输入 `/new` 可清除待用的检查并开始新会话。
如果带检查的任务在提交前中断，重启后重新输入原 `/check` 命令，再用 `/resume` 继续。
恢复会核对命令摘要，并保留已记录的工具操作。验收已经开始但任务状态尚未落定时，保留检查点供核实。

终端默认不绑定验收器。程序调用方可传入 `verifier(VerificationInput) -> CheckResult`，
优先复用任务已有的测试或检查命令：

~~~python
from verification import command_verifier

result = runtime.run(
    "修复已有测试发现的问题",
    verifier=command_verifier("py -3 -m unittest discover -s tests"),
)
~~~

这里的 `runtime` 是已创建的 Agent 实例。检查命令经过同一套工具、权限和沙箱。

- `pass`：通过给定条件。
- `fail`：退回主模型修正一次，继续使用原轮数；再次失败为 `verification_unmet`。
- `needs_input`：缺少信息或授权，结束本轮等待用户。
- `error`：验收失败或证据不足，明确未完成验证。

也可显式配置 `ModelVerifier(runtime.model_adapter, criteria, runtime.token_usage.record)`，
必须提供明确条件。模型验收只接收本任务证据、不执行工具，其判断仍需要独立评测检验。
验收落盘报告时，调用方可给 `ModelVerifier` 传入 `artifact_path="report.md"`。
每次提交经原有权限读取当前 UTF-8 文件。文件缺失或为空时退回补齐，读取失败或内容超出单次读取上限时报告验收错误。
能够完整读取时才会把报告正文交给验收模型。

## 测试与工程化证据

基础组不需要 API key、真实模型或本机 Codex；新增真实搜索集成测试需要已安装的 `rg`：

~~~powershell
.\.venv\Scripts\python.exe -B -X utf8 tests/run_offline.py
~~~

Windows 原生沙箱组需在具备兼容 `codex.exe` 和 MXC 的主机运行：

~~~powershell
.\.venv\Scripts\python.exe -B -X utf8 tests/run_offline.py --native
~~~

基础组使用本地 HTTP 模拟传输，保留 SDK 编解码、真实 Agent 循环、参数校验、权限和文件操作，
并覆盖摘要、用量、协议异常、最终消息一致性、检查点故障及跨进程中断。
原生组验证文件边界、实际读取范围、模拟凭证、localhost TCP/UDP 限制和超时后的子进程清理。
网络测试带宿主可达的正向对照；UDP 以监听器是否收到报文判定，不能仅依据发送函数的返回值。
覆盖仅限已测场景，不能推断所有网络通道均已隔离。
2026-11-12 本机存在两个已知未通过项：Git Bash/MSYS 启动、注册表模拟凭证读取隔离。
测试保留非零退出，不将已知失败改成跳过或预期通过。
原生组跳过会返回非零退出码，不能当作通过。

早期离线验证记录见 [验证记录](docs/validation/2026-11-07.json)。任务进度记录的原验证结果见 [任务进度验证记录](docs/validation/2026-11-13-task-progress.json)；取消每轮额外进度卡的测试、已知失败和源码 SHA-256 见 [本次验证记录](docs/validation/2026-11-14-progress-card-removal.json)。
精确编辑工具的本批测试和源码边界见 [v0.34 验证记录](docs/validation/2026-11-14-exact-edit.json)。
第二批搜索定位的测试和源码边界见 [v0.35 验证记录](docs/validation/2026-11-14-search-tools.json)。
第三批读取与续读的测试和源码边界见 [v0.36 验证记录](docs/validation/2026-11-14-read-continuation.json)。
第四批整文件可靠写入的测试和源码边界见 [v0.37 验证记录](docs/validation/2026-11-14-whole-file-write.json)。
第五批工具配置的测试和源码边界见 [v0.38 验证记录](docs/validation/2026-11-14-tool-config.json)。
测试通过说明所覆盖的程序行为符合约定；任务能力改善需要固定模型、任务、环境和评分方式进行版本对照，
保留逐题变化、原始轨迹、工具参数、最终状态和 token。离线测试数量与真实任务通过率分别报告。

BFCL 评测工程和冻结的旧 V0 独立维护，不随本仓库源码同步修改；其外部兼容检查也不等同于模型能力评测。
本次提交未调用真实模型。

## 当前边界与后续取舍

已补齐基础文件查找、内容定位和读取续接；超长单行读取仍有限制，通用失败恢复还需进一步完善。
这些是诊断出的方向，具体行为仍需评审后逐项实现。

多用户服务、异步并发、MCP 接入和完整文件读取隔离尚未实现。工程取舍先明确行为边界与验收标准，
再做小范围改动；每次保留可复现的基线、回归结果、改善与退步。

## 来源

项目起步参考了 [Nami 的 Agent 系列](https://namiai.tech/articles/agent-minimal-loop/)中的循环、
工具、权限和上下文示例。现以独立 harness 项目维护，README 按当前实现组织；
参考来源保留，功能行为以仓库源码及验证证据为准。
