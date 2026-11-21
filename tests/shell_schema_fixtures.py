LEGACY_SHELL_DESCRIPTION = (
    "通过配置的执行后端运行命令。内置后端为 Windows 原生沙箱，默认 cmd（for 变量 %%），可选 Git Bash。Windows .cmd 脚本使用 cmd。"
)
CURRENT_SHELL_DESCRIPTION = (
    '通过配置的执行后端运行命令。每次从 list_dir(".") 对应的工作区根目录启动；项目文件用相对于该目录的路径。'
    '解释器的绝对路径只用于调用解释器，直接传入相对的脚本路径即可，无需切换目录。'
    '内置后端为 Windows 原生沙箱，默认 cmd（for 变量 %%），可选 Git Bash。Windows .cmd 脚本使用 cmd。'
)
