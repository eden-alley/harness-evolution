"""进程内的 token 统计；实例由 Agent 持有，/new 不重置。"""


class TokenUsage:
    def __init__(self):
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def record(self, usage, source):
        if usage is None:
            print(f"   [{source} tokens] 未返回 usage，本次未计入累计。")
            return

        counts = {name: getattr(usage, name) for name in self.totals}
        for name, count in counts.items():
            self.totals[name] += count
        print(
            f"   [{source} tokens] 输入={counts['prompt_tokens']} "
            f"输出={counts['completion_tokens']} 合计={counts['total_tokens']} | "
            f"本次运行累计：输入={self.totals['prompt_tokens']} "
            f"输出={self.totals['completion_tokens']} 合计={self.totals['total_tokens']}"
        )
