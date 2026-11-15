"""Scoped read-only file discovery and text location using an installed ripgrep.

Interfaces/notices follow Pi find/grep at 827383a15262905f82e418483825a49dbb1adbf5.
Python enumeration, workspace guards and result storage adapt to this harness.
"""
from collections import deque
import base64
from contextlib import closing
import fnmatch
from functools import lru_cache
import json
import os
from pathlib import Path
import queue
import re
import stat
import subprocess
import threading
import time

from sandbox import _ENV_NAMES

SEARCH_TOOLS = frozenset({"find_files", "search_text"})
DEFAULT_FIND_LIMIT = 200
MAX_FIND_LIMIT = 2000
DEFAULT_SEARCH_LIMIT = 100
MAX_SEARCH_LIMIT = 1000
MAX_RESULT_BYTES = 64 * 1024
MAX_SEARCH_SECONDS = 30


class SearchError(ValueError):
    def __init__(self, code, message):
        encoded = message.encode("utf-8")
        if len(encoded) > 2048:
            message = encoded[:2048].decode("utf-8", errors="ignore") + "\n[错误说明过长，已省略]"
        super().__init__(message)
        self.code = code
        self.partial = None


def discover_rg(workspace):
    root = Path(workspace).resolve()
    for entry in os.get_exec_path():
        directory = Path(entry.strip('"'))
        if not directory.is_absolute():
            continue
        candidate = (directory / ("rg.exe" if os.name == "nt" else "rg")).resolve()
        if candidate.is_file() and not candidate.is_relative_to(root):
            return candidate
    return None


class _Glob:
    def __init__(self, pattern, ignore_case=False):
        pattern = pattern.replace("\\", "/") if os.name == "nt" else pattern
        while pattern.startswith("./"):
            pattern = pattern[2:]
        parts = pattern.split("/")
        if (not pattern or pattern.startswith(("/", "!")) or ":" in pattern
                or any(part in ("", "..") for part in parts)
                or "{" in pattern or "}" in pattern or re.search(r"[@+?!*]\(", pattern)):
            raise SearchError("invalid_pattern", "文件模式只支持工作区内的基础 glob：*、?、[]、**；不支持绝对路径、..、排除式或扩展模式。")
        self.basename = len(parts) == 1
        self.parts = parts
        self.matchers = [None if part == "**" else re.compile(fnmatch.translate(part), re.I if ignore_case else 0)
                         for part in parts]

    def matches(self, relative):
        pieces = relative.split("/")
        if self.basename:
            return self.parts[0] == "**" or self.matchers[0].fullmatch(pieces[-1]) is not None
        @lru_cache(None)
        def match(i, j):
            if j == len(self.parts):
                return i == len(pieces)
            if self.parts[j] == "**":
                return match(i, j + 1) or (i < len(pieces) and match(i + 1, j))
            return i < len(pieces) and self.matchers[j].fullmatch(pieces[i]) is not None and match(i + 1, j + 1)
        return match(0, 0)


def _deadline_check(deadline):
    if time.monotonic() >= deadline:
        raise SearchError("search_timeout", f"搜索超过 {MAX_SEARCH_SECONDS} 秒；返回内容仅是已收集部分，请缩小范围。")


def _fingerprint(info):
    # Windows stat/fstat ctime have different legacy meanings; don't compare it.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class _Report:
    def __init__(self, name, scope, limit, include_ignored):
        self.name, self.scope, self.limit = name, scope, limit
        self.include_ignored = include_ignored
        self.rows = {}
        self.count = 0
        self.size = 0
        self.complete = True
        self.reason = "已完成当前范围和过滤条件内的搜索；不包括被忽略和链接文件" + ("及非文本文件。" if name == "search_text" else "。")

    def stop(self, reason):
        self.complete = False
        self.reason = reason
        return False

    def add(self, key, text, *, match=True):
        previous = self.rows.get(key)
        if previous is not None:
            if match and not previous[1]:
                self.rows[key] = (text, True)
                self.count += 1
            return True
        if match and self.count >= self.limit:
            return self.stop("已确认还有更多结果；提高 limit 或缩小 path、pattern、glob 后重新搜索。")
        size = len(text.encode("utf-8")) + 1
        if self.size + size > max(0, MAX_RESULT_BYTES - 2048):
            return self.stop("结果容量达到上限；缩小 path、pattern、glob 或 context_lines 后重新搜索。")
        self.rows[key] = (text, match)
        self.size += size
        self.count += int(match)
        return True

    def text(self):
        complete = str(self.complete).lower()
        scope = json.dumps(self.scope, ensure_ascii=False)
        if len(scope.encode("utf-8")) > 512:
            scope = scope.encode("utf-8")[:512].decode("utf-8", errors="ignore") + "...(范围名过长，详见调用参数)"
        reason = self.reason.encode("utf-8")[:512].decode("utf-8", errors="ignore")
        if reason != self.reason:
            reason += "...(说明已省略)"
        header = (f"[{self.name} complete={complete} returned={self.count} limit={self.limit} "
                  f"ignored_rules={'off' if self.include_ignored else 'on'} scope={scope}]\n")
        body = "".join(value[0] + "\n" for _, value in sorted(self.rows.items()))
        footer = (f"[complete={complete} returned={self.count}；{reason} "
                  "若已启用 read_tool_result，可用它回查已保存的返回结果；未收集部分不能回查。"
                  "若已启用 read_file，定位后可按工作区相对路径和行号读取上下文。]\n")
        return header + body + footer


class FileSearch:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        self.rg_path = discover_rg(self.root)

    @property
    def available(self):
        return self.rg_path is not None and self.rg_path.is_file() and not self.rg_path.is_relative_to(self.root)

    def _resolve(self, path):
        p = Path(path).expanduser()
        p = (p if p.is_absolute() else self.root / p).resolve()
        if not p.is_relative_to(self.root):
            raise SearchError("policy_denied", "搜索路径在工作区之外，禁止访问。")
        if not p.exists():
            raise SearchError("path_not_found", "搜索路径不存在。")
        return p

    def _safe_file(self, candidate):
        if not candidate.is_relative_to(self.root):
            return False
        relative = candidate.relative_to(self.root)
        if any((part.casefold() if os.name == "nt" else part) == ".git" for part in relative.parts) or (os.name == "nt" and ":" in candidate.name):
            return False
        current = self.root
        for part in relative.parts:
            current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                return False
        if not candidate.resolve().is_relative_to(self.root):
            return False
        info = candidate.stat()
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1

    def _records(self, args, cwd, deadline, delimiter=b"\n"):
        _deadline_check(deadline)
        if not self.available:
            raise SearchError("dependency_unavailable", "已安装的 rg 不可用，未执行搜索；原有工具仍可使用。")
        environment = {k: v for k, v in os.environ.items() if k.upper() in _ENV_NAMES}
        process = subprocess.Popen([str(self.rg_path), "--no-config", *args], cwd=cwd,
            shell=False, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        messages = queue.Queue(maxsize=16)
        stopping = threading.Event()
        def send(item):
            while not stopping.is_set():
                try:
                    messages.put(item, timeout=0.05)
                    return
                except queue.Full:
                    pass
        def pump(stream, kind):
            try:
                while not stopping.is_set():
                    chunk = stream.read1(8192)
                    if not chunk:
                        break
                    send((kind, chunk))
            except OSError:
                pass
            finally:
                send((kind, None))
        readers = [threading.Thread(target=pump, args=(stream, kind), daemon=True,
                                    name="harness-rg-" + kind)
                   for stream, kind in [(process.stdout, "out"), (process.stderr, "err")]]
        for reader in readers:
            reader.start()
        pending, errors, ended = bytearray(), bytearray(), set()
        try:
            while len(ended) != 2:
                _deadline_check(deadline)
                try:
                    kind, chunk = messages.get(timeout=min(0.05, max(0.001, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                if chunk is None:
                    ended.add(kind)
                    continue
                if kind == "err":
                    errors.extend(chunk[:max(0, MAX_RESULT_BYTES - len(errors))])
                    continue
                pending.extend(chunk)
                while True:
                    boundary = pending.find(delimiter)
                    if boundary == -1:
                        break
                    if boundary > MAX_RESULT_BYTES:
                        raise SearchError("output_limit_exceeded", "单条搜索输出超过结果容量，未声称完整搜索；请缩小范围。")
                    record = bytes(pending[:boundary])
                    del pending[:boundary + len(delimiter)]
                    if record:
                        yield record
                if len(pending) > MAX_RESULT_BYTES:
                    raise SearchError("output_limit_exceeded", "单条搜索输出超过结果容量，未声称完整搜索；请缩小范围。")
            if pending:
                yield bytes(pending)
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            if code not in (0, 1):
                message = errors.decode("utf-8", errors="replace").strip() or f"rg 退出码 {code}"
                raise SearchError("invalid_pattern" if "regex parse error" in message else "search_error", message)
        except subprocess.TimeoutExpired as error:
            raise SearchError("search_timeout", "等待搜索进程结束时超时，请缩小范围。") from error
        finally:
            stopping.set()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
            for reader in readers:
                reader.join(timeout=1)
            process.stdout.close()
            process.stderr.close()

    def _candidates(self, target, deadline, include_ignored):
        single = target if target.is_file() else None
        if single is not None:
            if not self._safe_file(single):
                return
            target = single.parent
        args = ["--files", "--null", "--hidden", "--no-follow", "--no-require-git", "--no-ignore-global", "--sort", "path"]
        if include_ignored:
            args.append("--no-ignore")
        args.extend(["--iglob" if os.name == "nt" else "--glob", "!**/.git/**", "--", "."])
        with closing(self._records(args, target, deadline, b"\0")) as records:
            for raw in records:
                _deadline_check(deadline)
                try:
                    name = raw.decode("utf-8")
                except UnicodeError as error:
                    raise SearchError("unsupported_encoding", "搜索遇到无法按 UTF-8 还原的路径，未伪造路径。") from error
                candidate = target / name
                if single is not None and candidate != single:
                    continue
                if self._safe_file(candidate):
                    yield candidate
                    if single is not None:
                        return

    def find(self, pattern, path=".", *, limit=DEFAULT_FIND_LIMIT, ignore_case=False, include_ignored=False):
        report = _Report("find_files", path, limit, include_ignored)
        try:
            target = self._resolve(path)
            if not target.is_dir():
                raise SearchError("invalid_arguments", "find_files 的 path 必须是目录；若已启用 read_file，可用它读取文件。")
            matcher = _Glob(pattern, ignore_case)
            deadline = time.monotonic() + MAX_SEARCH_SECONDS
            with closing(self._candidates(target, deadline, include_ignored)) as candidates:
                for candidate in candidates:
                    if matcher.matches(candidate.relative_to(target).as_posix()):
                        relative = candidate.relative_to(self.root).as_posix()
                        if not report.add((relative, 0), relative):
                            break
            return report.text()
        except SearchError as error:
            report.stop(str(error)); error.partial = report.text(); raise
        except OSError as error:
            report.stop(str(error)); failure = SearchError("search_error", str(error)); failure.partial = report.text(); raise failure from error

    def search(self, pattern, path=".", *, glob=None, regex=False, ignore_case=False,
               context_lines=0, limit=DEFAULT_SEARCH_LIMIT, include_ignored=False):
        report = _Report("search_text", path, limit, include_ignored)
        try:
            if not pattern or "\n" in pattern or "\r" in pattern:
                raise SearchError("invalid_arguments", "pattern 必须是非空单行文本；本工具不做跨行匹配。")
            target = self._resolve(path)
            if target.is_file() and glob is not None:
                raise SearchError("invalid_arguments", "path 指定文件时不再接受 glob；目录搜索可用 glob 过滤。")
            matcher = _Glob(glob, ignore_case) if glob is not None else None
            deadline = time.monotonic() + MAX_SEARCH_SECONDS
            if regex:
                # Parse with rg itself even for an empty scope; '-' is closed stdin,
                # not an implicit directory search.
                for _ in self._records(["--json", "--crlf", "--no-encoding", "--", pattern, "-"], self.root, deadline):
                    pass
            with closing(self._candidates(target, deadline, include_ignored)) as candidates:
                batch, length = [], 0
                for candidate in candidates:
                    if matcher is not None and not matcher.matches(candidate.relative_to(target).as_posix()):
                        continue
                    if batch and (length + len(str(candidate)) + 3 > 16000 or len(batch) >= 128):
                        if not self._search_batch(batch, pattern, regex, ignore_case, context_lines, report, deadline):
                            return report.text()
                        batch, length = [], 0
                    batch.append(candidate); length += len(str(candidate)) + 3
                if batch:
                    self._search_batch(batch, pattern, regex, ignore_case, context_lines, report, deadline)
            return report.text()
        except SearchError as error:
            report.stop(str(error)); error.partial = report.text(); raise
        except OSError as error:
            report.stop(str(error)); failure = SearchError("search_error", str(error)); failure.partial = report.text(); raise failure from error

    def _search_batch(self, paths, pattern, regex, ignore_case, context, report, deadline):
        valid = [p for p in paths if self._safe_file(p)]
        if not valid or len(valid) != len(paths):
            raise SearchError("content_changed", "候选文件在内容搜索前已变化，未开始读取；请重新搜索。")
        before = {p: _fingerprint(p.stat()) for p in valid}
        bom_sizes, text_paths = {}, []
        # A binary end event can arrive after many match events. Verify NUL-free
        # input first so early N+1 termination never counts unconfirmed binary hits.
        for p in valid:
            binary = False
            with p.open("rb") as stream:
                prefix = stream.read(3)
                bom_sizes[p] = 3 if prefix == b"\xef\xbb\xbf" else 0
                chunk = prefix
                while chunk:
                    _deadline_check(deadline)
                    if b"\0" in chunk:
                        binary = True
                        break
                    chunk = stream.read(65536)
            if not self._safe_file(p) or _fingerprint(p.stat()) != before[p]:
                raise SearchError("content_changed", "文件在文本预检期间发生变化，请重新搜索。")
            if not binary:
                text_paths.append(p)
        if not text_paths:
            return True
        args = ["--json", "--crlf", "--line-number", "--color=never", "--no-follow", "--no-encoding", "--no-mmap", "--sort", "path"]
        if not regex:
            args.append("--fixed-strings")
        if ignore_case:
            args.append("--ignore-case")
        args.extend(["--", pattern, *(str(p) for p in text_paths)])
        offsets, matched_records = {}, 0
        with closing(self._records(args, self.root, deadline)) as records:
            for raw in records:
                try:
                    event = json.loads(raw)
                    if event["type"] == "match":
                        data = event["data"]
                        path_value = data["path"]
                        if "text" not in path_value:
                            raise SearchError("unsupported_encoding", "匹配路径不是 UTF-8，未伪造路径。")
                        p = Path(path_value["text"])
                        if p not in before or not self._safe_file(p):
                            raise SearchError("policy_denied", "搜索进程返回了候选范围以外的文件。")
                        starts = [item["start"] for item in data["submatches"]]
                        if not starts:
                            # rg emits a valid match with no submatches for some
                            # zero-width EOF matches. Locate the line's content end.
                            value = data["lines"]
                            line = value["text"].encode("utf-8") if "text" in value else base64.b64decode(value["bytes"], validate=True)
                            ending = 2 if line.endswith(b"\r\n") else int(line.endswith((b"\r", b"\n")))
                            starts = [len(line) - ending]
                        offsets.setdefault(p, set()).update(bom_sizes[p] + data["absolute_offset"] + start for start in starts)
                        matched_records += 1
                        if matched_records > report.limit - report.count:
                            break
                    elif event["type"] == "end" and event["data"].get("binary_offset") is not None:
                        name = event["data"]["path"].get("text")
                        if name:
                            offsets.pop(Path(name), None)
                except (ValueError, KeyError, TypeError) as error:
                    if isinstance(error, SearchError):
                        raise
                    raise SearchError("invalid_search_output", "rg 返回了无法校验的定位结果。") from error
        for p in valid:
            if not self._safe_file(p) or _fingerprint(p.stat()) != before[p]:
                raise SearchError("content_changed", "文件在搜索期间发生变化，请重新搜索。")
        for p in sorted(offsets):
            if not self._safe_file(p) or _fingerprint(p.stat()) != before[p]:
                raise SearchError("content_changed", "文件在搜索期间发生变化，请重新搜索。")
            if not self._render_file(p, sorted(offsets[p]), context, report, deadline):
                return False
            if _fingerprint(p.stat()) != before[p]:
                raise SearchError("content_changed", "文件在定位回读期间发生变化，请重新搜索。")
        return report.complete

    def _render_file(self, path, offsets, context, report, deadline):
        relative = path.relative_to(self.root).as_posix()
        prior, trailing, index = deque(maxlen=context), 0, 0
        def add(row, is_match):
            number, raw, oversized = row
            if oversized:
                return report.stop("匹配行或上下文行超过结果容量，未返回伪装成完整行的片段；请缩小查询。")
            try:
                text = raw.decode("utf-8-sig" if number == 1 else "utf-8")
            except UnicodeError as error:
                raise SearchError("unsupported_encoding", "匹配行或上下文不是 UTF-8，未自动转码。") from error
            formatted = f"{relative}:{number}: {text}" if is_match else f"{relative}-{number}- {text}"
            return report.add((relative, number), formatted, match=is_match)
        for start, end, number, raw, oversized in _logical_lines(path, deadline):
            row = (number, raw, oversized)
            hit = index < len(offsets) and start <= offsets[index] <= end
            if hit:
                while index < len(offsets) and offsets[index] <= end:
                    index += 1
                for previous in prior:
                    if not add(previous, False):
                        return False
                if not add(row, True):
                    return False
                trailing = context
            elif trailing:
                if not add(row, False):
                    return False
                trailing -= 1
            prior.append(row)
            if index == len(offsets) and trailing == 0:
                break
        return True


def _logical_lines(path, deadline):
    """Bounded UTF-8 byte lines, with offsets/CRLF/LF/CR numbering matching read_file."""
    start = base = 0
    number = 1
    line = bytearray()
    line_length = 0
    held = b""
    with path.open("rb") as stream:
        while True:
            _deadline_check(deadline)
            chunk = stream.read(65536)
            data, held = held + chunk, b""
            if chunk and data.endswith(b"\r"):
                data, held = data[:-1], b"\r"
            cursor = 0
            for ending in re.finditer(rb"\r\n|\r|\n", data):
                segment = data[cursor:ending.start()]
                line.extend(segment[:max(0, MAX_RESULT_BYTES + 1 - len(line))])
                line_length += len(segment)
                yield start, base + ending.start(), number, bytes(line), line_length > MAX_RESULT_BYTES
                start = base + ending.end()
                number += 1
                line.clear(); line_length = 0
                cursor = ending.end()
            segment = data[cursor:]
            line.extend(segment[:max(0, MAX_RESULT_BYTES + 1 - len(line))])
            line_length += len(segment)
            base += len(data)
            if not chunk:
                if line_length:
                    yield start, base, number, bytes(line), line_length > MAX_RESULT_BYTES
                break
