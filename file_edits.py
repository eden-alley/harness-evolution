"""Single-file exact edits and shared whole-file publication. Newline mapping is adapted from Pi (MIT).

Source: earendil-works/pi, 827383a15262905f82e418483825a49dbb1adbf5,
packages/coding-agent/src/core/tools/edit-diff.ts. See docs/third-party/pi-MIT.txt.
Strict matching, Python I/O and permission checks are local adaptations.
"""
from dataclasses import dataclass
import difflib
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile

MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_EDITS = 20


class FileEditError(ValueError):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class _Replacement:
    index: int
    start: int
    end: int
    text: str


def _normalize(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _match_edits(original, edits):
    if not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDITS:
        raise FileEditError("invalid_arguments", f"edits 必须包含 1–{MAX_EDITS} 处替换。")
    matched = []
    for index, edit in enumerate(edits, 1):
        if (not isinstance(edit, dict) or set(edit) != {"oldText", "newText"}
                or not all(isinstance(value, str) for value in edit.values()) or not edit["oldText"]):
            raise FileEditError("invalid_arguments", f"第 {index} 处需要非空 oldText 和字符串 newText。")
        old, new = _normalize(edit["oldText"]), _normalize(edit["newText"])
        try:
            new.encode("utf-8")
        except UnicodeError as error:
            raise FileEditError("invalid_arguments", f"第 {index} 处新内容不是合法 UTF-8 文本。") from error
        if "\x00" in new:
            raise FileEditError("invalid_arguments", f"第 {index} 处新内容包含 NUL，不是支持的文本。")
        start = original.find(old)
        if start == -1:
            raise FileEditError("edit_not_found", f"第 {index} 处原文未找到；请重读文件并提供准确原文，不会模糊替换。", edit_index=index)
        # Search from start+1 so overlapping occurrences such as aaa/aa are ambiguous too.
        if original.find(old, start + 1) != -1:
            raise FileEditError("ambiguous_edit", f"第 {index} 处原文至少匹配 2 次；请增加前后文。", edit_index=index, minimum_matches=2)
        matched.append(_Replacement(index, start, start + len(old), new))
    matched.sort(key=lambda item: item.start)
    for previous, current in zip(matched, matched[1:]):
        if previous.end > current.start:
            raise FileEditError("overlapping_edits", f"第 {previous.index} 和第 {current.index} 处替换范围重叠；请合并为一处。")
    return [item for item in matched if original[item.start:item.end] != item.text]


def _line_ending_at(text, offset):
    # Unlike Pi's LF-first rule, lone CR is a line ending even in mixed files,
    # consistently with this tool's CR/LF/CRLF matching and preservation contract.
    following = [position for position in (text.find("\r", offset), text.find("\n", offset))
                 if position != -1]
    if following:
        position = min(following)
    else:
        position = max(text.rfind("\r", 0, offset), text.rfind("\n", 0, offset))
        if position > 0 and text[position] == "\n" and text[position - 1] == "\r":
            position -= 1
    if position == -1:
        return "\n"
    return "\r\n" if text.startswith("\r\n", position) else text[position]


def _splice_original(original, replacements):
    # Pi's normalized->original offset mapping; only CRLF changes string length.
    crlf_before = 0
    next_crlf = original.find("\r\n")
    def to_original(offset):
        nonlocal crlf_before, next_crlf
        while next_crlf != -1 and next_crlf - crlf_before < offset:
            crlf_before += 1
            next_crlf = original.find("\r\n", next_crlf + 2)
        return offset + crlf_before

    parts = []
    def push(part):
        if not part:
            return
        # Keep two logical breaks when a splice puts lone CR immediately before LF.
        if parts and parts[-1].endswith("\r") and part.startswith("\n"):
            parts.append("\n")
        parts.append(part)

    position = 0
    for item in replacements:
        start = to_original(item.start)
        push(original[position:start])
        push(item.text.replace("\n", _line_ending_at(original, start)))
        position = to_original(item.end)
    push(original[position:])
    return "".join(parts)


def _lines(text):
    rows = text.split("\n")
    return [row + "\n" for row in rows[:-1]] + ([rows[-1]] if rows[-1] else [])


def _generate_diff(old, new, replacements, display_path):
    # Compare only merged windows around actual edits, not a whole repetitive file.
    windows = []
    for item in replacements:
        start = old.rfind("\n", 0, item.start) + 1
        for _ in range(3):
            start = old.rfind("\n", 0, max(0, start - 1)) + 1
        end = old.find("\n", item.end)
        end = len(old) if end == -1 else end + 1
        for _ in range(3):
            boundary = old.find("\n", end)
            end = len(old) if boundary == -1 else boundary + 1
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
        else:
            windows.append((start, end))
    output = [f"--- a/{display_path}\n", f"+++ b/{display_path}\n"]
    for start, end in windows:
        delta_before = sum(len(item.text) - (item.end - item.start) for item in replacements if item.end <= start)
        delta_inside = sum(len(item.text) - (item.end - item.start) for item in replacements if start <= item.start < end)
        new_start, new_end = start + delta_before, end + delta_before + delta_inside
        old_offset, new_offset = old.count("\n", 0, start), new.count("\n", 0, new_start)
        diff = difflib.unified_diff(_lines(old[start:end]), _lines(new[new_start:new_end]), n=3)
        for line in list(diff)[2:]:
            if line.startswith("@@"):
                match = re.fullmatch(r"@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@\n", line)
                if match is None:
                    raise ValueError("无法生成修改差异")
                line = (f"@@ -{int(match[1]) + old_offset}{match[2] or ''} "
                        f"+{int(match[3]) + new_offset}{match[4] or ''} @@\n")
            output.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    return "".join(output)


def _prepare(original_bytes, edits, display_path):
    if len(original_bytes) > MAX_FILE_BYTES:
        raise FileEditError("file_size_exceeded", f"原文件超过 {MAX_FILE_BYTES} 字节上限，不执行编辑。")
    try:
        original = original_bytes.decode("utf-8-sig")
    except UnicodeError as error:
        raise FileEditError("unsupported_encoding", "只支持 UTF-8（可带 BOM），不会自动转换其他编码。") from error
    if "\x00" in original:
        raise FileEditError("unsupported_encoding", "文件包含 NUL，不能作为 UTF-8 文本编辑。")
    normalized = _normalize(original)
    replacements = _match_edits(normalized, edits)
    if not replacements:
        raise FileEditError("no_changes", "替换前后内容相同，未写入文件。")
    normalized_parts, position = [], 0
    for item in replacements:
        normalized_parts.extend((normalized[position:item.start], item.text))
        position = item.end
    normalized_parts.append(normalized[position:])
    normalized_new = "".join(normalized_parts)
    new = _splice_original(original, replacements)
    if _normalize(new) != normalized_new:
        raise ValueError("换行映射与编辑结果不一致，未写入文件")
    prefix = b"\xef\xbb\xbf" if original_bytes.startswith(b"\xef\xbb\xbf") else b""
    new_bytes = prefix + new.encode("utf-8")
    if len(new_bytes) > MAX_FILE_BYTES:
        raise FileEditError("file_size_exceeded", f"修改结果超过 {MAX_FILE_BYTES} 字节上限，不执行编辑。")
    result = {"path": display_path, "changed": True, "edits_applied": len(replacements),
              "before_sha256": hashlib.sha256(original_bytes).hexdigest(),
              "after_sha256": hashlib.sha256(new_bytes).hexdigest(),
              "first_changed_line": normalized.count("\n", 0, replacements[0].start) + 1,
              "diff": _generate_diff(normalized, normalized_new, replacements, display_path)}
    return new_bytes, result


def _guard_path(path, root):
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise FileEditError("policy_denied", "目标路径在工作区之外，禁止访问。")
    if resolved != path:
        raise FileEditError("content_changed", "文件路径已发生变化；请重新读取后再编辑。")


def _signature(info):
    # Windows stat/fstat disagree on deprecated ctime; compare identity, mtime, mode and DACL instead.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, (0 if os.name == "nt" else info.st_ctime_ns),
            info.st_mode, info.st_nlink, getattr(info, "st_file_attributes", 0))


def _check_windows_streams(path):
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    # https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-findfirststreamw
    class StreamData(ctypes.Structure):
        _fields_ = [("size", ctypes.c_longlong), ("name", wintypes.WCHAR * 296)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    first, following, close = kernel.FindFirstStreamW, kernel.FindNextStreamW, kernel.FindClose
    first.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
    first.restype = wintypes.HANDLE
    following.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    following.restype = wintypes.BOOL
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    data = StreamData()
    handle = first(str(path), 0, ctypes.byref(data), 0)
    if handle == ctypes.c_void_p(-1).value:
        error = ctypes.get_last_error()
        if error == 87:  # Filesystems without streams have nothing extra to lose.
            return
        raise ctypes.WinError(error)
    try:
        while True:
            if data.name != "::$DATA":
                raise FileEditError("unsupported_file", "文件含 Windows 附加数据流，暂不支持替换，原文件与附加数据均保留。")
            if not following(handle, ctypes.byref(data)):
                error = ctypes.get_last_error()
                if error == 38:  # ERROR_HANDLE_EOF
                    break
                raise ctypes.WinError(error)
    finally:
        close(handle)


def _read_original(path, root):
    _guard_path(path, root)
    try:
        info = path.stat()
    except FileNotFoundError as error:
        raise FileEditError("file_not_found", "目标文件不存在；创建文件请使用 write_file。") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise FileEditError("unsupported_file", "只支持普通单链接文件；目录、设备和多硬链接文件不执行替换。")
    if not info.st_mode & stat.S_IWRITE:
        raise FileEditError("file_not_writable", "原文件不可写，不能通过替换绕过只读限制。")
    if os.name == "nt" and getattr(info, "st_file_attributes", 0) & ~(0x20 | 0x80):
        raise FileEditError("unsupported_file", "文件带有需要专门保留的 Windows 属性，不执行替换。")
    if info.st_size > MAX_FILE_BYTES:
        raise FileEditError("file_size_exceeded", f"原文件超过 {MAX_FILE_BYTES} 字节上限，不执行编辑。")
    _check_windows_streams(path)
    # Opening read/write checks the original file's write ACL without modifying its bytes.
    with path.open("r+b") as stream:
        opened = os.fstat(stream.fileno())
        raw = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if _signature(info) != _signature(opened) or _signature(opened) != _signature(after):
        raise FileEditError("content_changed", "读取期间文件已变化；请重新读取。")
    if len(raw) > MAX_FILE_BYTES:
        raise FileEditError("file_size_exceeded", f"原文件超过 {MAX_FILE_BYTES} 字节上限，不执行编辑。")
    return raw, info


def _security_descriptor(path):
    """Require matching owner/DACL rather than change ownership or broaden permissions."""
    if os.name != "nt":
        return b""
    import ctypes
    from ctypes import wintypes
    get_security = ctypes.WinDLL("advapi32", use_last_error=True).GetFileSecurityW
    get_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p,
                             wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    get_security.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    get_security(str(path), 5, None, 0, ctypes.byref(needed))  # OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
    if not needed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(needed.value)
    if not get_security(str(path), 5, buffer, needed.value, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw[:needed.value]


def _publish_file(path, root, new_bytes, *, original, info, security, create=False,
                  prefix=".agent-edit-"):
    """Publish prepared bytes once; checks do not provide cross-call CAS or a directory transaction."""
    _guard_path(path, root)
    fd, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        try:
            stream_context = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with stream_context as stream:
            if info is not None:
                os.chmod(temporary, stat.S_IMODE(info.st_mode))
            staged_security = _security_descriptor(temporary)
            if not create and staged_security != security:
                raise FileEditError("unsupported_file", "文件所有者或权限与同目录临时文件不同，不能安全替换；本工具暂不支持此文件。")
            if stream.write(new_bytes) != len(new_bytes):
                raise OSError("未完整写入临时文件")
            stream.flush()
            os.fsync(stream.fileno())
        if not create:
            try:
                current, current_info = _read_original(path, root)
            except FileEditError as error:
                if error.code == "policy_denied":
                    raise
                raise FileEditError("content_changed", "提交前目标文件或属性已变化，未覆盖；请重新读取。") from error
            if (current != original or _signature(current_info) != _signature(info)
                    or _security_descriptor(path) != security):
                raise FileEditError("content_changed", "提交前检测到文件被外部修改，未覆盖；请重新读取后合并修改。")
        try:
            staged, _ = _read_original(temporary, root)
        except FileEditError as error:
            raise FileEditError("content_changed", "提交前临时文件已变化，未覆盖原文件。") from error
        if staged != new_bytes or _security_descriptor(temporary) != staged_security:
            raise FileEditError("content_changed", "提交前临时内容或权限已变化，未覆盖原文件。")
        _guard_path(path, root)
        if create:
            # Windows rename rejects a destination that appeared after our initial snapshot.
            try:
                path.lstat()
            except FileNotFoundError:
                pass
            else:
                raise FileEditError("content_changed", "提交前目标已被其他执行者创建，未覆盖；请重新读取。")
            try:
                os.rename(temporary, path)
            except FileExistsError as error:
                raise FileEditError("content_changed", "发布时目标已被其他执行者创建，未覆盖；请重新读取。") from error
        else:
            os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Do not hide the original failure or turn a committed result into a retry.


def apply_file_edits(path, edits, *, workspace_root):
    """Validate all edits, prepare diff, then publish once. No retries or cross-file transaction."""
    path, root = Path(path), Path(workspace_root).resolve()
    original, info = _read_original(path, root)
    new_bytes, result = _prepare(original, edits, path.relative_to(root).as_posix())
    security = _security_descriptor(path)
    _publish_file(path, root, new_bytes, original=original, info=info, security=security)
    return result  # Prepare the receipt before any file is published.


def write_whole_file(path, content, *, workspace_root):
    """Write exact UTF-8 bytes, retaining only invocation-time conflict detection."""
    path, root = Path(path), Path(workspace_root).resolve()
    _guard_path(path, root)
    try:
        new_bytes = content.encode("utf-8")
    except UnicodeError as error:
        raise FileEditError("unsupported_encoding", "完整内容不能编码为 UTF-8，未写入文件。") from error
    if len(new_bytes) > MAX_FILE_BYTES:
        raise FileEditError("file_size_exceeded", f"完整内容超过 {MAX_FILE_BYTES} 字节上限，未写入文件。")
    try:
        original, info = _read_original(path, root)
    except FileEditError as error:
        if error.code != "file_not_found":
            raise
        original, info = None, None
    create = original is None
    if create and os.name != "nt":
        raise FileEditError("unsupported_platform", "安全新建文件暂仅验收 Windows；未启用可能覆盖目标的回退方式。")
    result = {"path": path.relative_to(root).as_posix(),
              "operation": "created" if create else ("unchanged" if original == new_bytes else "replaced"),
              "bytes": len(new_bytes), "sha256": hashlib.sha256(new_bytes).hexdigest()}
    if original == new_bytes:
        return result
    security = None if create else _security_descriptor(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _guard_path(path, root)
    _publish_file(path, root, new_bytes, original=original, info=info, security=security,
                  create=create, prefix=".agent-write-")
    return result
