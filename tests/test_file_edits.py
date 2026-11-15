"""精确编辑的真实文件与故障验证；不用模型或命令修改文件。"""
from copy import deepcopy
import hashlib
import contextlib
import re
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from permissions import Workspace
from tools import Toolset


class FileEditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="file-edit-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "sample.txt"
        self.tools = Toolset(Workspace(self.root), run_command=lambda *a: None)

    def edit(self, edits, **options):
        return self.tools.execute("edit_file", {"path": "sample.txt", "edits": edits, **options})

    def edit_module(self):
        self.assertIsNotNone(importlib.util.find_spec("file_edits"), "缺少精确编辑模块")
        import file_edits
        return file_edits

    def test_multiple_edits_preserve_complete_utf8_bytes_and_return_real_diff(self):
        original = b"\xef\xbb\xbf" + "标题\r\nleft=1\r\n中间😀\nright=2\n尾部".encode()
        expected = b"\xef\xbb\xbf" + "标题\r\nleft=10\r\n新增\r\n中间😀\nright=20\n尾部".encode()
        self.path.write_bytes(original)
        edits = [{"oldText": "right=2", "newText": "right=20"},
                 {"oldText": "left=1", "newText": "left=10\n新增"}]
        arguments = deepcopy(edits)
        result = self.edit(edits, max_output_tokens=1)
        self.assertTrue(result.ok, result)
        self.assertEqual(self.path.read_bytes(), expected)
        self.assertEqual(edits, arguments)
        self.assertEqual(result.data["edits_applied"], 2)
        self.assertEqual(result.data["before_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(result.data["after_sha256"], hashlib.sha256(expected).hexdigest())
        self.assertIn("-left=1", result.data["diff"])
        self.assertIn("+新增", result.data["diff"])
        self.assertIn("+right=20", result.data["diff"])
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])
        self.assertEqual(self.tools.execution_info("edit_file"), {"location": "host"})

    def test_pi_newline_cases_preserve_untouched_bytes_and_local_insert_style(self):
        # Pi tools.test.ts CRLF/BOM/mixed-ending/seam cases, adapted to strict Python matching.
        cases = [
            ("one\r\ntwo\r\nthree\r\n", "two\nthree", "TWO\nTHREE", "one\r\nTWO\r\nTHREE\r\n"),
            ("one\ntwo\nthree", "two\r\nthree", "TWO\r\nTHREE", "one\nTWO\nTHREE"),
            ("one\ntwo\r\nthree\r\n", "two", "TWO\nextra", "one\nTWO\r\nextra\r\nthree\r\n"),
            ("one\r\ntwo\nthree", "two", "TWO\nextra", "one\r\nTWO\nextra\nthree"),
            ("one\rtwo\rthree", "two", "TWO\nextra", "one\rTWO\rextra\rthree"),
            ("one\rtwo\nthree", "one", "ONE\nextra", "ONE\rextra\rtwo\nthree"),
            ("one\rtwo\r\nthree", "one", "ONE\nextra", "ONE\rextra\rtwo\r\nthree"),
            ("one\rX\nlast", "X", "", "one\r\n\nlast"),
            ("\ufeffone\r\ntwo", "two", "", "\ufeffone\r\n"),
        ]
        for original, old, new, expected in cases:
            with self.subTest(original=repr(original)):
                self.path.write_bytes(original.encode())
                result = self.edit([{"oldText": old, "newText": new}])
                self.assertTrue(result.ok, result)
                self.assertEqual(self.path.read_bytes(), expected.encode())

    def test_rejects_missing_ambiguous_overlapping_and_fuzzy_matches_without_writing(self):
        cases = [
            ("alpha beta", [{"oldText": "missing", "newText": "x"}], "edit_not_found"),
            ("aaa", [{"oldText": "aa", "newText": "x"}], "ambiguous_edit"),
            ("x\r\nx\n", [{"oldText": "x\n", "newText": "y\n"}], "ambiguous_edit"),
            ("abcdef", [{"oldText": "abc", "newText": "X"}, {"oldText": "bcde", "newText": "Y"}], "overlapping_edits"),
            ("a  b", [{"oldText": "a b", "newText": "x"}], "edit_not_found"),
            ("“quoted”", [{"oldText": '"quoted"', "newText": "x"}], "edit_not_found"),
        ]
        for original, edits, code in cases:
            with self.subTest(code=code, original=original):
                before = original.encode()
                self.path.write_bytes(before)
                result = self.edit(edits)
                self.assertFalse(result.ok)
                self.assertEqual(result.code, code)
                self.assertTrue(result.message)
                self.assertEqual(self.path.read_bytes(), before)

    def test_all_edits_match_original_and_never_partially_apply(self):
        self.path.write_bytes(b"alpha\nbeta\n")
        for edits in [
            [{"oldText": "alpha", "newText": "changed"}, {"oldText": "missing", "newText": "bad"}],
            [{"oldText": "alpha", "newText": "new-needle"}, {"oldText": "new-needle", "newText": "bad"}],
        ]:
            result = self.edit(edits)
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "edit_not_found")
            self.assertIn("2", result.message)
            self.assertEqual(self.path.read_bytes(), b"alpha\nbeta\n")
        # Adjacent replacements are allowed, with their order still based on the original file.
        self.path.write_bytes(b"abcd")
        result = self.edit([{"oldText": "cd", "newText": "Y"}, {"oldText": "ab", "newText": "X"}])
        self.assertTrue(result.ok, result)
        self.assertEqual(self.path.read_bytes(), b"XY")

    def test_no_change_does_not_replace_file_or_touch_modification_time(self):
        self.path.write_bytes(b"same\r\n")
        before = self.path.stat()
        result = self.edit([{"oldText": "same\n", "newText": "same\n"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "no_changes")
        self.assertEqual(self.path.read_bytes(), b"same\r\n")
        self.assertEqual((self.path.stat().st_ino, self.path.stat().st_mtime_ns),
                         (before.st_ino, before.st_mtime_ns))

    def test_rejects_unsupported_encoding_binary_missing_file_and_directory(self):
        for raw in ["中文".encode("gbk"), "text".encode("utf-16"), b"valid\x00binary"]:
            self.path.write_bytes(raw)
            result = self.edit([{"oldText": "text", "newText": "x"}])
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "unsupported_encoding")
            self.assertEqual(self.path.read_bytes(), raw)
        self.path.unlink()
        missing = self.edit([{"oldText": "text", "newText": "x"}])
        self.assertEqual(missing.code, "file_not_found")
        self.assertFalse(self.path.exists())
        self.path.mkdir()
        self.assertEqual(self.edit([{"oldText": "text", "newText": "x"}]).code, "unsupported_file")

    def test_size_limit_checks_both_original_and_result_without_truncating(self):
        module = self.edit_module()
        with patch.object(module, "MAX_FILE_BYTES", 16):
            for original, old, new in [(b"x" * 17, "x", "y"), (b"key", "key", "字" * 6)]:
                self.path.write_bytes(original)
                result = self.edit([{"oldText": old, "newText": new}])
                self.assertFalse(result.ok)
                self.assertEqual(result.code, "file_size_exceeded")
                self.assertEqual(self.path.read_bytes(), original)
            self.path.write_bytes(b"key")
            self.assertTrue(self.edit([{"oldText": "key", "newText": "x" * 16}]).ok)
            self.assertEqual(self.path.read_bytes(), b"x" * 16)

    def test_diff_generation_sync_and_publish_failures_leave_original_intact(self):
        module = self.edit_module()
        for target in ["_generate_diff", "os.fsync", "os.replace"]:
            self.path.write_bytes(b"original\r\n")
            owner, name = (module, target) if "." not in target else (module.os, target.split(".")[1])
            with self.subTest(target=target), patch.object(owner, name, side_effect=OSError("injected failure")):
                result = self.edit([{"oldText": "original", "newText": "changed"}])
            self.assertFalse(result.ok)
            self.assertEqual(self.path.read_bytes(), b"original\r\n")
            self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])

    def test_partial_temporary_write_failure_leaves_original_and_cleans_owned_temp(self):
        module = self.edit_module()
        self.path.write_bytes(b"original")
        fdopen = module.os.fdopen
        @contextlib.contextmanager
        def broken_writer(fd, mode):
            with fdopen(fd, mode) as stream:
                class Writer:
                    def write(self, value):
                        stream.write(value[:2])
                        raise OSError("partial temporary write")
                yield Writer()
        with patch.object(module.os, "fdopen", side_effect=broken_writer):
            result = self.edit([{"oldText": "original", "newText": "changed"}])
        self.assertFalse(result.ok)
        self.assertEqual(self.path.read_bytes(), b"original")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])

    def test_returned_unified_diff_reconstructs_multiple_distant_changes(self):
        rows = [f"line {i:02d}\n" for i in range(60)]
        self.path.write_bytes("".join(rows).encode())
        result = self.edit([{"oldText": "line 02", "newText": "new 02\nextra"},
                            {"oldText": "line 55\n", "newText": ""}])
        self.assertTrue(result.ok, result)
        expected = rows[:]
        expected[2] = "new 02\nextra\n"
        expected[55] = ""
        expected_text = "".join(expected)
        self.assertEqual(self.path.read_bytes(), expected_text.encode())
        # Independently apply the advertised unified patch to the original lines.
        output, cursor = [], 0
        for line in result.data["diff"].splitlines(keepends=True)[2:]:
            if line.startswith("@@"):
                header = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                self.assertIsNotNone(header)
                start = int(header[1]) - 1
                output.extend(rows[cursor:start])
                cursor = start
                self.assertEqual(len(output), int(header[2]) - 1)
            elif line.startswith((" ", "-")):
                self.assertEqual(rows[cursor], line[1:])
                if line.startswith(" "):
                    output.append(line[1:])
                cursor += 1
            elif line.startswith("+"):
                output.append(line[1:])
        output.extend(rows[cursor:])
        self.assertEqual("".join(output), expected_text)

    def test_detects_external_change_before_commit_and_keeps_external_content(self):
        module = self.edit_module()
        self.path.write_bytes(b"before")
        fsync = module.os.fsync
        def external_change(fd):
            fsync(fd)
            self.path.write_bytes(b"external")
        with patch.object(module.os, "fsync", side_effect=external_change):
            result = self.edit([{"oldText": "before", "newText": "ours"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "content_changed")
        self.assertEqual(self.path.read_bytes(), b"external")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])

    def test_detects_modified_staging_file_before_publish(self):
        module = self.edit_module()
        self.path.write_bytes(b"before")
        fsync = module.os.fsync
        def modify_staging(fd):
            fsync(fd)
            temporary = next(self.root.glob(".agent-edit-*.tmp"))
            temporary.write_bytes(b"unexpected")
        with patch.object(module.os, "fsync", side_effect=modify_staging):
            result = self.edit([{"oldText": "before", "newText": "after"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "content_changed")
        self.assertEqual(self.path.read_bytes(), b"before")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])

    def test_readonly_hardlinks_and_custom_windows_acl_are_not_bypassed(self):
        self.path.write_bytes(b"before")
        self.path.chmod(stat.S_IREAD)
        try:
            result = self.edit([{"oldText": "before", "newText": "after"}])
            self.assertFalse(result.ok)
            self.assertEqual(self.path.read_bytes(), b"before")
        finally:
            self.path.chmod(stat.S_IREAD | stat.S_IWRITE)
        alias = self.root / "alias.txt"
        os.link(self.path, alias)
        result = self.edit([{"oldText": "before", "newText": "after"}])
        self.assertEqual(result.code, "unsupported_file")
        self.assertEqual(alias.read_bytes(), b"before")
        alias.unlink()
        if os.name == "nt":
            changed = subprocess.run(["icacls", str(self.path), "/inheritance:d"], capture_output=True)
            self.assertEqual(changed.returncode, 0, changed.stderr)
            result = self.edit([{"oldText": "before", "newText": "after"}])
            self.assertFalse(result.ok)
            self.assertEqual(result.code, "unsupported_file")
            self.assertEqual(self.path.read_bytes(), b"before")

    def test_windows_extra_streams_are_rejected_without_losing_them(self):
        if os.name != "nt":
            return
        self.path.write_bytes(b"before")
        alternate = Path(str(self.path) + ":metadata")
        alternate.write_bytes(b"keep-stream")
        result = self.edit([{"oldText": "before", "newText": "after"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "unsupported_file")
        self.assertEqual(self.path.read_bytes(), b"before")
        self.assertEqual(alternate.read_bytes(), b"keep-stream")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["sample.txt"])

    def test_large_repetitive_file_produces_bounded_local_diff(self):
        original = b"repeated\n" * 150000 + b"target=old\n" + b"repeated\n" * 150000
        self.path.write_bytes(original)
        result = self.edit([{"oldText": "target=old", "newText": "target=new"}])
        self.assertTrue(result.ok, result)
        self.assertEqual(self.path.read_bytes(), original.replace(b"target=old", b"target=new"))
        self.assertLess(len(result.data["diff"]), 1000)
        self.assertIn("150001", result.data["diff"] + str(result.data["first_changed_line"]))


if __name__ == "__main__":
    unittest.main()
