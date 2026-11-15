"""搜索工具：真实 rg 与临时文件，验证定位、过滤及范围，无模型请求。"""
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from permissions import Workspace
from tools import Toolset


class FileSearchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="file-search-test-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "workspace"
        self.root.mkdir()
        self.tools = Toolset(Workspace(self.root), run_command=lambda *args: None)

    def write(self, name, content="needle\n"):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode("utf-8"))
        return p

    def find(self, pattern="*.py", **options):
        return self.tools.execute("find_files", {"pattern": pattern, **options})

    def search(self, pattern="needle", **options):
        return self.tools.execute("search_text", {"pattern": pattern, **options})

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("file_search"), "缺少搜索模块")
        import file_search
        return file_search

    def test_globs_ignore_hidden_and_workspace_relative_paths(self):
        for name in ["src/a.py", "src/sub/b.py", "pkg/src/c.py", "src/Upper.PY", ".hidden.py", "skip.py", "ignored/i.py"]:
            self.write(name)
        self.write(".git/config", "needle")
        self.write(".gitignore", "skip.py\nignored/\n")
        result = self.find()
        self.assertTrue(result.ok, result)
        for name in ["src/a.py", "src/sub/b.py", "pkg/src/c.py", ".hidden.py"]:
            self.assertIn("\n" + name + "\n", result.data)
        for name in ["skip.py", "ignored/i.py", "src/Upper.PY"]:
            self.assertNotIn("\n" + name + "\n", result.data)
        narrow = self.find("*.py", path="src")
        self.assertIn("\nsrc/a.py\n", narrow.data)
        self.assertNotIn("pkg/src/c.py", narrow.data)
        self.assertNotIn("src/sub/b.py", self.find("src/*.py").data)
        recursive = self.find("src/**/*.py")
        self.assertIn("src/a.py", recursive.data)
        self.assertIn("src/sub/b.py", recursive.data)
        self.assertNotIn("pkg/src/c.py", recursive.data)
        self.assertIn("skip.py", self.find(include_ignored=True).data)
        self.assertNotIn(".git/config", self.find("*", include_ignored=True).data)
        self.assertIn("Upper.PY", self.find("*.py", path="src", ignore_case=True).data)
        # Applying a glob must not silently reopen a .gitignore'd file.
        self.assertNotIn("skip.py:", self.search(glob="*.py").data)
        self.assertIn("skip.py:", self.search(glob="*.py", include_ignored=True).data)

    def test_literal_regex_case_and_special_characters_are_data(self):
        self.write("空 格/a.py", "foo.bar\nfooXbar\nFOO.BAR\n--flag\nx & echo nope\n")
        literal = self.search("foo.bar")
        self.assertTrue(literal.ok, literal)
        self.assertIn("空 格/a.py:1: foo.bar", literal.data)
        self.assertNotIn(":2: fooXbar", literal.data)
        self.assertNotIn(":3: FOO.BAR", literal.data)
        self.assertIn(":2: fooXbar", self.search("foo.bar", regex=True).data)
        self.assertIn(":3: FOO.BAR", self.search("foo.bar", ignore_case=True).data)
        self.assertIn(":4: --flag", self.search("--flag").data)
        self.assertIn(":5: x & echo nope", self.search("x & echo nope").data)
        self.assertFalse((self.root / "nope").exists())

    def test_line_numbers_and_context_agree_with_existing_reader(self):
        self.write("src/中文.py", "\ufeffhead\r\nneedle\rthird\nneedle2\nend")
        result = self.search("needle", path="src", context_lines=1)
        self.assertTrue(result.ok, result)
        self.assertIn("src/中文.py:2: needle", result.data)
        self.assertIn("src/中文.py:4: needle2", result.data)
        self.assertIn("src/中文.py-3- third", result.data)
        for number, text in [(2, "needle"), (4, "needle2")]:
            read = self.tools.execute("read_file", {"path": "src/中文.py", "start_line": number, "end_line": number})
            self.assertIn(f"{number}: {text}", read.data)
        self.assertEqual((self.root / "src/中文.py").read_bytes(), "\ufeffhead\r\nneedle\rthird\nneedle2\nend".encode())

    def test_exact_limit_and_proven_extra_result_are_different(self):
        self.write("a.py"); self.write("b.py")
        for result in [self.find(limit=2), self.search(limit=2)]:
            self.assertTrue(result.ok, result)
            self.assertIn("complete=true", result.data)
            self.assertIn("returned=2", result.data)
        self.write("c.py")
        for result in [self.find(limit=2), self.search(limit=2)]:
            self.assertTrue(result.ok, result)
            self.assertIn("complete=false", result.data)
            self.assertIn("returned=2", result.data)
            self.assertNotIn("\nc.py", result.data)
        self.assertEqual(self.find().data, self.find().data)
        self.assertEqual(self.search().data, self.search().data)

    def test_zero_matches_invalid_regex_and_invalid_path_are_distinct(self):
        self.write("a.py", "other\n")
        empty = self.search()
        self.assertTrue(empty.ok, empty)
        self.assertIn("returned=0", empty.data)
        self.assertIn("complete=true", empty.data)
        self.assertEqual(self.search("[", regex=True).code, "invalid_pattern")
        self.assertEqual(self.search(path="missing").code, "path_not_found")
        self.assertEqual(self.find(path="a.py").code, "invalid_arguments")
        self.assertEqual(self.find("{a,b}.py").code, "invalid_pattern")

    def test_workspace_escape_and_junction_and_hardlinks_cannot_expose_external_content(self):
        outside = self.base / "outside"
        outside.mkdir()
        secret = outside / "outside.py"
        secret.write_text("EXTERNAL_CANARY", encoding="utf-8")
        self.assertEqual(self.search("EXTERNAL_CANARY", path="../outside").code, "policy_denied")
        alias = self.root / "alias.py"
        os.link(secret, alias)
        if os.name == "nt":
            junction = self.root / "linked"
            created = subprocess.run([os.environ["COMSPEC"], "/d", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True)
            self.assertEqual(created.returncode, 0, created.stderr)
            self.addCleanup(os.rmdir, junction)
        else:
            (self.root / "linked").symlink_to(outside, target_is_directory=True)
        found = self.find()
        searched = self.search("EXTERNAL_CANARY", include_ignored=True)
        self.assertTrue(found.ok, found)
        self.assertTrue(searched.ok, searched)
        self.assertNotIn("outside.py", found.data)
        self.assertNotIn("alias.py", found.data)
        self.assertNotIn("EXTERNAL_CANARY", searched.data)
        self.assertIn("returned=0", searched.data)
        self.assertEqual(secret.read_text(encoding="utf-8"), "EXTERNAL_CANARY")

    def test_output_budget_is_an_operation_limit_not_a_claim_of_no_matches(self):
        self.module()
        for i in range(80):
            self.write(f"file-{i:03d}.py", "needle " + "中" * 500 + "\n")
        result = self.search(limit=100)
        self.assertTrue(result.ok, result)
        self.assertIn("complete=false", result.data)
        self.assertLessEqual(len(result.data.encode()), 64 * 1024)
        self.assertIn("read_tool_result", result.data)
        self.assertIn("容量", result.data)

    def test_missing_dependency_hides_and_rejects_only_new_tools(self):
        module = self.module()
        with patch.object(module, "discover_rg", return_value=None):
            toolset = Toolset(Workspace(self.root), run_command=lambda *args: None)
        names = {row["function"]["name"] for row in toolset.get_schemas()}
        self.assertNotIn("find_files", names)
        self.assertNotIn("search_text", names)
        self.assertEqual(toolset.execute("find_files", {"pattern": "*"}).code, "tool_unavailable")
        self.assertEqual(toolset.execution_info("find_files"), {"location": "unavailable"})
        self.write("ok.txt", "ok")
        self.assertEqual(toolset.execute("read_file", {"path": "ok.txt"}).data, "[行 1-1；文件已读完]\n1: ok")

    def test_candidate_becoming_unsafe_never_starts_content_search(self):
        module = self.module()
        path = self.write("a.py")
        outside = self.base / "external.py"
        outside.write_bytes(b"OUTSIDE")
        engine = self.tools.file_search
        original_batch, original_records = engine._search_batch, engine._records
        content_calls = []
        def changed(paths, *args):
            path.unlink()
            os.link(outside, path)
            return original_batch(paths, *args)
        def observe(args, *rest, **kwargs):
            if "--json" in args:
                content_calls.append(args)
            return original_records(args, *rest, **kwargs)
        with patch.object(engine, "_search_batch", side_effect=changed), patch.object(engine, "_records", side_effect=observe):
            result = self.search("OUTSIDE")
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "content_changed")
        self.assertEqual(content_calls, [])

    def test_error_reports_are_also_bounded(self):
        module = self.module()
        with patch.object(self.tools.file_search, "_candidates", side_effect=module.SearchError("search_error", "失败" * 40000)):
            result = self.find()
        self.assertFalse(result.ok)
        self.assertIn("complete=false", result.data)
        self.assertLessEqual(len(result.data.encode()), 64 * 1024)

    def test_timeout_terminates_reader_and_worker_without_retry(self):
        import threading
        import time
        import sys
        module = self.module()
        self.write("a.py")
        marker = self.base / "late.txt"
        real_popen = subprocess.Popen
        processes = []
        def slow_worker(argv, **options):
            code = "import time;from pathlib import Path;time.sleep(1);Path(" + repr(str(marker)) + ").write_text('late')"
            process = real_popen([sys.executable, "-B", "-c", code], **options)
            processes.append(process)
            return process
        before = {t.ident for t in threading.enumerate() if t.name.startswith("harness-rg-")}
        with patch.object(module, "MAX_SEARCH_SECONDS", 0.15), patch.object(module.subprocess, "Popen", side_effect=slow_worker):
            result = self.find()
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "search_timeout")
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertFalse(marker.exists())
        self.assertEqual({t.ident for t in threading.enumerate() if t.name.startswith("harness-rg-")}, before)

    def test_oversized_single_match_is_explicitly_incomplete(self):
        self.write("long.py", "needle" + "中" * 30000)
        result = self.search()
        self.assertFalse(result.ok)
        self.assertEqual(result.code, "output_limit_exceeded")
        self.assertIn("complete=false", result.data)
        self.assertLessEqual(len(result.data.encode()), 64 * 1024)

    def test_binary_files_do_not_consume_match_limit_or_appear_as_text(self):
        self.write("b.txt"); self.write("c.txt")
        binary = self.root / "a.bin"
        for prefix in [b"needle\n", b"needle\nneedle\n"]:
            binary.write_bytes(prefix + b"x" * 100000 + b"\0\n")
            result = self.search(limit=1)
            self.assertTrue(result.ok, result)
            self.assertIn("complete=false", result.data)
            self.assertIn("returned=1", result.data)
            self.assertIn("b.txt:1: needle", result.data)
            self.assertNotIn("a.bin:", result.data)

    def test_zero_width_regex_keeps_crlf_and_eof_locations_and_validates_empty_scope(self):
        for content, expected in [("a\r\nb\r\n", [1, 2]), ("a\nb\n", [1, 2]), ("abc", [1])]:
            self.write("anchors.txt", content)
            result = self.search("$", regex=True)
            self.assertTrue(result.ok, result)
            for number in expected:
                self.assertIn(f"anchors.txt:{number}: ", result.data)
            self.assertIn(f"returned={len(expected)}", result.data)
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self.search("[", path="empty", regex=True).code, "invalid_pattern")

    def test_windows_git_directory_exclusion_is_case_insensitive(self):
        if os.name != "nt":
            return
        self.write(".GIT/config", "needle")
        self.write("folder/.Git/objects/data", "needle")
        found = self.find("*", include_ignored=True)
        searched = self.search(include_ignored=True)
        self.assertTrue(found.ok, found)
        self.assertTrue(searched.ok, searched)
        self.assertNotIn(".git/", found.data.lower())
        self.assertIn("returned=0", searched.data)

    def test_explicit_ignored_file_uses_the_same_ignore_rule(self):
        self.write(".gitignore", "skip.py\n")
        self.write("skip.py")
        result = self.search(path="skip.py")
        self.assertTrue(result.ok, result)
        self.assertIn("returned=0", result.data)
        result = self.search(path="skip.py", include_ignored=True)
        self.assertTrue(result.ok, result)
        self.assertIn("skip.py:1: needle", result.data)

    def test_rg_discovery_rejects_workspace_and_relative_path_entries(self):
        module = self.module()
        real = module.discover_rg(self.root)
        self.assertIsNotNone(real, "离线搜索验收需要已安装的 rg")
        (self.root / ("rg.exe" if os.name == "nt" else "rg")).touch()
        with patch.dict(os.environ, {"PATH": os.pathsep.join([str(self.root), ".", str(real.parent)])}):
            self.assertEqual(module.discover_rg(self.root), real)
        with patch.dict(os.environ, {"PATH": os.pathsep.join([str(self.root), "."])}):
            self.assertIsNone(module.discover_rg(self.root))


if __name__ == "__main__":
    unittest.main()
