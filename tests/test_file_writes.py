"""Whole-file write contract, with real temporary files and injected failures."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from permissions import Workspace
from tools import Toolset
import file_edits


class WholeFileWriteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='whole-write-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'nested' / 'sample.txt'
        self.tools = Toolset(Workspace(self.root), run_command=lambda *a: None)

    def write(self, content, path='nested/sample.txt'):
        return self.tools.execute('write_file', {'path': path, 'content': content})

    def assert_receipt(self, result, operation, content):
        self.assertTrue(result.ok, result)
        raw = content.encode('utf-8')
        self.assertEqual(result.data, {'path': 'nested/sample.txt', 'operation': operation,
                                       'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
        self.assertEqual(self.path.read_bytes(), raw)

    def test_created_replaced_unchanged_and_exact_utf8(self):
        first = '\ufeff标题\r\n混合\n尾\r'
        self.assert_receipt(self.write(first), 'created', first)
        replacement = '无BOM\n中文😀\r\n尾\r'
        self.assert_receipt(self.write(replacement), 'replaced', replacement)
        before = self.path.stat()
        self.assert_receipt(self.write(replacement), 'unchanged', replacement)
        after = self.path.stat()
        self.assertEqual((before.st_ino, before.st_mtime_ns), (after.st_ino, after.st_mtime_ns))
        self.assert_receipt(self.write(''), 'replaced', '')
        self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [])

    def test_existing_non_utf8_bytes_can_be_replaced_and_nul_is_exact(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b'\xff\xfe\x00')
        self.assert_receipt(self.write('a\x00b'), 'replaced', 'a\x00b')

    def test_bad_encoding_and_size_do_not_create_parents_or_change_target(self):
        result = self.write('\ud800')
        self.assertEqual(result.code, 'unsupported_encoding')
        self.assertFalse(self.path.parent.exists())
        with patch.object(file_edits, 'MAX_FILE_BYTES', 4):
            result = self.write('中文字')
            self.assertEqual(result.code, 'file_size_exceeded')
            self.assertFalse(self.path.parent.exists())
            self.path.parent.mkdir()
            self.path.write_bytes(b'12345')
            result = self.write('a')
            self.assertEqual(result.code, 'file_size_exceeded')
            self.assertEqual(self.path.read_bytes(), b'12345')

    def test_fsync_and_replace_failures_preserve_original_and_clean_temp(self):
        self.path.parent.mkdir()
        for method in ('fsync', 'replace'):
            self.path.write_bytes(b'original')
            with self.subTest(method=method), patch.object(file_edits.os, method, side_effect=OSError('injected')):
                result = self.write('replacement')
            self.assertFalse(result.ok, result)
            self.assertEqual(self.path.read_bytes(), b'original')
            self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [])

    def test_read_before_external_change_is_not_a_compare_and_swap(self):
        self.path.parent.mkdir()
        self.path.write_bytes(b'old read')
        self.assertTrue(self.tools.execute('read_file', {'path': 'nested/sample.txt'}).ok)
        self.path.write_bytes(b'external before write starts')
        self.assert_receipt(self.write('user supplied'), 'replaced', 'user supplied')

    def test_partial_write_and_flush_failures_clean_only_owned_temp(self):
        import contextlib
        self.path.parent.mkdir()
        other = self.path.parent / '.agent-write-other.tmp'
        other.write_bytes(b'untouched')
        fdopen = file_edits.os.fdopen
        for stage in ('write', 'flush'):
            self.path.write_bytes(b'original')
            @contextlib.contextmanager
            def broken_writer(fd, mode):
                with fdopen(fd, mode) as stream:
                    class Writer:
                        def write(self, value):
                            if stage == 'write':
                                stream.write(value[:2])
                                raise OSError('partial write')
                            return stream.write(value)
                        def flush(self):
                            raise OSError('flush failure')
                        def fileno(self):
                            return stream.fileno()
                    yield Writer()
            with self.subTest(stage=stage), patch.object(file_edits.os, 'fdopen', side_effect=broken_writer):
                result = self.write('replacement')
            self.assertFalse(result.ok)
            self.assertEqual(self.path.read_bytes(), b'original')
            self.assertEqual(other.read_bytes(), b'untouched')
            self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [other])

    def test_target_or_staged_content_change_is_detected(self):
        self.path.parent.mkdir()
        fsync = file_edits.os.fsync
        for change in ('target', 'staged'):
            self.path.write_bytes(b'original')
            def mutate(fd):
                fsync(fd)
                if change == 'target':
                    self.path.write_bytes(b'external')
                else:
                    next(self.path.parent.glob('.agent-write-*.tmp')).write_bytes(b'tampered')
            with self.subTest(change=change), patch.object(file_edits.os, 'fsync', side_effect=mutate):
                result = self.write('replacement')
            self.assertEqual(result.code, 'content_changed')
            self.assertEqual(self.path.read_bytes(), b'external' if change == 'target' else b'original')
            self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [])

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows create-only publish')
    def test_create_race_and_rename_failure_do_not_overwrite(self):
        import os
        self.path.parent.mkdir()
        rename = file_edits.os.rename
        def race(source, destination):
            self.path.write_bytes(b'external')
            return rename(source, destination)
        with patch.object(file_edits.os, 'rename', side_effect=race):
            result = self.write('ours')
        self.assertEqual(result.code, 'content_changed')
        self.assertEqual(self.path.read_bytes(), b'external')
        self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [])
        self.path.unlink()
        with patch.object(file_edits.os, 'rename', side_effect=OSError('injected')):
            result = self.write('ours')
        self.assertFalse(result.ok)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.glob('.agent-write-*')), [])

    def test_readonly_is_not_bypassed(self):
        import stat
        self.path.parent.mkdir()
        self.path.write_bytes(b'original')
        self.path.chmod(stat.S_IREAD)
        try:
            self.assertFalse(self.write('replacement').ok)
            self.assertEqual(self.path.read_bytes(), b'original')
        finally:
            self.path.chmod(stat.S_IREAD | stat.S_IWRITE)

    def test_hardlinks_are_not_bypassed(self):
        import os
        self.path.parent.mkdir()
        self.path.write_bytes(b'original')
        alias = self.path.parent / 'alias.txt'
        os.link(self.path, alias)
        self.assertEqual(self.write('replacement').code, 'unsupported_file')
        self.assertEqual(alias.read_bytes(), b'original')

    @unittest.skipUnless(__import__('os').name == 'nt', 'Windows ACL and alternate streams')
    def test_custom_acl_and_alternate_streams_are_not_bypassed(self):
        import subprocess
        self.path.parent.mkdir()
        self.path.write_bytes(b'original')
        alternate = Path(str(self.path) + ':metadata')
        alternate.write_bytes(b'stream')
        self.assertEqual(self.write('replacement').code, 'unsupported_file')
        self.assertEqual(alternate.read_bytes(), b'stream')
        alternate.unlink()
        changed = subprocess.run(['icacls', str(self.path), '/inheritance:d'], capture_output=True)
        self.assertEqual(changed.returncode, 0, changed.stderr)
        self.assertEqual(self.write('replacement').code, 'unsupported_file')
        self.assertEqual(self.path.read_bytes(), b'original')

from test_file_reads import result_page
from scenario_support import OfflineScenario, permissions, answer, call, tool_reply


class WholeWriteLoopTests(OfflineScenario):
    def test_offline_loop_create_read_replace_recall_with_permission_and_usage(self):
        import json
        asked = []
        self.agent.permissions = permissions.PermissionManager(
            workspace=self.tools.workspace,
            ask=lambda name, args: (asked.append(name) or 'y', ''))
        self.model([
            (tool_reply(call('create', 'write_file', {'path': 'note.txt', 'content': '旧\n'})), 10),
            (tool_reply(call('read', 'read_file', {'path': 'note.txt'})), 20),
            (tool_reply(call('replace', 'write_file', {'path': 'note.txt', 'content': '新\r\n'})), 30),
            (tool_reply(call('recall', 'read_tool_result', {'call_id': 'replace', 'field': 'result', 'max_output_tokens': 4096})), 40),
            (answer('完成'), 50),
        ])
        self.run_agent('创建、读取、替换并回查', self.agent.permissions)
        self.assertEqual((self.workspace / 'note.txt').read_bytes(), '新\r\n'.encode())
        self.assertEqual(asked, ['write_file', 'write_file'])
        create, read, replace, recall = self.agent.run_state.tool_results
        self.assertEqual(create['data']['operation'], 'created')
        self.assertIn('旧', read['data'])
        self.assertEqual(replace['data']['operation'], 'replaced')
        self.assertEqual(json.loads(result_page(recall['data'])[1]), replace)
        self.assertEqual((self.last_result.stop_reason, self.agent.run_state.steps_used), ('end_turn', 5))
        self.assertEqual(len(self.all_requests), 5)
        self.assertEqual(self.agent.token_usage.totals['total_tokens'], 155)

class ParentFailureTests(unittest.TestCase):
    def test_failed_staging_may_leave_created_parent_but_no_owned_temp(self):
        with tempfile.TemporaryDirectory(prefix='write-parent-test-') as directory:
            root = Path(directory)
            toolset = Toolset(Workspace(root), run_command=lambda *args: None)
            with patch.object(file_edits.tempfile, 'mkstemp', side_effect=OSError('injected')):
                result = toolset.execute('write_file', {'path': 'new/inner/file.txt', 'content': 'x'})
            self.assertFalse(result.ok)
            self.assertFalse((root / 'new/inner/file.txt').exists())
            if (root / 'new/inner').exists():
                self.assertEqual(list((root / 'new/inner').iterdir()), [])
