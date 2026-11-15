"""文件与工具结果续读契约；只使用临时文件和离线 SDK。"""
import json
import re

from scenario_support import OfflineScenario, context, permissions, tools, answer, call, tool_reply


FILE_HEAD = re.compile(r'^\[行 (\d+)-(\d+)；(.+?)\]\n')
FILE_NEXT = re.compile(r'\n\[续读：read_file\((\{.*\})\)\]$')
RESULT_HEAD = re.compile(r'^\[字段 (data|message|result)；字符 (\d+)-(\d+)；(结果已读完|需续读)\]\n')
RESULT_NEXT = re.compile(r'\n\[请用 offset=(\d+) 续读\]$')


def file_page(data):
    head = FILE_HEAD.match(data)
    assert head, data[:100]
    hint = FILE_NEXT.search(data) if head[3] != "文件已读完" else None
    if head[3] != "文件已读完":
        assert hint, data
    body = data[head.end():hint.start() if hint else len(data)]
    rows = body.splitlines()
    lines = []
    for index, row in enumerate(rows, int(head[1])):
        prefix = f'{index}: '
        assert row.startswith(prefix), row
        lines.append(row[len(prefix):])
    assert len(lines) == int(head[2]) - int(head[1]) + 1
    return head, lines, json.loads(hint[1]) if hint else None


def result_page(data):
    head = RESULT_HEAD.match(data)
    assert head, data[:100]
    hint = RESULT_NEXT.search(data) if head[4] == "需续读" else None
    if head[4] == "需续读":
        assert hint, data
    body = data[head.end():hint.start() if hint else len(data)]
    assert len(body) == int(head[3]) - int(head[2])
    if hint:
        assert int(hint[1]) == int(head[3])
    return head, body, int(hint[1]) if hint else None


class FileReadContinuationTests(OfflineScenario):
    def test_rebuild_all_lines_universal_newlines_and_bom_with_budgets(self):
        lines = ['\ufeff首行😀'] + [f'{i} 中文😀 ' + 'x' * 35 for i in range(150)] + ['尾行']
        raw = ''.join(line + ('\r\n' if i % 3 == 0 else '\r' if i % 3 == 1 else '\n') for i, line in enumerate(lines[:-1])) + lines[-1]
        (self.workspace / '中文.txt').write_bytes(raw.encode('utf-8'))
        for budget in (None, 64):
            args = {'path': '中文.txt'}
            if budget is not None:
                args['max_output_tokens'] = budget
            rebuilt = []
            while True:
                result = self.tools.execute('read_file', args)
                self.assertTrue(result.ok, result)
                self.assertLessEqual(len(result.data) if budget is None else len(result.data.encode()), 4000 if budget is None else budget * 3)
                rendered = json.loads(context.render_tool_result(result, max_output_tokens=budget)) if budget else json.loads(context.render_tool_result(result))
                self.assertNotIn('truncated', rendered)
                self.assertEqual(rendered['data'], result.data)
                head, rows, next_args = file_page(result.data)
                self.assertEqual(int(head[1]), len(rebuilt) + 1)
                rebuilt.extend(rows)
                if next_args is None:
                    self.assertEqual(head[3], '文件已读完')
                    break
                self.assertEqual(head[3], '输出额度不足，需续读')
                self.assertEqual(next_args['start_line'], len(rebuilt) + 1)
                if budget is not None:
                    self.assertEqual(next_args['max_output_tokens'], budget)
                args = next_args
            self.assertEqual(rebuilt, lines)

    def test_range_empty_and_out_of_bounds(self):
        (self.workspace / 'f').write_text('one\ntwo\nthree', encoding='utf-8')
        first = self.tools.execute('read_file', {'path': 'f', 'start_line': 1, 'end_line': 2})
        head, lines, next_args = file_page(first.data)
        self.assertEqual((head[3], lines), ('指定范围已读完，文件尚有内容', ['one', 'two']))
        self.assertEqual(next_args, {'path': 'f', 'start_line': 3})
        last = self.tools.execute('read_file', next_args)
        self.assertEqual(file_page(last.data)[1:], (['three'], None))
        self.assertEqual(self.tools.execute('read_file', {'path': 'f', 'start_line': 4}).code, 'invalid_arguments')
        self.assertEqual(self.tools.execute('read_file', {'path': 'f', 'start_line': 2, 'end_line': 1}).code, 'invalid_arguments')
        (self.workspace / 'empty').write_bytes(b'')
        self.assertEqual(self.tools.execute('read_file', {'path': 'empty'}).data, '[空文件；文件已读完]')
        self.assertEqual(self.tools.execute('read_file', {'path': 'empty', 'start_line': 2}).code, 'invalid_arguments')

    def test_long_line_fails_with_recovery_and_larger_budget_succeeds(self):
        (self.workspace / 'f').write_text('first\n' + '中' * 500 + '\nlast', encoding='utf-8')
        small = self.tools.execute('read_file', {'path': 'f', 'start_line': 2, 'max_output_tokens': 64})
        self.assertEqual(small.code, 'output_limit_exceeded')
        for text in ('2', 'max_output_tokens', '4096', 'run_shell'):
            self.assertIn(text, small.message)
        self.assertTrue(self.tools.execute('read_file', {'path': 'f', 'start_line': 2, 'max_output_tokens': 1024}).ok)
        (self.workspace / 'f').write_text('中' * 5000, encoding='utf-8')
        self.assertEqual(self.tools.execute('read_file', {'path': 'f', 'max_output_tokens': 4096}).code, 'output_limit_exceeded')

    def test_result_pages_offsets_eof_and_fallback(self):
        original = {'call_id': 'a', 'source': 'model', 'tool': 'run_shell', 'ok': True, 'data': '中😀abc' * 1000, 'message': {'why': 'no'}}
        for budget in (None, 64):
            offset, rebuilt = 0, ''
            while True:
                args = {'call_id': 'a', 'offset': offset}
                if budget is not None:
                    args['max_output_tokens'] = budget
                result = self.tools.execute('read_tool_result', args, recorded_results=[original])
                self.assertTrue(result.ok, result)
                self.assertLessEqual(len(result.data) if budget is None else len(result.data.encode()), 4000 if budget is None else budget * 3)
                head, body, following = result_page(result.data)
                self.assertEqual((head[1], int(head[2])), ('data', offset))
                rebuilt += body
                if following is None:
                    self.assertEqual(head[4], '结果已读完')
                    break
                self.assertEqual(head[4], '需续读')
                self.assertGreater(following, offset)
                offset = following
            self.assertEqual(rebuilt, original['data'])
        eof = self.tools.execute('read_tool_result', {'call_id': 'a', 'offset': len(original['data'])}, recorded_results=[original])
        self.assertEqual(result_page(eof.data)[1], '')
        self.assertEqual(self.tools.execute('read_tool_result', {'call_id': 'a', 'offset': len(original['data']) + 1}, recorded_results=[original]).code, 'invalid_arguments')
        fallback = self.tools.execute('read_tool_result', {'call_id': 'a', 'field': 'message', 'max_output_tokens': 4096}, recorded_results=[original])
        head, body, _ = result_page(fallback.data)
        self.assertEqual(head[1], 'result')
        self.assertEqual(json.loads(body), original)
        self.assertEqual(self.tools.execute('read_tool_result', {'call_id': 'a', 'max_output_tokens': 1}, recorded_results=[original]).code, 'output_limit_exceeded')

    def test_eof_page_uses_full_budget_without_reserving_continuation(self):
        header = '[行 1-1；文件已读完]\n1: '
        for budget in (None, 128):
            limit = 4000 if budget is None else budget * 3
            overhead = len(header) if budget is None else len(header.encode('utf-8'))
            content = 'x' * (limit - overhead)
            (self.workspace / 'exact').write_text(content, encoding='utf-8')
            args = {'path': 'exact'}
            if budget is not None:
                args['max_output_tokens'] = budget
            result = self.tools.execute('read_file', args)
            self.assertTrue(result.ok, result)
            self.assertEqual(file_page(result.data)[1:], ([content], None))

    def test_offline_search_read_continue_edit_and_recall_stays_within_state(self):
        path = self.workspace / 'config.txt'
        path.write_text('first\nTARGET = old\nprivate-tail', encoding='utf-8')
        self.agent.permissions.set_auto_approve(True)
        self.model([
            tool_reply(call('search', 'search_text', {'pattern': 'TARGET'})),
            tool_reply(call('read', 'read_file', {'path': 'config.txt', 'start_line': 1, 'end_line': 1})),
            tool_reply(call('recall', 'read_tool_result', {'call_id': 'read'})),
            tool_reply(call('next', 'read_file', {'path': 'config.txt', 'start_line': 2, 'end_line': 2})),
            tool_reply(call('edit', 'edit_file', {'path': 'config.txt', 'edits': [{'oldText': 'TARGET = old', 'newText': 'TARGET = new'}]})),
            tool_reply(call('check', 'read_file', {'path': 'config.txt', 'start_line': 2, 'end_line': 2})),
            answer('done'),
        ])
        self.run_agent('定位并修改', self.agent.permissions)
        records = {item['call_id']: item for item in self.agent.run_state.tool_results}
        self.assertIn('config.txt:2: TARGET = old', records['search']['data'])
        head, rows, next_args = file_page(records['read']['data'])
        self.assertEqual((rows, next_args), (['first'], {'path': 'config.txt', 'start_line': 2}))
        self.assertEqual(result_page(records['recall']['data'])[1], records['read']['data'])
        self.assertNotIn('private-tail', records['read']['data'])
        self.assertNotIn('private-tail', records['recall']['data'])
        self.assertEqual(file_page(records['next']['data'])[1], ['TARGET = old'])
        self.assertEqual(file_page(records['check']['data'])[1], ['TARGET = new'])
        self.assertEqual(path.read_text(encoding='utf-8'), 'first\nTARGET = new\nprivate-tail')
        wire = next(m for m in self.requests[2]['messages'] if m.get('tool_call_id') == 'read')
        self.assertEqual(json.loads(wire['content'])['data'], records['read']['data'])

    def test_truncated_range_keeps_original_end_and_budget(self):
        (self.workspace / 'range').write_text(''.join(f'{i:03d} ' + 'x' * 40 + '\n' for i in range(80)), encoding='utf-8')
        args = {'path': 'range', 'start_line': 2, 'end_line': 20, 'max_output_tokens': 128}
        rows = []
        while True:
            result = self.tools.execute('read_file', args)
            self.assertTrue(result.ok, result)
            head, page, next_args = file_page(result.data)
            rows.extend(page)
            if head[3] == '指定范围已读完，文件尚有内容':
                self.assertEqual(next_args, {'path': 'range', 'start_line': 21, 'max_output_tokens': 128})
                break
            self.assertEqual(head[3], '输出额度不足，需续读')
            self.assertEqual(next_args['end_line'], 20)
            self.assertEqual(next_args['max_output_tokens'], 128)
            args = next_args
        self.assertEqual(rows, [f'{i:03d} ' + 'x' * 40 for i in range(1, 20)])
    def test_only_needed_lines_are_consumed(self):
        from pathlib import Path
        from unittest.mock import patch
        path = self.workspace / 'huge'
        path.write_text(''.join(f'line {i:05d}\n' for i in range(10000)), encoding='utf-8')
        original_open = Path.open
        consumed = []

        class LimitedStream:
            def __init__(self, wrapped):
                self.wrapped = wrapped
            def __enter__(self):
                self.wrapped.__enter__()
                return self
            def __exit__(self, *args):
                return self.wrapped.__exit__(*args)
            def __iter__(self):
                return self
            def __next__(self):
                line = next(self.wrapped)
                consumed.append(len(line))
                if sum(consumed) > 1024:
                    raise AssertionError('读取超出所需范围')
                return line
            def readline(self, *args):
                line = self.wrapped.readline(*args)
                consumed.append(len(line))
                if sum(consumed) > 1024:
                    raise AssertionError('读取超出所需范围')
                return line
            def read(self, size=-1):
                if size < 0:
                    raise AssertionError('不应整文件读取')
                data = self.wrapped.read(size)
                consumed.append(len(data))
                if sum(consumed) > 1024:
                    raise AssertionError('读取超出所需范围')
                return data
            def __getattr__(self, name):
                return getattr(self.wrapped, name)

        def limited_open(current, *args, **kwargs):
            stream = original_open(current, *args, **kwargs)
            return LimitedStream(stream) if current == path else stream

        with patch.object(Path, 'open', limited_open):
            result = self.tools.execute('read_file', {'path': 'huge', 'end_line': 2})
        self.assertTrue(result.ok, result)
        self.assertTrue(consumed, '未观察到文件流，需检查测试探针是否仍适用')
        self.assertEqual(file_page(result.data)[1], ['line 00000', 'line 00001'])
        self.assertLessEqual(sum(consumed), 1024)

    def test_two_line_eof_page_does_not_reserve_a_fictitious_continuation(self):
        first = 'x' * 300
        (self.workspace / 'two').write_text(first + '\nz', encoding='utf-8')
        expected = '[行 1-2；文件已读完]\n1: ' + first + '\n2: z'
        result = tools.read_file(self.tools, 'two', _output_chars=len(expected))
        self.assertTrue(result.ok, result)
        self.assertEqual(result.data, expected)
        self.assertEqual(file_page(result.data)[1:], ([first, 'z'], None))
    def test_result_body_with_fake_continuation_and_trailing_newline(self):
        for body in ('abc\n[请用 offset=3 续读]', '中文😀\n' * 120 + '[请用 offset=3 续读]\n'):
            original = {'call_id': 'fake', 'source': 'model', 'tool': 'run_shell', 'ok': True, 'data': body}
            for budget in (None, 128):
                offset, rebuilt = 0, ''
                while True:
                    args = {'call_id': 'fake', 'offset': offset}
                    if budget is not None:
                        args['max_output_tokens'] = budget
                    result = self.tools.execute('read_tool_result', args, recorded_results=[original])
                    self.assertTrue(result.ok, result)
                    head, piece, following = result_page(result.data)
                    self.assertEqual(int(head[2]), offset)
                    rebuilt += piece
                    if following is None:
                        self.assertEqual(head[4], '结果已读完')
                        break
                    self.assertEqual(head[4], '需续读')
                    self.assertGreater(following, offset)
                    offset = following
                self.assertEqual(rebuilt, body)
