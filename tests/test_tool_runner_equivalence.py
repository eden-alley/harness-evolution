"""Portable v0.38 differential oracle: separate imports, processes and effects.

The ZIP is immutable baseline source, never a wrapper around candidate Agent.
Only temporary absolute paths and checkpoint source fingerprints are normalized;
UUIDs are a fixed per-process sequence. Text, hashes of evidence, order, usage,
permission decisions and every checkpoint attempt remain comparison inputs.
Set TOOL_RUNNER_EQ_EVIDENCE to retain readable observations outside test tempdirs.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name('fixtures')
BASELINE_HASH = 'a168cb97b0ffd047cc96e671b8303af238514a0c944b17b23e619962a405d901'
CHILD = Path(__file__).with_name('tool_runner_equivalence_child.py')

def source_manifest(directory):
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob('*.py'))}
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=False, allow_nan=False,
                           separators=(',', ':')).encode('utf-8')
    return files, hashlib.sha256(canonical).hexdigest()

def normalize(value, source, directory):
    if isinstance(value, list):
        return [normalize(item, source, directory) for item in value]
    if isinstance(value, dict):
        result = {key: normalize(item, source, directory) for key, item in value.items()}
        binding = result.get('binding')
        if isinstance(binding, dict) and 'source' in binding:
            binding['source'] = '<SOURCE_FINGERPRINT>'
        return result
    if isinstance(value, str):
        for path, marker in ((source, '<SOURCE>'), (directory, '<SCENARIO>')):
            for spelling in (str(path), str(path).lower(), path.as_posix(), path.as_posix().lower()):
                value = value.replace(spelling, marker)
        return value
    return value

class ToolRunnerEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='tool-runner-equivalence-')
        cls.directory = Path(cls.temporary.name).resolve()
        cls.baseline = cls.directory / 'v038'
        cls.candidate = cls.directory / 'candidate'
        cls.baseline.mkdir()
        cls.candidate.mkdir()
        with zipfile.ZipFile(FIXTURES / 'tool_runner_v038.zip') as archive:
            names = archive.namelist()
            if len(names) != 21 or any(Path(name).name != name or not name.endswith('.py') for name in names):
                raise AssertionError('baseline must contain exactly 21 flat Python source files')
            archive.extractall(cls.baseline)
        manifest = json.loads((FIXTURES / 'tool_runner_v038_manifest.json').read_text(encoding='utf-8'))
        files, digest = source_manifest(cls.baseline)
        if digest != BASELINE_HASH or digest != manifest['source_sha256'] or files != manifest['files']:
            raise AssertionError('v0.38 baseline root-source fingerprint changed')
        for path in ROOT.glob('*.py'):
            shutil.copyfile(path, cls.candidate / path.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def verify_imports(self, source, directory, stage, *, candidate):
        provenance = json.loads((directory / f'{stage}-imports.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['stage'], stage)
        self.assertEqual(Path(provenance['source_root']), source)
        self.assertEqual(Path(provenance['scenario_root']), directory)
        files = sorted(path.name for path in source.glob('*.py'))
        self.assertEqual(provenance['root_python_files'], files)
        self.assertEqual(len(files), 22 if candidate else 21)
        for label, module in provenance['modules'].items():
            filename = 'checkpoint.py' if label == 'checkpoint' else f'{label}.py'
            self.assertEqual(Path(module['file']), source / filename)
        runner = provenance['runner_module']
        instance = provenance['runner_instance']
        if candidate:
            self.assertEqual(Path(runner['file']), source / 'tool_runner.py')
            self.assertEqual(runner['name'], provenance['modules']['models']['name'] + '.tool_runner')
            self.assertEqual(instance, {'class_name': 'ToolRunner', 'class_module': runner['name']})
        else:
            self.assertIsNone(runner)
            self.assertIsNone(instance)
        return provenance

    def compare(self, scenario, *, resume=False, crash=False):
        observations = []
        for label, source in (('baseline', self.baseline), ('candidate', self.candidate)):
            directory = self.directory / scenario / label
            directory.mkdir(parents=True)
            stages = ('start', 'resume') if resume else ('start',)
            executions = []
            for stage in stages:
                process = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(CHILD),
                    str(source), str(directory), scenario, stage], cwd=directory,
                    capture_output=True, text=True, encoding='utf-8', timeout=30)
                expected = 47 if crash and stage == 'start' else 0
                self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
                provenance = self.verify_imports(source, directory, stage, candidate=label == 'candidate')
                evidence = os.environ.get('TOOL_RUNNER_EQ_EVIDENCE')
                if evidence:
                    destination = Path(evidence)
                    destination.mkdir(parents=True, exist_ok=True)
                    (destination / f'{scenario}-{label}-{stage}-imports.json').write_text(
                        json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                executions.append({'stage': stage, 'returncode': process.returncode,
                                   'stdout': process.stdout, 'stderr': process.stderr})
            rows = [json.loads(line) for line in (directory / 'observations.jsonl').read_text(encoding='utf-8').splitlines()]
            result = normalize({'executions': executions, 'observations': rows}, source, directory)
            evidence = os.environ.get('TOOL_RUNNER_EQ_EVIDENCE')
            if evidence:
                destination = Path(evidence)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f'{scenario}-{label}.json').write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            observations.append(result)
        self.assertEqual(observations[0], observations[1], f'v0.38 behavior changed: {scenario}')
        return observations[0]['observations']

    def terminal(self, rows):
        return [row for row in rows if row['kind'] == 'terminal'][-1]

    def test_model_batch_permissions_errors_and_native_mcp_rendering(self):
        rows = self.compare('mixed')
        final = self.terminal(rows)
        self.assertEqual(final['result']['stop_reason'], 'end_turn')
        self.assertEqual(final['effects'], {'a.txt': 'A'})
        self.assertEqual(final['effect_count'], 1)
        self.assertEqual(final['usage']['total_tokens'], 24)
        results = {item['call_id']: item for item in final['state']['tool_results']}
        self.assertEqual(results['denied']['code'], 'user_denied')
        self.assertEqual(results['missing']['code'], 'tool_unavailable')
        self.assertEqual(results['json']['code'], 'invalid_arguments')
        self.assertEqual(results['array']['code'], 'invalid_arguments')
        self.assertEqual(results['mutate']['code'], 'execution_error')
        self.assertEqual(results['policy-error']['code'], 'invalid_arguments')
        self.assertEqual(results['native']['exit_code'], 7)
        replies = {m['tool_call_id']: json.loads(m['content']) for m in final['messages'] if m['role'] == 'tool'}
        self.assertLess(len(replies['native']['data']), len(replies['remote']['data']))
        self.assertEqual(len([row for row in rows if row['kind'] == 'permission']), 7)

    def test_verification_tools_keep_source_and_do_not_add_model_tool_messages(self):
        rows = self.compare('verification')
        final = self.terminal(rows)
        self.assertEqual(final['result']['verification']['status'], 'pass')
        verified = [r for r in final['state']['tool_results'] if r['source'] == 'verification']
        self.assertEqual(len(verified), 1)
        self.assertEqual([m['tool_call_id'] for m in final['messages'] if m['role'] == 'tool'], ['a'])

    def test_executing_checkpoint_failure_prevents_effects(self):
        rows = self.compare('save_executing')
        final = self.terminal(rows)
        self.assertEqual(final['result']['stop_reason'], 'error')
        self.assertEqual(final['effects'], {})
        self.assertEqual(final['effect_count'], 0)
        self.assertFalse(any(row['kind'] == 'execute' for row in rows))

    def test_recorded_checkpoint_failure_keeps_effect_and_blocks_independent_resume(self):
        rows = self.compare('save_recorded', resume=True)
        terminals = [row for row in rows if row['kind'] == 'terminal']
        self.assertEqual(terminals[0]['result']['stop_reason'], 'error')
        self.assertEqual(terminals[-1]['result']['stop_reason'], 'error')
        self.assertIn('禁止重放', terminals[-1]['result']['text'])
        self.assertEqual(terminals[-1]['effect_count'], 1)
        self.assertEqual(len([row for row in rows if row['kind'] == 'request']), 1)

    def test_hard_process_exit_after_real_effect_blocks_resume(self):
        rows = self.compare('hard_exit', resume=True, crash=True)
        final = self.terminal(rows)
        self.assertEqual(final['effects'], {'a.txt': 'A'})
        self.assertEqual(final['effect_count'], 1)
        self.assertEqual(final['result']['stop_reason'], 'error')
        self.assertIn('禁止重放', final['result']['text'])
        self.assertEqual(len([row for row in rows if row['kind'] == 'request']), 1)
        self.assertEqual(next(iter(final['checkpoint']['run_state']['call_records'].values()))['status'], 'executing')

    def test_event_callback_exception_preserves_existing_error_boundary(self):
        rows = self.compare('event_exception')
        final = self.terminal(rows)
        self.assertEqual(final['result']['stop_reason'], 'error')
        self.assertIn('RuntimeError: fixture event callback failure', final['result']['text'])
        self.assertEqual(final['effect_count'], 1)
        result_event = next(row for row in rows if row['kind'] == 'event' and row['event']['type'] == 'tool_result')
        self.assertEqual(result_event['observed']['pending_calls'], {})
        self.assertEqual(next(iter(result_event['observed']['call_records'].values()))['status'], 'recorded')

    def test_pending_resume_reuses_operations_and_skips_recorded_write(self):
        rows = self.compare('pending_resume', resume=True)
        terminals = [row for row in rows if row['kind'] == 'terminal']
        before, after = terminals[0], terminals[-1]
        self.assertEqual(after['result']['stop_reason'], 'end_turn')
        self.assertEqual(after['effects'], {'a.txt': 'external modification; never replay', 'b.txt': 'B'})
        self.assertEqual(after['effect_count'], 2)
        self.assertEqual(list(before['state']['call_records']), list(after['state']['call_records']))
        self.assertEqual(len(after['state']['call_records']), 2)
        self.assertEqual(len([row for row in rows if row['kind'] == 'permission']), 2)
        self.assertEqual(len([row for row in rows if row['kind'] == 'request']), 2)

    def test_progress_evidence_filter_and_full_result_recall(self):
        rows = self.compare('evidence_filter')
        final = self.terminal(rows)
        results = {r['call_id']: r for r in final['state']['tool_results']}
        self.assertFalse(results['progress-verified']['ok'])
        self.assertFalse(results['progress-orphan']['ok'])
        self.assertTrue(results['progress-recorded']['ok'])
        self.assertTrue(results['recall']['ok'])
        selected = next(row for row in rows if row['kind'] == 'execute' and row['name'] == 'update_progress')
        self.assertEqual([r['call_id'] for r in selected['recorded_results']], ['recorded'])

    def test_v038_checkpoint_is_rejected_by_candidate_without_migration_or_replay(self):
        # This is intentionally a migration-protection test, not an equality
        # comparison: both versions must run against the exact same old files.
        directory = self.directory / 'migration-protection'
        directory.mkdir()
        executions, provenance = [], []
        for source, stage, candidate in ((self.baseline, 'start', False), (self.candidate, 'resume', True)):
            process = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(CHILD),
                str(source), str(directory), 'pending_resume', stage], cwd=directory,
                capture_output=True, text=True, encoding='utf-8', timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            provenance.append(self.verify_imports(source, directory, stage, candidate=candidate))
            executions.append({'version': 'candidate' if candidate else 'v0.38', 'stage': stage,
                'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
            rows = [json.loads(line) for line in (directory / 'observations.jsonl').read_text(encoding='utf-8').splitlines()]
            if stage == 'start':
                old_rows = json.loads(json.dumps(rows))
                checkpoint_path = directory / 'checkpoints' / 'default.json'
                checkpoint_bytes = checkpoint_path.read_bytes()
                old_terminal = self.terminal(rows)
                self.assertIsNone(old_terminal['result'])
                self.assertEqual([r['status'] for r in old_terminal['checkpoint']['run_state']['call_records'].values()],
                                 ['recorded', 'pending'])
                old_files = {p.relative_to(directory / 'workspace').as_posix(): p.read_bytes()
                    for p in (directory / 'workspace').rglob('*') if p.is_file()}
                old_effect_count = (directory / 'effect-count.txt').read_bytes()
        self.assertEqual(rows[:len(old_rows)], old_rows)
        added = rows[len(old_rows):]
        self.assertEqual([r['kind'] for r in added], ['stdout', 'terminal'])
        final = self.terminal(rows)
        self.assertEqual(final['result']['stop_reason'], 'error')
        self.assertEqual(final['result']['text'], '检查点源码、工作区、工具、模型或额度配置不匹配')
        self.assertEqual(checkpoint_path.read_bytes(), checkpoint_bytes)
        self.assertEqual((directory / 'effect-count.txt').read_bytes(), old_effect_count)
        self.assertEqual({p.relative_to(directory / 'workspace').as_posix(): p.read_bytes()
            for p in (directory / 'workspace').rglob('*') if p.is_file()}, old_files)
        self.assertEqual(final['effects'], {'a.txt': 'external modification; never replay'})
        self.assertEqual(final['effect_count'], 1)
        self.assertFalse((directory / 'workspace' / 'b.txt').exists())
        # Verify that the persisted source binding is exactly v0.38's and differs
        # from the candidate; use checkpoint.py's fingerprint format verbatim.
        fingerprints = []
        for source in (self.baseline, self.candidate):
            files, _ = source_manifest(source)
            fingerprints.append(hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False,
                allow_nan=False).encode('utf-8')).hexdigest())
        self.assertEqual(old_terminal['checkpoint']['binding']['source'], fingerprints[0])
        self.assertNotEqual(fingerprints[0], fingerprints[1])
        evidence = os.environ.get('TOOL_RUNNER_EQ_EVIDENCE')
        if evidence:
            destination = Path(evidence)
            destination.mkdir(parents=True, exist_ok=True)
            artifact = {'test_kind': 'migration_protection', 'equal_behavior_comparison': False,
                'executions': executions, 'observations': rows, 'source_binding_fingerprints': fingerprints,
                'checkpoint_bytes_unchanged': True, 'workspace_bytes_unchanged': True,
                'effect_count_unchanged': True, 'new_model_requests': 0, 'new_tool_executions': 0,
                'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest()}
            (destination / 'migration-protection.json').write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            for item in provenance:
                (destination / f"migration-protection-{item['stage']}-imports.json").write_text(
                    json.dumps(item, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    unittest.main()
