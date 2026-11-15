import json
import subprocess
import unittest
from unittest.mock import patch

from execution_backend import CodexWindowsSandboxBackend, ExecutionBackend


class ExecutionBackendTests(unittest.TestCase):
    def test_default_shell_delegates_with_three_arguments(self):
        result = subprocess.CompletedProcess('dir', 0, 'output', '')
        with patch('sandbox.run_in_sandbox', return_value=result) as run:
            actual = CodexWindowsSandboxBackend().run('dir', 'workspace', 10)
        self.assertIs(actual, result)
        run.assert_called_once_with('dir', 'workspace', 10)

    def test_bash_shell_is_forwarded(self):
        result = subprocess.CompletedProcess('pwd', 0, 'output', '')
        with patch('sandbox.run_in_sandbox', return_value=result) as run:
            actual = CodexWindowsSandboxBackend().run('pwd', 'workspace', 10, shell='bash')
        self.assertIs(actual, result)
        run.assert_called_once_with('pwd', 'workspace', 10, shell='bash')

    def test_binding_is_serializable_and_contains_only_backend_identity(self):
        binding = CodexWindowsSandboxBackend().binding
        self.assertEqual(json.loads(json.dumps(binding)), {
            'name': 'codex_windows_mxc', 'policy_version': 1,
        })

    def test_execution_failure_propagates_without_host_fallback(self):
        with patch('sandbox.run_in_sandbox', side_effect=RuntimeError('sandbox failed')):
            with patch('subprocess.run') as host_run:
                with self.assertRaisesRegex(RuntimeError, 'sandbox failed'):
                    CodexWindowsSandboxBackend().run('dir', 'workspace', 10)
        host_run.assert_not_called()

    def test_alternative_backend_can_be_injected_against_protocol(self):
        class AlternativeBackend:
            @property
            def binding(self):
                return {'name': 'alternative', 'policy_version': 1}

            def run(self, command, workspace, timeout, *, shell='cmd'):
                return subprocess.CompletedProcess(command, 0, workspace, shell)

        def execute(backend: ExecutionBackend):
            return backend.run('example', 'workspace', 3, shell='bash')

        backend = AlternativeBackend()
        self.assertIsInstance(backend, ExecutionBackend)
        result = execute(backend)
        self.assertEqual((result.args, result.stdout, result.stderr),
                         ('example', 'workspace', 'bash'))


if __name__ == '__main__':
    unittest.main()
