"""Replaceable interface for command execution backends."""

import subprocess
from typing import Protocol, runtime_checkable

import sandbox


@runtime_checkable
class ExecutionBackend(Protocol):
    @property
    def binding(self) -> dict:
        """Return the serializable identity of this backend and policy."""
        ...

    def run(self, command, workspace, timeout, *, shell='cmd') -> subprocess.CompletedProcess:
        """Return exit status and stdout/stderr as bytes or text; failures propagate."""
        ...


class CodexWindowsSandboxBackend:
    @property
    def binding(self) -> dict:
        return {'name': 'codex_windows_mxc', 'policy_version': 1}

    def run(self, command, workspace, timeout, *, shell='cmd') -> subprocess.CompletedProcess:
        if shell == 'cmd':
            return sandbox.run_in_sandbox(command, workspace, timeout)
        return sandbox.run_in_sandbox(command, workspace, timeout, shell=shell)
