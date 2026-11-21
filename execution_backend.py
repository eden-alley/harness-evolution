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
    def __init__(self, read_only_paths=()):
        self.read_only_paths = tuple(read_only_paths)

    @property
    def binding(self) -> dict:
        identity = {'name': 'codex_windows_mxc', 'policy_version': 1}
        if self.read_only_paths:
            identity['read_only_paths'] = [str(path) for path in self.read_only_paths]
        return identity

    def run(self, command, workspace, timeout, *, shell='cmd') -> subprocess.CompletedProcess:
        if self.read_only_paths:
            return sandbox.run_in_sandbox(command, workspace, timeout, shell=shell,
                                          read_only_paths=self.read_only_paths)
        if shell == 'cmd':
            return sandbox.run_in_sandbox(command, workspace, timeout)
        return sandbox.run_in_sandbox(command, workspace, timeout, shell=shell)
