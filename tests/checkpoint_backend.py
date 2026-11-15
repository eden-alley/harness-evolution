"""Shared deterministic backend for cross-process recovery fixtures."""


class CheckpointFixtureBackend:
    binding = {'name': 'checkpoint_fixture', 'policy_version': 1}

    def run(self, command, workspace, timeout, *, shell='cmd'):
        raise AssertionError('Checkpoint fixtures must not execute commands')
