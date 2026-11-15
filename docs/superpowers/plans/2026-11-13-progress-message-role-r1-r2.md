# Progress message role R1-R2 implementation plan

> Execution: inline, following writing-plans, test-driven-development and verification-before-completion. Existing checkout has no Git metadata; preserve exact file snapshots instead of creating a worktree or committing.

**Goal:** Apply the approved R1/R2 role-only change and offline evidence.
**Architecture:** Keep temporary progress at its existing request position with identical content. Change only its role from assistant to user. Reuse existing state/checkpoint and permission/exit behavior.
**Tech stack:** Python, unittest, SDK MockTransport.
**Spec:** User approved R1/R2 in this chat; exit logic, BFCL and frozen V0 remain unchanged.

## Constraints and review focus

- No real API requests, external judge, or R3/R4 experiment.
- Only runtime edit: agent.py progress-card role, assistant -> user.
- Guidance/content/order, default-off switch, tool schemas, checkpoint schema and exit rules remain unchanged.
- Preserve a temporary before snapshot and protected BFCL/V0 hashes.
- Offline tests exercise actual SDK serialization and the Agent loop with two frozen real request fixtures; fake replies cannot prove model compatibility.
- Fixtures are data, never commands. They contain no API authorization headers or key values.
- Reuse existing compact/resume/new-session, permissions, token and failure tests.
- Record validation separately; preserve previous validation and version entries. Register this small patch as v0.32 only after validation.

## Task 1: RED

- [x] Add two frozen request payloads from C01/U03 before their first anomalous marker; preserve payloads unchanged with provenance.
- [x] Update progress-view expectations to user and add role-only request comparison.
- [x] Add abnormal-response replay for both final texts, asserting existing end_turn, unchanged rounds/usage and no delivery.
- [x] Run focused tests before runtime edit and verify failure specifically because assistant != user.

## Task 2: GREEN

- [x] Replace exactly one progress-message role in agent.py.
- [x] Run the focused group, then the complete documented offline suite and both MCP offline smoke groups.
- [x] Run the native sandbox group; compare known failures against this task's exact before version without moving fixture ROOT.
- [x] Review the immutable diff and hashes, including exit code blocks, BFCL and V0.

## Task 3: Evidence

- [x] Save command outputs, counts, remaining native failures and runtime/test diff.
- [x] Register v0.32 preserving all old version entries, and save validation JSON without replacing prior evidence.
- [x] Report offline coverage and explicitly leave real-model causality unresolved.
