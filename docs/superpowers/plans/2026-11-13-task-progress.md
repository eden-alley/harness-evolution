# Task progress implementation plan

> Execution: inline with superpowers:executing-plans and TDD; independent final review.

**Goal:** Implement the approved R1–R4 option using existing state and checkpoint only.
**Architecture:** update_progress returns a validated document stored in normal RunState.tool_results. Requests show the latest successful current-run document as temporary working data. No new state fields, storage, model judge or completion gate.
**Tech stack:** Existing Python, JSON Schema, unittest and SDK MockTransport.
**Spec:** User-approved R1–R4 in this conversation: default off; main model updates at meaningful points; four fields, 3KB limit; internal updates require no confirmation and cannot certify completion.

## Constraints and review focus

- Do not modify BFCL, frozen V0, checkpoint schemas, sandbox behavior, existing exit or verification rules.
- Preserve root sources and tests in a temporary baseline backup; there is no Git metadata in this checkout.
- Document fields: deliverables list[str]; findings list[{text: str, call_ids: list[str]}]; remaining list[str]; next_action str. All required, nonblank text, no extra fields.
- Canonical document limit: 3000 UTF-8 bytes (compact JSON, ensure_ascii=False). A failed update does not replace the latest valid one.
- References must resolve uniquely to recorded model tool results in the current run, excluding update_progress. Existence is checked, interpretation is not.
- Toolset(..., task_progress=False); CLI AGENT_TASK_PROGRESS=on|off, default off. Disabled tool is unpublished and cannot execute.
- Stable guidance is system text; progress content is assistant working data. Temporary context never accumulates in saved messages.
- Include guidance and current document in threshold, hard budget and 24–28K compaction-target calculations.
- Preserve existing step and usage accounting. No forced periodic updates, automatic Git work or task-specific evaluation rules.
- Cover malformed/oversized updates, stale/cross-run references, resume/compaction/new-task behavior, permissions and unchanged termination.

## Task 1: RED

- [x] Add grouped offline tests in tests/test_task_progress.py with existing OfflineScenario.
- [x] Cover normal updates, invalid updates, disabled/direct invocation, permission boundaries, input/compaction budgets, resume/new task, token counts, step limit and verifier.
- [x] Run new tests before implementation and confirm missing-feature failures.

## Task 2: GREEN

- [x] tools.py: optional publication/execution, schema, document/reference checks and recorded-result injection.
- [x] permissions.py: register only update_progress as SAFE.
- [x] prompts.py: separate optional guidance; preserve SYSTEM.
- [x] agent.py: pass recorded results, compose temporary latest-progress context and include it in budget checks.
- [x] context.py: optional extra_context for request/compaction budget calculations; empty default preserves behavior.
- [x] cli.py: validated on/off environment option before credential lookup.
- [x] README.md: activation, four fields, length and evidence limitations.
- [x] Focused tests green.

## Task 3: Validation

- [x] Run full basic offline group: .venv-mcp/Scripts/python.exe -B -X utf8 tests/run_offline.py.
- [x] Run independent native sandbox group; distinguish existing Bash/registry failures from regressions.
- [x] Independent review against baseline diff; meaningful fixes get RED–GREEN regression coverage.
- [x] Save validation command/results and hashes. No real-model evaluation or capability-improvement claim.

## Evidence

- Baseline basic offline group: 150 tests, OK, 2026-11-13. Full log in temporary backup directory.

- Final basic group: 160/160 passed; native group: 4/6 passed, same Bash/registry failures reproduced with baseline source and identical fixture ROOT.
- Independent review: fixed orphan progress/evidence after a RED regression; no new state/checkpoint schema. No real model calls.
