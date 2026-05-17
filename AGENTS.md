# Operating Model

This file is stable operating-model guidance for agents working in this repository. Do not use it as task-state scratch space.

Mutable working notes, handoff summaries, local investigation logs, and compaction-resistant task state belong in `.ai/`. That directory is intentionally gitignored so agents can record useful local context without adding noisy or stale scratch files to commits. Durable design decisions that should be reviewed and versioned belong in tracked docs such as `DESIGN.md`.

## Current Direction

This package provides reusable primitives for:

- SQLite-backed storage and migrations in `safe_fs_ops.sqlite_store`
- workspace-scoped leases, claims, and conflict coordination in `safe_fs_ops.workspace_state`
- operation batches, journals, checkpoints, and recovery in `safe_fs_ops.operation_journal`
- conservative cross-platform filesystem mutation helpers in `safe_fs_ops.filesystem_ops`

## Workflow

- Prefer characterization tests before correctness fixes.
- Keep tests free of monkeypatching. Prefer dependency injection and explicit seams.
- Make small, checkpointable changes.
- Run focused tests first, then the full suite before commits when practical.
- Treat residual risks explicitly; do not summarize uncertain behavior as solved.

## Validation Policy

- Dogfood mutating behavior in isolated temp workspaces.
- Do not mutate real user configuration during validation.
- Verify actual filesystem end state, state database records, and rollback or recovery behavior when working on journal/rollback code.
- Keep opt-in stress tests guarded so the default suite remains fast.

## Concurrency And State

- Design coordination code for free-threaded Python.
- Do not rely on the GIL for correctness.
- Do not share SQLite connections across threads.
- Protect mutable in-process runtime/coordinator state with real locks.
- Use SQLite transactions, lease tokens, and fencing tokens for cross-process correctness.
- Keep leases, claims, batches, journals, and checkpoints as separate concepts.
