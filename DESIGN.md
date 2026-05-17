# safe-fs-ops Design Notes

These notes capture the intended direction for extracting Prescribe's durable
filesystem and state-management machinery into a reusable package.

## Goals

- Provide cross-platform filesystem operations that are conservative by default.
- Provide a durable operation journal for filesystem mutations and recovery.
- Provide SQLite-backed primitives for migrations, transactions, leases, and ids.
- Support conflict management across threads, worker processes, and separate CLI
  invocations.
- Be usable by Prescribe during the extraction without removing the existing
  Prescribe implementation until the replacement path is proven.
- Design for free-threaded Python: do not rely on the GIL for correctness.
- Keep APIs explicit with keyword-only optional parameters.
- Avoid opaque raw SQL; prefer traceable schema/model helpers with named tables
  and columns.

## Proposed Package Responsibilities

```text
safe_fs_ops/
  sqlite_store/
    Connection and transaction primitives, schema migrations, id/time helpers,
    and low-level SQLite utilities.

  workspace_state/
    Workspace-scoped runtime coordination, lease acquisition, heartbeat state,
    resource claims, and conflict management.

  operation_journal/
    Operation batches, planned intent, attempt/failure records, recovery intent,
    and recovery outcomes.

  filesystem_ops/
    Path validation, snapshots, atomic-ish file writes, backup/restore helpers,
    and rollback execution primitives.
```

The public coordination API should be workspace-scoped, not a process-global bag
of mutable state:

```python
runtime = WorkspaceRuntime.open(root)

with runtime.exclusive_run("apply") as run:
    with run.resource_claims(["file:.env"]):
        with run.operation_batch("update .env") as batch:
            ...
```

Internally, `WorkspaceRuntime.open(path)` may maintain a memoized runtime cache,
but that cache must be guarded by a lock and keyed by resolved workspace path.

## Concurrency Model

Thread locks and SQLite leases have different jobs:

- Python locks protect in-process object invariants.
- SQLite transactions and lease tokens protect cross-process ownership.

Do not use a Python singleton as the authority for mutation safety. The database
lease is the durable authority. In-process runtime objects are only convenient
coordinators around that durable state.

### Free-Threaded Python Rules

- Do not share SQLite connections across threads.
- Protect mutable runtime/coordinator fields with `threading.RLock`.
- Protect any module-level runtime registry with a class/module lock.
- Prefer frozen dataclasses for public records.
- Avoid check-then-act flows unless guarded by a Python lock or a SQLite
  transaction.
- Require a current lease token for every lease-sensitive write.
- Communicate heartbeat state through locked fields and `threading.Event`, not
  unsynchronized mutable attributes.
- Validate state-machine transitions explicitly instead of assigning string
  phases freely.

### Schema and Query Style

SQLite remains the durable coordination layer, but SQL should be traceable:

- Define table and column names once in schema/model helpers.
- Build SQL from those definitions when SQL is necessary.
- Prefer small typed store methods over exposing callers to raw SQL.
- Keep schema ownership close to the package that owns behavior.
- Do not duplicate string table/column names across stores and tests.
- `sqlite_store` may expose a forward-only schema plan API with explicit schema
  identity and integer migration versions, while preserving a raw statement
  initializer for transitional callers that have not adopted versioned schemas.

If a future SQLAlchemy ORM/Core layer is introduced, it should preserve the same
ownership boundary: table definitions live with the package behavior they
support, and public APIs remain typed store/coordinator methods.

## Durable Coordination Concepts

The public mental model should stay small:

1. `Lease`: active authority to mutate a scope.
2. `Claim`: durable ownership of a resource.
3. `Batch`: a durable unit of intended work.
4. `Journal`: execution and recovery history for batches.
5. `Checkpoint`: enough resource state to recover safely.

Lower-level database concepts such as fencing tokens, epochs, reservations,
tombstones, idempotency keys, and outboxes can support those primitives without
becoming the main user-facing API.

### Leases

Leases are short-lived active authority to mutate a scope. They should include:

- name
- owner
- token
- fencing_token or epoch
- acquired_at
- heartbeat_at
- expires_at

A lease is current only when the owner and token match and `expires_at` is in the
future. Stale leases may be taken over by another owner through a SQLite
transaction.

Background lease heartbeats are a wall-clock behavior. Callers that pass fixed
`now` values or injected logical clocks are asking for deterministic logical
time, so mutation and recovery helpers should not run heartbeat threads that
repeatedly write the same logical expiry.

The token authenticates the holder. The fencing token/epoch orders holders. A
stale worker that still has an old in-memory lease object must not be allowed to
write after another worker acquires a newer lease.

### Resource Claims

Claims are durable ownership/conflict records. They answer "who owns this
resource over time?" rather than "who is actively writing right now?"

Examples:

- `file:C:/Users/me/.config/tool.toml`
- `json-key:C:/Users/me/.config/tool.toml::settings.theme`
- `asset:C:/Users/me/bin/tool.exe`

Claims and leases should stay separate. A project may need long-lived ownership
claims while still allowing short-lived leases to expire and be recovered.

### Batches

Batches are durable units of intent. A batch groups related operations even when
the filesystem cannot make the operations atomic as a group.

Batch records should be suitable for retry and recovery:

- batch_id
- idempotency_key
- owner/run id
- lease fencing token
- phase
- created_at
- updated_at

Idempotency keys let a retry find the existing logical operation instead of
creating duplicate recovery work.

### Operation Journal

The journal should record intent before mutation, then record outcomes:

1. record desired batch
2. mark batch attempting
3. attempt filesystem/database work
4. record success or failure, including observed failure state
5. if needed, record desired recovery state
6. attempt recovery
7. record recovery success or failure

Filesystem operations are not atomic as a group, so each stage should be a
durable SQLite transaction. If the process is not killed mid-step, the journal
should always describe the next safe action.

### Recovery Action Contract

Recovery execution should expose its own append-only contract without coupling
the journal to a filesystem runner:

- `start_recovery()` creates the active `recovery_attempt_id` for a batch.
- Recovery action writes must name that attempt id explicitly and are rejected
  unless it is still the latest `RECOVERING` recovery record for the batch.
- Each recovery action has a stable logical `action_id` plus append-only status
  records such as planned, attempting, succeeded, failed, skipped, and manual
  intervention required.
- Planned action intent is recorded before the side effect starts. Later status
  records reuse the same `action_id` so replay can distinguish intended work
  from observed recovery behavior.
- Recovery action records share the batch's monotonic sequence space with
  operations, checkpoints, and recovery phase records so a reader can replay one
  deterministic cross-table timeline.
- Completing or re-queuing recovery fences off the prior attempt; stale writers
  must not append more recovery action records after takeover or terminal
  recovery completion.

### Checkpoints

Checkpoints capture enough resource state to recover:

- existence
- content hash
- size and relevant metadata
- backup path or inline snapshot reference
- resource version before mutation
- resource version after mutation when available

Resource versions support optimistic concurrency checks. Rollback should avoid
overwriting unrelated external changes unless policy explicitly permits it.

### Recoverable Artifact Primitives

High-risk filesystem operations should build on small, named primitives instead
of arbitrary command rollback:

- `ContentAddressedStore` records large backup artifacts by hash and verifies
  them before restore.
- `snapshot_bundle()` captures durable evidence for one or more paths without
  mutating the filesystem.
- `rename_no_replace()` moves a file or directory only when the destination is
  absent and records the inverse rename needed for recovery.
- `capture_directory_to_quarantine()` moves a directory into an owned quarantine
  path and records the restore information needed for recovery.

Workspace transactions and phased operations expose these primitives through the
journal so intent, checkpoints, and recovery actions are durable before and after
the side effect. The package deliberately does not model arbitrary command
rollback; callers should add new typed primitives when they need another
recoverable mutation.

### Reservations

Reservations are temporary pre-commit claims. They are useful when planning a
multi-resource batch before making durable ownership changes.

Reservations may expire like leases. Persisted claims should represent committed
ownership; reservations should represent planned or in-progress ownership.

### Tombstones

Tombstones record intentional deletion or release. They help distinguish "never
existed" from "existed and was intentionally removed", which matters for
rollback and claim cleanup.

### Outbox and Compensation

Filesystem mutations are side effects. An outbox-style table can separate
"durably recorded work to do" from "side effect attempted". This makes recovery
more mechanical:

- record side effect intent
- attempt side effect
- record result
- enqueue compensation if recovery is required

Compensation should be modeled as first-class recovery work, not as ad hoc
cleanup code hidden in filesystem helpers.

### Invariants

The package should name and test important invariants:

- only one current lease exists for a resource
- a lease-sensitive write requires the current token and fencing token
- a resource cannot have conflicting active claims unless policy allows it
- a succeeded batch has no required recovery work
- a failed batch records observed failure state before recovery starts
- recovery work is idempotent or has an idempotency key

## Incremental Extraction Plan

1. Keep the existing Prescribe implementation intact.
2. Build and test `sqlite_store` primitives independently.
3. Add `workspace_state` with runtime caching, leases, heartbeat, and resource
   conflict tests.
4. Move `operation_journal` from an adapter over Prescribe state to a real
   package-level implementation.
5. Add `filesystem_ops` primitives behind focused tests.
6. Adapt Prescribe to use the new package one boundary at a time.
7. Remove old Prescribe-local code only after the new path has equivalent or
   stronger coverage.

## Initial Tests to Add

- `WorkspaceRuntime.open(path)` is thread-safe and returns one runtime per
  resolved workspace path.
- Runtime cache does not corrupt under concurrent opens.
- SQLite connections are opened per operation and are not reused across threads.
- Invalid coordinator state transitions are rejected.
- A fresh lease has a unique token and blocks competing acquisition.
- An expired lease can be taken over by exactly one contender.
- Heartbeat extends a lease while active.
- Heartbeat loss prevents later protected writes.
- Resource claims report conflicts across separate runtime instances.
- Operation batches persist intent before invoking filesystem mutations.

## Windows Backend Implementation Plan

The Windows backend slice should stay narrow and keep the backend primitive as a
guardrail, not as policy.

### Current Direction

- Keep backend capability routing explicit so unsupported Windows paths fail
  closed before mutation.
- Keep the coordinator as an orchestrator that delegates implementation, test,
  and review work to scoped subagents.
- Preserve the no-monkeypatch testing rule by using dependency injection and
  explicit seams.
- Keep files under about 500 lines when practical.
- Treat Windows primitives as mutation guardrails only.
- Keep durable rollback and recovery decisions in the journal and operation
  layers, not in the backend primitive.
- Land validated slices through checkpoint commits instead of batching the whole
  migration into one unreviewed change.

### Phases

1. Capability routing: expose backend selection and capability checks through
   explicit seams so unsupported Windows paths fail closed.
2. Windows primitives: implement the conservative Windows filesystem
   primitives and keep their guardrails separate from policy decisions.
3. Windows default ops: wire default Windows operations through the primitives
   and verify the supported mutation path end to end.
4. Rollback/dogfood tests: add rollback, recovery, and dogfood coverage in
   isolated temp workspaces and temporary state directories.
5. Adversarial review: review capability routing, rollback gaps, unsupported
   cases, and race conditions before widening scope.
6. Checkpoint commits: land each validated slice behind a checkpoint commit and
   update `.ai` notes before the next worker pass.

### Validation

- `uv run pytest -q -m safe_fs_ops`
- `uv run pytest -q -m safe_fs_ops_backend`
- `uv run mypy --no-incremental safe-fs-ops/src/safe_fs_ops`
- Windows or WSL backend smoke tests should run from an isolated temp workspace
  and a temporary state directory.

## Current Implementation State

This section records the current package state as of checkpoint `1c5b83c`
(`feat(safe-fs-ops): execute file rollback recovery`). More detailed mutable
coordination notes live in `.ai/file-rollback-handoff.md`.

### Implemented

- `SafeWorkspace` selects Windows default filesystem operations on Windows and
  keeps POSIX/default behavior available for platforms with descriptor-relative
  support.
- Journaled transaction rollback is no longer only a planned recovery surface:
  the package can execute automatic rollback for the currently supported
  high-level mutating operations.
- Recursive directory creation records enough ownership proof to remove only
  transaction-created directories during rollback.
- File rollback is implemented for the current public mutation surface:
  transaction-created files can be removed, overwritten files can be restored
  from durable backups, and deleted files can be restored from durable backups.
- Durable per-artifact cleanup rows are implemented and wired into commit and
  recovery cleanup. Keep-all retention is the default so the audit trail stays
  complete; compaction or archival can be added later if needed.
- Recovery planning and execution require expected-current proof before
  changing the filesystem. Missing, stale, tampered, or conflicting proof records
  manual intervention required instead of guessing.
- Custom injected filesystem backends remain conservative unless they
  explicitly opt into the automatic file rollback contract.
- The latest recorded validation for this slice had the focused safe-fs-ops
  test suite passing, package mypy clean, file-size checks clean, and isolated
  dogfood passing for covered rollback flows.

### Remaining Work

- Add transaction-level integration tests for missing or tampered backup
  artifacts during automatic rollback.
- Add transaction-level integration tests for current-file tampering between
  proof capture and restore execution.
- Decide whether `expected_after` checkpoints remain audit evidence or become
  executable recovery proof.
- Extend rollback only when new public mutating operations have explicit proof,
  planning, and recovery tests.
- Continue platform validation across native Windows, WSL/Linux, and macOS
  before treating the package as broadly portable.

### File Rollback Invariants

- Durable backup proof: rollback only restores or removes state when a durable
  backup/capture record exists and was journaled before mutation.
- Expected-current proof: rollback only proceeds when the observed current
  state matches the recorded expected current proof for the target resource.
- Fail manual intervention: if proof is missing, stale, tampered, or
  conflicting, record manual intervention required instead of guessing.
