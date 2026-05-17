# Public API

This document describes the intended public interface for `safe-fs-ops`.
Anything imported from underscored modules or names prefixed with `_` should be
treated as implementation detail.

## Stability Contract

The most stable surface is the top-level package:

```python
from safe_fs_ops import (
    DirectoryResource,
    DurabilityMode,
    FileResource,
    ResourceHandle,
    ResourceNotClaimedError,
    ResourceSet,
    SafeOperation,
    SafePhase,
    SafeTransaction,
    SafeWorkspace,
    SafeWorkspaceBusyError,
    SafeWorkspaceError,
    TreeResource,
)
```

The package is still pre-1.0. The top-level workspace API is the recommended
integration point. Lower-level package exports are available for advanced
callers, but exact journal payloads, SQLite schema details, private helpers, and
underscored modules can change without compatibility guarantees.

## Workspace

Use `SafeWorkspace` for application code.

```python
from pathlib import Path

from safe_fs_ops import SafeWorkspace

workspace = SafeWorkspace.open(
    Path(".safe-fs-ops/state.sqlite"),
    owner="worker-a",
)
```

`SafeWorkspace.open(...)` creates a coordinator around:

- one SQLite state database
- a lease store
- a resource claim store
- an operation journal
- platform default filesystem operations
- a content-addressed artifact store beside the state database

Important options:

- `owner`: required non-empty owner name recorded in leases, claims, and journal
  rows.
- `lease_name`: defaults to `"workspace"`; use a different name to isolate
  independent coordination scopes in one database.
- `lease_ttl`: defaults to 30 seconds. A background heartbeat keeps active
  transactions current unless deterministic `now` is injected.
- `durability`: `DurabilityMode.FSYNC` by default, with `BEST_EFFORT` and
  `NONE` available when a caller accepts weaker durability.
- `enable_automatic_file_rollback`: defaults to automatic only for the package
  default filesystem backend. Custom injected backends stay conservative unless
  explicitly opted in.
- `artifact_store`: custom content-addressed store object. The default is
  `state.sqlite.objects` beside the state database.
- `write_bytes_max_bytes`: default per-call byte write limit.
- `write_bytes_large_policy`: `"reject"` by default, or `"allow"` to permit
  over-limit byte writes unless a call overrides it.
- `captured_directory_cleanup`: `"retain"` by default, or `"automatic"` to clean
  committed captured-directory artifacts when safe cleanup has been recorded.

Workspace inspection and recovery helpers:

```python
workspace.filesystem_backend
workspace.lease_store
workspace.claim_store
workspace.journal_store

workspace.cleanup_outstanding_artifacts(run_id=None)
workspace.recover_pending_batches(run_id=None)
```

`cleanup_outstanding_artifacts` and `recover_pending_batches` return
`SafeWorkspaceError | None`; callers should raise or log a returned error.

## Resources

Resources are durable conflict keys. A transaction can only mutate resources it
claimed when it entered.

```python
settings = workspace.file("settings.toml")
worktree = workspace.directory(".git/worktrees/topic")
source_tree = workspace.tree("src")
custom = workspace.resource("tool:global-config")

resources = workspace.resources(
    {
        "settings": settings,
        "worktree": worktree,
        "source_tree": source_tree,
        "custom": custom,
    }
)
```

Resource helpers:

- `workspace.file(path)` creates a `FileResource`.
- `workspace.directory(path)` creates a `DirectoryResource`.
- `workspace.tree(path)` creates a `TreeResource`.
- `workspace.resource(resource_key)` creates a custom non-path resource.
- `workspace.resources({...})` creates a named `ResourceSet`.

Resource paths are converted to absolute lexical paths without resolving the
final target. That keeps the resource key stable without following symlinks.

Inside a transaction or phase, named resources are available through `tx.r` or
`phase.r`:

```python
tx.r.settings
tx.r["settings"]
```

`DirectoryResource` and `TreeResource` can create child handles that stay within
the parent:

```python
root = workspace.tree(project_root)
pyproject = root.file("pyproject.toml")
src_tree = root.tree("src")
```

Child paths must be relative and cannot escape the parent.

## Transactions

Use transactions for journaled filesystem mutation.

```python
with workspace.transaction(
    name="update-settings",
    resources={"settings": workspace.file("settings.toml")},
    rollback="automatic",
) as tx:
    tx.write_text(tx.r.settings, "enabled = true\n")
```

Transaction options:

- `name`: logical workflow name.
- `resources`: claimed resources for the transaction.
- `run_id`: caller-provided run id; generated when omitted.
- `rollback`: `"record-only"` or `"automatic"`.
- `now`: deterministic clock injection for tests and replay scenarios.
- `cleanup_clock`: optional clock used by cleanup/recovery finalization.

`record-only` records journal state for later inspection and recovery.
`automatic` attempts rollback on transaction-body failure when each touched
operation has explicit proof and a supported recovery path.

Transactions are single-use context managers. A transaction acquires one
workspace lease and all requested resource claims on enter, then releases claims
and the lease on exit. Concurrent conflicting claims raise
`SafeWorkspaceBusyError`. Mutating an unclaimed resource raises
`ResourceNotClaimedError`.

## Transaction Methods

All transaction methods return `JournaledFilesystemResult`.

### `write_text`

```python
tx.write_text(tx.r.settings, "enabled = true\n", encoding="utf-8")
```

Writes a text file through the configured atomic write operation and records
before/after checkpoints. Automatic rollback can restore overwritten file
content or remove transaction-created files on supported backends.

### `write_bytes`

```python
tx.write_bytes(tx.r.blob, b"\x00\x01", permissions=0o640)
tx.write_bytes(tx.r.archive, content, max_bytes=8 * 1024 * 1024, allow_large=True)
```

Writes bytes through the configured atomic bytes operation. Byte writes are
bounded by `write_bytes_max_bytes` unless the workspace or call explicitly
allows large writes.

Use `permissions` when the destination mode should be set as part of the atomic
write. The configured backend must support the `permissions` keyword.

### `delete_file`

```python
tx.delete_file(tx.r.settings, missing_ok=True)
```

Deletes a claimed regular file. Automatic rollback can restore the deleted file
from durable backup proof captured before the mutation.

### `make_directory`

```python
tx.make_directory(tx.r.output_dir, parents=False, exist_ok=False)
tx.make_directory(tx.r.nested_dir, parents=True, exist_ok=True)
```

Creates a claimed directory. `parents=True` uses the recursive mkdir workflow,
which records ownership proof for created parents so rollback removes only
transaction-owned state.

### `snapshot_bundle`

```python
tx.snapshot_bundle(
    tx.r.source_tree,
    paths=["src/package.py", "pyproject.toml"],
    include_children=False,
    hash_policy="metadata-only",
)
```

Records evidence for one or more paths without mutating the filesystem. This is
useful before invoking another tool.

Hash policies:

- `"metadata-only"` records file metadata without content hashes.
- `"small-files"` hashes regular files up to `small_file_max_bytes`.

`include_children=True` recursively records directory children without following
symlinked directories.

### `rename_no_replace`

```python
tx.rename_no_replace(tx.r.old_path, tx.r.new_path)
```

Renames a path only when the destination does not already exist. The source and
destination resources must both be claimed. The journal records inverse rename
information for recovery.

### `capture_directory`

```python
tx.capture_directory(
    tx.r.worktree,
    quarantine_path=".safe-fs-ops/quarantine/worktree-topic",
)
```

Moves a claimed directory into a caller-provided quarantine path and records the
directory identity needed to restore it. This is the preferred primitive before
tearing down a directory that may contain `.git` data.

The quarantine path is claimed automatically for the transaction.

### `backup_tree`

```python
result = tx.backup_tree(
    tx.r.source_tree,
    ["pyproject.toml", "src/safe_fs_ops/__init__.py"],
)
```

Records content-addressed backups for explicit file or symlink paths under a
claimed tree resource. It intentionally backs up named children; it does not
recursively snapshot a whole tree or preserve empty directories by itself.

Use this before handing a risky mutation to another tool. Recovery restores from
the recorded backup artifact, not from external tool state.

### `restore_tree_backup`

```python
tx.restore_tree_backup(tx.r.source_tree, backup, conflict_policy="no_replace")
```

Restores a `TreeBackup` into the claimed tree resource. If `destination_root` is
provided, it must resolve to the same path as the claimed tree resource; claim a
different `workspace.tree(...)` to restore somewhere else.

`restore_tree_backup` is journaled, but it is not supported inside
`rollback="automatic"` transactions. Use `rollback="record-only"` and inspect or
recover failures explicitly.

Conflict policies:

- `"no_replace"` refuses to overwrite existing non-matching destinations.
- `"replace"` can replace existing regular files where the platform can do so
  safely. Symlink replacement is intentionally conservative.

## Phased Operations

Use `workspace.operation(...)` when a workflow needs visible operation and
phase records.

```python
with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
    with op.phase("backup") as phase:
        phase.snapshot_bundle(phase.r.source_tree, include_children=True)
    with op.phase("mutate") as phase:
        phase.write_text(phase.r.settings, "enabled = true\n")
```

Phases expose the same public mutation methods as transactions:

- `write_text`
- `write_bytes`
- `delete_file`
- `make_directory`
- `snapshot_bundle`
- `rename_no_replace`
- `capture_directory`
- `backup_tree`
- `restore_tree_backup`

Phased operations currently support `rollback="record-only"` only. Phase names
must be unique within an operation, and only one phase may be active at a time.

## Lower-Level Modules

Application code should prefer `SafeWorkspace`, but lower-level packages are
exported for callers that need finer control.

### `safe_fs_ops.filesystem_ops`

Conservative path inspection, mutation primitives, snapshots, backups, content
addressed storage, no-replace renames, directory capture, and tree backups.

See [Filesystem primitives](FILESYSTEM_PRIMITIVES.md).

### `safe_fs_ops.operation_journal`

Durable operation batches, checkpoint records, operation runs/phases, recovery
records, recovery action records, and `JournaledFilesystemCoordinator`.

Use this directly only when building another workspace abstraction.

### `safe_fs_ops.workspace_state`

SQLite-backed leases, claims, heartbeat support, and runtime coordination.

Use this directly when you need durable conflict coordination without the
filesystem journal layer.

### `safe_fs_ops.sqlite_store`

Schema definition, migration, validation, and low-level SQLite store helpers.

### `safe_fs_ops.persistence`

SQLAlchemy engine/session helpers used by the stores. These are public for
advanced integration and testing, but the top-level workspace API should remain
the normal application interface.

## Error Model

Common top-level errors:

- `SafeWorkspaceError`: base workspace error.
- `SafeWorkspaceBusyError`: lease or resource conflict.
- `ResourceNotClaimedError`: attempted mutation of a resource not claimed by the
  active transaction.

Lower-level modules expose more specific errors such as `UnsafePathError`,
`UnsupportedFilesystemMutationError`, `RestoreConflictError`, journal state
transition errors, lease loss errors, and claim conflict errors.

## Recovery Summary

The package recovers from durable proof, not from guesses.

- Intent is recorded before mutation.
- Checkpoints or backup artifacts are recorded before risky filesystem changes.
- Recovery actions are planned and recorded before execution.
- If proof is missing, stale, tampered, or conflicts with current state,
  recovery records manual intervention required instead of overwriting unknown
  data.

See [Recovery model](RECOVERY_MODEL.md) for the full recovery contract.
