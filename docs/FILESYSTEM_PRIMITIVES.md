# Filesystem Primitives

The `safe_fs_ops.filesystem_ops` package exposes lower-level helpers for callers
that do not need the full `SafeWorkspace` coordination layer.

Most applications should use `SafeWorkspace` first. Use these primitives when
you need to build another coordinator, run isolated filesystem checks, or create
explicit backup/capture artifacts.

## Path Inspection

```python
from safe_fs_ops.filesystem_ops import inspect_path

safety = inspect_path("settings.toml")
```

`PathSafety` records:

- path
- existence
- file type: `"missing"`, `"file"`, `"directory"`, `"symlink"`, or `"other"`
- mount-point status
- Windows reparse-point status
- hardlink count
- file size when known

Safety helpers such as `ensure_safe_write_path`, `ensure_safe_delete_path`,
`ensure_safe_mkdir_path`, and `ensure_safe_parent_chain` reject paths that would
redirect through symlinks, Windows reparse points, unsafe parents, mount points,
or unsupported file types for the requested operation.

## Snapshots

```python
from safe_fs_ops.filesystem_ops import snapshot_resource

snapshot = snapshot_resource("settings.toml")
```

`ResourceSnapshot` records existence, file type, content hash for regular files,
size, mtime, symlink target, and platform identity fields when available.

Snapshots are evidence. They are not enough by themselves to restore file
content unless paired with a backup artifact.

## Atomic Writes

```python
from safe_fs_ops.filesystem_ops import DurabilityMode, atomic_write_text

atomic_write_text(
    "settings.toml",
    "enabled = true\n",
    durability=DurabilityMode.FSYNC,
)
```

Also available:

- `atomic_write_bytes`
- `delete_file`
- `make_directory`
- `remove_empty_directory`
- `remove_existing_empty_directory_by_identity`

The default operations are conservative:

- refuse unsafe parents and redirecting paths
- use same-directory temporary files where needed
- fsync according to `DurabilityMode`
- validate parent identity where platform support allows it
- fail closed when the required primitive is unavailable

On Windows, `SafeWorkspace` swaps in native Windows operations for default
write/delete/mkdir behavior.

## Durability

`DurabilityMode` values:

- `FSYNC`: request file and directory fsyncs where implemented.
- `BEST_EFFORT`: attempt fsyncs but suppress fsync-specific `OSError`s.
- `NONE`: skip durability flushes.

Durability cannot make a group of filesystem operations atomic. It only affects
how aggressively individual operations ask the OS to flush state.

## Content-Addressed Store

```python
from safe_fs_ops.filesystem_ops import ContentAddressedStore

store = ContentAddressedStore(".safe-fs-ops/objects")
ref = store.put_file("settings.toml")
assert store.verify(ref)
store.copy_to(ref, "restore/settings.toml", no_replace=True)
```

The store uses SHA-256 object paths and verifies content before reads and after
writes. `copy_to` can preserve permissions and mtime metadata when provided.

The store is used by tree backups and file rollback artifacts.

## File Backups

```python
from safe_fs_ops.filesystem_ops import capture_backup, restore_backup

backup = capture_backup("settings.toml")
restore_backup(backup, "settings.toml")
```

`capture_backup` captures either missing-file state or regular-file content.
`restore_backup` restores only from valid backup proof and raises typed errors
for conflicts or content mismatches.

For large or durable workflows, prefer workspace transactions or the
content-addressed store path rather than in-memory backup bytes.

## Snapshot Bundles

```python
from safe_fs_ops.filesystem_ops import snapshot_bundle

bundle = snapshot_bundle(
    ["pyproject.toml", "src"],
    include_children=True,
    hash_policy="small-files",
    small_file_max_bytes=1024 * 1024,
)
```

`SnapshotBundle` records snapshots for multiple paths. With
`include_children=True`, directories are traversed without following symlinked
directories.

Hash policies:

- `"metadata-only"`: do not hash file content.
- `"small-files"`: hash regular files up to the configured size.

## No-Replace Rename

```python
from safe_fs_ops.filesystem_ops import rename_no_replace, restore_inverse_rename

record = rename_no_replace("old.txt", "new.txt")
restore_inverse_rename(record)
```

`rename_no_replace` refuses to overwrite the destination and records a
`RenameRecord` that can be used for inverse recovery.

## Directory Capture

```python
from safe_fs_ops.filesystem_ops import (
    capture_directory_to_quarantine,
    restore_captured_directory,
)

record = capture_directory_to_quarantine(
    "worktree",
    quarantine_path=".safe-fs-ops/quarantine/worktree",
)
restore_captured_directory(record)
```

Directory capture moves a directory to a caller-provided quarantine location and
records ownership/identity information for recovery.

Use this before high-risk directory teardown, especially when the directory may
contain `.git` data.

## Tree Backups

```python
from safe_fs_ops.filesystem_ops import ContentAddressedStore, backup_tree, restore_tree_backup

store = ContentAddressedStore(".safe-fs-ops/objects")
backup = backup_tree(
    project_root,
    ["pyproject.toml", "src/package/__init__.py"],
    artifact_store=store,
)
restore_tree_backup(backup, destination_root=project_root, artifact_store=store)
```

`backup_tree` captures explicit relative file and symlink paths under a root.
It rejects absolute paths, `..`, missing paths, directories, and unsafe parent
chains.

`restore_tree_backup` preflights all entries before creating or replacing
anything. By default it refuses to overwrite existing non-matching destinations.
Use `conflict_policy="replace"` only when replacing regular files is intended
and supported.

Tree backups are not recursive whole-tree snapshots. They are explicit restore
sets for named paths.

## Backend Capabilities

```python
from safe_fs_ops.filesystem_ops import detect_default_backend_capabilities

capabilities = detect_default_backend_capabilities()
capabilities.require("write_text")
```

`FilesystemBackendCapabilities` reports support for the default mutation
backend. `SafeWorkspace.filesystem_backend` exposes the capabilities selected
for a workspace.

Capability checks are guardrails. They do not replace operation-level safety
checks.
