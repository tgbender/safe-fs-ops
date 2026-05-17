# safe-fs-ops

Conservative filesystem mutation primitives with SQLite-backed leases, resource
claims, operation journaling, checkpoints, and recovery hooks.

`safe-fs-ops` is for tools that need to change local files without pretending
the filesystem has real transactions. It records intent before mutation, keeps
durable proof for supported rollback paths, coordinates concurrent workers with
leases and claims, and fails closed when recovery would require guessing.

## Status

This package is pre-1.0 alpha. The current public surface is ready for careful
integration in tools that can tolerate explicit recovery semantics, especially
workspace-scoped writes, deletes, directory creation, no-replace renames,
directory capture, snapshot bundles, and tree backup artifacts.

It is not a sandbox, a general command rollback engine, or a generic `rm -rf`
replacement. If another tool mutates a directory after you hand it control,
`safe-fs-ops` can only restore state that you explicitly snapshotted, backed up,
captured, or journaled first.

## Install

```bash
uv add safe-fs-ops
```

Python 3.12 or newer is required.

## Quick Start

```python
from pathlib import Path

from safe_fs_ops import SafeWorkspace

workspace = SafeWorkspace.open(
    Path(".safe-fs-ops/state.sqlite"),
    owner="worker-a",
)

settings = workspace.file("settings.toml")

with workspace.transaction(
    name="update-settings",
    resources={"settings": settings},
    rollback="automatic",
) as tx:
    tx.write_text(tx.r.settings, "enabled = true\n")
```

The transaction acquires a workspace lease, claims `settings.toml`, records a
journal batch, captures rollback proof for supported operations, performs the
write, records checkpoints, and releases claims on exit.

## Common Patterns

Back up named files before invoking another tool:

```python
root = workspace.tree(project_root)

with workspace.transaction(
    name="prepare-risky-tool",
    resources={"root": root},
) as tx:
    backup_result = tx.backup_tree(
        tx.r.root,
        ["pyproject.toml", "src/package/__init__.py"],
    )
```

Capture a directory before tearing it down:

```python
worktree = workspace.directory(".git/worktrees/topic")
quarantine = ".safe-fs-ops/quarantine/worktree-topic"

with workspace.transaction(
    name="capture-worktree",
    resources={"worktree": worktree},
    rollback="automatic",
) as tx:
    tx.capture_directory(tx.r.worktree, quarantine_path=quarantine)
```

Resume pending recovery work after a crash or lease handoff:

```python
error = workspace.recover_pending_batches(run_id="run-123")
if error is not None:
    raise error
```

## Documentation

- [Docs index](docs/README.md): entry points for the documentation set.
- [Public API](docs/PUBLIC_API.md): supported imports, workspace usage,
  transactions, phased operations, and advanced modules.
- [Recovery model](docs/RECOVERY_MODEL.md): what automatic rollback can and
  cannot do, and how pending recovery is resumed.
- [Filesystem primitives](docs/FILESYSTEM_PRIMITIVES.md): lower-level path,
  mutation, snapshot, content-addressed, backup, capture, and restore helpers.
- [Development](docs/DEVELOPMENT.md): local setup, tests, slow-test markers,
  build, and release notes.
- [Source review](docs/SOURCE_REVIEW.md): current public-interface assessment,
  strengths, sharp edges, and remaining risks.
- [Design notes](DESIGN.md): longer-running architecture notes and historical
  extraction context.

## Development

```bash
uv sync
uv run pytest -q
uv run mypy src
uv build
```

The default test run skips opt-in stress tests and slow recovery integration
tests. Run the recovery-heavy tests with:

```bash
uv run pytest -q --slow-recovery -m slow_recovery
```

The project also defines task aliases:

```bash
mise run test
mise run test-slow
mise run test-full
mise run test-stress
```

## License

MIT. See [LICENSE](LICENSE).
