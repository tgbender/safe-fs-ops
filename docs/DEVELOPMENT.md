# Development

## Setup

```bash
uv sync
```

The package requires Python 3.12 or newer and uses the `uv_build` backend.

## Validation

Default tests:

```bash
uv run pytest -q
```

The default run skips opt-in stress tests and slow recovery integration tests.

Slow recovery and automatic rollback tests:

```bash
uv run pytest -q --slow-recovery -m slow_recovery
```

Full non-stress suite:

```bash
uv run pytest -q --slow-recovery
```

Stress tests:

```bash
uv run pytest -q --stress-locks -m stress_lock
```

Type check:

```bash
uv run mypy src
```

Build:

```bash
uv build
```

Whitespace check:

```bash
git diff --check
```

## Task Aliases

The repo uses `mise.toml` for task aliases:

```bash
mise run test
mise run test-slow
mise run test-full
mise run test-stress
```

`uv.toml` is reserved for uv settings. Do not put task aliases there.

## Test Policy

- Prefer characterization tests before correctness fixes.
- Keep tests free of monkeypatching; use dependency injection and explicit
  seams.
- Dogfood mutating behavior in isolated temp workspaces.
- Do not mutate real user configuration during validation.
- Verify filesystem end state, state database records, and rollback/recovery
  behavior when changing journal or rollback code.
- Keep stress tests opt-in so the default suite remains fast.

## Source Layout

- `src/safe_fs_ops/workspace.py`: public workspace factory, resource helpers,
  backend selection, recovery entry points.
- `src/safe_fs_ops/workspace_transaction.py`: transaction context manager and
  high-level journaled mutation methods.
- `src/safe_fs_ops/workspace_operation.py`: phased operation wrapper.
- `src/safe_fs_ops/filesystem_ops/`: conservative filesystem primitives and
  backup/capture helpers.
- `src/safe_fs_ops/operation_journal/`: durable operation batches,
  checkpoints, recovery records, and journaled filesystem coordinator.
- `src/safe_fs_ops/workspace_state/`: SQLite-backed leases, claims, heartbeat,
  and runtime coordination.
- `src/safe_fs_ops/sqlite_store/`: schema and migration primitives.
- `src/safe_fs_ops/persistence/`: SQLAlchemy engine/session helpers.

## Release Notes

The project metadata declares MIT licensing and uses `uv_build`.

The release workflow is expected to build artifacts in CI and publish through
PyPI trusted publishing. Local `dist/` artifacts are ignored and should not be
committed.

Before an alpha release, run:

```bash
uv run pytest -q --slow-recovery
uv run mypy src
uv build
```

Run opt-in stress tests separately when changing leases, claims, recovery
authority, or cross-process coordination.
