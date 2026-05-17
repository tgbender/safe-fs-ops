from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError
from safe_fs_ops.workspace_state import WorkspaceRuntime

pytestmark = pytest.mark.safe_fs_ops


def test_workspace_runtime_open_is_thread_safe_per_resolved_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    def open_runtime() -> WorkspaceRuntime:
        return WorkspaceRuntime.open(root)

    with ThreadPoolExecutor(max_workers=12) as pool:
        runtimes = list(pool.map(lambda _: open_runtime(), range(24)))

    assert len({id(runtime) for runtime in runtimes}) == 1
    assert runtimes[0].root == root.resolve()
    assert runtimes[0].state_path == root.resolve() / ".safe-fs-ops" / "state.db"


def test_workspace_runtime_registry_key_normalizes_final_state_path(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    first = WorkspaceRuntime.open(root, state_dir=".safe-fs-ops", state_filename="state.db")
    second = WorkspaceRuntime.open(root, state_dir=str(Path(".safe-fs-ops") / "."), state_filename="state.db")

    assert first is second


def test_workspace_runtime_rejects_symlink_state_directory(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / ".safe-fs-ops").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="redirects"):
        WorkspaceRuntime.open(root)

    assert list(outside.iterdir()) == []


def test_workspace_runtime_rejects_state_path_outside_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()

    with pytest.raises(UnsafePathError, match="inside workspace root"):
        WorkspaceRuntime.open(root, state_dir="..", state_filename="outside.db")


def test_workspace_runtime_keeps_authority_in_lease_store(tmp_path: Path) -> None:
    runtime = WorkspaceRuntime.open(tmp_path)
    contender = WorkspaceRuntime.open(tmp_path, state_dir=".safe-fs-ops", state_filename="state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)

    lease = runtime.acquire_lease("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    blocked = contender.acquire_lease(
        "workspace",
        owner="owner-b",
        ttl=timedelta(seconds=30),
        now=now + timedelta(seconds=1),
    )

    assert lease.acquired is True
    assert blocked.acquired is False
    assert blocked.owner == "owner-a"


def test_workspace_runtime_rejects_old_lease_after_stale_takeover(tmp_path: Path) -> None:
    runtime = WorkspaceRuntime.open(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old = runtime.acquire_lease("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    new = runtime.acquire_lease(
        "workspace",
        owner="owner-b",
        ttl=timedelta(seconds=30),
        now=now + timedelta(seconds=31),
    )

    assert new.acquired is True
    assert new.fencing_token == old.fencing_token + 1
    assert runtime.lease_store.is_current(old, now=now + timedelta(seconds=31)) is False
