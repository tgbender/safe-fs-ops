from __future__ import annotations

import os
from pathlib import Path
from stat import S_IMODE
from typing import cast

import pytest
from transaction_file_rollback_helpers import requires_backup_restore_support, workspace_with_portable_file_rollback

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.operation_journal import BatchPhase
from safe_fs_ops.resources import FileResource

pytestmark = pytest.mark.safe_fs_ops


def test_transaction_write_bytes_writes_content_and_permissions(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "payload.bin"

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-write-bytes-success",
    ) as tx:
        result = tx.write_bytes(tx.r.file, b"\x00safe\xff", permissions=0o640)

    assert target.read_bytes() == b"\x00safe\xff"
    if os.name != "nt":
        assert S_IMODE(target.stat().st_mode) == 0o640
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.payload["operation"] == "write_bytes"
    assert result.batch.payload["desired_size"] == 6
    assert result.batch.payload["permissions"] == 0o640


def test_transaction_write_bytes_default_backend_accepts_permissions(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "payload.bin"

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-write-bytes-default-permissions",
    ) as tx:
        tx.write_bytes(tx.r.file, b"default", permissions=0o640)

    assert target.read_bytes() == b"default"
    if os.name != "nt":
        assert S_IMODE(target.stat().st_mode) == 0o640


@requires_backup_restore_support
@pytest.mark.slow_recovery
def test_transaction_write_bytes_automatic_rollback_restores_previous_bytes(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "payload.bin"
    target.write_bytes(b"old\x00bytes")

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-write-bytes-rollback",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_bytes(tx.r.file, b"new\xffbytes", idempotency_key="write-bytes:rollback")
        raise RuntimeError("boom")

    assert target.read_bytes() == b"old\x00bytes"
    [batch] = workspace.journal_store.list_batches(run_id="run-write-bytes-rollback")
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert batch.payload["operation"] == "write_bytes"
    assert any(
        checkpoint.checkpoint_type == "backup"
        for checkpoint in workspace.journal_store.list_checkpoints(batch.batch_id)
    )
    assert getattr(excinfo.value, "__notes__", ()) == ()


def test_write_bytes_rejects_large_content_by_default_and_configured_max(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(
        tmp_path / "state.db",
        write_bytes_max_bytes=3,
    )
    target = tmp_path / "payload.bin"

    with (
        pytest.raises(ValueError, match="allow_large=True"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-write-bytes-reject-default",
        ) as tx,
    ):
        tx.write_bytes(tx.r.file, b"1234")

    assert target.exists() is False
    assert workspace.journal_store.list_batches(run_id="run-write-bytes-reject-default") == []

    with (
        pytest.raises(ValueError, match="max_bytes is 5"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-write-bytes-reject-override",
        ) as tx,
    ):
        tx.write_bytes(tx.r.file, b"123456", max_bytes=5)

    assert workspace.journal_store.list_batches(run_id="run-write-bytes-reject-override") == []


def test_write_bytes_allow_large_opt_in_writes_above_max(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(
        tmp_path / "state.db",
        write_bytes_max_bytes=3,
    )
    target = tmp_path / "payload.bin"

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-write-bytes-allow-large",
    ) as tx:
        result = tx.write_bytes(tx.r.file, b"1234", allow_large=True)

    assert target.read_bytes() == b"1234"
    assert result.batch.payload["large_write_allowed"] is True
    assert result.batch.payload["max_bytes"] == 3


def test_phase_write_bytes_links_batch_to_phase(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "payload.bin"
    resources = workspace.resources({"file": workspace.file(target)})

    with workspace.operation(name="sync", resources=resources, run_id="run-phase-write-bytes") as op:
        with op.phase("apply") as phase:
            file_resource = cast(FileResource, phase.r.file)
            phase.write_bytes(file_resource, b"phase")

    assert target.read_bytes() == b"phase"
    [batch] = workspace.journal_store.list_batches(run_id="run-phase-write-bytes")
    assert batch.idempotency_key == "sync:run-phase-write-bytes:apply:1:write_bytes:file"
    assert batch.operation_run_id == op.operation_run.operation_run_id
    assert batch.operation_phase_id == phase.phase_record.operation_phase_id
