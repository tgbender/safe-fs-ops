from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemCoordinator,
    JournaledFilesystemRecoveryContext,
    OperationJournalStore,
    file_resource_key,
)
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def _coordinator(
    state_path: Path,
    *,
    journal: OperationJournalStore | None = None,
    snapshot=None,
    write_text_operation=None,
    delete_file_operation=None,
    clock=None,
) -> JournaledFilesystemCoordinator:
    kwargs = {
        "snapshot": _portable_snapshot,
        "write_text_operation": _portable_write_text,
        "delete_file_operation": _portable_delete_file,
    }
    if snapshot is not None:
        kwargs["snapshot"] = snapshot
    if write_text_operation is not None:
        kwargs["write_text_operation"] = write_text_operation
    if delete_file_operation is not None:
        kwargs["delete_file_operation"] = delete_file_operation
    return JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal or OperationJournalStore(state_path),
        clock=clock,
        **kwargs,
    )


def _portable_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    Path(path).write_text(content, encoding=encoding, newline=newline)


def _portable_delete_file(path: Path | str, *, missing_ok: bool = False) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def _portable_snapshot(path: Path | str) -> ResourceSnapshot:
    target = Path(path)
    if target.is_symlink():
        stat_result = target.lstat()
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="symlink",
            content_hash=None,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=os.readlink(target),
        )
    if not target.exists():
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
        )
    stat_result = target.lstat()
    if target.is_file():
        content_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        file_type = "file"
    elif target.is_dir():
        content_hash = None
        file_type = "directory"
    else:
        content_hash = None
        file_type = "other"
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type=file_type,
        content_hash=content_hash,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
    )


def _acquire_and_claim(state_path: Path, resource_key: str, *, now: datetime) -> LeaseRecord:
    lease = LeaseStore(state_path).acquire("workspace", owner="runner-a", ttl=timedelta(seconds=30), now=now)
    ClaimStore(state_path).upsert(resource_key, lease=lease, owner="owner-a", now=now)
    return lease


def _sequence_clock(values: list[datetime]):
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        if calls >= len(values):
            return values[-1]
        value = values[calls]
        calls += 1
        return value

    return clock


def test_recover_batch_passes_deterministic_context_to_callback(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal, clock=_sequence_clock([now]))

    batch = journal.create_batch(
        idempotency_key="recover:config",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload={"operation": "write_text"},
        batch_id="batch-1",
        now=now,
    )
    op1 = journal.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload={"step": 1},
        operation_id="operation-1",
        now=now,
    )
    journal.mark_attempting(batch.batch_id, lease=lease, now=now)
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        resource_key=resource_key,
        checkpoint_type="before",
        payload={"step": 1},
        checkpoint_id="checkpoint-1",
        operation_id=op1.operation_id,
        now=now,
    )
    op2 = journal.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload={"step": 2},
        operation_id="operation-2",
        now=now,
    )
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        resource_key=resource_key,
        checkpoint_type="after",
        payload={"step": 2},
        checkpoint_id="checkpoint-2",
        operation_id=op2.operation_id,
        now=now,
    )
    journal.mark_failed(batch.batch_id, lease=lease, error="write failed", observed_state={"step": 2}, now=now)
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="retry recovery",
        payload={"step": "desired-1"},
        recovery_id="recovery-1",
        now=now,
    )
    _started, first_recovering = journal.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-started-1",
        now=now + timedelta(microseconds=1),
    )
    journal.record_recovery_failed(
        batch.batch_id,
        lease=lease,
        recovery_attempt_id=first_recovering.recovery_id,
        reason="first recovery failed",
        payload={"step": "failed-1"},
        recovery_id="recovery-2",
        now=now,
    )
    takeover_now = now + timedelta(seconds=31)
    takeover_lease = LeaseStore(state_path).acquire(
        "workspace", owner="runner-b", ttl=timedelta(seconds=30), now=takeover_now
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        reason="retry recovery",
        payload={"step": "desired-2"},
        recovery_id="recovery-3",
        now=takeover_now,
    )

    seen_contexts: list[JournaledFilesystemRecoveryContext] = []

    def capture_context(context: JournaledFilesystemRecoveryContext) -> None:
        seen_contexts.append(context)

    result = coordinator.recover_batch(batch.batch_id, lease=takeover_lease, recover=capture_context, now=takeover_now)

    assert len(seen_contexts) == 1
    context = seen_contexts[0]
    assert context.batch.phase == BatchPhase.RECOVERING
    assert [operation.operation_id for operation in context.operations] == ["operation-1", "operation-2"]
    assert [checkpoint.checkpoint_id for checkpoint in context.checkpoints] == ["checkpoint-1", "checkpoint-2"]
    assert len(context.recovery_records) == 6
    assert [recovery_record.recovery_id for recovery_record in context.recovery_records[:2]] == [
        "recovery-1",
        "recovery-started-1",
    ]
    assert context.recovery_records[2].recovery_id == "recovery-2"
    assert context.recovery_records[2].phase == BatchPhase.RECOVERY_FAILED
    assert context.recovery_records[3].reason == "recovery takeover"
    assert context.recovery_records[4].phase == BatchPhase.RECOVERY_DESIRED
    assert context.recovery_records[4].recovery_id == "recovery-3"
    assert context.recovery_records[5].phase == BatchPhase.RECOVERING
    assert result.recovery_record.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert [record.phase for record in journal.list_recovery_records(batch.batch_id)] == [
        BatchPhase.RECOVERY_DESIRED,
        BatchPhase.RECOVERING,
        BatchPhase.RECOVERY_FAILED,
        BatchPhase.RECOVERY_DESIRED,
        BatchPhase.RECOVERY_DESIRED,
        BatchPhase.RECOVERING,
        BatchPhase.RECOVERY_SUCCEEDED,
    ]
    assert result.context == context
    assert result.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
