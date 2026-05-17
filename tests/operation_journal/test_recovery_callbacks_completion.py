from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread

import pytest

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemCoordinator,
    JournaledFilesystemRecoveryContext,
    JournaledFilesystemRecoveryResult,
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


def test_recover_batch_allows_only_one_callback_entry_per_batch_lease(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal, clock=_sequence_clock([now]))

    batch = journal.create_batch(
        idempotency_key="recover-race:config",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload={"operation": "write_text"},
        batch_id="batch-1",
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload={"step": 1},
        now=now,
    )
    journal.mark_failed(batch.batch_id, lease=lease, error="write failed", observed_state={"step": 1}, now=now)
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore after failure",
        payload={"step": "desired"},
        recovery_id="recovery-1",
        now=now,
    )

    first_entered = Event()
    release_first = Event()
    callback_count = 0
    callback_lock = Lock()
    thread_result: list[JournaledFilesystemRecoveryResult] = []
    thread_error: list[BaseException] = []

    def blocking_recover(context: JournaledFilesystemRecoveryContext) -> str:
        nonlocal callback_count
        with callback_lock:
            callback_count += 1
            current_count = callback_count
        assert context.recovery_attempt is not None
        assert context.recovery_attempt_id == context.recovery_attempt.recovery_id
        if current_count == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
            return "restored-by-first"
        return "restored-by-second"

    def run_first_recovery() -> None:
        try:
            thread_result.append(
                coordinator.recover_batch(batch.batch_id, lease=lease, recover=blocking_recover, now=now),
            )
        except BaseException as exc:
            thread_error.append(exc)

    worker = Thread(target=run_first_recovery)
    worker.start()
    assert first_entered.wait(timeout=5)

    with pytest.raises(JournaledFilesystemBatchStateError, match="cannot be recovered"):
        coordinator.recover_batch(batch.batch_id, lease=lease, recover=blocking_recover, now=now)

    release_first.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert thread_error == []
    assert callback_count == 1
    assert thread_result[0].batch.phase == BatchPhase.RECOVERY_SUCCEEDED
