from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier, Lock, Thread

import pytest

from safe_fs_ops.operation_journal import (
    BatchIdempotencyMismatchError,
    BatchPhase,
    BatchStartConflictError,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
    OperationJournalStore,
)
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_resource_bound_batch_rejects_operation_for_different_resource(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        batch_id="batch-1",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="different resource"):
        store.append_operation(
            batch.batch_id,
            lease=lease,
            operation_type="write_text",
            resource_key="file:b",
            now=now,
        )

    assert store.list_operations(batch.batch_id) == []


def test_resource_bound_batch_rejects_checkpoint_without_operation_for_different_resource(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        batch_id="batch-1",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="different resource"):
        store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=None,
            resource_key="file:b",
            checkpoint_type="failure",
            now=now,
        )

    assert store.list_checkpoints(batch.batch_id) == []


def test_same_idempotency_concurrent_starts_record_one_operation(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    barrier = Barrier(2)
    lock = Lock()
    successes: list[str] = []
    conflicts: list[str] = []
    unexpected: list[BaseException] = []

    def start(index: int) -> None:
        try:
            barrier.wait()
            local_store = OperationJournalStore(state_path)
            batch = local_store.create_batch(
                idempotency_key="apply:file:a",
                lease=lease,
                owner="owner-a",
                run_id="run-1",
                resource_key="file:a",
                claim_owner="owner-a",
                payload={"operation": "write_text"},
                batch_id=f"batch-{index}",
                now=now,
            )
            _started, operation = local_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type="write_text",
                resource_key="file:a",
                payload={"operation": "write_text"},
                operation_id=f"operation-{index}",
                now=now,
            )
        except BatchStartConflictError as exc:
            with lock:
                conflicts.append(str(exc))
        except BaseException as exc:
            with lock:
                unexpected.append(exc)
        else:
            with lock:
                successes.append(operation.operation_id)

    threads = [Thread(target=start, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    store = OperationJournalStore(state_path)
    batches = store.list_batches()
    assert unexpected == []
    assert len(batches) == 1
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert len(store.list_operations(batches[0].batch_id)) == 1


def test_recovery_phase_records_update_batch_and_list_deterministically(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    with pytest.raises(InvalidBatchPhaseTransitionError, match="planned.*recovery_desired"):
        store.record_recovery_desired(batch.batch_id, lease=lease, reason="no failure yet", now=now)

    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    failed = store.mark_failed(
        batch.batch_id,
        lease=lease,
        error="write failed",
        observed_state={"path": "a", "exists": False},
        now=now,
    )
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        payload={"backup": "backup/a"},
        recovery_id="recovery-1",
        now=now,
    )
    started, recovering = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=now + timedelta(microseconds=1),
    )
    recovery_failed = store.record_recovery_failed(
        batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovering.recovery_id,
        reason="backup missing",
        recovery_id="recovery-2",
        now=now + timedelta(seconds=1),
    )

    assert failed.phase == BatchPhase.FAILED
    assert failed.status_message == "write failed"
    assert failed.status_payload["exists"] is False
    assert desired.sequence == 1
    assert started.phase == BatchPhase.RECOVERING
    assert recovering.sequence == 2
    assert recovery_failed.sequence == 3
    assert store.get_batch(batch.batch_id).phase == BatchPhase.RECOVERY_FAILED  # type: ignore[union-attr]
    assert [record.recovery_id for record in store.list_recovery_records(batch.batch_id)] == [
        "recovery-1",
        "recovery-started",
        "recovery-2",
    ]


def test_lease_lost_recovery_marker_rejects_forged_lease_record(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)
    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    lost_now = now + timedelta(seconds=31)
    LeaseStore(state_path).acquire("workspace", owner="runner-b", ttl=timedelta(seconds=30), now=lost_now)
    forged = LeaseRecord(
        name=lease.name,
        owner=lease.owner,
        token="forged-token",
        fencing_token=lease.fencing_token,
        acquired_at=lease.acquired_at,
        heartbeat_at=lease.heartbeat_at,
        expires_at=lease.expires_at,
        acquired=True,
    )

    with pytest.raises(JournalLeaseMismatchError, match="authority does not match"):
        store.record_lease_lost_recovery_desired(
            batch.batch_id,
            lease=forged,
            reason="forged marker",
            now=lost_now,
        )

    assert store.get_batch(batch.batch_id).phase == BatchPhase.ATTEMPTING  # type: ignore[union-attr]
    assert store.list_recovery_records(batch.batch_id) == []


def test_lease_lost_recovery_marker_accepts_original_lost_lease_authority(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)
    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    lost_now = now + timedelta(seconds=31)
    LeaseStore(state_path).acquire("workspace", owner="runner-b", ttl=timedelta(seconds=30), now=lost_now)

    recovery = store.record_lease_lost_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="original lease lost",
        recovery_id="recovery-1",
        now=lost_now,
    )

    assert recovery.recovery_id == "recovery-1"
    assert store.get_batch(batch.batch_id).phase == BatchPhase.RECOVERY_DESIRED  # type: ignore[union-attr]
    assert [record.reason for record in store.list_recovery_records(batch.batch_id)] == ["original lease lost"]


def _acquire_lease(
    state_path: Path,
    *,
    now: datetime,
    owner: str = "runner-a",
    name: str = "workspace",
) -> LeaseRecord:
    return LeaseStore(state_path).acquire(name, owner=owner, ttl=timedelta(seconds=30), now=now)


def _create_batch(store: OperationJournalStore, *, lease: LeaseRecord, now: datetime):
    return store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )


def _start_recovery_attempt(
    store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    now: datetime,
):
    batch = _create_batch(store, lease=lease, now=now)
    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", now=now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        recovery_id="recovery-1",
        now=now,
    )
    return store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-2",
        now=now + timedelta(microseconds=1),
    )
