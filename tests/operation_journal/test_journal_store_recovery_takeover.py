from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import (
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    OperationJournalStore,
)
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_empty_planned_batch_from_expired_lease_restarts_without_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=first_now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        claim_owner="owner-a",
        batch_id="batch-1",
        now=first_now,
    )
    new_now = first_now + timedelta(seconds=31)
    new_lease = _acquire_lease(state_path, owner="runner-b", now=new_now)

    with pytest.raises(InvalidBatchPhaseTransitionError, match="empty planned"):
        store.record_interrupted_recovery_desired(
            batch.batch_id,
            lease=new_lease,
            resource_key="file:a",
            reason="lost lease before start",
            now=new_now,
        )

    started, operation = store.start_batch_operation(
        batch.batch_id,
        lease=new_lease,
        operation_type="write_text",
        resource_key="file:a",
        operation_id="operation-1",
        now=new_now,
    )

    assert started.phase == BatchPhase.ATTEMPTING
    assert started.lease_fencing_token == new_lease.fencing_token
    assert operation.sequence == 1
    assert store.list_recovery_records(batch.batch_id) == []


def test_newer_lease_can_mark_abandoned_partial_batch_for_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=first_now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        claim_owner="owner-a",
        payload={"operation": "write_text"},
        batch_id="batch-1",
        now=first_now,
    )
    store.append_operation(
        batch.batch_id,
        lease=old_lease,
        operation_type="write_text",
        resource_key="file:a",
        now=first_now,
    )
    store.mark_attempting(batch.batch_id, lease=old_lease, now=first_now)
    new_now = first_now + timedelta(seconds=31)
    new_lease = _acquire_lease(state_path, owner="runner-b", now=new_now)

    recovery = store.record_interrupted_recovery_desired(
        batch.batch_id,
        lease=new_lease,
        resource_key="file:a",
        reason="lost lease during mutation",
        payload={"source": "retry"},
        recovery_id="recovery-1",
        now=new_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.status_message == "lost lease during mutation"
    assert recovery.sequence == 2
    assert store.list_recovery_records(batch.batch_id) == [recovery]


def test_takeover_recovery_can_complete_with_lease_that_recorded_intent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=first_now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        claim_owner="owner-a",
        batch_id="batch-1",
        now=first_now,
    )
    store.append_operation(
        batch.batch_id,
        lease=old_lease,
        operation_type="write_text",
        resource_key="file:a",
        now=first_now,
    )
    store.mark_attempting(batch.batch_id, lease=old_lease, now=first_now)
    takeover_now = first_now + timedelta(seconds=31)
    takeover_lease = _acquire_lease(state_path, owner="runner-b", now=takeover_now)

    store.record_interrupted_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        resource_key="file:a",
        reason="lost lease during mutation",
        now=takeover_now,
    )
    started, recovering = store.start_recovery(
        batch.batch_id,
        lease=takeover_lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=takeover_now,
    )
    succeeded = store.record_recovery_succeeded(
        batch.batch_id,
        lease=takeover_lease,
        recovery_attempt_id=recovering.recovery_id,
        reason="restored",
        recovery_id="recovery-succeeded",
        now=takeover_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert updated.lease_fencing_token == takeover_lease.fencing_token
    assert started.phase == BatchPhase.RECOVERING
    assert recovering.sequence == 3
    assert succeeded.sequence == 4

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.record_recovery_failed(
            batch.batch_id,
            lease=old_lease,
            recovery_attempt_id=recovering.recovery_id,
            reason="stale writer",
            now=takeover_now,
        )


def test_same_lease_cannot_rewind_an_inflight_recovery_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=first_now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        claim_owner="owner-a",
        batch_id="batch-1",
        now=first_now,
    )
    store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        now=first_now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=first_now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", now=first_now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        recovery_id="recovery-desired",
        now=first_now,
    )
    store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=first_now + timedelta(microseconds=1),
    )

    with pytest.raises(InvalidBatchPhaseTransitionError, match="recovering.*recovery_desired"):
        store.record_recovery_desired(
            batch.batch_id,
            lease=lease,
            reason="duplicate retry",
            now=first_now,
        )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERING
    assert updated.lease_fencing_token == lease.fencing_token
    assert [record.phase for record in store.list_recovery_records(batch.batch_id)] == [
        BatchPhase.RECOVERY_DESIRED,
        BatchPhase.RECOVERING,
    ]


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
