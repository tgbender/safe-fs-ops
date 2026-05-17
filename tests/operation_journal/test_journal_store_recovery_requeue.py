from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import (
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    OperationJournalStore,
    RecoveryAttemptMismatchError,
)
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_newer_lease_can_requeue_abandoned_recovering_batch(tmp_path: Path) -> None:
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
    store.mark_failed(batch.batch_id, lease=old_lease, error="write failed", now=first_now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=old_lease,
        reason="restore backup",
        recovery_id="recovery-desired",
        now=first_now,
    )
    store.start_recovery(
        batch.batch_id,
        lease=old_lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=first_now + timedelta(microseconds=1),
    )

    takeover_now = first_now + timedelta(seconds=31)
    takeover_lease = _acquire_lease(state_path, owner="runner-b", now=takeover_now)
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        reason="retry recovery",
        recovery_id="recovery-takeover",
        now=takeover_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.lease_fencing_token == takeover_lease.fencing_token
    assert desired.sequence == 5

    recovery_records = store.list_recovery_records(batch.batch_id)
    handoff = recovery_records[2]
    assert [record.recovery_id for record in recovery_records] == [
        "recovery-desired",
        "recovery-started",
        handoff.recovery_id,
        "recovery-takeover",
    ]
    assert handoff.phase == BatchPhase.RECOVERY_DESIRED
    assert handoff.reason == "recovery takeover"
    assert handoff.sequence == 4
    assert handoff.payload["previous_phase"] == BatchPhase.RECOVERING
    assert handoff.payload["previous_lease"]["fencing_token"] == old_lease.fencing_token
    assert handoff.payload["current_lease"]["fencing_token"] == takeover_lease.fencing_token


def test_newer_lease_rejects_stale_recovery_attempt_after_requeue(tmp_path: Path) -> None:
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
    store.mark_failed(batch.batch_id, lease=old_lease, error="write failed", now=first_now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=old_lease,
        reason="restore backup",
        recovery_id="recovery-desired",
        now=first_now,
    )
    store.start_recovery(
        batch.batch_id,
        lease=old_lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=first_now + timedelta(microseconds=1),
    )

    takeover_now = first_now + timedelta(seconds=31)
    takeover_lease = _acquire_lease(state_path, owner="runner-b", now=takeover_now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        reason="retry recovery",
        recovery_id="recovery-takeover",
        now=takeover_now,
    )

    with pytest.raises((RecoveryAttemptMismatchError, InvalidBatchPhaseTransitionError)):
        store.record_recovery_succeeded(
            batch.batch_id,
            lease=takeover_lease,
            recovery_attempt_id="recovery-started",
            reason="stale completion",
            recovery_id="recovery-succeeded",
            now=takeover_now + timedelta(seconds=1),
        )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.lease_fencing_token == takeover_lease.fencing_token
    recovery_records = store.list_recovery_records(batch.batch_id)
    handoff = recovery_records[2]
    assert [record.recovery_id for record in recovery_records] == [
        "recovery-desired",
        "recovery-started",
        handoff.recovery_id,
        "recovery-takeover",
    ]
    assert handoff.phase == BatchPhase.RECOVERY_DESIRED
    assert handoff.reason == "recovery takeover"


def test_newer_lease_can_complete_abandoned_recovery_desired_batch(tmp_path: Path) -> None:
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

    desired = store.record_interrupted_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        resource_key="file:a",
        reason="lost lease during mutation",
        payload={"source": "retry"},
        recovery_id="recovery-desired",
        now=takeover_now,
    )
    final_now = takeover_now + timedelta(seconds=31)
    final_lease = _acquire_lease(state_path, owner="runner-c", now=final_now)
    started, recovering = store.start_recovery(
        batch.batch_id,
        lease=final_lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=final_now,
    )

    succeeded = store.record_recovery_succeeded(
        batch.batch_id,
        lease=final_lease,
        recovery_attempt_id=recovering.recovery_id,
        reason="restored",
        payload={"restored": True},
        recovery_id="recovery-succeeded",
        now=final_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert updated.lease_fencing_token == final_lease.fencing_token
    assert desired.sequence == 2
    assert started.phase == BatchPhase.RECOVERING
    assert recovering.sequence == 4
    assert succeeded.sequence == 5

    recovery_records = store.list_recovery_records(batch.batch_id)
    handoff = recovery_records[1]
    assert [record.recovery_id for record in recovery_records] == [
        "recovery-desired",
        handoff.recovery_id,
        "recovery-started",
        "recovery-succeeded",
    ]
    assert handoff.phase == BatchPhase.RECOVERY_DESIRED
    assert handoff.reason == "recovery takeover"
    assert handoff.sequence == 3
    assert handoff.payload["event_type"] == "recovery_takeover"
    assert handoff.payload["previous_lease"]["fencing_token"] == takeover_lease.fencing_token
    assert handoff.payload["current_lease"]["fencing_token"] == final_lease.fencing_token

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.record_recovery_failed(
            batch.batch_id,
            lease=takeover_lease,
            recovery_attempt_id=recovering.recovery_id,
            reason="stale writer",
            now=final_now,
        )


def test_newer_lease_can_retry_failed_batch_for_recovery(tmp_path: Path) -> None:
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
    store.mark_failed(batch.batch_id, lease=old_lease, error="write failed", now=first_now)

    retry_now = first_now + timedelta(seconds=31)
    retry_lease = _acquire_lease(state_path, owner="runner-b", now=retry_now)
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=retry_lease,
        reason="retry recovery",
        recovery_id="recovery-retry",
        now=retry_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.lease_fencing_token == retry_lease.fencing_token
    assert desired.sequence == 3

    recovery_records = store.list_recovery_records(batch.batch_id)
    handoff = recovery_records[0]
    assert [record.recovery_id for record in recovery_records] == [
        handoff.recovery_id,
        "recovery-retry",
    ]
    assert handoff.reason == "recovery takeover"
    assert handoff.payload["previous_phase"] == BatchPhase.FAILED
    assert handoff.payload["previous_lease"]["fencing_token"] == old_lease.fencing_token
    assert handoff.payload["current_lease"]["fencing_token"] == retry_lease.fencing_token

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.record_recovery_desired(
            batch.batch_id,
            lease=old_lease,
            reason="stale retry",
            now=retry_now,
        )


def test_newer_lease_can_retry_recovery_failed_batch(tmp_path: Path) -> None:
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
    store.mark_attempting(batch.batch_id, lease=old_lease, now=first_now)
    store.mark_failed(batch.batch_id, lease=old_lease, error="write failed", now=first_now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=old_lease,
        reason="restore backup",
        recovery_id="recovery-1",
        now=first_now,
    )
    _started, recovering = store.start_recovery(
        batch.batch_id,
        lease=old_lease,
        reason="recovery reserved",
        recovery_id="recovery-started",
        now=first_now + timedelta(microseconds=1),
    )
    store.record_recovery_failed(
        batch.batch_id,
        lease=old_lease,
        recovery_attempt_id=recovering.recovery_id,
        reason="backup missing",
        recovery_id="recovery-2",
        now=first_now,
    )

    retry_now = first_now + timedelta(seconds=31)
    retry_lease = _acquire_lease(state_path, owner="runner-b", now=retry_now)
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=retry_lease,
        reason="retry recovery",
        recovery_id="recovery-3",
        now=retry_now,
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.lease_fencing_token == retry_lease.fencing_token
    assert desired.sequence == 5

    recovery_records = store.list_recovery_records(batch.batch_id)
    handoff = recovery_records[3]
    assert [record.recovery_id for record in recovery_records] == [
        "recovery-1",
        "recovery-started",
        "recovery-2",
        handoff.recovery_id,
        "recovery-3",
    ]
    assert handoff.reason == "recovery takeover"
    assert handoff.payload["previous_phase"] == BatchPhase.RECOVERY_FAILED
    assert handoff.payload["previous_lease"]["fencing_token"] == old_lease.fencing_token
    assert handoff.payload["current_lease"]["fencing_token"] == retry_lease.fencing_token

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.record_recovery_desired(
            batch.batch_id,
            lease=old_lease,
            reason="stale retry",
            now=retry_now,
        )


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
