from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import (
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    OperationJournalStore,
    RecoveryActionStatus,
    RecoveryAttemptMismatchError,
)
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


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


def test_recovery_action_flow_records_monotonic_status_sequence(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    _batch, recovery_attempt = _start_recovery_attempt(store, lease=lease, now=now)

    planned = store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="restore_backup",
        resource_key="file:a",
        payload={"backup_path": "backup/a"},
        action_id="action-1",
        action_record_id="action-record-1",
        now=now + timedelta(seconds=1),
    )
    attempting = store.mark_recovery_action_attempting(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-1",
        payload={"worker": "runner-a"},
        action_record_id="action-record-2",
        now=now + timedelta(seconds=2),
    )
    succeeded = store.record_recovery_action_succeeded(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-1",
        payload={"restored": True},
        action_record_id="action-record-3",
        now=now + timedelta(seconds=3),
    )

    assert [planned.sequence, attempting.sequence, succeeded.sequence] == [3, 4, 5]
    assert [planned.status, attempting.status, succeeded.status] == [
        RecoveryActionStatus.PLANNED,
        RecoveryActionStatus.ATTEMPTING,
        RecoveryActionStatus.SUCCEEDED,
    ]
    assert [record.action_record_id for record in store.list_recovery_actions("batch-1")] == [
        "action-record-1",
        "action-record-2",
        "action-record-3",
    ]
    assert [
        record.action_record_id for record in store.list_recovery_actions("batch-1", recovery_attempt_id="recovery-2")
    ] == [
        "action-record-1",
        "action-record-2",
        "action-record-3",
    ]


def test_recovery_action_failure_and_non_runner_terminal_states_are_recorded(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    _batch, recovery_attempt = _start_recovery_attempt(store, lease=lease, now=now)

    store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-1",
        now=now + timedelta(seconds=1),
    )
    store.mark_recovery_action_attempting(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-1",
        now=now + timedelta(seconds=2),
    )
    failed = store.record_recovery_action_failed(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-1",
        reason="backup missing",
        payload={"backup_path": "backup/a"},
        action_record_id="action-record-3",
        now=now + timedelta(seconds=3),
    )
    skipped_planned = store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="cleanup_temp",
        action_id="action-2",
        now=now + timedelta(seconds=4),
    )
    skipped = store.record_recovery_action_skipped(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-2",
        reason="already clean",
        action_record_id="action-record-5",
        now=now + timedelta(seconds=5),
    )
    manual_planned = store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="restore_permissions",
        action_id="action-3",
        now=now + timedelta(seconds=6),
    )
    manual = store.record_recovery_action_manual_intervention_required(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_id="action-3",
        reason="chmod unsupported",
        payload={"path": "file:a"},
        action_record_id="action-record-7",
        now=now + timedelta(seconds=7),
    )

    assert failed.status == RecoveryActionStatus.FAILED
    assert failed.reason == "backup missing"
    assert skipped_planned.status == RecoveryActionStatus.PLANNED
    assert skipped.status == RecoveryActionStatus.SKIPPED
    assert manual_planned.status == RecoveryActionStatus.PLANNED
    assert manual.status == RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED
    assert [record.status for record in store.list_recovery_actions("batch-1")] == [
        RecoveryActionStatus.PLANNED,
        RecoveryActionStatus.ATTEMPTING,
        RecoveryActionStatus.FAILED,
        RecoveryActionStatus.PLANNED,
        RecoveryActionStatus.SKIPPED,
        RecoveryActionStatus.PLANNED,
        RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED,
    ]


def test_recovery_action_writes_reject_missing_wrong_and_stale_attempt_ids(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    _batch, recovery_attempt = _start_recovery_attempt(store, lease=lease, now=now)

    with pytest.raises(RecoveryAttemptMismatchError, match="require the active recovery_attempt_id"):
        store.record_recovery_action_planned(
            batch_id="batch-1",
            lease=lease,
            recovery_attempt_id="",
            action_type="restore_backup",
            now=now + timedelta(seconds=1),
        )

    with pytest.raises(RecoveryAttemptMismatchError, match="is not recoverable"):
        store.record_recovery_action_planned(
            batch_id="batch-1",
            lease=lease,
            recovery_attempt_id="recovery-1",
            action_type="restore_backup",
            now=now + timedelta(seconds=1),
        )

    store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-1",
        now=now + timedelta(seconds=2),
    )
    store.record_recovery_failed(
        "batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        reason="backup missing",
        recovery_id="recovery-3",
        now=now + timedelta(seconds=3),
    )
    store.record_recovery_desired(
        "batch-1",
        lease=lease,
        reason="retry recovery",
        recovery_id="recovery-4",
        now=now + timedelta(seconds=4),
    )
    _retry_batch, retry_attempt = store.start_recovery(
        "batch-1",
        lease=lease,
        reason="retry reserved",
        recovery_id="recovery-5",
        now=now + timedelta(seconds=4, microseconds=1),
    )

    with pytest.raises(RecoveryAttemptMismatchError, match="is not the active recovery attempt"):
        store.mark_recovery_action_attempting(
            batch_id="batch-1",
            lease=lease,
            recovery_attempt_id=recovery_attempt.recovery_id,
            action_id="action-1",
            now=now + timedelta(seconds=5),
        )

    planned = store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=retry_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-2",
        now=now + timedelta(seconds=6),
    )
    assert planned.recovery_attempt_id == retry_attempt.recovery_id


def test_recovery_actions_reject_writes_after_recovery_completion(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    _batch, recovery_attempt = _start_recovery_attempt(store, lease=lease, now=now)
    store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-1",
        now=now + timedelta(seconds=1),
    )
    store.record_recovery_succeeded(
        "batch-1",
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        reason="restored",
        recovery_id="recovery-3",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(InvalidBatchPhaseTransitionError, match="cannot append recovery actions.*recovery_succeeded"):
        store.mark_recovery_action_attempting(
            batch_id="batch-1",
            lease=lease,
            recovery_attempt_id=recovery_attempt.recovery_id,
            action_id="action-1",
            now=now + timedelta(seconds=3),
        )


def test_recovery_action_write_rejects_non_recovering_takeover_without_side_effects(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=first_now)
    batch = _create_batch(store, lease=old_lease, now=first_now)
    store.mark_attempting(batch.batch_id, lease=old_lease, now=first_now)
    store.mark_failed(batch.batch_id, lease=old_lease, error="write failed", now=first_now)
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=old_lease,
        reason="restore backup",
        recovery_id="recovery-1",
        now=first_now,
    )

    takeover_now = first_now + timedelta(seconds=31)
    takeover_lease = _acquire_lease(state_path, owner="runner-b", now=takeover_now)

    with pytest.raises(InvalidBatchPhaseTransitionError, match="cannot append recovery actions.*recovery_desired"):
        store.record_recovery_action_planned(
            batch_id=batch.batch_id,
            lease=takeover_lease,
            recovery_attempt_id="recovery-1",
            action_type="restore_backup",
            action_id="action-1",
            now=takeover_now,
        )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert updated.lease_fencing_token == old_lease.fencing_token
    assert store.list_recovery_records(batch.batch_id) == [desired]
    assert store.list_recovery_actions(batch.batch_id) == []


def test_recovery_takeover_invalidates_old_attempt_action_writes(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=first_now)
    _batch, old_attempt = _start_recovery_attempt(store, lease=old_lease, now=first_now)
    store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=old_lease,
        recovery_attempt_id=old_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-1",
        now=first_now + timedelta(seconds=1),
    )
    takeover_now = first_now + timedelta(seconds=31)
    takeover_lease = _acquire_lease(state_path, owner="runner-b", now=takeover_now)
    store.record_recovery_desired(
        "batch-1",
        lease=takeover_lease,
        reason="retry recovery",
        recovery_id="recovery-3",
        now=takeover_now,
    )
    _updated_batch, new_attempt = store.start_recovery(
        "batch-1",
        lease=takeover_lease,
        reason="retry reserved",
        recovery_id="recovery-4",
        now=takeover_now + timedelta(microseconds=1),
    )

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.mark_recovery_action_attempting(
            batch_id="batch-1",
            lease=old_lease,
            recovery_attempt_id=old_attempt.recovery_id,
            action_id="action-1",
            now=takeover_now + timedelta(seconds=1),
        )

    planned = store.record_recovery_action_planned(
        batch_id="batch-1",
        lease=takeover_lease,
        recovery_attempt_id=new_attempt.recovery_id,
        action_type="restore_backup",
        action_id="action-2",
        action_record_id="action-record-2",
        now=takeover_now + timedelta(seconds=1),
    )

    assert planned.recovery_attempt_id == "recovery-4"
    assert [record.recovery_attempt_id for record in store.list_recovery_actions("batch-1")] == [
        old_attempt.recovery_id,
        new_attempt.recovery_id,
    ]
