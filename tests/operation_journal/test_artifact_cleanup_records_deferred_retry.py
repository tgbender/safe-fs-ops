from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from artifact_cleanup_records_helpers import _acquire_lease, _create_succeeded_batch

from safe_fs_ops.operation_journal import (
    ArtifactCleanupStatus,
    ArtifactCleanupTrigger,
    InvalidBatchPhaseTransitionError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.models import RecoveryRecord
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


@pytest.mark.parametrize(
    ("terminal_status", "record_terminal_status"),
    [
        (
            ArtifactCleanupStatus.FAILED,
            lambda store, *, batch_id, lease, now, artifact_id: store.record_artifact_cleanup_failed(
                batch_id=batch_id,
                lease=lease,
                artifact_id=artifact_id,
                reason="delete failed",
                cleanup_record_id=f"{artifact_id}-terminal",
                now=now,
            ),
        ),
        (
            ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
            lambda store, *, batch_id, lease, now, artifact_id: (
                store.record_artifact_cleanup_manual_intervention_required(
                    batch_id=batch_id,
                    lease=lease,
                    artifact_id=artifact_id,
                    reason="manual cleanup required",
                    cleanup_record_id=f"{artifact_id}-terminal",
                    now=now,
                )
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    ("terminal_trigger", "finish_batch"),
    [
        (
            ArtifactCleanupTrigger.COMMIT_CLEANUP,
            lambda store, *, batch_id, lease, now: store.mark_succeeded(
                batch_id,
                lease=lease,
                now=now,
            ),
        ),
        (
            ArtifactCleanupTrigger.RECOVERY_CLEANUP,
            lambda store, *, batch_id, lease, now: _finish_batch_recovery_succeeded(
                store,
                batch_id=batch_id,
                lease=lease,
                now=now,
            ),
        ),
    ],
)
def test_artifact_cleanup_retry_after_deferred_failure_uses_requested_terminal_trigger(
    tmp_path: Path,
    terminal_status: str,
    record_terminal_status,
    terminal_trigger: str,
    finish_batch,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:batch-1",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now + timedelta(microseconds=1))
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(microseconds=2),
    )
    record_terminal_status(
        store,
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        now=now + timedelta(microseconds=3),
    )
    finish_batch(
        store,
        batch_id=batch.batch_id,
        lease=lease,
        now=now + timedelta(microseconds=4),
    )

    retry = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=terminal_trigger,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(microseconds=5),
    )
    records = store.list_artifact_cleanup_records(batch.batch_id, artifact_id="artifact-1")

    assert [record.status for record in records] == [
        ArtifactCleanupStatus.PLANNED,
        terminal_status,
        ArtifactCleanupStatus.PLANNED,
    ]
    assert [record.trigger for record in records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        terminal_trigger,
    ]
    assert retry.trigger == terminal_trigger


@pytest.mark.parametrize("terminal_status", [ArtifactCleanupStatus.SUCCEEDED, ArtifactCleanupStatus.SKIPPED])
def test_artifact_cleanup_cannot_reopen_succeeded_or_skipped_cleanup_with_new_planned_row(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(microseconds=1),
    )
    if terminal_status == ArtifactCleanupStatus.SUCCEEDED:
        store.record_artifact_cleanup_succeeded(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            cleanup_record_id="cleanup-record-2",
            now=now + timedelta(microseconds=2),
        )
    else:
        store.record_artifact_cleanup_skipped(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            reason="already removed",
            cleanup_record_id="cleanup-record-2",
            now=now + timedelta(microseconds=2),
        )

    with pytest.raises(InvalidBatchPhaseTransitionError, match="cannot transition artifact cleanup"):
        store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
            resource_key="file:a",
            cleanup_record_id="cleanup-record-3",
            now=now + timedelta(microseconds=3),
        )


@pytest.mark.parametrize(
    ("terminal_trigger", "finish_batch"),
    [
        (
            ArtifactCleanupTrigger.COMMIT_CLEANUP,
            lambda store, *, batch_id, lease, now: store.mark_succeeded(
                batch_id,
                lease=lease,
                now=now,
            ),
        ),
        (
            ArtifactCleanupTrigger.RECOVERY_CLEANUP,
            lambda store, *, batch_id, lease, now: _finish_batch_recovery_succeeded(
                store,
                batch_id=batch_id,
                lease=lease,
                now=now,
            ),
        ),
    ],
)
def test_newer_current_lease_can_retry_deferred_cleanup_debt_on_terminal_batch(
    tmp_path: Path,
    terminal_trigger: str,
    finish_batch,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:batch-1",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=old_lease, now=now + timedelta(microseconds=1))
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=old_lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(microseconds=2),
    )
    store.record_artifact_cleanup_failed(
        batch_id=batch.batch_id,
        lease=old_lease,
        artifact_id="artifact-1",
        reason="locked file",
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(microseconds=3),
    )
    finish_batch(
        store,
        batch_id=batch.batch_id,
        lease=old_lease,
        now=now + timedelta(microseconds=4),
    )
    later_now = now + timedelta(seconds=31)
    new_lease = _acquire_lease(state_path, owner="runner-b", now=later_now)

    retry = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=new_lease,
        artifact_id="artifact-1",
        trigger=terminal_trigger,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-3",
        now=later_now,
    )
    resolved = store.record_artifact_cleanup_succeeded(
        batch_id=batch.batch_id,
        lease=new_lease,
        artifact_id="artifact-1",
        cleanup_record_id="cleanup-record-4",
        now=later_now + timedelta(seconds=1),
    )

    assert retry.status == ArtifactCleanupStatus.PLANNED
    assert retry.trigger == terminal_trigger
    assert resolved.status == ArtifactCleanupStatus.SUCCEEDED
    assert store.list_unresolved_artifact_cleanup_debt() == []
    assert store.list_outstanding_artifact_cleanup_records() == []


def _finish_batch_recovery_succeeded(
    store: OperationJournalStore,
    *,
    batch_id: str,
    lease: LeaseRecord,
    now: datetime,
) -> RecoveryRecord:
    store.mark_failed(
        batch_id,
        lease=lease,
        error="write failed",
        now=now,
    )
    store.record_recovery_desired(
        batch_id,
        lease=lease,
        reason="recover",
        recovery_id=f"{batch_id}-recovery-1",
        now=now + timedelta(microseconds=1),
    )
    _updated_batch, recovery_attempt = store.start_recovery(
        batch_id,
        lease=lease,
        reason="attempt recovery",
        recovery_id=f"{batch_id}-recovery-2",
        now=now + timedelta(microseconds=2),
    )
    return store.record_recovery_succeeded(
        batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        reason="recovered",
        recovery_id=f"{batch_id}-recovery-3",
        now=now + timedelta(microseconds=3),
    )
