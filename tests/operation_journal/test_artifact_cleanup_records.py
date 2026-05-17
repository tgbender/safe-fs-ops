from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from artifact_cleanup_records_helpers import (
    _acquire_lease,
    _create_succeeded_batch,
)

from safe_fs_ops.operation_journal import (
    ArtifactCleanupStatus,
    ArtifactCleanupTrigger,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
    OperationJournalStore,
)
from safe_fs_ops.workspace_state import LeaseLostError

pytestmark = pytest.mark.safe_fs_ops


def test_artifact_cleanup_retry_appends_new_planned_row_after_failed_or_manual(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-failed",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=1),
    )
    store.record_artifact_cleanup_failed(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-failed",
        reason="delete failed",
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=2),
    )
    retry_failed = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-failed",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(seconds=3),
    )
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-manual",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-4",
        now=now + timedelta(seconds=4),
    )
    store.record_artifact_cleanup_manual_intervention_required(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-manual",
        reason="path escaped root",
        cleanup_record_id="cleanup-record-5",
        now=now + timedelta(seconds=5),
    )
    retry_manual = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-manual",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-6",
        now=now + timedelta(seconds=6),
    )

    failed_records = store.list_artifact_cleanup_records(batch.batch_id, artifact_id="artifact-failed")
    manual_records = store.list_artifact_cleanup_records(batch.batch_id, artifact_id="artifact-manual")

    assert [record.status for record in failed_records] == [
        ArtifactCleanupStatus.PLANNED,
        ArtifactCleanupStatus.FAILED,
        ArtifactCleanupStatus.PLANNED,
    ]
    assert retry_failed.sequence > failed_records[1].sequence
    assert [record.status for record in manual_records] == [
        ArtifactCleanupStatus.PLANNED,
        ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
        ArtifactCleanupStatus.PLANNED,
    ]
    assert retry_manual.sequence > manual_records[1].sequence


def test_artifact_cleanup_does_not_duplicate_active_work(tmp_path: Path) -> None:
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
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(InvalidBatchPhaseTransitionError, match="active work"):
        store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
            resource_key="file:a",
            cleanup_record_id="cleanup-record-2",
            now=now + timedelta(seconds=2),
        )


def test_stale_or_noncurrent_lease_cannot_append_artifact_cleanup_rows(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=old_lease, now=now)
    later_now = now + timedelta(seconds=31)
    _new_lease = _acquire_lease(state_path, owner="runner-b", now=later_now)

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=old_lease,
            artifact_id="artifact-1",
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
            resource_key="file:a",
            cleanup_record_id="cleanup-record-1",
            now=later_now,
        )

    mismatched_name_lease = _acquire_lease(
        state_path,
        name="other-workspace",
        owner="runner-c",
        now=later_now + timedelta(seconds=1),
    )
    with pytest.raises(JournalLeaseMismatchError, match="different lease"):
        store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=mismatched_name_lease,
            artifact_id="artifact-2",
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
            resource_key="file:a",
            cleanup_record_id="cleanup-record-2",
            now=later_now + timedelta(seconds=1),
        )


def test_artifact_cleanup_rejects_invalid_trigger_and_status(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)

    with pytest.raises(ValueError, match="unsupported artifact cleanup trigger"):
        store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            trigger="invalid-trigger",
            cleanup_record_id="cleanup-record-1",
            now=now + timedelta(seconds=1),
        )

    with pytest.raises(ValueError, match="unsupported artifact cleanup status"):
        store._record_artifact_cleanup(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id="artifact-1",
            status="invalid-status",
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
            resource_key=None,
            reason=None,
            payload=None,
            cleanup_record_id="cleanup-record-2",
            now=now + timedelta(seconds=2),
        )


def test_artifact_cleanup_debt_and_outstanding_queries_have_distinct_semantics(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)
    planned = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-planned",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=1),
    )
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-attempting",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=2),
    )
    attempting = store.mark_artifact_cleanup_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-attempting",
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(seconds=3),
    )
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-failed",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        cleanup_record_id="cleanup-record-4",
        now=now + timedelta(seconds=4),
    )
    failed = store.record_artifact_cleanup_failed(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-failed",
        reason="delete failed",
        cleanup_record_id="cleanup-record-5",
        now=now + timedelta(seconds=5),
    )

    assert store.list_unresolved_artifact_cleanup_debt() == [failed]
    assert store.list_outstanding_artifact_cleanup_records() == [planned, attempting, failed]
