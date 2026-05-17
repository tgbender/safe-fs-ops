from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from artifact_cleanup_records_helpers import (
    _acquire_lease,
    _create_recovery_succeeded_batch,
    _create_succeeded_batch,
)

from safe_fs_ops.operation_journal import (
    ArtifactCleanupStatus,
    ArtifactCleanupTrigger,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _ARTIFACT_CLEANUP_TABLE,
    _ArtifactCleanupColumn,
)

pytestmark = pytest.mark.safe_fs_ops


def test_initialize_creates_artifact_cleanup_table(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)

    store.initialize()

    with sqlite3.connect(state_path) as connection:
        column_names = {
            row[1] for row in connection.execute(f"PRAGMA table_info({_ARTIFACT_CLEANUP_TABLE})").fetchall()
        }

    assert column_names == {
        _ArtifactCleanupColumn.CLEANUP_RECORD_ID,
        _ArtifactCleanupColumn.ARTIFACT_ID,
        _ArtifactCleanupColumn.BATCH_ID,
        _ArtifactCleanupColumn.SEQUENCE,
        _ArtifactCleanupColumn.TRIGGER,
        _ArtifactCleanupColumn.STATUS,
        _ArtifactCleanupColumn.RESOURCE_KEY,
        _ArtifactCleanupColumn.REASON,
        _ArtifactCleanupColumn.PAYLOAD,
        _ArtifactCleanupColumn.CREATED_AT,
    }


def test_artifact_cleanup_records_are_append_only_and_filterable_by_artifact(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)

    planned = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        payload={"content_path": "artifacts/a.bin"},
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=1),
    )
    attempting = store.mark_artifact_cleanup_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        payload={"worker": "runner-a"},
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=2),
    )
    succeeded = store.record_artifact_cleanup_succeeded(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        payload={"deleted": True},
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(seconds=3),
    )
    second_planned = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-2",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        payload={"content_path": "artifacts/b.bin"},
        cleanup_record_id="cleanup-record-4",
        now=now + timedelta(seconds=4),
    )

    assert [record.status for record in store.list_artifact_cleanup_records(batch.batch_id)] == [
        ArtifactCleanupStatus.PLANNED,
        ArtifactCleanupStatus.ATTEMPTING,
        ArtifactCleanupStatus.SUCCEEDED,
        ArtifactCleanupStatus.PLANNED,
    ]
    assert store.list_artifact_cleanup_records(batch.batch_id, artifact_id="artifact-1") == [
        planned,
        attempting,
        succeeded,
    ]
    assert store.list_artifact_cleanup_records(batch.batch_id, artifact_id="artifact-2") == [second_planned]


def test_artifact_cleanup_records_share_batch_sequence_with_other_journal_rows(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )

    operation = store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        operation_id="operation-1",
        now=now + timedelta(seconds=1),
    )
    checkpoint = store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key="file:a",
        checkpoint_type="backup",
        checkpoint_id="checkpoint-1",
        now=now + timedelta(seconds=2),
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now + timedelta(seconds=3))
    store.mark_succeeded(batch.batch_id, lease=lease, now=now + timedelta(seconds=4))
    cleanup = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=5),
    )

    assert operation.sequence == 1
    assert checkpoint.sequence == 2
    assert cleanup.sequence == 3


def test_deferred_artifact_cleanup_plan_can_be_recorded_before_terminal_success(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now + timedelta(seconds=1))

    planned = store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=2),
    )
    store.mark_succeeded(batch.batch_id, lease=lease, now=now + timedelta(seconds=3))
    attempting = store.mark_artifact_cleanup_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=4),
    )
    succeeded = store.record_artifact_cleanup_succeeded(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-1",
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(seconds=5),
    )

    assert [record.status for record in store.list_artifact_cleanup_records(batch.batch_id)] == [
        ArtifactCleanupStatus.PLANNED,
        ArtifactCleanupStatus.ATTEMPTING,
        ArtifactCleanupStatus.SUCCEEDED,
    ]
    assert planned.trigger == ArtifactCleanupTrigger.DEFERRED_CLEANUP
    assert attempting.trigger == ArtifactCleanupTrigger.DEFERRED_CLEANUP
    assert succeeded.trigger == ArtifactCleanupTrigger.DEFERRED_CLEANUP


def test_list_unresolved_artifact_cleanup_debt_returns_latest_unresolved_per_artifact(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    first_batch = _create_succeeded_batch(store, lease=lease, now=now, batch_id="batch-1")
    store.record_artifact_cleanup_planned(
        batch_id=first_batch.batch_id,
        lease=lease,
        artifact_id="artifact-a",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key="file:a",
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=1),
    )
    failed = store.record_artifact_cleanup_failed(
        batch_id=first_batch.batch_id,
        lease=lease,
        artifact_id="artifact-a",
        reason="artifact missing",
        payload={"content_path": "artifacts/a.bin"},
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=2),
    )
    later_lease = _acquire_lease(state_path, now=now + timedelta(minutes=1))
    second_batch = _create_recovery_succeeded_batch(
        store,
        lease=later_lease,
        now=now + timedelta(minutes=1),
        batch_id="batch-2",
    )
    store.record_artifact_cleanup_planned(
        batch_id=second_batch.batch_id,
        lease=later_lease,
        artifact_id="artifact-b",
        trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
        resource_key="file:b",
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(minutes=1, seconds=1),
    )
    manual = store.record_artifact_cleanup_manual_intervention_required(
        batch_id=second_batch.batch_id,
        lease=later_lease,
        artifact_id="artifact-b",
        reason="unsafe artifact path",
        payload={"content_path": "artifacts/b.bin"},
        cleanup_record_id="cleanup-record-4",
        now=now + timedelta(minutes=1, seconds=2),
    )

    assert store.list_unresolved_artifact_cleanup_debt() == [failed, manual]


def test_list_unresolved_artifact_cleanup_debt_excludes_latest_succeeded_and_skipped(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_succeeded_batch(store, lease=lease, now=now)

    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-success",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        cleanup_record_id="cleanup-record-1",
        now=now + timedelta(seconds=1),
    )
    store.record_artifact_cleanup_succeeded(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-success",
        cleanup_record_id="cleanup-record-2",
        now=now + timedelta(seconds=2),
    )
    store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-skip",
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        cleanup_record_id="cleanup-record-3",
        now=now + timedelta(seconds=3),
    )
    store.record_artifact_cleanup_skipped(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id="artifact-skip",
        reason="already removed",
        cleanup_record_id="cleanup-record-4",
        now=now + timedelta(seconds=4),
    )

    assert store.list_unresolved_artifact_cleanup_debt() == []
