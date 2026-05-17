from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import (
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    OperationJournalStore,
    RecoveryAttemptMismatchError,
    RecoveryContextError,
)
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_read_recovery_context_is_atomic(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", observed_state={"path": "a"}, now=now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        payload={"backup": "backup/a"},
        recovery_id="recovery-1",
        now=now,
    )

    mutations: list[str] = []

    def after_batch_read() -> None:
        mutations.append("after-batch-read")
        _started, recovering = store.start_recovery(
            batch.batch_id,
            lease=lease,
            reason="interrupted recovery start",
            recovery_id="injected-recovery-started",
            now=now + timedelta(seconds=1),
        )
        store.record_recovery_failed(
            batch.batch_id,
            lease=lease,
            recovery_attempt_id=recovering.recovery_id,
            reason="backup missing",
            recovery_id="injected-recovery-failed",
            now=now + timedelta(seconds=2),
        )
        store.record_recovery_desired(
            batch.batch_id,
            lease=lease,
            reason="retry restore",
            recovery_id="injected-recovery-desired",
            now=now + timedelta(seconds=3),
        )

    context = store.read_recovery_context(batch.batch_id, _after_batch_read=after_batch_read)

    assert mutations == ["after-batch-read"]
    assert context.batch.phase == BatchPhase.RECOVERY_DESIRED
    assert [record.recovery_id for record in context.recovery_records] == ["recovery-1"]
    assert context.recovery_attempt is None
    assert context.recovery_attempt_id is None
    assert [record.recovery_id for record in store.list_recovery_records(batch.batch_id)] == [
        "recovery-1",
        "injected-recovery-started",
        "injected-recovery-failed",
        "injected-recovery-desired",
    ]


def test_read_recovery_context_includes_active_recovery_attempt(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", observed_state={"path": "a"}, now=now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        payload={"backup": "backup/a"},
        recovery_id="recovery-1",
        now=now,
    )
    _started_batch, recovering = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-2",
        now=now + timedelta(microseconds=1),
    )

    context = store.read_recovery_context(batch.batch_id)

    assert context.batch.phase == BatchPhase.RECOVERING
    assert context.recovery_attempt is not None
    assert context.recovery_attempt.recovery_id == recovering.recovery_id
    assert context.recovery_attempt_id == recovering.recovery_id
    assert [record.recovery_id for record in context.recovery_records] == [
        "recovery-1",
        "recovery-2",
    ]


def test_read_recovery_context_requires_active_recovery_attempt_for_recovering_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", observed_state={"path": "a"}, now=now)
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        payload={"backup": "backup/a"},
        recovery_id="recovery-1",
        now=now,
    )

    with store.sqlite_store.transaction() as connection:
        connection.execute(
            """
            UPDATE operation_batches
            SET phase = ?, updated_at = ?
            WHERE batch_id = ?
            """,
            (BatchPhase.RECOVERING, now.isoformat(), batch.batch_id),
        )
        connection.execute(
            """
            DELETE FROM operation_recovery_records
            WHERE batch_id = ?
              AND phase = ?
            """,
            (batch.batch_id, BatchPhase.RECOVERING),
        )

    with pytest.raises(RecoveryContextError, match="has no active recovery attempt record"):
        store.read_recovery_context(batch.batch_id)


def test_start_recovery_reserves_batch_and_blocks_second_attempt(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", now=now)
    desired = store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore backup",
        recovery_id="recovery-1",
        now=now,
    )

    started_batch, started_recovery = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        payload={"attempt": 1},
        recovery_id="recovery-2",
        now=now + timedelta(seconds=1),
    )

    assert desired.phase == BatchPhase.RECOVERY_DESIRED
    assert started_batch.phase == BatchPhase.RECOVERING
    assert started_recovery.phase == BatchPhase.RECOVERING
    assert started_recovery.sequence == 2

    with pytest.raises(InvalidBatchPhaseTransitionError, match="recovering.*recovering"):
        store.start_recovery(
            batch.batch_id,
            lease=lease,
            reason="duplicate reservation",
            recovery_id="recovery-3",
            now=now + timedelta(seconds=2),
        )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERING
    assert [record.recovery_id for record in store.list_recovery_records(batch.batch_id)] == [
        "recovery-1",
        "recovery-2",
    ]


def test_recovery_completion_requires_active_recovery_attempt_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
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
    _started_batch, recovering = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-2",
        now=now + timedelta(microseconds=1),
    )

    with pytest.raises(RecoveryAttemptMismatchError, match="requires the active recovery_attempt_id"):
        store.record_recovery_succeeded(
            batch.batch_id,
            lease=lease,
            reason="missing attempt id",
            recovery_id="recovery-3",
            now=now + timedelta(seconds=1),
        )

    with pytest.raises(RecoveryAttemptMismatchError, match="is not recoverable"):
        store.record_recovery_failed(
            batch.batch_id,
            lease=lease,
            recovery_attempt_id="recovery-1",
            reason="wrong attempt id",
            recovery_id="recovery-4",
            now=now + timedelta(seconds=2),
        )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERING
    assert [record.recovery_id for record in store.list_recovery_records(batch.batch_id)] == [
        "recovery-1",
        "recovery-2",
    ]
    assert recovering.phase == BatchPhase.RECOVERING


def test_recovery_completion_accepts_active_recovery_attempt_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
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
    _started_batch, recovering = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-2",
        now=now + timedelta(microseconds=1),
    )

    succeeded = store.record_recovery_succeeded(
        batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovering.recovery_id,
        reason="restored",
        recovery_id="recovery-3",
        now=now + timedelta(seconds=1),
    )

    updated = store.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert succeeded.recovery_id == "recovery-3"
    assert [record.recovery_id for record in store.list_recovery_records(batch.batch_id)] == [
        "recovery-1",
        "recovery-2",
        "recovery-3",
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
