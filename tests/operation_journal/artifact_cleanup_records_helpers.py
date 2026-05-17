from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from safe_fs_ops.operation_journal import OperationJournalStore
from safe_fs_ops.operation_journal.models import OperationBatchRecord, RecoveryRecord
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord


def _acquire_lease(
    state_path: Path,
    *,
    now: datetime,
    owner: str = "runner-a",
    name: str = "workspace",
) -> LeaseRecord:
    return LeaseStore(state_path).acquire(name, owner=owner, ttl=timedelta(seconds=30), now=now)


def _create_succeeded_batch(
    store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    now: datetime,
    batch_id: str = "batch-1",
) -> OperationBatchRecord:
    batch = store.create_batch(
        idempotency_key=f"apply:{batch_id}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id=batch_id,
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now + timedelta(microseconds=1))
    return store.mark_succeeded(batch.batch_id, lease=lease, now=now + timedelta(microseconds=2))


def _create_recovery_succeeded_batch(
    store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    now: datetime,
    batch_id: str = "batch-1",
) -> RecoveryRecord:
    batch = store.create_batch(
        idempotency_key=f"recover:{batch_id}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id=batch_id,
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now + timedelta(microseconds=1))
    store.mark_failed(batch.batch_id, lease=lease, error="write failed", now=now + timedelta(microseconds=2))
    store.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="recover",
        recovery_id=f"{batch_id}-recovery-1",
        now=now + timedelta(microseconds=3),
    )
    _updated_batch, recovery_attempt = store.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="attempt recovery",
        recovery_id=f"{batch_id}-recovery-2",
        now=now + timedelta(microseconds=4),
    )
    return store.record_recovery_succeeded(
        batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt.recovery_id,
        reason="recovered",
        recovery_id=f"{batch_id}-recovery-3",
        now=now + timedelta(microseconds=5),
    )
