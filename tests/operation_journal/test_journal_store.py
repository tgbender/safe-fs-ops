from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import (
    BatchIdempotencyMismatchError,
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
    OperationJournalStore,
)
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_create_batch_is_idempotent_and_copies_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    payload = {"paths": ["a"], "plan": {"step": 1}}

    first = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        payload=payload,
        batch_id="batch-1",
        now=now,
    )
    payload["paths"].append("b")
    payload["plan"]["step"] = 2
    second = store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        payload={"paths": ["a"], "plan": {"step": 1}},
        batch_id="batch-2",
        now=now + timedelta(seconds=1),
    )

    assert second == first
    assert len(store.list_batches()) == 1
    assert first.payload["paths"] == ("a",)
    assert first.payload["plan"]["step"] == 1
    with pytest.raises(TypeError):
        first.payload["new"] = "value"  # type: ignore[index]


def test_create_batch_rejects_idempotent_retry_with_different_payload_or_authority(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    store.create_batch(
        idempotency_key="apply:file:a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key="file:a",
        claim_owner="owner-a",
        claim_scope="install",
        payload={"operation": "write_text", "sha256": "aaa"},
        batch_id="batch-1",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="payload"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            resource_key="file:a",
            claim_owner="owner-a",
            claim_scope="install",
            payload={"operation": "write_text", "sha256": "bbb"},
            now=now,
        )

    with pytest.raises(BatchIdempotencyMismatchError, match="claim_scope"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            resource_key="file:a",
            claim_owner="owner-a",
            claim_scope="upgrade",
            payload={"operation": "write_text", "sha256": "aaa"},
            now=now,
        )

    assert len(store.list_batches()) == 1


def test_batch_creation_requires_current_lease_and_rejects_stale_without_write(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    stale_lease = _acquire_lease(state_path, now=first_now)
    _acquire_lease(state_path, owner="runner-b", now=first_now + timedelta(seconds=31))

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=stale_lease,
            owner="owner-a",
            run_id="run-1",
            batch_id="batch-1",
            now=first_now + timedelta(seconds=31),
        )

    assert store.list_batches() == []


def test_journal_writes_require_current_matching_batch_lease(tmp_path: Path) -> None:
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
    other_lease = LeaseStore(state_path).acquire("other", owner="runner-a", ttl=timedelta(seconds=30), now=now)

    with pytest.raises(JournalLeaseMismatchError, match="different lease"):
        store.append_operation(batch.batch_id, lease=other_lease, operation_type="write", now=now)

    _acquire_lease(state_path, owner="runner-b", now=now + timedelta(seconds=31))
    with pytest.raises(LeaseLostError, match="no longer current"):
        store.append_operation(batch.batch_id, lease=lease, operation_type="write", now=now + timedelta(seconds=31))

    assert store.list_operations(batch.batch_id) == []


def test_valid_and_invalid_phase_transitions_are_enforced(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    with pytest.raises(InvalidBatchPhaseTransitionError, match="planned.*succeeded"):
        store.mark_succeeded(batch.batch_id, lease=lease, now=now)

    attempting = store.mark_attempting(batch.batch_id, lease=lease, payload={"pid": 123}, now=now)
    assert attempting.phase == BatchPhase.ATTEMPTING
    assert attempting.status_payload["pid"] == 123

    succeeded = store.mark_succeeded(batch.batch_id, lease=lease, result={"changed": True}, now=now)
    assert succeeded.phase == BatchPhase.SUCCEEDED
    assert succeeded.status_payload["changed"] is True

    with pytest.raises(InvalidBatchPhaseTransitionError, match="succeeded.*failed"):
        store.mark_failed(batch.batch_id, lease=lease, error="late error", now=now)


def test_operations_and_checkpoints_share_monotonic_batch_ordering(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)

    first_operation = store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        payload={"content": "one"},
        operation_id="operation-1",
        now=now,
    )
    checkpoint = store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=first_operation.operation_id,
        resource_key="file:a",
        checkpoint_type="before",
        payload={"exists": True, "hash": "abc"},
        checkpoint_id="checkpoint-1",
        now=now,
    )
    second_operation = store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="chmod",
        resource_key="file:a",
        operation_id="operation-2",
        now=now,
    )

    assert first_operation.sequence == 1
    assert checkpoint.sequence == 2
    assert second_operation.sequence == 3
    assert [operation.operation_id for operation in store.list_operations(batch.batch_id)] == [
        "operation-1",
        "operation-2",
    ]
    assert store.list_checkpoints(batch.batch_id) == [checkpoint]


def test_terminal_batches_reject_late_operations_and_checkpoints(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)
    operation = store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        operation_id="operation-1",
        now=now,
    )
    store.mark_attempting(batch.batch_id, lease=lease, now=now)
    store.mark_succeeded(batch.batch_id, lease=lease, now=now)

    with pytest.raises(InvalidBatchPhaseTransitionError, match="append.*succeeded"):
        store.append_operation(batch.batch_id, lease=lease, operation_type="chmod", now=now)
    with pytest.raises(InvalidBatchPhaseTransitionError, match="append.*succeeded"):
        store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key="file:a",
            checkpoint_type="after",
            now=now,
        )

    assert [record.operation_id for record in store.list_operations(batch.batch_id)] == ["operation-1"]
    assert store.list_checkpoints(batch.batch_id) == []


def test_checkpoint_operation_resource_key_must_match(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    batch = _create_batch(store, lease=lease, now=now)
    operation = store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        operation_id="operation-1",
        now=now,
    )

    with pytest.raises(ValueError, match="resource_key"):
        store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key="file:b",
            checkpoint_type="before",
            now=now,
        )

    assert store.list_checkpoints(batch.batch_id) == []


def test_checkpoint_operation_id_must_belong_to_same_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    first = _create_batch(store, lease=lease, now=now)
    second = store.create_batch(
        idempotency_key="apply:file:b",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-2",
        now=now,
    )
    operation = store.append_operation(
        first.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key="file:a",
        operation_id="operation-1",
        now=now,
    )

    with pytest.raises(ValueError, match="different batch"):
        store.record_checkpoint(
            second.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key="file:b",
            checkpoint_type="before",
            now=now,
        )

    assert store.list_checkpoints(second.batch_id) == []


def test_batches_list_deterministically_with_filters(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    store.create_batch(
        idempotency_key="z",
        lease=lease,
        owner="owner-a",
        run_id="run-2",
        batch_id="batch-z",
        now=now + timedelta(seconds=2),
    )
    first = store.create_batch(
        idempotency_key="a",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-a",
        now=now,
    )
    second = store.create_batch(
        idempotency_key="m",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-m",
        now=now + timedelta(seconds=1),
    )
    store.mark_attempting(second.batch_id, lease=lease, now=now + timedelta(seconds=1))

    assert [batch.batch_id for batch in store.list_batches()] == ["batch-a", "batch-m", "batch-z"]
    assert [batch.batch_id for batch in store.list_batches(run_id="run-1")] == ["batch-a", "batch-m"]
    assert store.list_batches(run_id="run-1", phase=BatchPhase.PLANNED) == [first]


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
