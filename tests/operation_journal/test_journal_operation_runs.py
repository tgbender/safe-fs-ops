from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _journal_test_helpers import acquire_lease

from safe_fs_ops.operation_journal import (
    BatchIdempotencyMismatchError,
    JournalLeaseMismatchError,
    OperationJournalStore,
)
from safe_fs_ops.workspace_state import LeaseLostError
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_create_batch_rejects_unknown_operation_run_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)

    with pytest.raises(ValueError, match="operation_run_id 'operation-run-missing' does not exist"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            operation_run_id="operation-run-missing",
            now=now,
        )


def test_create_batch_rejects_unknown_or_mismatched_operation_phase_links(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)
    first_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    second_run = store.create_operation_run(
        run_id="run-2",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-2",
        now=now + timedelta(seconds=1),
    )
    phase = store.create_operation_phase(
        operation_run_id=first_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(ValueError, match="operation_phase_id requires operation_run_id"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            operation_phase_id=phase.operation_phase_id,
            now=now + timedelta(seconds=3),
        )

    with pytest.raises(ValueError, match="operation_phase_id 'phase-missing' does not exist"):
        store.create_batch(
            idempotency_key="apply:file:b",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            operation_run_id=first_run.operation_run_id,
            operation_phase_id="phase-missing",
            now=now + timedelta(seconds=4),
        )

    with pytest.raises(ValueError, match="belongs to operation_run_id 'operation-run-1'"):
        store.create_batch(
            idempotency_key="apply:file:c",
            lease=lease,
            owner="owner-a",
            run_id="run-2",
            operation_run_id=second_run.operation_run_id,
            operation_phase_id=phase.operation_phase_id,
            now=now + timedelta(seconds=5),
        )


def test_create_batch_rejects_operation_run_link_for_different_run_id_and_idempotent_link_mismatch(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="belongs to run_id 'run-1'"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-2",
            operation_run_id=operation_run.operation_run_id,
            now=now + timedelta(seconds=2),
        )

    store.create_batch(
        idempotency_key="apply:file:b",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        operation_run_id=operation_run.operation_run_id,
        operation_phase_id=phase.operation_phase_id,
        now=now + timedelta(seconds=3),
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="operation_phase_id"):
        store.create_batch(
            idempotency_key="apply:file:b",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            operation_run_id=operation_run.operation_run_id,
            operation_phase_id=None,
            now=now + timedelta(seconds=4),
        )


def test_create_batch_rejects_operation_run_link_for_different_owner_with_same_run_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )

    with pytest.raises(ValueError, match="belongs to owner 'owner-a'"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-b",
            run_id="run-1",
            operation_run_id=operation_run.operation_run_id,
            now=now + timedelta(seconds=1),
        )


@pytest.mark.parametrize("status", ["succeeded", "failed", "finalization_failed"])
def test_create_batch_rejects_terminal_operation_links(tmp_path: Path, status: str) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status=status,
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status=status,
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match=f"status {status!r} is terminal"):
        store.create_batch(
            idempotency_key="apply:file:a",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            operation_run_id=operation_run.operation_run_id,
            operation_phase_id=operation_phase.operation_phase_id,
            now=now + timedelta(seconds=2),
        )


def test_terminal_diagnostic_can_be_recorded_after_lease_takeover(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first_lease = acquire_lease(state_path, now=now, owner="owner-a")
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=first_lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=first_lease,
        phase_name="prepare",
        status="succeeded",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )
    takeover_lease = acquire_lease(
        state_path,
        now=now + timedelta(seconds=31),
        owner="owner-b",
    )

    updated_run = store.record_operation_run_terminal_diagnostic(
        operation_run.operation_run_id,
        lease=takeover_lease,
        status="finalization_failed",
        now=now + timedelta(seconds=32),
    )
    updated_phase = store.record_operation_phase_terminal_diagnostic(
        operation_phase.operation_phase_id,
        lease=takeover_lease,
        status="finalization_failed",
        now=now + timedelta(seconds=33),
    )

    assert updated_run.status == "finalization_failed"
    assert updated_phase.status == "finalization_failed"


def test_terminal_diagnostic_rejects_stale_lease_after_takeover(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first_lease = acquire_lease(state_path, now=now, owner="owner-a")
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=first_lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=first_lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )
    acquire_lease(state_path, now=now + timedelta(seconds=31), owner="owner-b")

    with pytest.raises(LeaseLostError, match="lease 'workspace' is no longer current"):
        store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=first_lease,
            status="failed",
            now=now + timedelta(seconds=32),
        )

    with pytest.raises(LeaseLostError, match="lease 'workspace' is no longer current"):
        store.record_operation_phase_terminal_diagnostic(
            operation_phase.operation_phase_id,
            lease=first_lease,
            status="failed",
            now=now + timedelta(seconds=33),
        )

    assert store.get_operation_run(operation_run.operation_run_id).status == "active"
    assert store.get_operation_phase(operation_phase.operation_phase_id).status == "active"


def test_terminal_diagnostic_rejects_forged_newer_fencing_token(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now, owner="owner-a")
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )
    forged_lease = LeaseRecord(
        name=lease.name,
        owner=lease.owner,
        token="forged-token",
        fencing_token=lease.fencing_token + 100,
        acquired_at=now + timedelta(seconds=2),
        heartbeat_at=now + timedelta(seconds=2),
        expires_at=now + timedelta(seconds=300),
        acquired=True,
    )

    with pytest.raises(LeaseLostError, match="lease 'workspace' is no longer current"):
        store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=forged_lease,
            status="failed",
            now=now + timedelta(seconds=3),
        )

    with pytest.raises(LeaseLostError, match="lease 'workspace' is no longer current"):
        store.record_operation_phase_terminal_diagnostic(
            operation_phase.operation_phase_id,
            lease=forged_lease,
            status="failed",
            now=now + timedelta(seconds=4),
        )

    assert store.get_operation_run(operation_run.operation_run_id).status == "active"
    assert store.get_operation_phase(operation_phase.operation_phase_id).status == "active"


def test_terminal_diagnostic_rejects_current_lease_for_different_lease_name(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace_lease = acquire_lease(state_path, now=now, owner="owner-a", name="workspace-a")
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=workspace_lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=workspace_lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )
    other_lease = acquire_lease(
        state_path,
        now=now + timedelta(seconds=2),
        owner="owner-b",
        name="workspace-b",
    )

    with pytest.raises(JournalLeaseMismatchError, match="uses lease 'workspace-a', not 'workspace-b'"):
        store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=other_lease,
            status="failed",
            now=now + timedelta(seconds=3),
        )

    with pytest.raises(JournalLeaseMismatchError, match="uses lease 'workspace-a', not 'workspace-b'"):
        store.record_operation_phase_terminal_diagnostic(
            operation_phase.operation_phase_id,
            lease=other_lease,
            status="failed",
            now=now + timedelta(seconds=4),
        )

    assert store.get_operation_run(operation_run.operation_run_id).status == "active"
    assert store.get_operation_phase(operation_phase.operation_phase_id).status == "active"


def test_terminal_diagnostic_does_not_rewrite_failed_to_finalization_failed(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="failed",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="failed",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(ValueError, match="from 'failed' to 'finalization_failed'"):
        store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=lease,
            status="finalization_failed",
            now=now + timedelta(seconds=2),
        )

    with pytest.raises(ValueError, match="from 'failed' to 'finalization_failed'"):
        store.record_operation_phase_terminal_diagnostic(
            operation_phase.operation_phase_id,
            lease=lease,
            status="finalization_failed",
            now=now + timedelta(seconds=3),
        )

    assert store.get_operation_run(operation_run.operation_run_id).status == "failed"
    assert store.get_operation_phase(operation_phase.operation_phase_id).status == "failed"
