from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.operation_journal import BatchIdempotencyMismatchError, OperationJournalStore
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_create_operation_run_creates_distinct_rows_for_same_public_run_and_owner(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    first = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        payload={"step": 1},
        now=now,
    )
    second = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        payload={"step": 1},
        now=now + timedelta(seconds=1),
    )

    assert second != first
    assert second.operation_run_id != first.operation_run_id
    assert store.get_operation_run(first.operation_run_id) == first
    assert store.list_operation_runs(run_id="run-1", owner="owner-a") == [first, second]


def test_create_operation_run_is_idempotent_by_explicit_operation_run_id(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    first = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        payload={"step": 1},
        operation_run_id="operation-run-1",
        now=now,
    )
    second = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        payload={"step": 1},
        operation_run_id="operation-run-1",
        now=now + timedelta(seconds=1),
    )

    assert second == first
    assert store.list_operation_runs(run_id="run-1", owner="owner-a") == [first]


def test_create_operation_run_rejects_conflicting_explicit_operation_run_id_on_retry(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        payload={"step": 1},
        operation_run_id="operation-run-1",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="different status"):
        store.create_operation_run(
            run_id="run-1",
            lease=lease,
            owner="owner-a",
            status="running",
            payload={"step": 1},
            operation_run_id="operation-run-1",
            now=now + timedelta(seconds=1),
        )


def test_create_operation_run_rejects_explicit_operation_run_id_collision_for_different_identity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        operation_run_id="operation-run-1",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="different run_id"):
        store.create_operation_run(
            run_id="run-2",
            lease=lease,
            owner="owner-b",
            status="planned",
            operation_run_id="operation-run-1",
            now=now + timedelta(seconds=1),
        )


def test_create_and_list_operation_phases_are_idempotent_by_run_and_phase_name(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        operation_run_id="operation-run-1",
        now=now,
    )

    second = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="apply",
        status="planned",
        phase_order=2,
        operation_phase_id="phase-2",
        now=now + timedelta(seconds=2),
    )
    first = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="done",
        phase_order=1,
        payload={"count": 1},
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )
    replay = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="done",
        phase_order=1,
        payload={"count": 1},
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=2),
    )
    replay_without_id = store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="done",
        phase_order=1,
        payload={"count": 1},
        now=now + timedelta(seconds=3),
    )

    assert store.get_operation_phase(first.operation_phase_id) == first
    assert replay == first
    assert replay_without_id == first
    assert store.list_operation_phases(operation_run.operation_run_id) == [first, second]

    with pytest.raises(BatchIdempotencyMismatchError, match="different status"):
        store.create_operation_phase(
            operation_run_id=operation_run.operation_run_id,
            lease=lease,
            phase_name="prepare",
            status="retrying",
            phase_order=3,
            now=now + timedelta(seconds=4),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"status": "retrying"}, "different status"),
        ({"phase_order": 2}, "different phase_order"),
        ({"payload": {"count": 2}}, "different payload"),
        ({"operation_phase_id": "phase-2"}, "operation_phase_id"),
    ],
)
def test_create_operation_phase_rejects_conflicting_retries(
    tmp_path: Path,
    kwargs: dict[str, object],
    match: str,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        operation_run_id="operation-run-1",
        now=now,
    )

    store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="done",
        phase_order=1,
        payload={"count": 1},
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(BatchIdempotencyMismatchError, match=match):
        store.create_operation_phase(
            operation_run_id=operation_run.operation_run_id,
            lease=lease,
            phase_name="prepare",
            status=kwargs.get("status", "done"),
            phase_order=int(kwargs.get("phase_order", 1)),
            payload=kwargs.get("payload", {"count": 1}),
            operation_phase_id=kwargs.get("operation_phase_id"),
            now=now + timedelta(seconds=2),
        )


def test_create_operation_phase_rejects_explicit_operation_phase_id_collision_for_different_identity(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    operation_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="planned",
        operation_run_id="operation-run-1",
        now=now,
    )

    store.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="done",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="operation_phase_id"):
        store.create_operation_phase(
            operation_run_id=operation_run.operation_run_id,
            lease=lease,
            phase_name="apply",
            status="planned",
            phase_order=2,
            operation_phase_id="phase-1",
            now=now + timedelta(seconds=2),
        )


def _acquire_lease(
    state_path: Path,
    *,
    now: datetime,
    owner: str = "runner-a",
    name: str = "workspace",
) -> LeaseRecord:
    return LeaseStore(state_path).acquire(name, owner=owner, ttl=timedelta(seconds=30), now=now)
