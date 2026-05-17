from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from safe_fs_ops.operation_journal import BatchPhase, OperationJournalStore
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.stress_lock]


def _lease_payload(lease: LeaseRecord) -> dict[str, Any]:
    return {
        "name": lease.name,
        "owner": lease.owner,
        "token": lease.token,
        "fencing_token": lease.fencing_token,
        "acquired_at": lease.acquired_at.isoformat(),
        "heartbeat_at": lease.heartbeat_at.isoformat(),
        "expires_at": lease.expires_at.isoformat(),
        "acquired": lease.acquired,
    }


def _lease_record(payload: dict[str, Any]) -> LeaseRecord:
    return LeaseRecord(
        name=str(payload["name"]),
        owner=str(payload["owner"]),
        token=str(payload["token"]),
        fencing_token=int(payload["fencing_token"]),
        acquired_at=datetime.fromisoformat(str(payload["acquired_at"])),
        heartbeat_at=datetime.fromisoformat(str(payload["heartbeat_at"])),
        expires_at=datetime.fromisoformat(str(payload["expires_at"])),
        acquired=bool(payload["acquired"]),
    )


def _collect_results(results: Any, count: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for _ in range(count):
        try:
            collected.append(results.get(timeout=10))
        except Empty:
            break
    return collected


def _create_batch_worker(
    state_path: str,
    lease_payload: dict[str, Any],
    start: Any,
    results: Any,
    *,
    now_iso: str,
    requested_batch_id: str,
) -> None:
    try:
        start.wait(timeout=5)
        now = datetime.fromisoformat(now_iso)
        lease = _lease_record(lease_payload)
        store = OperationJournalStore(Path(state_path))
        batch = store.create_batch(
            idempotency_key="apply:config",
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            batch_id=requested_batch_id,
            now=now,
        )
        results.put(
            {
                "requested_batch_id": requested_batch_id,
                "batch_id": batch.batch_id,
                "phase": batch.phase,
            }
        )
    except BaseException as exc:
        results.put(
            {
                "requested_batch_id": requested_batch_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _start_recovery_worker(
    state_path: str,
    batch_id: str,
    lease_payload: dict[str, Any],
    start: Any,
    results: Any,
    *,
    now_iso: str,
    recovery_id: str,
) -> None:
    try:
        start.wait(timeout=5)
        now = datetime.fromisoformat(now_iso)
        lease = _lease_record(lease_payload)
        store = OperationJournalStore(Path(state_path))
        batch, recovery = store.start_recovery(
            batch_id,
            lease=lease,
            reason="recovery reserved",
            recovery_id=recovery_id,
            now=now,
        )
        results.put(
            {
                "recovery_id": recovery.recovery_id,
                "batch_phase": batch.phase,
                "batch_lease_fencing_token": batch.lease_fencing_token,
            }
        )
    except BaseException as exc:
        results.put({"recovery_id": recovery_id, "error": f"{type(exc).__name__}: {exc}"})


def test_multiprocess_idempotent_batch_creation_has_single_record(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = LeaseStore(state_path).acquire("workspace", owner="runner-a", ttl=timedelta(seconds=30), now=now)
    journal = OperationJournalStore(state_path)
    journal.initialize()
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    process_count = 6
    processes = [
        ctx.Process(
            target=_create_batch_worker,
            args=(str(state_path), _lease_payload(lease), start, results),
            kwargs={
                "now_iso": now.isoformat(),
                "requested_batch_id": f"batch-{index}",
            },
        )
        for index in range(process_count)
    ]

    for process in processes:
        process.start()
    start.set()
    collected = _collect_results(results, process_count)
    for process in processes:
        process.join(timeout=10)

    assert len(collected) == process_count
    assert [item for item in collected if "error" in item] == []
    assert len({item["batch_id"] for item in collected}) == 1
    assert len(journal.list_batches()) == 1
    stored_batch = journal.list_batches()[0]
    assert stored_batch.phase == BatchPhase.PLANNED
    assert stored_batch.idempotency_key == "apply:config"
    assert all(process.exitcode == 0 for process in processes)


def test_multiprocess_recovery_start_has_single_winner_and_preserves_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = LeaseStore(state_path).acquire("workspace", owner="runner-a", ttl=timedelta(seconds=30), now=now)
    journal = OperationJournalStore(state_path)
    batch = journal.create_batch(
        idempotency_key="apply:config",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        batch_id="batch-1",
        now=now,
    )
    journal.mark_attempting(batch.batch_id, lease=lease, now=now)
    journal.mark_failed(batch.batch_id, lease=lease, error="seed failure", now=now)
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="retry recovery",
        recovery_id="recovery-seed",
        now=now,
    )
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    process_count = 6
    processes = [
        ctx.Process(
            target=_start_recovery_worker,
            args=(str(state_path), batch.batch_id, _lease_payload(lease), start, results),
            kwargs={
                "now_iso": (now + timedelta(microseconds=1)).isoformat(),
                "recovery_id": f"recovery-{index}",
            },
        )
        for index in range(process_count)
    ]

    for process in processes:
        process.start()
    start.set()
    collected = _collect_results(results, process_count)
    for process in processes:
        process.join(timeout=10)

    assert len(collected) == process_count
    winners = [item for item in collected if "error" not in item]
    losers = [item for item in collected if "error" in item]
    assert len(winners) == 1
    assert len(losers) == process_count - 1
    assert {item["error"].split(":", 1)[0] for item in losers} == {"InvalidBatchPhaseTransitionError"}
    updated_batch = journal.get_batch(batch.batch_id)
    assert updated_batch is not None
    assert updated_batch.phase == BatchPhase.RECOVERING
    assert winners[0]["batch_phase"] == BatchPhase.RECOVERING
    assert winners[0]["batch_lease_fencing_token"] == lease.fencing_token
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert [record.recovery_id for record in recovery_records] == ["recovery-seed", winners[0]["recovery_id"]]
    assert [record.phase for record in recovery_records] == [BatchPhase.RECOVERY_DESIRED, BatchPhase.RECOVERING]
    assert all(process.exitcode == 0 for process in processes)
