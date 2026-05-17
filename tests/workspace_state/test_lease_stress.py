from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

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


def _collect_results(results: Any, count: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for _ in range(count):
        try:
            collected.append(results.get(timeout=10))
        except Empty:
            break
    return collected


def _acquire_lease_worker(
    state_path: str,
    lease_name: str,
    owner: str,
    start: Any,
    results: Any,
    *,
    now_iso: str,
) -> None:
    try:
        start.wait(timeout=5)
        now = datetime.fromisoformat(now_iso)
        lease = LeaseStore(Path(state_path)).acquire(
            lease_name,
            owner=owner,
            ttl=timedelta(seconds=30),
            now=now,
        )
        results.put(
            {
                "requested_owner": owner,
                "lease": _lease_payload(lease),
            }
        )
    except BaseException as exc:
        results.put({"requested_owner": owner, "error": f"{type(exc).__name__}: {exc}"})


def test_multiprocess_lease_acquire_has_single_winner(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    results = ctx.Queue()
    process_count = 6
    now_iso = datetime(2026, 1, 1, tzinfo=UTC).isoformat()
    processes = [
        ctx.Process(
            target=_acquire_lease_worker,
            args=(str(state_path), "workspace", f"owner-{index}", start, results),
            kwargs={"now_iso": now_iso},
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

    winners = [item for item in collected if item["lease"]["acquired"]]
    losers = [item for item in collected if not item["lease"]["acquired"]]
    assert len(winners) == 1
    assert len(losers) == process_count - 1

    winner = winners[0]
    assert {item["lease"]["owner"] for item in losers} == {winner["lease"]["owner"]}
    assert {item["lease"]["token"] for item in losers} == {""}
    assert all(item["lease"]["fencing_token"] == 0 for item in losers)
    assert all(process.exitcode == 0 for process in processes)
