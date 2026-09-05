from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from typing import Any

import pytest
from workspace_api_helpers import workspace_with_portable_ops

from safe_fs_ops import SafeWorkspaceError
from safe_fs_ops.operation_journal import OperationJournalStore
from safe_fs_ops.operation_journal.models import OperationPhaseRecord
from safe_fs_ops.workspace_operation import SafePhase
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_operation_serializes_phase_entry_across_threads(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    blocking_store = _BlockingPhaseCreateJournalStore(workspace.journal_store)
    workspace._journal_store = blocking_store
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    phase_orders: dict[str, int] = {}
    errors: dict[str, BaseException] = {}
    result_lock = Lock()
    start_barrier = Barrier(2)
    second_attempt_started = Event()
    allow_first_phase_exit = Event()

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        prepare = operation.phase("prepare")
        apply_phase = operation.phase("apply")

        threads = [
            Thread(
                target=_enter_phase_in_thread,
                args=(
                    "prepare",
                    prepare,
                    start_barrier,
                    Event(),
                    allow_first_phase_exit,
                    phase_orders,
                    errors,
                    result_lock,
                ),
            ),
            Thread(
                target=_enter_phase_in_thread,
                args=(
                    "apply",
                    apply_phase,
                    start_barrier,
                    second_attempt_started,
                    allow_first_phase_exit,
                    phase_orders,
                    errors,
                    result_lock,
                ),
            ),
        ]
        for thread in threads:
            thread.start()

        assert blocking_store.first_create_started.wait(timeout=5)
        assert second_attempt_started.wait(timeout=5)
        time.sleep(0.1)
        blocking_store.release_first_create.set()

        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

    assert len(phase_orders) == 1
    assert sorted(phase_orders.values()) == [1]
    assert len(errors) == 1
    assert isinstance(next(iter(errors.values())), SafeWorkspaceError)
    assert "another phase is already active" in str(next(iter(errors.values())))
    assert operation.operation_run is not None
    assert [
        (record.phase_name, record.phase_order, record.status)
        for record in workspace.journal_store.list_operation_phases(operation.operation_run.operation_run_id)
    ] == [(next(iter(phase_orders)), 1, "succeeded")]


def _enter_phase_in_thread(
    name: str,
    phase: SafePhase,
    start_barrier: Barrier,
    attempt_started: Event,
    allow_phase_exit: Event,
    phase_orders: dict[str, int],
    errors: dict[str, BaseException],
    result_lock: Lock,
) -> None:
    try:
        start_barrier.wait(timeout=5)
        attempt_started.set()
        with phase as active_phase:
            assert active_phase.phase_record is not None
            with result_lock:
                phase_orders[name] = active_phase.phase_record.phase_order
            assert allow_phase_exit.wait(timeout=5)
    except BaseException as exc:
        with result_lock:
            errors[name] = exc
        # Keep the successful phase active until the competing entry is rejected.
        allow_phase_exit.set()


class _BlockingPhaseCreateJournalStore:
    def __init__(self, delegate: OperationJournalStore) -> None:
        self._delegate = delegate
        self._lock = Lock()
        self.first_create_started = Event()
        self.release_first_create = Event()
        self._create_call_count = 0

    def create_operation_phase(
        self,
        *,
        operation_run_id: str,
        lease: LeaseRecord,
        phase_name: str,
        status: str,
        phase_order: int,
        payload: Mapping[str, Any] | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationPhaseRecord:
        with self._lock:
            self._create_call_count += 1
            call_number = self._create_call_count
            if call_number == 1:
                self.first_create_started.set()
        if call_number == 1:
            assert self.release_first_create.wait(timeout=5)
        return self._delegate.create_operation_phase(
            operation_run_id=operation_run_id,
            lease=lease,
            phase_name=phase_name,
            status=status,
            phase_order=phase_order,
            payload=payload,
            operation_phase_id=operation_phase_id,
            now=now,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)
