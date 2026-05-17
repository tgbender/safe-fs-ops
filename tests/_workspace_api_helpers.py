from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from threading import Event, Lock

from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_snapshot,
    portable_write_text,
)

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.operation_journal import OperationJournalStore


def _workspace(state_path: Path, *, lease_ttl: timedelta = timedelta(seconds=30)) -> SafeWorkspace:
    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        lease_ttl=lease_ttl,
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
    )


class _StatusFailureJournalStore:
    def __init__(
        self,
        delegate: OperationJournalStore,
        *,
        fail_phase_statuses: set[str] | None = None,
        fail_run_statuses: set[str] | None = None,
    ) -> None:
        self._delegate = delegate
        self._fail_phase_statuses = fail_phase_statuses or set()
        self._fail_run_statuses = fail_run_statuses or set()

    def update_operation_phase_status(
        self, operation_phase_id: str, *, lease: object, status: str, now: object = None
    ) -> object:
        if status in self._fail_phase_statuses:
            raise RuntimeError("injected phase status failure")
        return self._delegate.update_operation_phase_status(operation_phase_id, lease=lease, status=status, now=now)

    def update_operation_run_status(
        self, operation_run_id: str, *, lease: object, status: str, now: object = None
    ) -> object:
        if status in self._fail_run_statuses:
            raise RuntimeError("injected operation status failure")
        return self._delegate.update_operation_run_status(operation_run_id, lease=lease, status=status, now=now)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _BlockingPortableMkdir:
    def __init__(self) -> None:
        self.first_step_started = Event()
        self.allow_first_step_finish = Event()
        self._lock = Lock()
        self._call_count = 0

    def make_directory(self, path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        portable_make_directory(path, parents=parents, exist_ok=exist_ok)
        with self._lock:
            self._call_count += 1
            call_count = self._call_count
            if call_count == 1:
                self.first_step_started.set()
        if call_count == 1:
            assert self.allow_first_step_finish.wait(timeout=5)
