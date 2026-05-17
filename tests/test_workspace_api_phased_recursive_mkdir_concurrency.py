from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock, Thread

import pytest
from workspace_api_helpers import portable_delete_file, portable_make_directory, portable_snapshot, portable_write_text

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.operation_journal import BatchPhase

pytestmark = pytest.mark.safe_fs_ops


def test_phase_recursive_mkdir_blocks_phase_exit_until_recursive_steps_finish(tmp_path: Path) -> None:
    blocker = _BlockingRecursiveMkdir()
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=blocker.make_directory,
    )
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    errors: list[BaseException] = []
    result_lock = Lock()

    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")
    phase = operation.phase("prepare")
    operation.__enter__()
    phase.__enter__()

    def run_recursive_mkdir() -> None:
        try:
            phase.make_directory(phase.r.state_dir, parents=True)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    phase_exit_started = Event()

    def exit_phase() -> None:
        phase_exit_started.set()
        try:
            phase.__exit__(None, None, None)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    mkdir_thread = Thread(target=run_recursive_mkdir)
    exit_thread = Thread(target=exit_phase)

    mkdir_thread.start()
    assert blocker.first_step_started.wait(timeout=5)
    exit_thread.start()
    assert phase_exit_started.wait(timeout=5)
    time.sleep(0.1)
    assert exit_thread.is_alive()

    blocker.allow_first_step_finish.set()
    mkdir_thread.join(timeout=5)
    exit_thread.join(timeout=5)
    assert not mkdir_thread.is_alive()
    assert not exit_thread.is_alive()
    assert errors == []

    assert phase.phase_record is not None
    assert phase.phase_record.status == "succeeded"
    assert operation.__exit__(None, None, None) is False
    assert [batch.phase for batch in workspace.journal_store.list_batches()] == [
        BatchPhase.SUCCEEDED,
        BatchPhase.SUCCEEDED,
        BatchPhase.SUCCEEDED,
    ]


class _BlockingRecursiveMkdir:
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
