from __future__ import annotations

import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import cast

import pytest
from _workspace_api_helpers import _BlockingPortableMkdir, _workspace
from workspace_api_helpers import portable_delete_file, portable_snapshot, portable_write_text

from safe_fs_ops import SafeWorkspace, SafeWorkspaceError
from safe_fs_ops.operation_journal import directory_resource_key
from safe_fs_ops.resources import FileResource

pytestmark = pytest.mark.safe_fs_ops


def test_operation_phases_share_single_transaction_lifecycle(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    target_path = tmp_path / "config.txt"
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        assert op.operation_run is not None
        assert op.operation_run.run_id == "run-1"
        assert op.operation_run.owner == "owner-a"
        assert op.operation_run.payload == {"name": "sync", "rollback": "record-only"}

        with op.phase("prepare") as prepare:
            assert prepare.phase_record is not None
            assert prepare.phase_record.phase_name == "prepare"
            assert prepare.phase_record.phase_order == 1
            config = cast(FileResource, prepare.r.config)
            assert config.path == target_path
            prepare.write_text(config, "prepare\n")
            lease_during_prepare = op.lease
            assert workspace.claim_store.get(resources.config.resource_key) is not None

        assert workspace.claim_store.get(resources.config.resource_key) is not None

        with op.phase("apply") as apply_phase:
            assert apply_phase.phase_record is not None
            assert apply_phase.phase_record.phase_name == "apply"
            assert apply_phase.phase_record.phase_order == 2
            config = cast(FileResource, apply_phase.r.config)
            assert config.path == target_path
            apply_phase.write_text(config, "apply\n")
            assert op.lease == lease_during_prepare

    assert target_path.read_text(encoding="utf-8") == "apply\n"
    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None
    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0] == op.operation_run
    assert operation_runs[0].status == "succeeded"
    assert workspace.journal_store.list_operation_phases(op.operation_run.operation_run_id) == [
        prepare.phase_record,
        apply_phase.phase_record,
    ]
    assert prepare.phase_record is not None
    assert apply_phase.phase_record is not None
    assert prepare.phase_record.status == "succeeded"
    assert apply_phase.phase_record.status == "succeeded"
    assert [
        (batch.idempotency_key, batch.operation_run_id, batch.operation_phase_id)
        for batch in workspace.journal_store.list_batches()
    ] == [
        (
            "sync:run-1:prepare:1:write_text:config",
            op.operation_run.operation_run_id,
            prepare.phase_record.operation_phase_id,
        ),
        (
            "sync:run-1:apply:1:write_text:config",
            op.operation_run.operation_run_id,
            apply_phase.phase_record.operation_phase_id,
        ),
    ]


def test_phase_creates_claimed_directory_resource(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state"
    target_path.parent.mkdir()
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op, op.phase("prepare") as phase:
        result = phase.make_directory(phase.r.state_dir)

    assert target_path.is_dir()
    assert result.batch.phase == "succeeded"
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None
    assert [batch.idempotency_key for batch in workspace.journal_store.list_batches()] == [
        "sync:run-1:prepare:1:make_directory:state_dir",
    ]


def test_legacy_transaction_batches_with_null_operation_links_still_read_normally(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    target_path = tmp_path / "config.txt"
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx:
        tx.write_text(cast(FileResource, tx.r.config), "value\n")

    assert [
        (batch.idempotency_key, batch.operation_run_id, batch.operation_phase_id)
        for batch in workspace.journal_store.list_batches()
    ] == [("sync:run-1:1:write_text:config", None, None)]


def test_phase_recursive_directory_creation_creates_ancestor_batches(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op, op.phase("prepare") as phase:
        result = phase.make_directory(phase.r.state_dir, parents=True)

    assert target_path.is_dir()
    assert result.batch.resource_key == resources.state_dir.resource_key
    assert [batch.resource_key for batch in workspace.journal_store.list_batches()] == [
        directory_resource_key(tmp_path / "config"),
        resources.state_dir.resource_key,
    ]


def test_operation_rejects_nested_active_phases(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op, op.phase("prepare"):
        with pytest.raises(SafeWorkspaceError, match="phase is already active"):
            with op.phase("apply"):
                pass


def test_operation_rejects_duplicate_serial_phase_names_before_entry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    target_path = tmp_path / "config.txt"
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        prepare = op.phase("prepare")

        with prepare:
            config = cast(FileResource, prepare.r.config)
            assert config.path == target_path

        with pytest.raises(SafeWorkspaceError, match="duplicate phase name 'prepare'"):
            op.phase("prepare")

    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert op.operation_run is not None
    assert workspace.journal_store.list_operation_phases(op.operation_run.operation_run_id) == [prepare.phase_record]


def test_phase_rejects_use_after_exit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        with op.phase("prepare") as phase:
            config = cast(FileResource, phase.r.config)
            assert config.path == tmp_path / "config.txt"
            phase.write_text(config, "prepare\n")

        with pytest.raises(SafeWorkspaceError, match="phase is not active"):
            phase.write_text(config, "after-exit\n")


def test_phase_rejects_raw_paths_for_directory_operations(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op, op.phase("prepare") as phase:
        with pytest.raises(TypeError, match="DirectoryResource"):
            phase.make_directory(tmp_path / "config" / "state")  # type: ignore[arg-type]


def test_operation_rejects_reentry_after_exit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")

    with operation:
        pass

    with pytest.raises(SafeWorkspaceError, match="operation is single-use"), operation:
        pass


def test_operation_duplicate_exit_after_success_is_harmless(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")
    transaction = operation._transaction

    operation.__enter__()
    assert transaction._entered is True
    assert transaction._closed is False

    assert operation.__exit__(None, None, None) is False
    assert transaction._entered is False
    assert transaction._closed is True
    assert operation.__exit__(None, None, None) is False
    assert transaction._entered is False
    assert transaction._closed is True

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "succeeded"
    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None


def test_phase_rejects_reentry_after_exit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        phase = op.phase("prepare")
        with phase:
            pass

        with pytest.raises(SafeWorkspaceError, match="phase is single-use"), phase:
            pass


def test_phase_duplicate_exit_after_success_is_harmless(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        phase = op.phase("prepare")

        phase.__enter__()

        assert phase.__exit__(None, None, None) is False
        assert phase.__exit__(None, None, None) is False
        assert phase.phase_record is not None
        assert phase.phase_record.status == "succeeded"

    assert op.operation_run is not None
    assert workspace.journal_store.list_operation_phases(op.operation_run.operation_run_id) == [phase.phase_record]


def test_operation_exit_reconciles_outstanding_active_phase_before_delayed_phase_exit(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")

    operation.__enter__()
    assert operation.operation_run is not None
    phase = operation.phase("prepare")
    phase.__enter__()

    with pytest.raises(SafeWorkspaceError, match="operation cannot exit while phase 'prepare' is still active"):
        operation.__exit__(None, None, None)

    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None
    assert workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")[0].status == "failed"
    assert [
        record.status
        for record in workspace.journal_store.list_operation_phases(operation.operation_run.operation_run_id)
    ] == ["failed"]

    assert phase.__exit__(None, None, None) is False
    assert phase.phase_record is not None
    assert phase.phase_record.status == "failed"


def test_operation_exit_waits_for_recursive_phase_mutation_before_reconciling_active_phase(tmp_path: Path) -> None:
    blocker = _BlockingPortableMkdir()
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=blocker.make_directory,
    )
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state" / "cache")})
    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")

    operation.__enter__()
    assert operation.operation_run is not None
    phase = operation.phase("prepare")
    phase.__enter__()
    errors: list[BaseException] = []
    result_lock = Lock()

    def run_recursive_mkdir() -> None:
        try:
            phase.make_directory(phase.r.state_dir, parents=True)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    operation_exit_started = Event()

    def exit_operation() -> None:
        operation_exit_started.set()
        try:
            operation.__exit__(None, None, None)
        except BaseException as exc:
            with result_lock:
                errors.append(exc)

    mkdir_thread = Thread(target=run_recursive_mkdir)
    exit_thread = Thread(target=exit_operation)

    mkdir_thread.start()
    assert blocker.first_step_started.wait(timeout=5)
    exit_thread.start()
    assert operation_exit_started.wait(timeout=5)
    time.sleep(0.1)
    assert exit_thread.is_alive()

    blocker.allow_first_step_finish.set()
    mkdir_thread.join(timeout=5)
    exit_thread.join(timeout=5)
    assert not mkdir_thread.is_alive()
    assert not exit_thread.is_alive()

    assert len(errors) == 1
    assert isinstance(errors[0], SafeWorkspaceError)
    assert "operation cannot exit while phase 'prepare' is still active" in str(errors[0])
    assert phase.phase_record is not None
    assert phase.phase_record.status == "failed"
    assert [batch.phase for batch in workspace.journal_store.list_batches()] == ["succeeded", "succeeded", "succeeded"]
    assert phase.__exit__(None, None, None) is False
