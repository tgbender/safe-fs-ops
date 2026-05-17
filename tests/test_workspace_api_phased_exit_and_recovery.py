from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest
from _workspace_api_helpers import _StatusFailureJournalStore, _workspace

from safe_fs_ops import SafeWorkspaceError
from safe_fs_ops.resources import FileResource
from safe_fs_ops.workspace_state import LeaseLostError

pytestmark = pytest.mark.safe_fs_ops


def test_operation_releases_claims_and_lease_on_phase_exception(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with pytest.raises(RuntimeError, match="boom"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
            with op.phase("prepare") as phase:
                config = cast(FileResource, phase.r.config)
                assert config.path == tmp_path / "config.txt"
                phase.write_text(config, "prepare\n")
                raise RuntimeError("boom")

    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None
    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "failed"
    phases = workspace.journal_store.list_operation_phases(operation_runs[0].operation_run_id)
    assert [record.phase_name for record in phases] == ["prepare"]
    assert [record.status for record in phases] == ["failed"]


def test_operation_does_not_record_success_when_cleanup_fails_after_lease_expiry(tmp_path: Path) -> None:
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    cleanup_now = first_now + timedelta(seconds=2)
    workspace = _workspace(tmp_path / "state.db", lease_ttl=timedelta(seconds=1))
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with (
        pytest.raises(SafeWorkspaceError, match="failed to release transaction lease"),
        workspace.operation(
            name="sync",
            resources=resources,
            run_id="run-1",
            now=first_now,
            cleanup_clock=lambda: cleanup_now,
        ),
    ):
        pass

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "active"
    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name, now=cleanup_now) is None


def test_operation_rejects_stale_terminal_diagnostics_after_lease_takeover(tmp_path: Path) -> None:
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    takeover_now = first_now + timedelta(seconds=31)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with (
        pytest.raises(LeaseLostError, match="lease 'workspace' is no longer current"),
        workspace.operation(
            name="sync",
            resources=resources,
            run_id="run-1",
            now=first_now,
            cleanup_clock=lambda: takeover_now,
        ) as op,
        op.phase("prepare") as phase,
    ):
        phase.write_text(cast(FileResource, phase.r.config), "prepare\n")
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-b",
            ttl=workspace.lease_ttl,
            now=takeover_now,
        )

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "active"
    phases = workspace.journal_store.list_operation_phases(operation_runs[0].operation_run_id)
    assert [record.phase_name for record in phases] == ["prepare"]
    assert [record.status for record in phases] == ["active"]
    assert workspace.claim_store.get(resources.config.resource_key) is None
    active_lease = workspace.lease_store.active(workspace.lease_name, now=takeover_now)
    assert active_lease is not None
    assert active_lease.owner == "owner-b"


def test_operation_cleanup_failure_after_phase_success_marks_phase_finalization_failed(tmp_path: Path) -> None:
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    cleanup_now_values = iter(
        (
            first_now + timedelta(milliseconds=1),
            first_now + timedelta(seconds=2),
            first_now + timedelta(seconds=2),
            first_now + timedelta(seconds=2),
        )
    )
    workspace = _workspace(tmp_path / "state.db", lease_ttl=timedelta(seconds=1))
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with (
        pytest.raises(SafeWorkspaceError, match="failed to release transaction lease"),
        workspace.operation(
            name="sync",
            resources=resources,
            run_id="run-1",
            now=first_now,
            cleanup_clock=lambda: next(cleanup_now_values),
        ) as op,
        op.phase("prepare"),
    ):
        pass

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "active"
    phases = workspace.journal_store.list_operation_phases(operation_runs[0].operation_run_id)
    assert [record.phase_name for record in phases] == ["prepare"]
    assert [record.status for record in phases] == ["succeeded"]


def test_phase_exit_records_terminal_diagnostic_after_non_lease_status_write_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    workspace._journal_store = _StatusFailureJournalStore(workspace.journal_store, fail_phase_statuses={"succeeded"})
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with pytest.raises(RuntimeError, match="injected phase status failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
            with op.phase("prepare"):
                pass

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "failed"
    phases = workspace.journal_store.list_operation_phases(operation_runs[0].operation_run_id)
    assert [record.phase_name for record in phases] == ["prepare"]
    assert [record.status for record in phases] == ["finalization_failed"]


def test_operation_exit_records_terminal_diagnostic_after_non_lease_run_status_failure(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    workspace._journal_store = _StatusFailureJournalStore(workspace.journal_store, fail_run_statuses={"failed"})
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with pytest.raises(RuntimeError, match="boom"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1"):
            raise RuntimeError("boom")

    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 1
    assert operation_runs[0].status == "failed"
    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None


def test_operation_phase_rows_are_ordered_by_phase_entry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op:
        with op.phase("prepare") as prepare:
            assert prepare.phase_record is not None
        with op.phase("apply") as apply_phase:
            assert apply_phase.phase_record is not None
        with op.phase("finalize") as finalize:
            assert finalize.phase_record is not None

    assert op.operation_run is not None
    assert [
        (record.phase_name, record.phase_order)
        for record in workspace.journal_store.list_operation_phases(op.operation_run.operation_run_id)
    ] == [("prepare", 1), ("apply", 2), ("finalize", 3)]


def test_operation_retry_with_same_run_id_creates_distinct_operation_runs(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as first:
        pass

    retry_workspace = _workspace(tmp_path / "state.db")
    retry_resources = retry_workspace.resources({"config": retry_workspace.file(tmp_path / "config.txt")})

    with retry_workspace.operation(name="sync", resources=retry_resources, run_id="run-1") as second:
        pass

    operation_runs = retry_workspace.journal_store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert len(operation_runs) == 2
    assert first.operation_run is not None
    assert second.operation_run is not None
    assert [record.operation_run_id for record in operation_runs] == [
        first.operation_run.operation_run_id,
        second.operation_run.operation_run_id,
    ]


def test_operation_rejects_unsupported_rollback_mode(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with pytest.raises(NotImplementedError, match="rollback='full'"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1", rollback="full"):
            pass


def test_operation_does_not_expose_public_transaction_field(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    operation = workspace.operation(name="sync", resources=resources, run_id="run-1")

    with pytest.raises(AttributeError):
        _transaction = operation.transaction

    assert operation._transaction.run_id == "run-1"
