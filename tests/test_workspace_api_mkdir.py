from __future__ import annotations

from pathlib import Path

import pytest
from workspace_api_helpers import (
    portable_delete_file,
    portable_snapshot,
    portable_write_text,
    workspace_with_portable_ops,
)

from safe_fs_ops import ResourceNotClaimedError, SafeWorkspace
from safe_fs_ops.operation_journal import BatchPhase, JournaledFilesystemMutationError
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key

pytestmark = pytest.mark.safe_fs_ops


def test_transaction_creates_claimed_directory_resource(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state"
    target_path.parent.mkdir()
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        result = tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:state")

    assert target_path.is_dir()
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.resource_key == resources.state_dir.resource_key
    assert tx.operation_run is None
    assert tx.operation_phase is None
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


def test_transaction_recursive_mkdir_creates_missing_parents_with_implicit_operation_context(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        result = tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        operation_run = tx.operation_run
        operation_phase = tx.operation_phase

    assert operation_run is not None
    assert operation_phase is not None
    assert target_path.is_dir()
    assert result.batch.phase == BatchPhase.SUCCEEDED
    batches = workspace.journal_store.list_batches()
    assert [batch.resource_key for batch in batches] == [
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
        resources.state_dir.resource_key,
    ]
    assert {(batch.operation_run_id, batch.operation_phase_id) for batch in batches} == {
        (operation_run.operation_run_id, operation_phase.operation_phase_id)
    }
    assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "succeeded"
    assert workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "succeeded"
    for resource_key in [batch.resource_key for batch in batches]:
        assert resource_key is not None
        assert workspace.claim_store.get(resource_key) is None


def test_transaction_recursive_mkdir_honors_explicit_operation_links(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        operation_run = workspace.journal_store.create_operation_run(
            run_id=tx.run_id,
            lease=tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
            now=tx.now,
        )
        operation_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=operation_run.operation_run_id,
            lease=tx.lease,
            phase_name="transaction",
            status="active",
            phase_order=1,
            payload={"implicit": True, "rollback": tx.rollback},
            now=tx.now,
        )

        result = tx.make_directory(
            tx.r.state_dir,
            parents=True,
            idempotency_key="mkdir:state",
            operation_run_id=operation_run.operation_run_id,
            operation_phase_id=operation_phase.operation_phase_id,
        )

    assert target_path.is_dir()
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert tx.operation_run is None
    assert tx.operation_phase is None
    assert {(batch.operation_run_id, batch.operation_phase_id) for batch in workspace.journal_store.list_batches()} == {
        (operation_run.operation_run_id, operation_phase.operation_phase_id)
    }
    assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "succeeded"
    assert workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "succeeded"


def test_transaction_recursive_mkdir_failure_cleans_dynamic_claims_and_marks_implicit_operation_failed(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    def fail_on_leaf(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        target = Path(path)
        calls.append(target)
        if target.name == "cache":
            raise RuntimeError("injected mkdir failure")
        target.mkdir()

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=fail_on_leaf,
    )
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"), workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
    ) as tx:
        tx_ref = tx
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert calls == [
        tmp_path / "config",
        tmp_path / "config" / "state",
        tmp_path / "config" / "state" / "cache",
    ]
    assert tx_ref is not None
    assert tx_ref.operation_run is not None
    assert tx_ref.operation_phase is not None
    assert workspace.journal_store.get_operation_run(tx_ref.operation_run.operation_run_id).status == "failed"
    assert workspace.journal_store.get_operation_phase(tx_ref.operation_phase.operation_phase_id).status == "failed"
    assert [batch.phase for batch in workspace.journal_store.list_batches()] == [
        BatchPhase.SUCCEEDED,
        BatchPhase.SUCCEEDED,
        BatchPhase.RECOVERY_DESIRED,
    ]
    for resource_key in [
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
        resources.state_dir.resource_key,
    ]:
        assert workspace.claim_store.get(resource_key) is None


def test_transaction_rejects_unclaimed_directory_resource(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})
    other = workspace.directory(tmp_path / "other")

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(
        ResourceNotClaimedError,
        match="was not claimed",
    ):
        tx.make_directory(other)


def test_transaction_rejects_directory_alias_handle_with_same_resource_key(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})
    alias = workspace.directory(tmp_path / "config" / "state")

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(
        ResourceNotClaimedError,
        match="claimed ResourceSet member",
    ):
        tx.make_directory(alias)


def test_transaction_rejects_raw_paths_for_directory_operations(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(
        TypeError,
        match="DirectoryResource",
    ):
        tx.make_directory(tmp_path / "config" / "state")  # type: ignore[arg-type]
