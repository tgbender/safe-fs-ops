from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_text,
)

from safe_fs_ops import SafeWorkspace, SafeWorkspaceError
from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal import BatchPhase, JournaledFilesystemMutationError, OperationJournalStore
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_transaction_automatic_rollback_removes_recursive_mkdir_parents(tmp_path: Path) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert (tmp_path / "config").exists() is False
    assert (tmp_path / "config" / "state").exists() is False
    batches = workspace.journal_store.list_batches()
    assert [batch.phase for batch in batches] == [
        BatchPhase.RECOVERY_SUCCEEDED,
        BatchPhase.RECOVERY_SUCCEEDED,
        BatchPhase.RECOVERY_SUCCEEDED,
    ]
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batches[0].batch_id)] == [
        "planned",
        "attempting",
        "skipped",
    ]
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batches[1].batch_id)] == [
        "planned",
        "attempting",
        "skipped",
    ]
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batches[2].batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "succeeded",
    ]


def test_transaction_record_only_rollback_leaves_recovery_desired_unexecuted(tmp_path: Path) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="record-only",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "config" / "state").is_dir()
    failed_batch = workspace.journal_store.list_batches()[-1]
    assert failed_batch.phase == BatchPhase.RECOVERY_DESIRED
    assert workspace.journal_store.list_recovery_actions(failed_batch.batch_id) == []


def test_transaction_automatic_rollback_manual_intervention_notes_original_exception(tmp_path: Path) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        try:
            tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        except JournaledFilesystemMutationError:
            blocker = tmp_path / "config" / "state" / "blocker.txt"
            blocker.write_text("keep\n", encoding="utf-8")
            raise RuntimeError("boom") from None

    notes = getattr(excinfo.value, "__notes__", None)
    assert notes is not None
    assert any("automatic rollback also failed" in note for note in notes)
    failed_batch = workspace.journal_store.list_batches()[-1]
    assert failed_batch.phase == BatchPhase.RECOVERY_FAILED
    recovery_actions = workspace.journal_store.list_recovery_actions(failed_batch.batch_id)
    assert recovery_actions[-1].status == "manual_intervention_required"
    assert recovery_actions[-1].payload["exception_payload"]["reason_code"] == "directory_not_empty"
    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "config" / "state").is_dir()
    for resource_key in (
        resources.state_dir.resource_key,
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
    ):
        assert workspace.claim_store.get(resource_key) is not None
    assert workspace.lease_store.active(workspace.lease_name) is not None


def test_transaction_pending_automatic_rollback_failure_keeps_claims_and_lease(tmp_path: Path) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(SafeWorkspaceError, match="automatic rollback failed"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        try:
            tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        except JournaledFilesystemMutationError:
            blocker = tmp_path / "config" / "state" / "blocker.txt"
            blocker.write_text("keep\n", encoding="utf-8")

    failed_batch = workspace.journal_store.list_batches()[-1]
    recovery_actions = workspace.journal_store.list_recovery_actions(failed_batch.batch_id)
    assert failed_batch.phase == BatchPhase.RECOVERY_FAILED
    assert recovery_actions[-1].status == "manual_intervention_required"
    assert workspace.lease_store.active(workspace.lease_name) is not None
    assert workspace.claim_store.get(resources.state_dir.resource_key) is not None


def test_transaction_automatic_rollback_releases_claims_and_lease(tmp_path: Path) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
        rollback="automatic",
    )

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"), tx:
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert tx.cleanup_error is None
    assert workspace.lease_store.active(workspace.lease_name) is None
    for resource_key in (
        resources.state_dir.resource_key,
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
    ):
        assert workspace.claim_store.get(resource_key) is None


def test_transaction_automatic_rollback_does_not_recover_prior_record_only_batch_with_same_run_id(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_post_create_failure(tmp_path / "state.db")
    stale_path = tmp_path / "alpha"
    fresh_path = tmp_path / "beta"
    stale_resources = workspace.resources({"state_dir": workspace.directory(stale_path)})
    fresh_resources = workspace.resources({"state_dir": workspace.directory(fresh_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=stale_resources,
            run_id="shared-run",
            rollback="record-only",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:stale")

    stale_batch = workspace.journal_store.list_batches()[-1]
    assert stale_batch.phase == BatchPhase.RECOVERY_DESIRED
    assert stale_path.is_dir()

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=fresh_resources,
            run_id="shared-run",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:fresh")

    fresh_batch = workspace.journal_store.list_batches()[-1]
    stale_batch = workspace.journal_store.get_batch(stale_batch.batch_id)
    assert fresh_batch.phase == BatchPhase.RECOVERY_DESIRED
    assert stale_batch is not None
    assert stale_batch.phase == BatchPhase.RECOVERY_DESIRED
    assert stale_path.is_dir()
    assert fresh_path.is_dir()


def test_transaction_automatic_rollback_recovers_explicit_phase_recursive_failure_in_same_transaction(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    implicit_path = tmp_path / "implicit" / "state" / "ready"
    explicit_path = tmp_path / "explicit" / "state" / "cache"
    resources = workspace.resources(
        {
            "implicit_dir": workspace.directory(implicit_path),
            "explicit_dir": workspace.directory(explicit_path),
        }
    )

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.implicit_dir, parents=True, idempotency_key="mkdir:implicit")
        implicit_run = tx.operation_run
        implicit_phase = tx.operation_phase
        assert implicit_run is not None
        assert implicit_phase is not None
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=implicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=2,
            payload={"explicit": True},
            operation_phase_id="operation-phase-explicit",
            now=tx.now,
        )
        tx.make_directory(
            tx.r.explicit_dir,
            parents=True,
            idempotency_key="mkdir:explicit",
            operation_run_id=implicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    explicit_batch = workspace.journal_store.list_batches()[-1]
    assert explicit_batch.operation_run_id is not None
    assert explicit_batch.operation_phase_id == "operation-phase-explicit"
    assert explicit_batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert workspace.journal_store.get_operation_run(implicit_run.operation_run_id).status == "failed"
    assert workspace.journal_store.get_operation_phase(implicit_phase.operation_phase_id).status == "failed"
    assert workspace.journal_store.get_operation_phase(explicit_phase.operation_phase_id).status == "failed"
    assert explicit_path.exists() is False
    assert explicit_path.parent.exists() is False
    assert explicit_path.parent.parent.exists() is False
    assert implicit_path.exists() is False
    assert implicit_path.parent.exists() is False


def test_transaction_automatic_rollback_marks_explicit_only_operation_links_failed(tmp_path: Path) -> None:
    workspace = _workspace_with_post_create_failure(tmp_path / "state.db")
    target_path = tmp_path / "explicit" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        explicit_run = workspace.journal_store.create_operation_run(
            run_id=tx.run_id,
            lease=tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
            operation_run_id="operation-run-explicit",
            now=tx.now,
        )
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=explicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=1,
            payload={"explicit": True},
            operation_phase_id="operation-phase-explicit",
            now=tx.now,
        )
        tx.make_directory(
            tx.r.state_dir,
            parents=True,
            idempotency_key="mkdir:explicit",
            operation_run_id=explicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    assert target_path.exists() is False
    assert target_path.parent.exists() is False
    failed_batch = workspace.journal_store.list_batches()[-1]
    assert failed_batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert workspace.journal_store.get_operation_run("operation-run-explicit").status == "failed"
    assert workspace.journal_store.get_operation_phase("operation-phase-explicit").status == "failed"


def test_transaction_automatic_rollback_uses_supplied_time_for_recovery_calls(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace_with_leaf_failure(tmp_path / "state.db")
    workspace.lease_ttl = timedelta(seconds=1)
    workspace._coordinator._clock = lambda: now + timedelta(seconds=5)
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            rollback="automatic",
            now=now,
            cleanup_clock=lambda: now,
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:state")

    assert workspace.journal_store.list_batches()[-1].phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target_path.exists() is False


def test_transaction_automatic_rollback_removes_successful_direct_mkdir(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    target_path = tmp_path / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-unsupported-mkdir",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:direct")
        raise RuntimeError("boom")

    assert target_path.exists() is False
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert getattr(excinfo.value, "__notes__", ()) == ()


def test_transaction_automatic_rollback_treats_recursive_mkdir_noop_as_recovered(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    target_path = tmp_path / "state"
    target_path.mkdir()
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-noop-mkdir",
            rollback="automatic",
        ) as tx,
    ):
        result = tx.make_directory(
            tx.r.state_dir,
            parents=True,
            exist_ok=True,
            idempotency_key="mkdir:noop",
        )
        assert result.skipped is True
        raise RuntimeError("boom")

    assert target_path.is_dir()
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []
    assert getattr(excinfo.value, "__notes__", ()) == ()


def test_transaction_automatic_rollback_refuses_direct_mkdir_without_identity_proof(tmp_path: Path) -> None:
    def snapshot_without_directory_identity(path: Path | str) -> ResourceSnapshot:
        snapshot = portable_snapshot(path)
        if snapshot.file_type != "directory":
            return snapshot
        return ResourceSnapshot(
            path=snapshot.path,
            exists=snapshot.exists,
            file_type=snapshot.file_type,
            content_hash=snapshot.content_hash,
            size=snapshot.size,
            mtime_ns=snapshot.mtime_ns,
            symlink_target=snapshot.symlink_target,
            device=None,
            inode=None,
        )

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=snapshot_without_directory_identity,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    target_path = tmp_path / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-mkdir-without-identity",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:direct")
        raise RuntimeError("boom")

    assert target_path.is_dir()
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []
    notes = getattr(excinfo.value, "__notes__", ())
    assert any("make_directory" in note and "does not support automatic rollback" in note for note in notes)


def test_transaction_keeps_claims_and_lease_when_automatic_rollback_preparation_fails(tmp_path: Path) -> None:
    class FailingRollbackPreparationJournal(OperationJournalStore):
        def record_recovery_desired(self, *args: object, **kwargs: object) -> object:
            payload = kwargs.get("payload")
            if isinstance(payload, dict) and "automatic_transaction_rollback" in payload:
                raise RuntimeError("rollback preparation failed")
            return super().record_recovery_desired(*args, **kwargs)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    journal = FailingRollbackPreparationJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    target_path = tmp_path / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-rollback-prep-fails",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:direct")
        claim_scope = tx.claim_scope
        raise RuntimeError("boom")

    assert target_path.is_dir()
    assert workspace.lease_store.active(workspace.lease_name) is not None
    assert workspace.claim_store.get(resources.state_dir.resource_key) is not None
    assert getattr(excinfo.value, "__notes__", None) is not None
    assert any("rollback preparation failed" in note for note in excinfo.value.__notes__)
    [batch] = journal.list_batches()
    assert batch.claim_scope == claim_scope
    assert batch.phase == BatchPhase.SUCCEEDED


def test_transaction_automatic_rollback_handles_checkpoint_preparation_failure(tmp_path: Path) -> None:
    class FailingExpectedAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args: object, **kwargs: object) -> object:
            if kwargs.get("checkpoint_type") == "expected_after":
                raise RuntimeError("expected-after checkpoint failed")
            return super().record_checkpoint(*args, **kwargs)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
    )
    journal = FailingExpectedAfterCheckpointJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    target_path = tmp_path / "config.txt"
    target_path.write_text("old\n", encoding="utf-8")
    resources = workspace.resources({"file": workspace.file(target_path)})

    with (
        pytest.raises(RuntimeError, match="expected-after checkpoint failed"),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-checkpoint-prep-fails",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:prep-fails")

    [batch] = journal.list_batches()
    recovery_actions = journal.list_recovery_actions(batch.batch_id)

    assert target_path.read_text(encoding="utf-8") == "old\n"
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert recovery_actions != []
    assert all(record.status != "manual_intervention_required" for record in recovery_actions)


def test_transaction_automatic_rollback_does_not_remove_direct_mkdir_failure_observation(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "state"

    def externally_created_before_failure(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        Path(path).mkdir()
        raise FileExistsError(path)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=externally_created_before_failure,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with (
        pytest.raises(JournaledFilesystemMutationError),
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-direct-mkdir-failure-observation",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:direct-failure")

    assert target_path.is_dir()
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []


def test_transaction_automatic_rollback_rejects_unknown_mutating_operation(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    target_path = tmp_path / "config.txt"
    target_path.write_text("old\n", encoding="utf-8")
    resources = workspace.resources({"config": workspace.file(target_path)})
    now = datetime.now(UTC)

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id="run-unknown-operation",
            rollback="automatic",
            now=now,
        ) as tx,
    ):
        payload = {
            "operation": "move_tree",
            "path": str(target_path),
            "resource_key": resources.config.resource_key,
        }
        batch = workspace.journal_store.create_batch(
            idempotency_key="move-tree:unknown",
            lease=tx.lease,
            owner=workspace.owner,
            run_id=tx.run_id,
            resource_key=resources.config.resource_key,
            claim_owner=workspace.owner,
            claim_scope=tx.claim_scope,
            payload=payload,
            now=now,
        )
        batch, _operation = workspace.journal_store.start_batch_operation(
            batch.batch_id,
            lease=tx.lease,
            operation_type="move_tree",
            resource_key=resources.config.resource_key,
            payload=payload,
            now=now,
        )
        workspace.journal_store.mark_succeeded(batch.batch_id, lease=tx.lease, now=now)
        raise RuntimeError("boom")

    [batch] = workspace.journal_store.list_batches(run_id="run-unknown-operation")
    notes = getattr(excinfo.value, "__notes__", ())
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []
    assert any("move_tree" in note and "does not support automatic rollback" in note for note in notes)


@pytest.mark.parametrize("operation", ["rename_no_replace", "capture_directory"])
def test_transaction_automatic_rollback_refuses_artifact_operations_without_durable_proof(
    tmp_path: Path,
    operation: str,
) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    source = tmp_path / "repo" / ".git"
    source.parent.mkdir()
    source.mkdir()
    resource_key = directory_resource_key(source)
    resources = workspace.resources({"git": workspace.directory(source)})
    now = datetime.now(UTC)

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources=resources,
            run_id=f"run-{operation}-missing-proof",
            rollback="automatic",
            now=now,
        ) as tx,
    ):
        payload = {
            "operation": operation,
            "path": str(source),
            "source_path": str(source),
            "destination_path": str(tmp_path / "repo" / ".git.moved"),
            "quarantine_path": str(tmp_path / "repo" / ".safe" / "git-captured"),
            "resource_key": resource_key,
        }
        batch = workspace.journal_store.create_batch(
            idempotency_key=f"{operation}:missing-proof",
            lease=tx.lease,
            owner=workspace.owner,
            run_id=tx.run_id,
            resource_key=resource_key,
            claim_owner=workspace.owner,
            claim_scope=tx.claim_scope,
            payload=payload,
            now=now,
        )
        workspace.journal_store.start_batch_operation(
            batch.batch_id,
            lease=tx.lease,
            operation_type=operation,
            resource_key=resource_key,
            payload=payload,
            now=now,
        )
        workspace.journal_store.mark_succeeded(batch.batch_id, lease=tx.lease, now=now)
        raise RuntimeError("boom")

    [batch] = workspace.journal_store.list_batches(run_id=f"run-{operation}-missing-proof")
    notes = getattr(excinfo.value, "__notes__", ())
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []
    assert any(operation in note and "does not support automatic rollback" in note for note in notes)


def _workspace_with_leaf_failure(state_path: Path) -> SafeWorkspace:
    def fail_on_leaf(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        target = Path(path)
        if target.name == "cache":
            raise RuntimeError("injected mkdir failure")
        target.mkdir()

    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=fail_on_leaf,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )


def _workspace_with_post_create_failure(
    state_path: Path,
    *,
    lease_ttl: timedelta = timedelta(seconds=30),
) -> SafeWorkspace:
    def fail_after_create(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        target = Path(path)
        target.mkdir(parents=parents, exist_ok=exist_ok)
        raise RuntimeError("injected mkdir failure")

    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        lease_ttl=lease_ttl,
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=fail_after_create,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )


def _hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()
