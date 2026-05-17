from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from workspace_api_helpers import (
    portable_delete_file,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_text,
)

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.filesystem_ops import DirectoryIdentity
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemMutationError,
    JournaledFilesystemRecoveryError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_phase_recursive_mkdir_replacement_after_after_checkpoint_does_not_authorize_automatic_root_rollback(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "config" / "state" / "cache"
    replaced_root = tmp_path / "config"

    def hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
        del missing_ok
        target = Path(path)
        if hooks is not None:
            hooks.after_rmdir_validation(target)
        target.rmdir()

    class ReplacingAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args, **kwargs):
            checkpoint = super().record_checkpoint(*args, **kwargs)
            if kwargs.get("checkpoint_type") == "after" and kwargs.get("resource_key") == directory_resource_key(
                replaced_root
            ):
                replaced_root.rmdir()
                replaced_root.mkdir()
            return checkpoint

    def fail_on_leaf(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        target = Path(path)
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
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
    )
    journal = ReplacingAfterCheckpointJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    workspace._coordinator._make_directory = fail_on_leaf
    workspace._coordinator._remove_directory = hook_capable_remove_directory
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True)

    batches = journal.list_batches()
    failed_batch = batches[-1]
    failed_recovery = journal.list_recovery_records(failed_batch.batch_id)[0]
    prior_created_steps = list(failed_recovery.payload["recursive_group"]["prior_created_steps"])
    root_step = prior_created_steps[0]
    root_after_checkpoint = journal.list_checkpoints(batches[0].batch_id)[1]
    current_root_identity = DirectoryIdentity.from_stat(replaced_root.stat())

    assert root_step["created_directory_identity"]["inode"] == root_after_checkpoint.payload["inode"]
    assert root_step["created_directory_identity"]["device"] == root_after_checkpoint.payload["device"]
    if root_step["created_directory_identity"]["inode"] == current_root_identity.inode:
        pytest.skip("filesystem reused directory identity for remove/recreate")
    assert root_step["created_directory_identity"]["inode"] != current_root_identity.inode

    recovery_lease = workspace.lease_store.acquire("workspace", owner="owner-a", ttl=workspace.lease_ttl)
    assert recovery_lease.acquired

    with pytest.raises(JournaledFilesystemRecoveryError, match="recovery callback failed"):
        workspace._coordinator.recover_batch(
            failed_batch.batch_id,
            lease=recovery_lease,
            recover=lambda context: workspace._coordinator.run_recovery_actions(context, lease=recovery_lease),
        )

    assert (replaced_root / "state").exists() is False
    assert replaced_root.is_dir()
    assert journal.list_recovery_actions(failed_batch.batch_id)[-1].status == "manual_intervention_required"
    assert journal.list_recovery_actions(failed_batch.batch_id)[-1].payload["exception_payload"]["reason_code"] == (
        "created_directory_identity_mismatch"
    )


def test_phase_recursive_mkdir_missing_after_checkpoint_identity_requires_manual_recovery(
    tmp_path: Path,
) -> None:
    target_path = tmp_path / "config" / "state" / "cache"
    created_root = tmp_path / "config"
    remove_calls: list[Path] = []
    identity_remove_calls: list[Path] = []

    def snapshot_without_root_identity(path: Path | str):
        snapshot = portable_snapshot(path)
        if snapshot.path == created_root and snapshot.exists and snapshot.file_type == "directory":
            return replace(snapshot, device=None, inode=None)
        return snapshot

    def fail_on_leaf(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        del parents, exist_ok
        target = Path(path)
        if target.name == "cache":
            raise RuntimeError("injected mkdir failure")
        target.mkdir()

    def hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
        del missing_ok
        target = Path(path)
        remove_calls.append(target)
        if hooks is not None:
            hooks.after_rmdir_validation(target)
        target.rmdir()

    def identity_remove_directory(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        identity_remove_calls.append(Path(path))
        portable_remove_empty_directory_by_identity(path, expected_identity=expected_identity)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=snapshot_without_root_identity,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=fail_on_leaf,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=identity_remove_directory,
    )
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True)

    failed_batch = workspace.journal_store.list_batches()[-1]
    failed_recovery = workspace.journal_store.list_recovery_records(failed_batch.batch_id)[0]
    prior_created_steps = list(failed_recovery.payload["recursive_group"]["prior_created_steps"])
    assert [step["step_path"] for step in prior_created_steps] == [str(created_root / "state")]

    recovery_lease = workspace.lease_store.acquire("workspace", owner="owner-a", ttl=workspace.lease_ttl)
    assert recovery_lease.acquired

    with pytest.raises(JournaledFilesystemRecoveryError, match="recovery callback failed"):
        workspace._coordinator.recover_batch(
            failed_batch.batch_id,
            lease=recovery_lease,
            recover=lambda context: workspace._coordinator.run_recovery_actions(context, lease=recovery_lease),
        )

    recovered_batch = workspace.journal_store.get_batch(failed_batch.batch_id)
    assert recovered_batch is not None
    assert recovered_batch.phase == BatchPhase.RECOVERY_FAILED
    assert remove_calls == []
    assert identity_remove_calls == []
    assert created_root.is_dir()
    assert (created_root / "state").is_dir()
    assert target_path.exists() is False
    actions = workspace.journal_store.list_recovery_actions(failed_batch.batch_id)
    assert [action.status for action in actions] == ["planned", "attempting", "manual_intervention_required"]
    assert actions[-1].payload["manual_intervention"]["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"
