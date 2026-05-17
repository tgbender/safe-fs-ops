from __future__ import annotations

from pathlib import Path

import pytest
from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_text,
    workspace_with_portable_ops,
)

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.filesystem_ops import DirectoryIdentity, PathSafety, UnsafePathError
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemMutationError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_phase_recursive_mkdir_creates_missing_parents_shallow_to_deep(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            result = phase.make_directory(phase.r.state_dir, parents=True)

    assert target_path.is_dir()
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.resource_key == resources.state_dir.resource_key
    batches = workspace.journal_store.list_batches()
    assert [batch.resource_key for batch in batches] == [
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
        resources.state_dir.resource_key,
    ]
    assert [batch.phase for batch in batches] == [BatchPhase.SUCCEEDED, BatchPhase.SUCCEEDED, BatchPhase.SUCCEEDED]
    assert {(batch.operation_run_id, batch.operation_phase_id) for batch in batches} == {
        (operation.operation_run.operation_run_id, phase.phase_record.operation_phase_id)
    }
    assert [batch.payload["recursive_mkdir"]["step_index"] for batch in batches] == [1, 2, 3]
    assert [batch.payload["recursive_mkdir"]["step_count"] for batch in batches] == [3, 3, 3]
    assert all(batch.payload["rollback_diagnostic"]["automatic_recursive_delete"] is False for batch in batches)
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None
    for resource_key in [batch.resource_key for batch in batches]:
        assert resource_key is not None
        assert workspace.claim_store.get(resource_key) is None


def test_phase_recursive_mkdir_existing_target_with_exist_ok_is_noop(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state"
    target_path.mkdir(parents=True)
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            result = phase.make_directory(phase.r.state_dir, parents=True, exist_ok=True)

    assert result.skipped is True
    assert target_path.is_dir()
    batches = workspace.journal_store.list_batches()
    assert [batch.phase for batch in batches] == [BatchPhase.SUCCEEDED]
    assert batches[0].batch_id == result.batch.batch_id
    assert batches[0].payload["noop"] is True
    assert batches[0].payload["recursive_mkdir"]["step_index"] == 0
    assert batches[0].operation_run_id == operation.operation_run.operation_run_id
    assert batches[0].operation_phase_id == phase.phase_record.operation_phase_id
    operations = workspace.journal_store.list_operations(result.batch.batch_id)
    checkpoints = workspace.journal_store.list_checkpoints(result.batch.batch_id)
    assert len(operations) == 1
    assert operations[0].operation_type == "make_directory_noop"
    assert len(checkpoints) == 2
    assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == ["before", "after"]
    assert checkpoints[0].payload == checkpoints[1].payload


def test_phase_recursive_mkdir_noop_rechecks_runtime_target_safety_before_journaling(tmp_path: Path) -> None:
    target_path = tmp_path / "config" / "state"
    target_path.mkdir(parents=True)
    target_checks = {"count": 0}

    def inspect_runtime(path: Path) -> PathSafety:
        checked = Path(path)
        if checked == target_path:
            target_checks["count"] += 1
            is_mount = target_checks["count"] > 2
        else:
            is_mount = False
        if checked.exists():
            file_type = "directory" if checked.is_dir() else "file" if checked.is_file() else "other"
            size = checked.lstat().st_size
        else:
            file_type = "missing"
            size = None
        return PathSafety(
            path=checked,
            exists=checked.exists(),
            file_type=file_type,
            is_mount=is_mount,
            is_windows_reparse_point=False,
            hardlink_count=1,
            size=size,
        )

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        path_inspector=inspect_runtime,
    )
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            with pytest.raises(UnsafePathError, match="mount point"):
                phase.make_directory(phase.r.state_dir, parents=True, exist_ok=True)

    assert target_checks["count"] >= 3
    assert target_path.is_dir()
    assert workspace.journal_store.list_batches() == []


def test_phase_recursive_mkdir_noop_rechecks_runtime_existing_ancestors_before_journaling(tmp_path: Path) -> None:
    target_path = tmp_path / "config" / "state"
    target_path.mkdir(parents=True)
    target_checks = {"count": 0}

    def inspect_runtime(path: Path) -> PathSafety:
        checked = Path(path)
        if checked == target_path:
            target_checks["count"] += 1
        if checked.exists():
            file_type = "directory" if checked.is_dir() else "file" if checked.is_file() else "other"
            size = checked.lstat().st_size
        else:
            file_type = "missing"
            size = None
        return PathSafety(
            path=checked,
            exists=checked.exists(),
            file_type=file_type,
            is_mount=checked == target_path.parent.parent and target_checks["count"] > 2,
            is_windows_reparse_point=False,
            hardlink_count=1,
            size=size,
        )

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        path_inspector=inspect_runtime,
    )
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            with pytest.raises(UnsafePathError, match="mount point"):
                phase.make_directory(phase.r.state_dir, parents=True, exist_ok=True)

    assert target_checks["count"] >= 3
    assert target_path.is_dir()
    assert workspace.journal_store.list_batches() == []


def test_phase_recursive_mkdir_rechecks_runtime_ancestor_safety_before_each_step(tmp_path: Path) -> None:
    mutated_parent = tmp_path / "config"
    state = {"mount_after_first_step": False}

    def inspect_runtime(path: Path) -> PathSafety:
        checked = Path(path)
        if checked.exists():
            file_type = "directory" if checked.is_dir() else "file" if checked.is_file() else "other"
            size = checked.lstat().st_size
        else:
            file_type = "missing"
            size = None
        return PathSafety(
            path=checked,
            exists=checked.exists(),
            file_type=file_type,
            is_mount=checked == mutated_parent and state["mount_after_first_step"],
            is_windows_reparse_point=False,
            hardlink_count=1,
            size=size,
        )

    def mutate_and_flip(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        portable_make_directory(path, parents=parents, exist_ok=exist_ok)
        if Path(path) == mutated_parent:
            state["mount_after_first_step"] = True

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=mutate_and_flip,
        path_inspector=inspect_runtime,
    )
    target_path = mutated_parent / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with pytest.raises(UnsafePathError, match="mount point"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True)

    assert mutated_parent.is_dir()
    assert not (mutated_parent / "state").exists()
    batches = workspace.journal_store.list_batches()
    assert [batch.resource_key for batch in batches] == [directory_resource_key(mutated_parent)]
    assert [batch.phase for batch in batches] == [BatchPhase.SUCCEEDED]


def test_phase_recursive_mkdir_noop_after_checkpoint_failure_records_recovery_state(tmp_path: Path) -> None:
    target_path = tmp_path / "config" / "state"
    target_path.mkdir(parents=True)

    class FailingAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args, **kwargs):
            if kwargs.get("checkpoint_type") == "after":
                raise RuntimeError("injected after checkpoint failure")
            return super().record_checkpoint(*args, **kwargs)

    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    journal = FailingAfterCheckpointJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with pytest.raises(RuntimeError, match="injected after checkpoint failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True, exist_ok=True)

    assert target_path.is_dir()
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem post-mutation journal failed"
    assert batch.status_payload["failure"]["stage"] == "after_checkpoint"
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == ["before"]
    operations = journal.list_operations(batch.batch_id)
    assert len(operations) == 1
    assert operations[0].operation_type == "make_directory_noop"


def test_phase_recursive_mkdir_rejects_unsafe_existing_component_before_side_effects(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    file_parent = tmp_path / "config"
    file_parent.write_text("not-a-directory\n", encoding="utf-8")
    target_path = file_parent / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            with pytest.raises(UnsafePathError, match="not a directory"):
                phase.make_directory(phase.r.state_dir, parents=True)

    assert workspace.journal_store.list_batches() == []
    assert not target_path.exists()


def test_phase_recursive_mkdir_failure_cleans_dynamic_claims_and_records_step_diagnostics(tmp_path: Path) -> None:
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

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True)

    assert calls == [
        tmp_path / "config",
        tmp_path / "config" / "state",
        tmp_path / "config" / "state" / "cache",
    ]
    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "config" / "state").is_dir()
    assert not target_path.exists()
    batches = workspace.journal_store.list_batches()
    assert [batch.phase for batch in batches] == [
        BatchPhase.SUCCEEDED,
        BatchPhase.SUCCEEDED,
        BatchPhase.RECOVERY_DESIRED,
    ]
    for index, batch in enumerate(batches[:2], start=1):
        assert batch.payload["recursive_mkdir"]["step_index"] == index
        assert batch.payload["rollback_diagnostic"]["manual_intervention_possible"] is True
        assert batch.payload["rollback_diagnostic"]["remove_only_if"]["directory_is_empty"] is True
    failed_recovery = workspace.journal_store.list_recovery_records(batches[-1].batch_id)[0]
    assert failed_recovery.payload["recursive_mkdir"]["step_index"] == 3
    assert failed_recovery.payload["rollback_diagnostic"]["remove_only_if"]["created_by_exact_step"] is True
    prior_created_steps = list(failed_recovery.payload["recursive_group"]["prior_created_steps"])
    for index, step_path in enumerate(
        [tmp_path / "config", tmp_path / "config" / "state"],
        start=1,
    ):
        prior_step = prior_created_steps[index - 1]
        step_batch = batches[index - 1]
        step_operation = workspace.journal_store.list_operations(step_batch.batch_id)[0]
        step_after_checkpoint = workspace.journal_store.list_checkpoints(step_batch.batch_id)[1]
        assert prior_step == {
            "step_index": index,
            "step_path": str(step_path),
            "step_resource_key": directory_resource_key(step_path),
            "ownership_class": "created_by_transaction",
            "created_directory_identity": {
                "path": str(step_path),
                "resource_key": directory_resource_key(step_path),
                "file_type": "directory",
                "device": DirectoryIdentity.from_stat(step_path.stat()).device,
                "inode": DirectoryIdentity.from_stat(step_path.stat()).inode,
            },
            "journaled_creation_proof": {
                "batch_id": step_batch.batch_id,
                "operation_id": step_operation.operation_id,
                "operation_type": "make_directory",
                "checkpoint_id": step_after_checkpoint.checkpoint_id,
                "checkpoint_type": "after",
                "path": str(step_path),
                "resource_key": directory_resource_key(step_path),
            },
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
            "cleanup": {
                "manual_intervention_possible": True,
                "cleanup_policy": "owned_empty_directory_safe_ish",
                "backend_guarantee": "identity_conditional_remove",
                "remove_only_if": {
                    "created_by_exact_step": True,
                    "directory_is_empty": True,
                    "path_is_still_safe_directory": True,
                    "resource_key_still_matches": True,
                },
            },
        }
    for resource_key in [
        directory_resource_key(tmp_path / "config"),
        directory_resource_key(tmp_path / "config" / "state"),
        resources.state_dir.resource_key,
    ]:
        assert workspace.claim_store.get(resource_key) is None


def test_phase_recursive_mkdir_recovery_cleanup_removes_owned_directories(
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

    def hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
        del missing_ok
        target = Path(path)
        if hooks is not None:
            hooks.after_rmdir_validation(target)
        target.rmdir()

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
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with pytest.raises(JournaledFilesystemMutationError, match="injected mkdir failure"):
        with workspace.operation(name="sync", resources=resources, run_id="run-1") as operation:
            with operation.phase("prepare") as phase:
                phase.make_directory(phase.r.state_dir, parents=True)

    failed_batch = workspace.journal_store.list_batches()[-1]
    recovery_lease = workspace.lease_store.acquire("workspace", owner="owner-a", ttl=workspace.lease_ttl)
    assert recovery_lease.acquired

    recovery_result = workspace._coordinator.recover_batch(
        failed_batch.batch_id,
        lease=recovery_lease,
        recover=lambda context: workspace._coordinator.run_recovery_actions(context, lease=recovery_lease),
    )

    assert calls == [
        tmp_path / "config",
        tmp_path / "config" / "state",
        tmp_path / "config" / "state" / "cache",
    ]
    assert failed_batch.batch_id == workspace.journal_store.list_batches()[-1].batch_id
    assert recovery_result.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert workspace.journal_store.get_batch(failed_batch.batch_id).phase == BatchPhase.RECOVERY_SUCCEEDED  # type: ignore[union-attr]
    assert (tmp_path / "config").exists() is False
    assert (tmp_path / "config" / "state").exists() is False
    assert target_path.exists() is False
    assert [record.status for record in workspace.journal_store.list_recovery_actions(failed_batch.batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "succeeded",
    ]


def test_phase_recursive_mkdir_scopes_recursive_step_idempotency_by_path_shape(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    first_target = tmp_path / "alpha" / "state"
    second_target = tmp_path / "beta" / "state"
    first_resources = workspace.resources({"state_dir": workspace.directory(first_target)})
    second_resources = workspace.resources({"state_dir": workspace.directory(second_target)})

    with workspace.operation(name="sync", resources=first_resources, run_id="run-1") as operation:
        with operation.phase("prepare") as phase:
            phase.make_directory(phase.r.state_dir, parents=True, idempotency_key="shared-recursive-key")

    with workspace.operation(name="sync", resources=second_resources, run_id="run-2") as operation:
        with operation.phase("prepare") as phase:
            phase.make_directory(phase.r.state_dir, parents=True, idempotency_key="shared-recursive-key")

    assert first_target.is_dir()
    assert second_target.is_dir()
    assert [batch.resource_key for batch in workspace.journal_store.list_batches()] == [
        directory_resource_key(tmp_path / "alpha"),
        directory_resource_key(first_target),
        directory_resource_key(tmp_path / "beta"),
        directory_resource_key(second_target),
    ]
