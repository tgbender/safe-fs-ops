from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from workspace_api_helpers import workspace_with_portable_ops

from safe_fs_ops import ResourceNotClaimedError, SafeWorkspaceError
from safe_fs_ops.filesystem_ops import (
    ContentAddressedPutResult,
    ContentAddressedStore,
    ContentRef,
)
from safe_fs_ops.filesystem_ops import (
    backup_tree as create_tree_backup,
)
from safe_fs_ops.operation_journal import BatchPhase, JournaledFilesystemMutationError
from safe_fs_ops.resources import TreeResource
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload

pytestmark = pytest.mark.safe_fs_ops


def _claim_details(lease) -> str:
    return json.dumps(
        lease_claim_details_payload(lease, claim_id=f"claim:{lease.name}:{lease.fencing_token}"),
        separators=(",", ":"),
        sort_keys=True,
    )


def test_workspace_uses_default_tree_backup_artifact_store_root(tmp_path: Path) -> None:
    state_path = tmp_path / "workspace.sqlite"
    workspace = workspace_with_portable_ops(state_path)

    assert isinstance(workspace.artifact_store, ContentAddressedStore)
    assert workspace.artifact_store.root == tmp_path / "workspace.sqlite.objects"


def test_workspace_allows_custom_tree_backup_artifact_store(tmp_path: Path) -> None:
    store = object()
    workspace = workspace_with_portable_ops(tmp_path / "state.db", artifact_store=store)

    assert workspace.artifact_store is store


def test_transaction_backup_tree_requires_claimed_tree_resource(tmp_path: Path) -> None:
    workspace = workspace_with_tree_backup_coordinator(tmp_path)
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})
    other = workspace.tree(tmp_path / "other")

    with (
        workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx,
        pytest.raises(ResourceNotClaimedError, match="was not claimed"),
    ):
        tx.backup_tree(other, ["config.toml"])


def test_transaction_backup_tree_rejects_alias_handle_with_same_resource_key(tmp_path: Path) -> None:
    workspace = workspace_with_tree_backup_coordinator(tmp_path)
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})
    alias = workspace.tree(tmp_path / "tree")

    with (
        workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx,
        pytest.raises(ResourceNotClaimedError, match="claimed ResourceSet member"),
    ):
        tx.backup_tree(alias, ["config.toml"])


def test_transaction_backup_tree_delegates_to_coordinator_with_default_store(tmp_path: Path) -> None:
    workspace = workspace_with_tree_backup_coordinator(tmp_path)
    coordinator = cast(_TreeBackupCoordinatorSpy, workspace._coordinator)
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})

    with workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx:
        result = tx.backup_tree(tx.r.tree, ["config.toml"], idempotency_key="backup:key")

    assert result.kind == "backup_tree"
    assert coordinator.calls == [
        {
            "method": "backup_tree",
            "path": tmp_path / "tree",
            "relative_paths": (Path("config.toml"),),
            "artifact_store": workspace.artifact_store,
            "resource_key": resources.tree.resource_key,
            "owner": "owner-a",
            "run_id": "run-1",
            "idempotency_key": "backup:key",
            "operation_run_id": None,
            "operation_phase_id": None,
            "claim_scope": tx.claim_scope,
        }
    ]


def test_transaction_restore_tree_backup_delegates_to_coordinator_with_destination(tmp_path: Path) -> None:
    workspace = workspace_with_tree_backup_coordinator(tmp_path)
    coordinator = cast(_TreeBackupCoordinatorSpy, workspace._coordinator)
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})
    backup = object()
    store = object()

    with workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx:
        result = tx.restore_tree_backup(
            tx.r.tree,
            backup,
            destination_root=tmp_path / "tree",
            artifact_store=store,
            conflict_policy="replace",
            idempotency_key="restore:key",
        )

    assert result.kind == "restore_tree_backup"
    assert coordinator.calls == [
        {
            "method": "restore_tree_backup",
            "backup": backup,
            "destination_root": tmp_path / "tree",
            "artifact_store": store,
            "conflict_policy": "replace",
            "resource_key": resources.tree.resource_key,
            "owner": "owner-a",
            "run_id": "run-1",
            "idempotency_key": "restore:key",
            "operation_run_id": None,
            "operation_phase_id": None,
            "claim_scope": tx.claim_scope,
        }
    ]


def test_transaction_restore_tree_backup_rejects_unclaimed_destination_root(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})

    with (
        workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx,
        pytest.raises(ValueError, match="destination_root must match"),
    ):
        tx.restore_tree_backup(tx.r.tree, object(), destination_root=tmp_path / "restore")


def test_phase_tree_backup_methods_pass_operation_links(tmp_path: Path) -> None:
    workspace = workspace_with_tree_backup_coordinator(tmp_path)
    coordinator = cast(_TreeBackupCoordinatorSpy, workspace._coordinator)
    resources = workspace.resources({"tree": workspace.tree(tmp_path / "tree")})

    with workspace.operation(name="sync", resources=resources, run_id="run-1") as op, op.phase("backup") as phase:
        tree = cast(TreeResource, phase.r.tree)
        phase.backup_tree(tree, ["config.toml"])
        phase.restore_tree_backup(tree, object())
        assert op.operation_run is not None
        assert phase.phase_record is not None
        operation_run_id = op.operation_run.operation_run_id
        operation_phase_id = phase.phase_record.operation_phase_id

    assert [
        (call["method"], call["operation_run_id"], call["operation_phase_id"], call["idempotency_key"])
        for call in coordinator.calls
    ] == [
        ("backup_tree", operation_run_id, operation_phase_id, "sync:run-1:backup:1:backup_tree:tree"),
        ("restore_tree_backup", operation_run_id, operation_phase_id, "sync:run-1:backup:2:restore_tree_backup:tree"),
    ]


def test_transaction_backup_tree_records_checkpoint_with_real_coordinator(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "tree"
    root.mkdir()
    (root / "config.toml").write_text("enabled = true\n", encoding="utf-8")
    workspace = workspace_with_portable_ops(state_path)
    resources = workspace.resources({"tree": workspace.tree(root)})

    with workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx:
        result = tx.backup_tree(tx.r.tree, ["config.toml"], idempotency_key="backup:key")

    assert result.batch.phase == "succeeded"
    checkpoints = workspace.journal_store.list_checkpoints(result.batch.batch_id)
    tree_backup_checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint.checkpoint_type == "tree_backup"]
    assert len(tree_backup_checkpoints) == 1
    assert tree_backup_checkpoints[0].resource_key == resources.tree.resource_key
    assert tree_backup_checkpoints[0].payload["root"] == str(root)


def test_transaction_restore_tree_backup_records_checkpoints_with_real_coordinator(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.toml").write_text("enabled = true\n", encoding="utf-8")
    restore_root = tmp_path / "restore"
    workspace = workspace_with_portable_ops(state_path)
    backup = create_tree_backup(source, ["config.toml"], artifact_store=workspace.artifact_store)
    resources = workspace.resources({"tree": workspace.tree(restore_root)})

    with workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx:
        result = tx.restore_tree_backup(tx.r.tree, backup, idempotency_key="restore:key")

    assert result.batch.phase == "succeeded"
    assert (restore_root / "config.toml").read_text(encoding="utf-8") == "enabled = true\n"
    checkpoints = workspace.journal_store.list_checkpoints(result.batch.batch_id)
    assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == ["before", "after"]
    assert {checkpoint.resource_key for checkpoint in checkpoints} == {resources.tree.resource_key}


def test_transaction_automatic_rollback_rejects_restore_tree_backup_before_mutation(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.toml").write_text("enabled = true\n", encoding="utf-8")
    restore_root = tmp_path / "restore"
    workspace = workspace_with_portable_ops(state_path)
    backup = create_tree_backup(source, ["config.toml"], artifact_store=workspace.artifact_store)
    resources = workspace.resources({"tree": workspace.tree(restore_root)})

    with pytest.raises(SafeWorkspaceError, match="restore_tree_backup is not supported"):
        with workspace.transaction(name="sync", resources=resources, run_id="run-1", rollback="automatic") as tx:
            tx.restore_tree_backup(tx.r.tree, backup, idempotency_key="restore:key")

    assert restore_root.exists() is False
    assert workspace.journal_store.list_batches() == []


def test_transaction_automatic_rollback_does_not_restore_read_only_tree_backup(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "config.toml"
    target.write_text("enabled = true\n", encoding="utf-8")
    workspace = workspace_with_portable_ops(state_path)
    resources = workspace.resources({"tree": workspace.tree(root)})

    with pytest.raises(RuntimeError, match="boom"):
        with workspace.transaction(name="sync", resources=resources, run_id="run-1", rollback="automatic") as tx:
            tx.backup_tree(tx.r.tree, ["config.toml"], idempotency_key="backup:key")
            target.unlink()
            raise RuntimeError("boom")

    assert target.exists() is False
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.SUCCEEDED
    assert workspace.journal_store.list_recovery_records(batch.batch_id) == []
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []


def test_public_recovery_cleans_partial_tree_backup_artifact_before_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "config.toml"
    target.write_text("enabled = true\n", encoding="utf-8")
    workspace = workspace_with_portable_ops(state_path)
    resource = workspace.tree(root)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )
    written_paths: list[Path] = []

    def write_artifact_then_fail(
        path: Path | str,
        relative_paths: Iterable[Path | str],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        [relative_path] = list(relative_paths)
        ref = artifact_store.put_file(Path(path) / relative_path)
        written_paths.append(ref.path)
        raise RuntimeError("simulated crash after artifact write")

    with pytest.raises(JournaledFilesystemMutationError, match="simulated crash"):
        workspace._coordinator.backup_tree(
            root,
            ["config.toml"],
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-partial-tree-backup",
            idempotency_key="backup-tree:partial-before-checkpoint",
            backup_tree_operation=write_artifact_then_fail,
            now=crash_time,
        )

    [batch] = workspace.journal_store.list_batches(run_id="run-partial-tree-backup")
    checkpoints = workspace.journal_store.list_checkpoints(batch.batch_id)
    assert batch.phase == BatchPhase.FAILED
    assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == ["tree_backup_partial_artifacts"]
    assert written_paths and written_paths[0].exists()

    recovery_error = workspace.recover_pending_batches(run_id="run-partial-tree-backup")

    recovered = workspace.journal_store.get_batch(batch.batch_id)
    assert recovery_error is None
    assert recovered is not None
    assert recovered.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert written_paths[0].exists() is False
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert cleanup_records[0].payload["artifact_type"] == "tree_backup_object"
    assert target.read_text(encoding="utf-8") == "enabled = true\n"


def test_partial_tree_backup_tracks_actual_ref_when_source_changes_during_put(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "config.toml"
    target.write_text("enabled = true\n", encoding="utf-8")
    store = _MutatingBeforePutStore(tmp_path / "cas", content="enabled = false\n")
    workspace = workspace_with_portable_ops(state_path, artifact_store=store)
    resource = workspace.tree(root)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )
    written_refs: list[ContentRef] = []

    def write_artifact_then_fail(
        path: Path | str,
        relative_paths: Iterable[Path | str],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        [relative_path] = list(relative_paths)
        ref = artifact_store.put_file(Path(path) / relative_path)
        written_refs.append(ref)
        raise RuntimeError("simulated crash after changed artifact write")

    with pytest.raises(JournaledFilesystemMutationError, match="simulated crash"):
        workspace._coordinator.backup_tree(
            root,
            ["config.toml"],
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-partial-tree-backup-changed",
            idempotency_key="backup-tree:partial-before-checkpoint-changed",
            artifact_store=store,
            backup_tree_operation=write_artifact_then_fail,
            now=crash_time,
        )

    [batch] = workspace.journal_store.list_batches(run_id="run-partial-tree-backup-changed")
    [checkpoint] = workspace.journal_store.list_checkpoints(batch.batch_id)
    changed_digest = hashlib.sha256(b"enabled = false\n").hexdigest()
    [written_ref] = written_refs
    assert batch.phase == BatchPhase.FAILED
    assert checkpoint.checkpoint_type == "tree_backup_partial_artifacts"
    assert written_ref.digest == changed_digest
    assert [dict(ref_payload) for ref_payload in checkpoint.payload["content_refs"]] == [
        {
            "algo": "sha256",
            "digest": changed_digest,
            "size": len(b"enabled = false\n"),
            "path": str(written_ref.path),
        }
    ]


def test_transaction_backup_tree_rejects_custom_artifact_store_without_recovery_path(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "tree"
    root.mkdir()
    (root / "config.toml").write_text("enabled = true\n", encoding="utf-8")
    workspace = workspace_with_portable_ops(state_path, artifact_store=object())
    resources = workspace.resources({"tree": workspace.tree(root)})

    with (
        workspace.transaction(name="sync", resources=resources, run_id="run-1") as tx,
        pytest.raises(TypeError, match="ContentAddressedStore"),
    ):
        tx.backup_tree(tx.r.tree, ["config.toml"])


class _MutatingBeforePutStore(ContentAddressedStore):
    def __init__(self, root: Path | str, *, content: str) -> None:
        super().__init__(root)
        self._content = content

    def put_file(self, path: Path | str) -> ContentRef:
        return self.put_file_with_status(path).ref

    def put_file_with_status(self, path: Path | str) -> ContentAddressedPutResult:
        Path(path).write_text(self._content, encoding="utf-8", newline="")
        return super().put_file_with_status(path)


def workspace_with_tree_backup_coordinator(tmp_path: Path) -> Any:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    workspace._coordinator = _TreeBackupCoordinatorSpy()
    return workspace


class _TreeBackupCoordinatorSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def backup_tree(
        self,
        path: Path | str,
        relative_paths: Iterable[Path | str],
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "method": "backup_tree",
                "path": Path(path),
                "relative_paths": tuple(Path(relative_path) for relative_path in relative_paths),
                **_selected_kwargs(kwargs),
            }
        )
        return SimpleNamespace(kind="backup_tree")

    def restore_tree_backup(
        self,
        backup: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "method": "restore_tree_backup",
                "backup": backup,
                **_selected_kwargs(kwargs),
            }
        )
        return SimpleNamespace(kind="restore_tree_backup")


def _selected_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    return {
        key: Path(value) if key == "destination_root" and isinstance(value, str | Path) else value
        for key, value in kwargs.items()
        if key
        in {
            "artifact_store",
            "claim_scope",
            "conflict_policy",
            "destination_root",
            "idempotency_key",
            "operation_phase_id",
            "operation_run_id",
            "owner",
            "resource_key",
            "run_id",
        }
    }
