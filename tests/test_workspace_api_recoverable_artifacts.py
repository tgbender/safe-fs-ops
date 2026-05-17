from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from workspace_api_helpers import workspace_with_portable_ops

from safe_fs_ops import DirectoryResource
from safe_fs_ops.operation_journal import BatchPhase

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_transaction_records_snapshot_bundle_for_claimed_resource(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    resources = workspace.resources({"root": workspace.directory(root)})

    with workspace.transaction(name="inspect", resources=resources, run_id="run-1") as tx:
        result = tx.snapshot_bundle(
            tx.r.root,
            include_children=True,
            hash_policy="small-files",
            idempotency_key="snapshot:root",
        )

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.before_checkpoint is not None
    assert result.before_checkpoint.checkpoint_type == "snapshot_bundle"
    assert len(result.before_checkpoint.payload["entries"]) == 2


def test_transaction_renames_claimed_directory_no_replace(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    destination = tmp_path / "repo" / ".bare"
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    resources = workspace.resources(
        {
            "git": workspace.directory(source),
            "bare": workspace.directory(destination),
        }
    )

    with workspace.transaction(name="move-git", resources=resources, run_id="run-1") as tx:
        result = tx.rename_no_replace(tx.r.git, tx.r.bare, idempotency_key="rename:git")

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert not source.exists()
    assert (destination / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


def test_transaction_renames_claimed_file_no_replace(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    source = tmp_path / "repo" / "settings.toml"
    destination = tmp_path / "repo" / "settings.toml.bak"
    source.parent.mkdir(parents=True)
    source.write_text("enabled = true\n", encoding="utf-8")
    resources = workspace.resources(
        {
            "settings": workspace.file(source),
            "backup": workspace.file(destination),
        }
    )

    with workspace.transaction(name="move-settings", resources=resources, run_id="run-file-rename") as tx:
        result = tx.rename_no_replace(tx.r.settings, tx.r.backup, idempotency_key="rename:settings")

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.resource_key == resources.settings.resource_key
    assert result.batch.payload["resource_key"] == resources.settings.resource_key
    assert result.batch.payload["destination_resource_key"] == resources.backup.resource_key
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "enabled = true\n"


def test_transaction_automatic_rollback_restores_renamed_directory(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    destination = tmp_path / "repo" / ".bare"
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    resources = workspace.resources(
        {
            "git": workspace.directory(source),
            "bare": workspace.directory(destination),
        }
    )

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="move-git",
            resources=resources,
            run_id="run-rollback-rename",
            rollback="automatic",
        ) as tx,
    ):
        tx.rename_no_replace(tx.r.git, tx.r.bare, idempotency_key="rename:git")
        raise RuntimeError("boom")

    [batch] = workspace.journal_store.list_batches(run_id="run-rollback-rename")
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not destination.exists()


def test_transaction_automatic_rollback_restores_renamed_file(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    source = tmp_path / "repo" / "settings.toml"
    destination = tmp_path / "repo" / "settings.toml.bak"
    source.parent.mkdir(parents=True)
    source.write_text("enabled = true\n", encoding="utf-8")
    resources = workspace.resources(
        {
            "settings": workspace.file(source),
            "backup": workspace.file(destination),
        }
    )

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="move-settings",
            resources=resources,
            run_id="run-rollback-file-rename",
            rollback="automatic",
        ) as tx,
    ):
        tx.rename_no_replace(tx.r.settings, tx.r.backup, idempotency_key="rename:settings")
        raise RuntimeError("boom")

    [batch] = workspace.journal_store.list_batches(run_id="run-rollback-file-rename")
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert source.read_text(encoding="utf-8") == "enabled = true\n"
    assert not destination.exists()


def test_transaction_captures_claimed_directory(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db", captured_directory_cleanup="retain")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    resources = workspace.resources({"git": workspace.directory(source)})

    with workspace.transaction(name="capture-git", resources=resources, run_id="run-1") as tx:
        result = tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert not source.exists()
    assert (quarantine / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"


def test_transaction_automatic_rollback_restores_captured_directory(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    resources = workspace.resources({"git": workspace.directory(source)})

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="capture-git",
            resources=resources,
            run_id="run-rollback-capture",
            rollback="automatic",
        ) as tx,
    ):
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )
        raise RuntimeError("boom")

    [batch] = workspace.journal_store.list_batches(run_id="run-rollback-capture")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not quarantine.exists()
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "skipped"]
    assert cleanup_records[-1].reason == "captured_directory_missing"
