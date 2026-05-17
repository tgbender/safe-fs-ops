from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pytest
from workspace_api_helpers import (
    portable_capture_directory,
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_rename_no_replace,
    portable_snapshot,
    portable_write_text,
)

from safe_fs_ops import DirectoryResource, SafeWorkspace
from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    IdentitySafeRemoveDirectoryUnavailableError,
    UnsafePathError,
)
from safe_fs_ops.operation_journal import ArtifactCleanupTrigger, BatchPhase, file_resource_key
from safe_fs_ops.operation_journal.file_backup_artifacts import plan_backup_artifact_cleanup_candidates
from safe_fs_ops.operation_journal.filesystem_mutation_checkpoints import backup_content_path
from safe_fs_ops.operation_journal.filesystem_mutations import _captured_directory_payload
from safe_fs_ops.workspace_rollback import (
    _cleanup_artifacts_for_batches,
    cleanup_committed_transaction_artifacts,
)
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_transaction_commit_records_debt_for_non_empty_captured_directory_cleanup(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (source / "objects").mkdir()
    (source / "objects" / "pack").write_text("pack\n", encoding="utf-8")

    with (
        pytest.raises(Exception, match="captured directory cleanup debt") as excinfo,
        workspace.transaction(
            name="capture-git",
            resources={"git": workspace.directory(source)},
            run_id="run-commit-captured-directory-cleanup",
            rollback="automatic",
        ) as tx,
    ):
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    batch = _latest_batch_for_run(workspace, "run-commit-captured-directory-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert "safe recursive deleter" in str(excinfo.value)
    assert source.exists() is False
    assert (quarantine / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "manual_intervention_required"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
    ]
    assert cleanup_records[0].payload["artifact_type"] == "captured_directory"
    assert cleanup_records[-1].reason == "captured_directory_cleanup_unsafe"


def test_transaction_commit_cleanup_runs_before_claim_and_lease_release(tmp_path: Path) -> None:
    observations: list[tuple[bool, bool]] = []
    workspace: SafeWorkspace
    resource_key: str

    def cleanup_backup(path: Path) -> None:
        observations.append(
            (
                workspace.claim_store.get(resource_key) is not None,
                workspace.lease_store.active(workspace.lease_name) is not None,
            )
        )
        path.unlink()

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
        backup_artifact_cleanup_operation=cleanup_backup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource_key = workspace.file(target).resource_key

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-cleanup-before-release",
        rollback="automatic",
    ) as tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-cleanup-before-release")

    batch = _latest_batch_for_run(workspace, "run-commit-cleanup-before-release")
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)
    assert observations == [(True, True)]
    assert backup_path.exists() is False
    assert workspace.claim_store.get(resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None


def test_transaction_commit_retains_captured_directory_when_configured(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    workspace = _workspace_with_portable_file_rollback(
        state_path,
        captured_directory_cleanup="retain",
    )
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-retain-captured-directory-cleanup",
        rollback="automatic",
    ) as tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    batch = _latest_batch_for_run(workspace, "run-retain-captured-directory-cleanup")

    assert source.exists() is False
    assert (quarantine / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == ["planned", "skipped"]
    assert cleanup_records[-1].reason == "captured_directory_retained"
    assert cleanup_records[-1].payload["reason_code"] == "captured_directory_retained"

    reopened = _workspace_with_portable_file_rollback(state_path)
    cleanup_lease = reopened.lease_store.acquire(
        reopened.lease_name,
        owner=reopened.owner,
        ttl=reopened.lease_ttl,
    )
    assert cleanup_lease.acquired
    try:
        cleanup_transaction = _CleanupTransactionStub(
            workspace=reopened,
            run_id=batch.run_id,
            claim_scope=batch.claim_scope,
            lease=cleanup_lease,
            _lease=cleanup_lease,
        )
        cleanup_error = _cleanup_artifacts_for_batches(
            cleanup_transaction,
            [batch],
            trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        )
    finally:
        reopened.lease_store.release(cleanup_lease)

    assert cleanup_error is None
    assert (quarantine / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert reopened.journal_store.list_artifact_cleanup_records(batch.batch_id) == cleanup_records


def test_transaction_commit_follows_durable_captured_cleanup_intent_after_policy_drift(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)

    with workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-policy-drift-captured-directory-cleanup",
        rollback="automatic",
    ) as tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )
        workspace.captured_directory_cleanup = "retain"

    batch = _latest_batch_for_run(workspace, "run-policy-drift-captured-directory-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert source.exists() is False
    assert quarantine.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]


def test_transaction_commit_retained_captured_directory_ignores_later_automatic_policy(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_portable_file_rollback(
        tmp_path / "state.db",
        captured_directory_cleanup="retain",
    )
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)

    with workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-retain-policy-drift-captured-directory-cleanup",
        rollback="automatic",
    ) as tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )
        workspace.captured_directory_cleanup = "automatic"

    batch = _latest_batch_for_run(workspace, "run-retain-policy-drift-captured-directory-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert source.exists() is False
    assert quarantine.is_dir()
    assert [record.status for record in cleanup_records] == ["planned", "skipped"]
    assert cleanup_records[-1].reason == "captured_directory_retained"


def test_capture_directory_uses_mutated_workspace_cleanup_policy_before_capture(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    workspace.captured_directory_cleanup = "retain"
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)

    with workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-mutated-retain-before-capture",
        rollback="automatic",
    ) as tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    batch = _latest_batch_for_run(workspace, "run-mutated-retain-before-capture")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert source.exists() is False
    assert quarantine.exists() is True
    assert [record.status for record in cleanup_records] == ["planned", "skipped"]
    assert cleanup_records[-1].reason == "captured_directory_retained"


def test_automatic_transaction_rolls_back_caught_capture_failure(tmp_path: Path) -> None:
    def move_then_fail(path: Path | str, *, quarantine_path: Path | str) -> CapturedDirectoryRecord:
        Path(path).rename(quarantine_path)
        raise UnsafePathError("post-capture verification failed")

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        rename_no_replace_operation=portable_rename_no_replace,
        capture_directory_operation=move_then_fail,
        enable_automatic_file_rollback=True,
    )
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with (
        workspace.transaction(
            name="capture-git",
            resources={"git": workspace.directory(source)},
            run_id="run-caught-capture-failure",
            rollback="automatic",
        ) as tx,
        pytest.raises(Exception, match="post-capture verification failed"),
    ):
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    batch = _latest_batch_for_run(workspace, "run-caught-capture-failure")

    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert quarantine.exists() is False
    assert workspace.claim_store.get(workspace.directory(source).resource_key) is None


def test_transaction_commit_surfaces_debt_when_identity_remove_unavailable(tmp_path: Path) -> None:
    def identity_remove_unavailable(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        del path, expected_identity
        raise IdentitySafeRemoveDirectoryUnavailableError(
            "identity-safe rmdir refused because Python exposes no atomic identity-conditional "
            "directory removal API for this backend"
        )

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=identity_remove_unavailable,
        rename_no_replace_operation=portable_rename_no_replace,
        capture_directory_operation=portable_capture_directory,
        enable_automatic_file_rollback=True,
        captured_directory_cleanup="automatic",
    )
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)

    tx = workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-skip-unsupported-captured-directory-cleanup",
        rollback="automatic",
    )
    with pytest.raises(Exception, match="identity_safe_remove_directory_unavailable"), tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )

    batch = _latest_batch_for_run(workspace, "run-skip-unsupported-captured-directory-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert source.exists() is False
    assert quarantine.is_dir()
    assert tx.cleanup_error is not None
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "manual_intervention_required"]
    assert cleanup_records[-1].reason == "identity_safe_remove_directory_unavailable"
    assert workspace.journal_store.list_unresolved_artifact_cleanup_debt() == [cleanup_records[-1]]
    assert workspace.journal_store.list_outstanding_artifact_cleanup_records() == [cleanup_records[-1]]


def test_transaction_recovery_records_captured_directory_cleanup_after_restore(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="capture-git",
            resources={"git": workspace.directory(source)},
            run_id="run-recovery-captured-directory-cleanup",
            rollback="automatic",
        ) as tx,
    ):
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )
        raise RuntimeError("boom")

    batch = _latest_batch_for_run(workspace, "run-recovery-captured-directory-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert quarantine.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "skipped"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
    ]
    assert cleanup_records[-1].reason == "captured_directory_missing"


def test_operation_commit_cleans_backup_artifacts(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        workspace.operation(
            name="sync",
            resources={"file": workspace.file(target)},
            run_id="run-operation-commit-cleanup",
        ) as operation,
        operation.phase("write") as phase,
    ):
        phase.write_text(phase.r.file, "new\n", idempotency_key="phase:write")

    batch = _latest_batch_for_run(workspace, "run-operation-commit-cleanup")
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert target.read_text(encoding="utf-8") == "new\n"
    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
    ]


def test_transaction_commit_records_manual_debt_when_captured_directory_identity_changes(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    tx = workspace.transaction(
        name="capture-git",
        resources={"git": workspace.directory(source)},
        run_id="run-captured-directory-cleanup-debt",
        rollback="automatic",
    )

    with pytest.raises(Exception, match="captured directory cleanup debt"), tx:
        tx.capture_directory(
            cast(DirectoryResource, tx.r.git),
            quarantine_path=quarantine,
            idempotency_key="capture:git",
        )
        shutil.rmtree(quarantine)
        quarantine.mkdir()

    batch = _latest_batch_for_run(workspace, "run-captured-directory-cleanup-debt")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert tx.cleanup_error is not None
    assert "captured directory cleanup debt" in str(tx.cleanup_error)
    assert quarantine.is_dir()
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "manual_intervention_required"]
    assert cleanup_records[-1].reason == "captured_directory_cleanup_unsafe"


def test_commit_cleanup_resumes_latest_attempting_artifact_cleanup_row(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-resume-attempting-cleanup",
        rollback="automatic",
    ) as tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-resume-attempting")
        batch = _latest_batch_for_run(workspace, "run-commit-resume-attempting-cleanup")
        cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
        assert [record.status for record in cleanup_records] == ["planned"]
        workspace.journal_store.mark_artifact_cleanup_attempting(
            batch_id=batch.batch_id,
            lease=tx.lease,
            artifact_id=cleanup_records[0].artifact_id,
            reason="resume from prior delete attempt",
            payload=dict(cleanup_records[0].payload),
            now=tx.now,
        )

    batch = _latest_batch_for_run(workspace, "run-commit-resume-attempting-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)

    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.artifact_id for record in cleanup_records] == [
        cleanup_records[0].artifact_id,
        cleanup_records[0].artifact_id,
        cleanup_records[0].artifact_id,
    ]
    assert [record.status for record in cleanup_records].count("attempting") == 1


def test_commit_cleanup_backfills_legacy_terminal_batch_without_planned_rows(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("new\n", encoding="utf-8")
    lease = workspace.lease_store.acquire(workspace.lease_name, owner=workspace.owner, ttl=workspace.lease_ttl)
    run_id = "run-legacy-terminal-cleanup"
    claim_scope = "transaction:legacy-terminal-cleanup"
    batch = workspace.journal_store.create_batch(
        idempotency_key="legacy:write-text",
        lease=lease,
        owner=workspace.owner,
        run_id=run_id,
        resource_key=file_resource_key(target),
        claim_owner=workspace.owner,
        claim_scope=claim_scope,
        batch_id="legacy-batch-1",
    )
    operation = workspace.journal_store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        operation_id="legacy-op-1",
    )
    backup_path = backup_content_path(
        workspace.journal_store,
        batch_id=batch.batch_id,
        operation_id=operation.operation_id,
        path=target,
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text("old\n", encoding="utf-8")
    workspace.journal_store.mark_attempting(batch.batch_id, lease=lease)
    workspace.journal_store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=file_resource_key(target),
        checkpoint_type="backup",
        payload={
            "path": str(target),
            "content_path": str(backup_path),
            "content_hash": hashlib.sha256(backup_path.read_bytes()).hexdigest(),
            "size": backup_path.stat().st_size,
            "existed": True,
            "file_type": "file",
            "snapshot": {
                "path": str(target),
                "exists": True,
                "file_type": "file",
            },
        },
        checkpoint_id="legacy-backup-checkpoint-1",
    )
    workspace.journal_store.mark_succeeded(batch.batch_id, lease=lease)

    cleanup_error = cleanup_committed_transaction_artifacts(
        _CleanupTransactionStub(
            workspace=workspace,
            run_id=run_id,
            claim_scope=claim_scope,
            lease=lease,
            _lease=lease,
        )
    )

    cleanup_records = workspace.journal_store.list_artifact_cleanup_records("legacy-batch-1")

    assert cleanup_error is None
    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.trigger for record in cleanup_records] == [
        "commit_cleanup",
        "commit_cleanup",
        "commit_cleanup",
    ]


def test_commit_cleanup_retries_failed_deferred_cleanup_with_commit_trigger(tmp_path: Path) -> None:
    failed_paths: list[Path] = []

    def fail_once_backup_cleanup(path: Path) -> None:
        if not failed_paths:
            failed_paths.append(path)
            raise PermissionError("injected backup cleanup failure")
        path.unlink()

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
        backup_artifact_cleanup_operation=fail_once_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(Exception, match="backup artifact cleanup debt"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-commit-retry-failed-deferred-cleanup",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-retry-failed-deferred-cleanup")

    batch = _latest_batch_for_run(workspace, "run-commit-retry-failed-deferred-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "failed"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
    ]
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)
    assert backup_path.exists() is True

    cleanup_error = cleanup_committed_transaction_artifacts(
        _CleanupTransactionStub(
            workspace=workspace,
            run_id="run-commit-retry-failed-deferred-cleanup",
            claim_scope=batch.claim_scope,
            lease=(
                retry_lease := workspace.lease_store.acquire(
                    workspace.lease_name,
                    owner=workspace.owner,
                    ttl=workspace.lease_ttl,
                )
            ),
            _lease=retry_lease,
        )
    )

    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert cleanup_error is None
    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == [
        "planned",
        "attempting",
        "failed",
        "planned",
        "attempting",
        "succeeded",
    ]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
    ]


def test_recovery_cleanup_retries_failed_deferred_cleanup_after_automatic_rollback(tmp_path: Path) -> None:
    failed_paths: list[Path] = []

    def fail_once_backup_cleanup(path: Path) -> None:
        if not failed_paths:
            failed_paths.append(path)
            raise PermissionError("injected backup cleanup failure")
        path.unlink()

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
        backup_artifact_cleanup_operation=fail_once_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-recovery-retry-failed-deferred-cleanup",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:recovery-retry-failed-deferred-cleanup")
        raise RuntimeError("boom")

    notes = getattr(excinfo.value, "__notes__", None)
    assert notes is not None
    assert any("artifact cleanup also failed" in note for note in notes)

    batch = _latest_batch_for_run(workspace, "run-recovery-retry-failed-deferred-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)

    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is True
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "failed"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
    ]

    retry_lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=workspace.lease_ttl,
    )
    cleanup_error = _cleanup_artifacts_for_batches(
        _CleanupTransactionStub(
            workspace=workspace,
            run_id="run-recovery-retry-failed-deferred-cleanup",
            claim_scope=batch.claim_scope,
            lease=retry_lease,
            _lease=retry_lease,
        ),
        [batch],
        trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
    )

    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert cleanup_error is None
    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == [
        "planned",
        "attempting",
        "failed",
        "planned",
        "attempting",
        "succeeded",
    ]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.RECOVERY_CLEANUP,
        ArtifactCleanupTrigger.RECOVERY_CLEANUP,
        ArtifactCleanupTrigger.RECOVERY_CLEANUP,
    ]


@pytest.mark.parametrize(
    ("trigger",),
    [
        (ArtifactCleanupTrigger.COMMIT_CLEANUP,),
        (ArtifactCleanupTrigger.RECOVERY_CLEANUP,),
    ],
)
def test_legacy_attempting_cleanup_rows_resume_for_commit_and_recovery_triggers(
    tmp_path: Path,
    trigger: str,
) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("new\n", encoding="utf-8")
    run_id = f"run-legacy-attempting-{trigger}"
    claim_scope = f"transaction:legacy-attempting-{trigger}"
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=workspace.lease_ttl,
    )
    batch = workspace.journal_store.create_batch(
        idempotency_key=f"legacy:{trigger}:write-text",
        lease=lease,
        owner=workspace.owner,
        run_id=run_id,
        resource_key=file_resource_key(target),
        claim_owner=workspace.owner,
        claim_scope=claim_scope,
        batch_id=f"legacy-{trigger}-batch-1",
    )
    operation = workspace.journal_store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        operation_id=f"legacy-{trigger}-op-1",
    )
    backup_path = backup_content_path(
        workspace.journal_store,
        batch_id=batch.batch_id,
        operation_id=operation.operation_id,
        path=target,
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text("old\n", encoding="utf-8")
    workspace.journal_store.mark_attempting(batch.batch_id, lease=lease)
    workspace.journal_store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=file_resource_key(target),
        checkpoint_type="backup",
        payload={
            "path": str(target),
            "content_path": str(backup_path),
            "content_hash": hashlib.sha256(backup_path.read_bytes()).hexdigest(),
            "size": backup_path.stat().st_size,
            "existed": True,
            "file_type": "file",
            "snapshot": {
                "path": str(target),
                "exists": True,
                "file_type": "file",
            },
        },
        checkpoint_id=f"legacy-{trigger}-backup-checkpoint-1",
    )
    if trigger == ArtifactCleanupTrigger.COMMIT_CLEANUP:
        workspace.journal_store.mark_succeeded(batch.batch_id, lease=lease)
    else:
        workspace.journal_store.mark_failed(batch.batch_id, lease=lease, error="recovery needed")
        workspace.journal_store.record_recovery_desired(
            batch.batch_id,
            lease=lease,
            reason="recover",
            recovery_id=f"legacy-{trigger}-recovery-1",
        )
        _, recovery_attempt = workspace.journal_store.start_recovery(
            batch.batch_id,
            lease=lease,
            reason="attempt recovery",
            recovery_id=f"legacy-{trigger}-recovery-2",
        )
        workspace.journal_store.record_recovery_succeeded(
            batch.batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt.recovery_id,
            reason="recovered",
            recovery_id=f"legacy-{trigger}-recovery-3",
        )
    cleanup_candidate = plan_backup_artifact_cleanup_candidates(
        workspace.journal_store.list_checkpoints(batch.batch_id),
        state_path=workspace.state_path,
    )[0]
    planned_cleanup = workspace.journal_store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=cleanup_candidate.artifact_id,
        trigger=trigger,
        resource_key=file_resource_key(target),
        payload={
            "checkpoint_id": cleanup_candidate.checkpoint_id,
            "content_path": str(backup_path),
        },
    )
    workspace.journal_store.mark_artifact_cleanup_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=planned_cleanup.artifact_id,
        reason="resume from prior delete attempt",
        payload=dict(planned_cleanup.payload),
    )

    cleanup_transaction = _CleanupTransactionStub(
        workspace=workspace,
        run_id=run_id,
        claim_scope=claim_scope,
        lease=lease,
        _lease=lease,
    )
    if trigger == ArtifactCleanupTrigger.COMMIT_CLEANUP:
        cleanup_error = cleanup_committed_transaction_artifacts(cleanup_transaction)
    else:
        cleanup_error = _cleanup_artifacts_for_batches(
            cleanup_transaction,
            [batch],
            trigger=trigger,
        )

    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)

    assert cleanup_error is None
    assert backup_path.exists() is False
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.trigger for record in cleanup_records] == [trigger, trigger, trigger]


@pytest.mark.parametrize("terminal_status", ["succeeded", "skipped"])
def test_commit_cleanup_rerun_after_terminal_artifact_cleanup_is_noop_and_continues(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    run_id = f"run-terminal-cleanup-rerun-{terminal_status}"
    claim_scope = f"transaction:terminal-cleanup-rerun-{terminal_status}"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=workspace.lease_ttl,
        now=now,
    )
    terminal_target = tmp_path / "terminal.txt"
    pending_target = tmp_path / "pending.txt"
    terminal_batch = _create_succeeded_write_batch_with_backup(
        workspace,
        lease=lease,
        run_id=run_id,
        claim_scope=claim_scope,
        batch_id=f"terminal-{terminal_status}-batch",
        target=terminal_target,
        now=now,
    )
    pending_batch = _create_succeeded_write_batch_with_backup(
        workspace,
        lease=lease,
        run_id=run_id,
        claim_scope=claim_scope,
        batch_id=f"pending-{terminal_status}-batch",
        target=pending_target,
        now=now + timedelta(seconds=1),
    )
    terminal_candidate = plan_backup_artifact_cleanup_candidates(
        workspace.journal_store.list_checkpoints(terminal_batch.batch_id),
        state_path=workspace.state_path,
    )[0]
    terminal_backup_path = _backup_content_path_for_batch(workspace, terminal_batch.batch_id)
    pending_backup_path = _backup_content_path_for_batch(workspace, pending_batch.batch_id)
    terminal_backup_path.unlink()
    workspace.journal_store.record_artifact_cleanup_planned(
        batch_id=terminal_batch.batch_id,
        lease=lease,
        artifact_id=terminal_candidate.artifact_id,
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
        resource_key=terminal_candidate.resource_key,
        payload=_artifact_cleanup_candidate_payload(terminal_candidate),
        now=now + timedelta(seconds=2),
    )
    if terminal_status == "succeeded":
        workspace.journal_store.record_artifact_cleanup_succeeded(
            batch_id=terminal_batch.batch_id,
            lease=lease,
            artifact_id=terminal_candidate.artifact_id,
            payload=_artifact_cleanup_candidate_payload(terminal_candidate),
            now=now + timedelta(seconds=3),
        )
    else:
        workspace.journal_store.record_artifact_cleanup_skipped(
            batch_id=terminal_batch.batch_id,
            lease=lease,
            artifact_id=terminal_candidate.artifact_id,
            reason="artifact_missing",
            payload=_artifact_cleanup_candidate_payload(terminal_candidate),
            now=now + timedelta(seconds=3),
        )

    cleanup_error = cleanup_committed_transaction_artifacts(
        _CleanupTransactionStub(
            workspace=workspace,
            run_id=run_id,
            claim_scope=claim_scope,
            lease=lease,
        )
    )

    terminal_records = workspace.journal_store.list_artifact_cleanup_records(terminal_batch.batch_id)
    pending_records = workspace.journal_store.list_artifact_cleanup_records(pending_batch.batch_id)

    assert cleanup_error is None
    assert terminal_backup_path.exists() is False
    assert pending_backup_path.exists() is False
    assert [record.status for record in terminal_records] == ["planned", terminal_status]
    assert [record.status for record in pending_records] == ["planned", "attempting", "succeeded"]


def test_workspace_cleanup_outstanding_artifacts_replays_committed_cleanup(tmp_path: Path) -> None:
    workspace = _workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    def fail_cleanup(path: Path) -> None:
        raise OSError(f"held open: {path}")

    workspace.backup_artifact_cleanup_operation = fail_cleanup
    with (
        pytest.raises(Exception, match="artifact_cleanup_failed"),
        workspace.transaction(
            name="write-config",
            resources={"config": workspace.file(target)},
            run_id="run-public-cleanup-replay",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.config, "new\n", idempotency_key="write-config")

    batch = _latest_batch_for_run(workspace, "run-public-cleanup-replay")
    backup_path = _backup_content_path_for_batch(workspace, batch.batch_id)
    assert backup_path.exists()
    assert workspace.journal_store.list_outstanding_artifact_cleanup_records()

    workspace.backup_artifact_cleanup_operation = None
    cleanup_error = workspace.cleanup_outstanding_artifacts(run_id="run-public-cleanup-replay")

    assert cleanup_error is None
    assert backup_path.exists() is False
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert cleanup_records[-1].status == "succeeded"


def test_checkpoint_only_captured_directory_replay_keeps_durable_automatic_cleanup_policy(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    workspace = _workspace_with_portable_file_rollback(state_path)
    source = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.parent.mkdir(parents=True)
    source.mkdir(parents=True)
    resource_key = workspace.directory(source).resource_key
    captured = portable_capture_directory(source, quarantine_path=quarantine)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=workspace.lease_ttl,
        now=now,
    )
    assert lease.acquired
    try:
        batch = _create_succeeded_capture_batch_with_checkpoint_only(
            workspace,
            lease=lease,
            run_id="run-checkpoint-only-captured-cleanup",
            batch_id="batch-checkpoint-only-captured-cleanup",
            resource_key=resource_key,
            captured=captured,
            cleanup_policy="automatic",
            now=now,
        )
    finally:
        workspace.lease_store.release(lease, now=now + timedelta(microseconds=5))

    replay_workspace = _workspace_with_portable_file_rollback(
        state_path,
        captured_directory_cleanup="retain",
    )
    cleanup_error = replay_workspace.cleanup_outstanding_artifacts(run_id=batch.run_id)

    assert cleanup_error is None
    assert source.exists() is False
    assert quarantine.exists() is False
    cleanup_records = replay_workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.trigger for record in cleanup_records] == [
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
    ]
    assert cleanup_records[0].payload["captured_directory_cleanup"] == "automatic"


def _workspace_with_portable_file_rollback(
    state_path: Path,
    *,
    captured_directory_cleanup: Literal["automatic", "retain"] = "automatic",
) -> SafeWorkspace:
    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        rename_no_replace_operation=portable_rename_no_replace,
        capture_directory_operation=portable_capture_directory,
        enable_automatic_file_rollback=True,
        captured_directory_cleanup=captured_directory_cleanup,
    )


def _hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()


def _latest_batch_for_run(workspace: SafeWorkspace, run_id: str):
    matches = [batch for batch in workspace.journal_store.list_batches() if batch.run_id == run_id]
    assert matches, f"expected batch for run {run_id!r}"
    return matches[-1]


def _backup_content_path_for_batch(workspace: SafeWorkspace, batch_id: str) -> Path:
    for checkpoint in workspace.journal_store.list_checkpoints(batch_id):
        if checkpoint.checkpoint_type != "backup":
            continue
        content_path = checkpoint.payload.get("content_path")
        if content_path is not None:
            return Path(str(content_path))
    raise AssertionError(f"expected backup content path for batch {batch_id!r}")


def _create_succeeded_write_batch_with_backup(
    workspace: SafeWorkspace,
    *,
    lease: LeaseRecord,
    run_id: str,
    claim_scope: str,
    batch_id: str,
    target: Path,
    now: datetime,
):
    target.write_text("new\n", encoding="utf-8")
    resource_key = file_resource_key(target)
    batch = workspace.journal_store.create_batch(
        idempotency_key=f"{batch_id}:write-text",
        lease=lease,
        owner=workspace.owner,
        run_id=run_id,
        resource_key=resource_key,
        claim_owner=workspace.owner,
        claim_scope=claim_scope,
        batch_id=batch_id,
        now=now,
    )
    operation = workspace.journal_store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload={"path": str(target)},
        operation_id=f"{batch_id}-operation",
        now=now + timedelta(microseconds=1),
    )
    backup_path = backup_content_path(
        workspace.journal_store,
        batch_id=batch.batch_id,
        operation_id=operation.operation_id,
        path=target,
    )
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_text("old\n", encoding="utf-8")
    workspace.journal_store.mark_attempting(
        batch.batch_id,
        lease=lease,
        now=now + timedelta(microseconds=2),
    )
    workspace.journal_store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=resource_key,
        checkpoint_type="backup",
        payload={
            "path": str(target),
            "content_path": str(backup_path),
            "content_hash": hashlib.sha256(backup_path.read_bytes()).hexdigest(),
            "size": backup_path.stat().st_size,
            "existed": True,
            "file_type": "file",
            "snapshot": {
                "path": str(target),
                "exists": True,
                "file_type": "file",
            },
        },
        checkpoint_id=f"{batch_id}-backup-checkpoint",
        now=now + timedelta(microseconds=3),
    )
    return workspace.journal_store.mark_succeeded(
        batch.batch_id,
        lease=lease,
        now=now + timedelta(microseconds=4),
    )


def _create_succeeded_capture_batch_with_checkpoint_only(
    workspace: SafeWorkspace,
    *,
    lease: LeaseRecord,
    run_id: str,
    batch_id: str,
    resource_key: str,
    captured: CapturedDirectoryRecord,
    cleanup_policy: Literal["automatic", "retain"],
    now: datetime,
):
    payload = {
        "operation": "capture_directory",
        "path": str(captured.original_path),
        "quarantine_path": str(captured.quarantine_path),
        "resource_key": resource_key,
        "captured_directory_cleanup": cleanup_policy,
    }
    batch = workspace.journal_store.create_batch(
        idempotency_key=f"{batch_id}:capture-directory",
        lease=lease,
        owner=workspace.owner,
        run_id=run_id,
        resource_key=resource_key,
        claim_owner=workspace.owner,
        batch_id=batch_id,
        payload=payload,
        now=now,
    )
    operation = workspace.journal_store.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="capture_directory",
        resource_key=resource_key,
        payload=payload,
        operation_id=f"{batch_id}-operation",
        now=now + timedelta(microseconds=1),
    )
    workspace.journal_store.mark_attempting(
        batch.batch_id,
        lease=lease,
        now=now + timedelta(microseconds=2),
    )
    capture_payload = _captured_directory_payload(
        captured,
        resource_key=resource_key,
        captured_directory_cleanup=cleanup_policy,
    )
    workspace.journal_store.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=resource_key,
        checkpoint_type="captured_directory",
        payload=capture_payload,
        checkpoint_id=f"{batch_id}-captured-directory-checkpoint",
        now=now + timedelta(microseconds=3),
    )
    return workspace.journal_store.mark_succeeded(
        batch.batch_id,
        lease=lease,
        result={"operation_id": operation.operation_id, "captured_directory": capture_payload},
        now=now + timedelta(microseconds=4),
    )


def _artifact_cleanup_candidate_payload(candidate) -> dict[str, object]:
    return {
        "checkpoint_id": candidate.checkpoint_id,
        "artifact_type": "backup_file",
        "content_path": str(candidate.content_path),
        "expected_content_path": str(candidate.expected_content_path),
    }


@dataclass(frozen=True, slots=True)
class _CleanupTransactionStub:
    workspace: SafeWorkspace
    run_id: str
    claim_scope: str
    lease: LeaseRecord
    _lease: LeaseRecord | None = None
    cleanup_clock: object = None
