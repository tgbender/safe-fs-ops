from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from transaction_file_rollback_helpers import (
    backup_content_path_for_batch,
    hook_capable_remove_directory,
    latest_batch_for_run,
    requires_backup_restore_support,
    rewrite_backup_checkpoint_payload,
    workspace_with_portable_file_rollback,
)
from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_text,
)

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.operation_journal import BatchPhase

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


@requires_backup_restore_support
def test_transaction_automatic_rollback_restores_repeated_writes_newest_first(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    fixed_now = datetime.now(UTC)

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-repeated-write-rollback",
            rollback="automatic",
            now=fixed_now,
        ) as tx,
    ):
        tx.write_text(tx.r.file, "first\n", idempotency_key="write:first")
        tx.write_text(tx.r.file, "second\n", idempotency_key="write:second")
        raise RuntimeError("boom")

    batches = workspace.journal_store.list_batches(run_id="run-repeated-write-rollback")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [batch.phase for batch in batches] == [
        BatchPhase.RECOVERY_SUCCEEDED,
        BatchPhase.RECOVERY_SUCCEEDED,
    ]
    assert batches[0].created_at == batches[1].created_at == fixed_now
    batches_by_key = {batch.idempotency_key: batch for batch in batches}
    assert batches_by_key["write:first"].storage_order < batches_by_key["write:second"].storage_order
    assert getattr(excinfo.value, "__notes__", ()) == ()


@requires_backup_restore_support
def test_transaction_commit_removes_backup_artifact_for_overwritten_file(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-overwrite",
        rollback="automatic",
    ) as tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-overwrite")
        batch = latest_batch_for_run(workspace, "run-commit-overwrite")
        backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
        assert backup_path.is_file()
        planned_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
        assert [record.status for record in planned_records] == ["planned"]
        assert planned_records[0].trigger == "deferred_cleanup"

    batch = latest_batch_for_run(workspace, "run-commit-overwrite")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert batch.phase == BatchPhase.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "new\n"
    assert backup_path.exists() is False
    assert [record.status for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]
    assert [record.trigger for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "deferred_cleanup",
        "deferred_cleanup",
        "deferred_cleanup",
    ]


@requires_backup_restore_support
def test_transaction_commit_backup_artifact_cleanup_is_idempotent_when_artifact_is_missing(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-missing-backup",
        rollback="automatic",
    ) as tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-missing")
        batch = latest_batch_for_run(workspace, "run-commit-missing-backup")
        backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
        backup_path.unlink()

    batch = latest_batch_for_run(workspace, "run-commit-missing-backup")
    assert batch.phase == BatchPhase.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "new\n"
    assert [record.status for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "planned",
        "skipped",
    ]


@requires_backup_restore_support
def test_transaction_commit_surfaces_backup_artifact_cleanup_debt_after_data_commit(tmp_path: Path) -> None:
    backup_paths: list[Path] = []

    def fail_backup_cleanup(path: Path) -> None:
        backup_paths.append(path)
        raise PermissionError("injected backup cleanup failure")

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
        backup_artifact_cleanup_operation=fail_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    tx = workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-cleanup-debt",
        rollback="automatic",
    )

    with pytest.raises(Exception, match="backup artifact cleanup debt"), tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-cleanup-debt")

    batch = latest_batch_for_run(workspace, "run-commit-cleanup-debt")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert batch.phase == BatchPhase.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "new\n"
    assert tx.cleanup_error is not None
    assert "backup artifact cleanup debt" in str(tx.cleanup_error)
    assert backup_path.exists() is True
    assert backup_paths == [backup_path]
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == [
        "planned",
        "attempting",
        "failed",
    ]
    assert [record.trigger for record in cleanup_records] == [
        "deferred_cleanup",
        "deferred_cleanup",
        "deferred_cleanup",
    ]
    assert [record.status for record in workspace.journal_store.list_outstanding_artifact_cleanup_records()] == [
        "failed"
    ]


@requires_backup_restore_support
def test_transaction_commit_backup_artifact_cleanup_debt_marks_operation_finalization_failed(tmp_path: Path) -> None:
    backup_paths: list[Path] = []
    tx_ref = None
    explicit_run_id = "operation-run-cleanup-debt"
    explicit_phase_id = "operation-phase-cleanup-debt"

    def fail_backup_cleanup(path: Path) -> None:
        backup_paths.append(path)
        raise PermissionError("injected backup cleanup failure")

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
        backup_artifact_cleanup_operation=fail_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(Exception, match="backup artifact cleanup debt"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-commit-cleanup-status",
            rollback="automatic",
        ) as tx,
    ):
        tx_ref = tx
        explicit_run = workspace.journal_store.create_operation_run(
            run_id=tx.run_id,
            lease=tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
            operation_run_id=explicit_run_id,
            now=tx.now,
        )
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=explicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=1,
            payload={"explicit": True},
            operation_phase_id=explicit_phase_id,
            now=tx.now,
        )
        tx.write_text(
            tx.r.file,
            "new\n",
            idempotency_key="write:commit-cleanup-status",
            operation_run_id=explicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    batch = latest_batch_for_run(workspace, "run-commit-cleanup-status")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert tx_ref is not None
    assert batch.phase == BatchPhase.SUCCEEDED
    assert tx_ref.cleanup_error is not None
    assert "backup artifact cleanup debt" in str(tx_ref.cleanup_error)
    assert workspace.journal_store.get_operation_run(explicit_run_id).status == "finalization_failed"
    assert workspace.journal_store.get_operation_phase(explicit_phase_id).status == "finalization_failed"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert backup_path.exists() is True
    assert backup_paths == [backup_path]
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert [record.status for record in cleanup_records] == [
        "planned",
        "attempting",
        "failed",
    ]
    assert [record.status for record in cleanup_records].count("planned") == 1


@requires_backup_restore_support
def test_transaction_commit_records_manual_intervention_for_malformed_backup_checkpoint(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    tx = workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-malformed-backup",
        rollback="automatic",
    )
    backup_path: Path | None = None

    with pytest.raises(Exception, match="backup artifact cleanup debt"), tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:commit-malformed")
        batch = latest_batch_for_run(workspace, "run-commit-malformed-backup")
        backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
        rewrite_backup_checkpoint_payload(
            workspace,
            batch_id=batch.batch_id,
            content_path=None,
            existed=True,
            file_type="file",
        )
        assert backup_path.exists() is True

    batch = latest_batch_for_run(workspace, "run-commit-malformed-backup")
    assert backup_path is not None
    assert batch.phase == BatchPhase.SUCCEEDED
    assert target.read_text(encoding="utf-8") == "new\n"
    assert backup_path.exists() is True
    assert tx.cleanup_error is not None
    assert [record.status for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "planned",
        "manual_intervention_required",
    ]


@requires_backup_restore_support
def test_transaction_commit_missing_source_backup_creates_no_cleanup_row_or_debt(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-commit-missing-source",
        rollback="automatic",
    ) as tx:
        tx.delete_file(tx.r.file, missing_ok=True, idempotency_key="delete:missing-source")

    batch = latest_batch_for_run(workspace, "run-commit-missing-source")
    assert batch.phase == BatchPhase.SUCCEEDED
    assert workspace.journal_store.list_artifact_cleanup_records(batch.batch_id) == []


@requires_backup_restore_support
def test_transaction_recovery_cleanup_reuses_preplanned_artifact_cleanup_row(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-recovery-preplanned-cleanup",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:recovery-preplanned")
        batch = latest_batch_for_run(workspace, "run-recovery-preplanned-cleanup")
        cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
        assert [record.status for record in cleanup_records] == ["planned"]
        assert cleanup_records[0].trigger == "deferred_cleanup"
        raise RuntimeError("boom")

    batch = latest_batch_for_run(workspace, "run-recovery-preplanned-cleanup")
    cleanup_records = workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert [record.status for record in cleanup_records] == ["planned", "attempting", "succeeded"]
    assert [record.trigger for record in cleanup_records] == [
        "deferred_cleanup",
        "deferred_cleanup",
        "deferred_cleanup",
    ]


def test_transaction_automatic_rollback_reports_missing_file_rollback_proof(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=False,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-missing-file-proof",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:missing-proof")
        raise RuntimeError("boom")

    assert target.read_text(encoding="utf-8") == "new\n"
    [batch] = workspace.journal_store.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    notes = getattr(excinfo.value, "__notes__", ())
    assert any("write_text" in note and "does not support automatic rollback" in note for note in notes)
