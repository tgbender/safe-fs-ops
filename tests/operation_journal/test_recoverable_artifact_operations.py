from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    RenameRecord,
    ResourceSnapshot,
    UnsafePathError,
    capture_directory_to_quarantine,
    rename_no_replace,
)
from safe_fs_ops.operation_journal import (
    ArtifactCleanupTrigger,
    BatchPhase,
    DirectoryResourceKeyMismatchError,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemCoordinator,
    JournaledFilesystemMutationError,
    JournaledFilesystemRecoveryError,
    MissingResourceClaimError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.filesystem import directory_resource_key, file_resource_key
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def _rename_portable(source: Path | str, destination: Path | str) -> RenameRecord:
    return rename_no_replace(
        source,
        destination,
        _rename_no_replace=lambda src, dst, *, operation: src.rename(dst),
    )


def _capture_portable(source: Path | str, *, quarantine_path: Path | str) -> CapturedDirectoryRecord:
    return capture_directory_to_quarantine(
        source,
        quarantine_path=quarantine_path,
        _rename_no_replace=lambda src, dst, *, operation: src.rename(dst),
    )


def _coordinator(
    state_path: Path,
    journal: OperationJournalStore,
    *,
    rename_no_replace_operation=None,
    capture_directory_operation=None,
) -> JournaledFilesystemCoordinator:
    return JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal,
        rename_no_replace_operation=rename_no_replace_operation or _rename_portable,
        capture_directory_operation=capture_directory_operation or _capture_portable,
        snapshot=_portable_snapshot,
        captured_directory_cleanup="automatic",
    )


def _portable_snapshot(path: Path | str) -> ResourceSnapshot:
    target = Path(path)
    if target.is_symlink():
        stat_result = target.lstat()
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="symlink",
            content_hash=None,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=os.readlink(target),
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
            ctime_ns=stat_result.st_ctime_ns,
        )
    if not target.exists():
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
        )
    stat_result = target.lstat()
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type="file" if target.is_file() else "directory" if target.is_dir() else "other",
        content_hash=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        ctime_ns=stat_result.st_ctime_ns,
    )


def _lease_and_claim(state_path: Path, *resource_keys: str) -> LeaseRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = LeaseStore(state_path).acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    claim_store = ClaimStore(state_path)
    for resource_key in resource_keys:
        claim_store.upsert(resource_key, lease=lease, owner="owner-a", details=_claim_details(lease), now=now)
    return lease


def _claim_details(lease: LeaseRecord) -> str:
    return json.dumps(
        lease_claim_details_payload(lease, claim_id=f"claim:{lease.name}:{lease.fencing_token}"),
        separators=(",", ":"),
        sort_keys=True,
    )


def test_rename_no_replace_records_inverse_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source.mkdir()
    (source / "settings.toml").write_text("enabled = true\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    destination_key = directory_resource_key(destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    result = coordinator.rename_no_replace(
        source,
        destination,
        resource_key=source_key,
        destination_resource_key=destination_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="rename-config",
        now=now,
    )

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert not source.exists()
    assert (destination / "settings.toml").is_file()
    journal.record_recovery_desired(
        result.batch.batch_id,
        lease=lease,
        reason="automatic rollback",
        payload={"batch_id": result.batch.batch_id},
        now=now + timedelta(seconds=1),
    )
    recovered = coordinator.recover_batch(
        result.batch.batch_id,
        lease=lease,
        recover=lambda context: coordinator.run_recovery_actions(
            context,
            lease=lease,
            now=now + timedelta(seconds=2),
        ),
        now=now + timedelta(seconds=2),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "settings.toml").read_text(encoding="utf-8") == "enabled = true\n"
    assert not destination.exists()


def test_rename_no_replace_rejects_paths_unrelated_to_claimed_keys(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    claimed_source = tmp_path / "claimed-source"
    claimed_destination = tmp_path / "claimed-destination"
    victim_source = tmp_path / "victim-source"
    victim_destination = tmp_path / "victim-destination"
    claimed_source.mkdir()
    victim_source.mkdir()
    (victim_source / "settings.toml").write_text("keep\n", encoding="utf-8")
    source_key = directory_resource_key(claimed_source)
    destination_key = directory_resource_key(claimed_destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)

    with pytest.raises(DirectoryResourceKeyMismatchError, match="resource_key"):
        coordinator.rename_no_replace(
            victim_source,
            victim_destination,
            resource_key=source_key,
            destination_resource_key=destination_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="rename-victim",
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )

    assert claimed_source.is_dir()
    assert claimed_destination.exists() is False
    assert (victim_source / "settings.toml").read_text(encoding="utf-8") == "keep\n"
    assert victim_destination.exists() is False
    assert journal.list_batches() == []


def test_capture_directory_records_restore_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    quarantine_key = directory_resource_key(quarantine)
    lease = _lease_and_claim(state_path, source_key, quarantine_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    result = coordinator.capture_directory(
        source,
        quarantine_path=quarantine,
        resource_key=source_key,
        destination_resource_key=quarantine_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="capture-git",
        now=now,
    )

    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.payload["captured_directory_cleanup"] == "automatic"
    assert not source.exists()
    assert (quarantine / "HEAD").is_file()
    captured_checkpoint = next(
        checkpoint
        for checkpoint in journal.list_checkpoints(result.batch.batch_id)
        if checkpoint.checkpoint_type == "captured_directory"
    )
    assert captured_checkpoint.payload["captured_directory_cleanup"] == "automatic"
    cleanup_records = journal.list_artifact_cleanup_records(result.batch.batch_id)
    assert [record.status for record in cleanup_records] == ["planned"]
    assert cleanup_records[0].trigger == ArtifactCleanupTrigger.DEFERRED_CLEANUP
    assert cleanup_records[0].payload["artifact_type"] == "captured_directory"
    assert cleanup_records[0].payload["quarantine_path"] == str(quarantine)
    assert cleanup_records[0].payload["captured_directory_cleanup"] == "automatic"
    journal.record_recovery_desired(
        result.batch.batch_id,
        lease=lease,
        reason="automatic rollback",
        payload={"batch_id": result.batch.batch_id},
        now=now + timedelta(seconds=1),
    )
    recovered = coordinator.recover_batch(
        result.batch.batch_id,
        lease=lease,
        recover=lambda context: coordinator.run_recovery_actions(
            context,
            lease=lease,
            now=now + timedelta(seconds=2),
        ),
        now=now + timedelta(seconds=2),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not quarantine.exists()


def test_capture_directory_requires_claimed_quarantine_resource(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    lease = _lease_and_claim(state_path, source_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)

    with pytest.raises(ValueError, match="destination_resource_key"):
        coordinator.capture_directory(
            source,
            quarantine_path=quarantine,
            resource_key=source_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="capture-git",
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )

    assert source.exists()
    assert not quarantine.exists()
    assert journal.list_batches() == []


def test_capture_directory_requires_claim_authority_for_quarantine_destination(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    quarantine_key = directory_resource_key(quarantine)
    lease = _lease_and_claim(state_path, source_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)

    with pytest.raises(MissingResourceClaimError, match="not claimed"):
        coordinator.capture_directory(
            source,
            quarantine_path=quarantine,
            resource_key=source_key,
            destination_resource_key=quarantine_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="capture-git",
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )

    assert source.exists()
    assert not quarantine.exists()
    assert journal.list_batches() == []


def test_coordinator_rejects_invalid_captured_directory_cleanup_policy(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)

    with pytest.raises(ValueError, match="unsupported captured_directory_cleanup"):
        JournaledFilesystemCoordinator(
            lease_store=LeaseStore(state_path),
            claim_store=ClaimStore(state_path),
            journal_store=journal,
            captured_directory_cleanup="automtic",  # type: ignore[arg-type]
        )


def test_capture_directory_intent_normalizes_relative_quarantine_path(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    source_key = directory_resource_key(source)
    quarantine_key = directory_resource_key(quarantine)
    lease = _lease_and_claim(state_path, source_key, quarantine_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    old_cwd = Path.cwd()

    try:
        os.chdir(tmp_path)
        result = coordinator.capture_directory(
            Path(".git"),
            quarantine_path=Path(".safe") / "git-captured",
            resource_key=source_key,
            destination_resource_key=quarantine_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="capture-git-relative",
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        os.chdir(old_cwd)

    assert result.batch.payload["quarantine_path"] == str(quarantine)


def test_rename_post_move_verification_failure_records_inverse_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source.write_text("enabled = true\n", encoding="utf-8")
    source_key = file_resource_key(source)
    destination_key = file_resource_key(destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def move_then_fail(source_path: Path | str, destination_path: Path | str) -> RenameRecord:
        Path(source_path).rename(destination_path)
        raise UnsafePathError("post-rename verification failed")

    coordinator = _coordinator(state_path, journal, rename_no_replace_operation=move_then_fail)

    with pytest.raises(JournaledFilesystemMutationError, match="post-rename verification failed"):
        coordinator.rename_no_replace(
            source,
            destination,
            resource_key=source_key,
            destination_resource_key=destination_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="rename-config",
            now=now,
        )

    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    recovery_payload = journal.list_recovery_records(batch.batch_id)[0].payload
    assert set(recovery_payload["rename_record"]) >= {"source_path", "destination_path", "device", "inode"}
    assert not source.exists()
    assert destination.read_text(encoding="utf-8") == "enabled = true\n"

    recovered = coordinator.recover_batch(
        batch.batch_id,
        lease=lease,
        recover=lambda context: coordinator.run_recovery_actions(
            context,
            lease=lease,
            now=now + timedelta(seconds=1),
        ),
        now=now + timedelta(seconds=1),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert source.read_text(encoding="utf-8") == "enabled = true\n"
    assert not destination.exists()


def test_rename_post_move_snapshot_failure_still_records_recovery_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source.mkdir()
    (source / "settings.toml").write_text("enabled = true\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    destination_key = directory_resource_key(destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def fail_missing_source_snapshot(path: Path | str) -> ResourceSnapshot:
        if Path(path) == source and not source.exists():
            raise UnsafePathError("source snapshot failed after rename")
        return _portable_snapshot(path)

    coordinator = JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal,
        rename_no_replace_operation=_rename_portable,
        snapshot=fail_missing_source_snapshot,
    )

    with pytest.raises(UnsafePathError, match="source snapshot failed after rename"):
        coordinator.rename_no_replace(
            source,
            destination,
            resource_key=source_key,
            destination_resource_key=destination_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="rename-config",
            now=now,
        )

    batch = journal.list_batches()[0]
    recovery_payload = journal.list_recovery_records(batch.batch_id)[0].payload
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert "rename_record" in recovery_payload
    assert recovery_payload["failure"]["snapshot_error"]["error"] == "source snapshot failed after rename"


def test_capture_post_move_verification_failure_records_restore_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    quarantine_key = directory_resource_key(quarantine)
    lease = _lease_and_claim(state_path, source_key, quarantine_key)
    journal = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def move_then_fail(source_path: Path | str, *, quarantine_path: Path | str) -> CapturedDirectoryRecord:
        Path(source_path).rename(quarantine_path)
        raise UnsafePathError("post-capture verification failed")

    coordinator = _coordinator(state_path, journal, capture_directory_operation=move_then_fail)

    with pytest.raises(JournaledFilesystemMutationError, match="post-capture verification failed"):
        coordinator.capture_directory(
            source,
            quarantine_path=quarantine,
            resource_key=source_key,
            destination_resource_key=quarantine_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="capture-git",
            now=now,
        )

    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    recovery_payload = journal.list_recovery_records(batch.batch_id)[0].payload
    captured_restore = recovery_payload["captured_directory_restore"]
    assert captured_restore["ownership_class"] == "captured_by_transaction"
    assert not source.exists()
    assert (quarantine / "HEAD").is_file()

    recovered = coordinator.recover_batch(
        batch.batch_id,
        lease=lease,
        recover=lambda context: coordinator.run_recovery_actions(
            context,
            lease=lease,
            now=now + timedelta(seconds=1),
        ),
        now=now + timedelta(seconds=1),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not quarantine.exists()


def test_capture_post_move_snapshot_failure_still_records_recovery_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    quarantine_key = directory_resource_key(quarantine)
    lease = _lease_and_claim(state_path, source_key, quarantine_key)
    journal = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def fail_missing_source_snapshot(path: Path | str) -> ResourceSnapshot:
        if Path(path) == source and not source.exists():
            raise UnsafePathError("source snapshot failed after capture")
        return _portable_snapshot(path)

    coordinator = JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal,
        capture_directory_operation=_capture_portable,
        snapshot=fail_missing_source_snapshot,
    )

    with pytest.raises(UnsafePathError, match="source snapshot failed after capture"):
        coordinator.capture_directory(
            source,
            quarantine_path=quarantine,
            resource_key=source_key,
            destination_resource_key=quarantine_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="capture-git",
            now=now,
        )

    batch = journal.list_batches()[0]
    recovery_payload = journal.list_recovery_records(batch.batch_id)[0].payload
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert "captured_directory_restore" in recovery_payload
    assert recovery_payload["failure"]["snapshot_error"]["error"] == "source snapshot failed after capture"


def test_rename_post_move_verification_failure_without_identity_proof_plans_manual_action(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source.mkdir()
    source_key = directory_resource_key(source)
    destination_key = directory_resource_key(destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def move_replace_then_fail(source_path: Path | str, destination_path: Path | str) -> RenameRecord:
        Path(source_path).rename(destination_path)
        Path(destination_path).rmdir()
        Path(destination_path).mkdir()
        raise UnsafePathError("post-rename verification failed")

    coordinator = _coordinator(state_path, journal, rename_no_replace_operation=move_replace_then_fail)

    with pytest.raises(JournaledFilesystemMutationError):
        coordinator.rename_no_replace(
            source,
            destination,
            resource_key=source_key,
            destination_resource_key=destination_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="rename-config",
            now=now,
        )

    batch = journal.list_batches()[0]
    recovery_payload = journal.list_recovery_records(batch.batch_id)[0].payload
    assert recovery_payload["manual_intervention_action"]["payload"]["reason_code"] == (
        "rename_ambiguous_success_unverified"
    )

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.recover_batch(
            batch.batch_id,
            lease=lease,
            recover=lambda context: coordinator.run_recovery_actions(
                context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    actions = journal.list_recovery_actions(batch.batch_id)
    assert actions[-1].status == "manual_intervention_required"
    assert actions[-1].payload["manual_intervention"]["reason_code"] == "rename_ambiguous_success_unverified"


def test_rename_recovery_without_durable_inverse_proof_requires_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source_key = directory_resource_key(source)
    destination_key = directory_resource_key(destination)
    lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    payload = {
        "operation": "rename_no_replace",
        "source_path": str(source),
        "destination_path": str(destination),
        "resource_key": source_key,
        "destination_resource_key": destination_key,
    }
    batch = journal.create_batch(
        idempotency_key="rename:missing-proof",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=source_key,
        claim_owner="owner-a",
        payload=payload,
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="rename_no_replace",
        resource_key=source_key,
        payload=payload,
        now=now,
    )
    journal.mark_failed(batch.batch_id, lease=lease, error="interrupted after move", now=now)
    journal.record_recovery_desired(batch.batch_id, lease=lease, reason="recover", now=now)

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.recover_batch(
            batch.batch_id,
            lease=lease,
            recover=lambda context: coordinator.run_recovery_actions(
                context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    actions = journal.list_recovery_actions(batch.batch_id)
    assert [action.status for action in actions] == ["planned", "attempting", "manual_intervention_required"]
    assert actions[-1].payload["manual_intervention"]["reason_code"] == "missing_rename_rollback_proof"


def test_capture_recovery_without_durable_restore_proof_requires_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    source_key = directory_resource_key(source)
    lease = _lease_and_claim(state_path, source_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    payload = {
        "operation": "capture_directory",
        "path": str(source),
        "quarantine_path": str(quarantine),
        "resource_key": source_key,
    }
    batch = journal.create_batch(
        idempotency_key="capture:missing-proof",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=source_key,
        claim_owner="owner-a",
        payload=payload,
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="capture_directory",
        resource_key=source_key,
        payload=payload,
        now=now,
    )
    journal.mark_failed(batch.batch_id, lease=lease, error="interrupted after capture", now=now)
    journal.record_recovery_desired(batch.batch_id, lease=lease, reason="recover", now=now)

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.recover_batch(
            batch.batch_id,
            lease=lease,
            recover=lambda context: coordinator.run_recovery_actions(
                context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    actions = journal.list_recovery_actions(batch.batch_id)
    assert [action.status for action in actions] == ["planned", "attempting", "manual_intervention_required"]
    assert actions[-1].payload["manual_intervention"]["reason_code"] == ("missing_captured_directory_restore_proof")


def test_capture_recovery_derives_restore_proof_from_before_snapshot_and_quarantine(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / ".git"
    quarantine = tmp_path / ".safe" / "git-captured"
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    source_key = directory_resource_key(source)
    lease = _lease_and_claim(state_path, source_key)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    payload = {
        "operation": "capture_directory",
        "path": str(source),
        "quarantine_path": str(quarantine),
        "resource_key": source_key,
    }
    batch = journal.create_batch(
        idempotency_key="capture:interrupted-after-move",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=source_key,
        claim_owner="owner-a",
        payload=payload,
        now=now,
    )
    operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="capture_directory",
        resource_key=source_key,
        payload=payload,
        now=now,
    )[1]
    before_snapshot = _portable_snapshot(source)
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=source_key,
        checkpoint_type="before",
        payload={
            "path": str(before_snapshot.path),
            "exists": before_snapshot.exists,
            "file_type": before_snapshot.file_type,
            "content_hash": before_snapshot.content_hash,
            "size": before_snapshot.size,
            "mtime_ns": before_snapshot.mtime_ns,
            "symlink_target": before_snapshot.symlink_target,
            "device": before_snapshot.device,
            "inode": before_snapshot.inode,
        },
        now=now,
    )
    source.rename(quarantine)
    journal.mark_failed(batch.batch_id, lease=lease, error="interrupted after move", now=now)
    journal.record_recovery_desired(batch.batch_id, lease=lease, reason="recover", now=now)

    recovered = coordinator.recover_batch(
        batch.batch_id,
        lease=lease,
        recover=lambda context: coordinator.run_recovery_actions(
            context,
            lease=lease,
            now=now + timedelta(seconds=1),
        ),
        now=now + timedelta(seconds=1),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert (source / "HEAD").read_text(encoding="utf-8") == "ref: refs/heads/main\n"
    assert not quarantine.exists()
    actions = journal.list_recovery_actions(batch.batch_id)
    assert actions[0].action_type == "restore_captured_directory"
    assert actions[0].action_id.startswith("restore-captured-directory-from-before:")


def test_artifact_mutation_retry_with_newer_lease_marks_abandoned_attempt_for_recovery(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "config"
    destination = tmp_path / "config.moved"
    source.mkdir()
    source_key = directory_resource_key(source)
    destination_key = directory_resource_key(destination)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    old_lease = _lease_and_claim(state_path, source_key, destination_key)
    journal = OperationJournalStore(state_path)
    intent_payload = {
        "operation": "rename_no_replace",
        "source_path": str(source),
        "destination_path": str(destination),
        "resource_key": source_key,
        "destination_resource_key": destination_key,
    }
    batch = journal.create_batch(
        idempotency_key="rename-config",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=source_key,
        claim_owner="owner-a",
        payload=intent_payload,
        batch_id="batch-1",
        now=first_now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=old_lease,
        operation_type="rename_no_replace",
        resource_key=source_key,
        payload=intent_payload,
        now=first_now,
    )
    new_now = first_now + timedelta(seconds=31)
    new_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=new_now,
    )
    claim_store = ClaimStore(state_path)
    for resource_key in (source_key, destination_key):
        claim_store.upsert(
            resource_key, lease=new_lease, owner="owner-a", details=_claim_details(new_lease), now=new_now
        )

    with pytest.raises(JournaledFilesystemBatchStateError, match="cannot start a rename mutation"):
        _coordinator(state_path, journal).rename_no_replace(
            source,
            destination,
            resource_key=source_key,
            destination_resource_key=destination_key,
            lease=new_lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="rename-config",
            now=new_now,
        )

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert journal.list_recovery_records(batch.batch_id)[0].reason == (
        "retry found abandoned batch outside planned phase"
    )
    assert source.exists()
    assert not destination.exists()
