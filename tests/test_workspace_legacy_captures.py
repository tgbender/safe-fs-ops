from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops import SafeWorkspace, SafeWorkspaceError
from safe_fs_ops.filesystem_ops import UnsafePathError
from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token
from safe_fs_ops.operation_journal import BatchPhase, directory_resource_key
from safe_fs_ops.operation_journal.legacy_captures import ADOPTION, APPROVAL


def _legacy_workspace(tmp_path: Path, *, before_only: bool = False):
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    source, quarantine = tmp_path / "original", tmp_path / "quarantine"
    source.mkdir()
    (source / "data.txt").write_text("original contents")
    value = source.stat()
    now = datetime.now(UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name, owner=workspace.owner, ttl=timedelta(seconds=60), now=now
    )
    key = directory_resource_key(source)
    for claim_key in (key, directory_resource_key(quarantine)):
        workspace.claim_store.upsert(claim_key, lease=lease, owner=workspace.owner, scope="legacy", now=now)
    intent = {
        "operation": "capture_directory",
        "path": str(source),
        "quarantine_path": str(quarantine),
        "resource_key": key,
    }
    journal = workspace.journal_store
    batch = journal.create_batch(
        idempotency_key="old-capture",
        lease=lease,
        owner=workspace.owner,
        run_id="legacy-run",
        resource_key=key,
        claim_owner=workspace.owner,
        claim_scope="legacy",
        payload=intent,
        now=now,
    )
    operation = journal.start_batch_operation(
        batch.batch_id, lease=lease, operation_type="capture_directory", resource_key=key, payload=intent, now=now
    )[1]
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=key,
        checkpoint_type="before",
        payload={
            "path": str(source),
            "file_type": "directory",
            "exists": True,
            "device": value.st_dev,
            "inode": value.st_ino,
        },
        now=now,
    )
    source.rename(quarantine)
    if not before_only:
        identity = {"file_type": "directory", "resource_key": key, "device": value.st_dev, "inode": value.st_ino}
        journal.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=key,
            checkpoint_type="captured_directory",
            payload={
                "path": str(source),
                "step_resource_key": key,
                "ownership_class": "captured_by_transaction",
                "captured_directory_cleanup": "automatic",
                "captured_directory": {
                    "original_path": str(source),
                    "quarantine_path": str(quarantine),
                    "original_identity": {**identity, "path": str(source)},
                    "captured_identity": {**identity, "path": str(quarantine)},
                    "ownership_class": "captured_by_transaction",
                },
            },
            now=now,
        )
    journal.mark_failed(batch.batch_id, lease=lease, error="old process stopped", now=now)
    workspace.lease_store.release(lease)
    return workspace, batch.batch_id, source, quarantine


def _token(path: Path):
    value = path.stat()
    return directory_capture_token(path, device=value.st_dev, inode=value.st_ino)


@pytest.mark.parametrize("before_only", [False, True])
def test_explicit_legacy_recovery_preserves_history_and_restores(tmp_path: Path, before_only: bool):
    workspace, batch_id, source, quarantine = _legacy_workspace(tmp_path, before_only=before_only)
    original_checkpoints = workspace.journal_store.list_checkpoints(batch_id)
    candidates = workspace.list_legacy_captures()
    assert len(candidates) == 1
    assert candidates[0].status == "needs_confirmation"
    assert _token(quarantine) is None
    assert workspace.recover_pending_batches() is not None
    assert quarantine.is_dir()
    workspace.recover_legacy_capture(candidates[0], confirm_ownership=True, reason="Reviewed quarantined data")
    assert (source / "data.txt").read_text() == "original contents"
    assert not quarantine.exists()
    assert _token(source) is not None
    journal = workspace.journal_store
    assert journal.get_batch(batch_id).phase == BatchPhase.RECOVERY_SUCCEEDED
    assert journal.list_checkpoints(batch_id)[: len(original_checkpoints)] == original_checkpoints
    checkpoints = journal.list_checkpoints(batch_id)
    assert [c.checkpoint_type for c in checkpoints[-2:]] == [APPROVAL, ADOPTION]
    assert checkpoints[-1].payload["approved_by"] == "owner-a"
    assert checkpoints[-1].payload["reason"] == "Reviewed quarantined data"
    assert workspace.claim_store.list_claims() == []
    assert workspace.cleanup_outstanding_artifacts() is None
    workspace.recover_legacy_capture(candidates[0], confirm_ownership=True, reason="Retry after interruption")
    assert journal.list_checkpoints(batch_id) == checkpoints


@pytest.mark.parametrize("stage", ["approval", "tag", "proof"])
def test_legacy_recovery_retries_interrupted_adoption(tmp_path: Path, stage: str):
    workspace, batch_id, source, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]

    def interrupt(actual: str):
        if actual == stage:
            raise RuntimeError("injected interruption")

    with pytest.raises(SafeWorkspaceError) as raised:
        workspace.recover_legacy_capture(
            candidate, confirm_ownership=True, reason="Verified contents", _after_stage=interrupt
        )
    assert str(raised.value.__cause__.__cause__) == "injected interruption"
    assert quarantine.is_dir() and not source.exists()
    assert (_token(quarantine) is None) is (stage == "approval")
    if stage == "proof":
        assert workspace.recover_pending_batches() is None
    else:
        assert workspace.recover_pending_batches() is not None
        workspace.recover_legacy_capture(candidate, confirm_ownership=True, reason="Retry confirmed capture")
    assert (source / "data.txt").read_text() == "original contents"
    assert workspace.journal_store.get_batch(batch_id).phase == BatchPhase.RECOVERY_SUCCEEDED


@pytest.mark.parametrize("problem", ["unconfirmed", "occupied", "replaced", "edited", "lease", "claim"])
def test_legacy_adoption_refuses_unsafe_or_unapproved_requests(tmp_path: Path, problem: str):
    workspace, batch_id, source, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]
    active = None
    if problem == "occupied":
        source.mkdir()
        (source / "unrelated.txt").write_text("leave me alone")
    elif problem == "replaced":
        quarantine.rename(tmp_path / "saved-original")
        quarantine.mkdir()
    elif problem == "edited":
        (quarantine / "new.txt").write_text("changed since review")
    elif problem in {"lease", "claim"}:
        active = workspace.lease_store.acquire(workspace.lease_name, owner="other-worker", ttl=timedelta(seconds=60))
        if problem == "claim":
            key = directory_resource_key(quarantine)
            workspace.claim_store.release(key, lease=active, owner=workspace.owner, scope="legacy")
            workspace.claim_store.upsert(key, lease=active, owner="other-owner", scope="other")
            workspace.lease_store.release(active)
            active = None
    try:
        with pytest.raises((ValueError, SafeWorkspaceError, UnsafePathError)):
            workspace.recover_legacy_capture(candidate, confirm_ownership=problem != "unconfirmed", reason="Reviewed")
        assert quarantine.is_dir()
        assert _token(quarantine) is None
        assert not any(c.checkpoint_type == ADOPTION for c in workspace.journal_store.list_checkpoints(batch_id))
        if problem == "occupied":
            assert (source / "unrelated.txt").read_text() == "leave me alone"
    finally:
        if active is not None:
            workspace.lease_store.release(active)
