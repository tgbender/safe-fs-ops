import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_workspace_legacy_captures import _legacy_workspace, _token

from safe_fs_ops import SafeWorkspaceError
from safe_fs_ops.operation_journal import RecoveryAttemptMismatchError
from safe_fs_ops.operation_journal.legacy_captures import ADOPTION, APPROVAL


@pytest.mark.slow_recovery
@pytest.mark.parametrize("stage", ["approval", "tag", "proof"])
def test_legacy_capture_survives_process_exit(tmp_path: Path, stage: str):
    workspace, batch_id, source, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]
    script = """
import os, sys
from datetime import timedelta
from safe_fs_ops import SafeWorkspace
workspace = SafeWorkspace.open(sys.argv[1], owner="owner-a", lease_ttl=timedelta(seconds=5))
def crash(stage):
    if stage == sys.argv[2]:
        os._exit(73)
workspace.recover_legacy_capture(workspace.list_legacy_captures()[0], confirm_ownership=True,
                                reason="Operator verified quarantined contents", _after_stage=crash)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(workspace.state_path), stage],
        capture_output=True,
        text=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert child.returncode == 73, child.stderr
    assert quarantine.is_dir() and not source.exists()
    assert (_token(quarantine) is None) is (stage == "approval")
    assert workspace.journal_store.list_checkpoints(batch_id)[-1].checkpoint_type == (
        ADOPTION if stage == "proof" else APPROVAL
    )
    active = workspace.lease_store.active(workspace.lease_name)
    if active is not None:
        time.sleep(max(0, (active.expires_at - datetime.now(UTC)).total_seconds()) + 0.1)
    if stage == "proof":
        assert workspace.recover_pending_batches() is None
    else:
        assert workspace.recover_pending_batches() is not None
        workspace.recover_legacy_capture(candidate, confirm_ownership=True, reason="Reviewed after process exit")
    assert (source / "data.txt").read_text() == "original contents"
    assert not quarantine.exists()
    assert workspace.claim_store.list_claims() == []


def test_recovery_checkpoints_require_the_active_attempt(tmp_path: Path):
    workspace, batch_id, _, _ = _legacy_workspace(tmp_path)
    journal = workspace.journal_store
    lease = workspace.lease_store.acquire(workspace.lease_name, owner=workspace.owner, ttl=timedelta(seconds=60))
    try:
        journal.record_recovery_desired(batch_id, lease=lease)
        _, attempt = journal.start_recovery(batch_id, lease=lease)
        key = journal.get_batch(batch_id).resource_key
        original_count = len(journal.list_checkpoints(batch_id))
        with pytest.raises(RecoveryAttemptMismatchError):
            journal.record_checkpoint(
                batch_id, lease=lease, resource_key=key, checkpoint_type=APPROVAL, recovery_attempt_id="stale-attempt"
            )
        assert len(journal.list_checkpoints(batch_id)) == original_count
        journal.record_checkpoint(
            batch_id, lease=lease, resource_key=key, checkpoint_type=APPROVAL, recovery_attempt_id=attempt.recovery_id
        )
        journal.record_recovery_failed(batch_id, lease=lease, recovery_attempt_id=attempt.recovery_id)
        journal.record_recovery_desired(batch_id, lease=lease)
        journal.start_recovery(batch_id, lease=lease)
        with pytest.raises(RecoveryAttemptMismatchError):
            journal.record_checkpoint(
                batch_id,
                lease=lease,
                resource_key=key,
                checkpoint_type=ADOPTION,
                recovery_attempt_id=attempt.recovery_id,
            )
        assert len(journal.list_checkpoints(batch_id)) == original_count + 1
    finally:
        workspace.lease_store.release(lease)


def test_legacy_adoption_stops_after_lease_takeover(tmp_path: Path):
    workspace, batch_id, _, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]

    def takeover(stage):
        if stage == "approval":
            later = datetime.now(UTC) + timedelta(minutes=2)
            replacement = workspace.lease_store.acquire(
                workspace.lease_name, owner="new-worker", ttl=timedelta(seconds=60), now=later
            )
            assert replacement.acquired
            workspace.lease_store.release(replacement, now=later)

    with pytest.raises(SafeWorkspaceError):
        workspace.recover_legacy_capture(candidate, confirm_ownership=True, reason="Reviewed", _after_stage=takeover)
    assert _token(quarantine) is None
    assert not any(c.checkpoint_type == ADOPTION for c in workspace.journal_store.list_checkpoints(batch_id))


def test_legacy_inspection_cannot_be_redirected_to_another_path(tmp_path: Path):
    workspace, _, _, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]
    with pytest.raises(ValueError, match="paths and resource key"):
        workspace.recover_legacy_capture(
            replace(candidate, original_path=tmp_path / "other"), confirm_ownership=True, reason="Reviewed"
        )
    assert _token(quarantine) is None


def test_legacy_replay_recognizes_restore_before_completion(tmp_path: Path):
    workspace, _, source, quarantine = _legacy_workspace(tmp_path)
    candidate = workspace.list_legacy_captures()[0]

    def stop_after_proof(stage):
        if stage == "proof":
            raise RuntimeError("stop before restore")

    with pytest.raises(SafeWorkspaceError):
        workspace.recover_legacy_capture(
            candidate, confirm_ownership=True, reason="Reviewed", _after_stage=stop_after_proof
        )
    quarantine.rename(source)  # Simulate a crash after the physical restore.
    assert workspace.recover_pending_batches() is None
    assert (source / "data.txt").read_text() == "original contents"
    assert workspace.list_legacy_captures()[0].status == "restored"
