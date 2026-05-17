from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from recovery_runner_helpers import (
    _acquire_and_claim,
    _coordinator,
    _current_recovery_attempt_id,
    _restore_backup_payload,
    _skip_handler,
    _start_recovering_batch,
    requires_backup_restore_support,
)

from safe_fs_ops.filesystem_ops import ResourceSnapshot, UnsafePathError, capture_backup, snapshot_resource
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemRecoveryError,
    OperationJournalStore,
    file_resource_key,
)
from safe_fs_ops.operation_journal.file_recovery_planning import file_recovery_action_plans

pytestmark = pytest.mark.safe_fs_ops


def _start_recovery_for_succeeded_batch(
    journal: OperationJournalStore,
    *,
    lease,
    batch_id: str,
    now: datetime,
) -> None:
    journal.record_recovery_desired(
        batch_id,
        lease=lease,
        reason="automatic transaction rollback",
        payload={"batch_id": batch_id},
        recovery_id=f"{batch_id}-desired",
        now=now,
    )
    journal.start_recovery(
        batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id=f"{batch_id}-attempt",
        now=now + timedelta(microseconds=1),
    )


@requires_backup_restore_support
def test_recovery_runner_executes_restore_backup_and_records_skipped_actions(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    backup_content_file = tmp_path / "backup-content.bin"
    target.write_text("old\n", encoding="utf-8")
    backup = capture_backup(target, content_path=backup_content_file)
    target.write_text("mutated\n", encoding="utf-8")

    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="restore_backup",
        resource_key=file_resource_key(target),
        payload=_restore_backup_payload(backup, expected_current=snapshot_resource(target)),
        action_id="action-restore",
        now=now + timedelta(seconds=1),
    )
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="cleanup_temp",
        resource_key=file_resource_key(target),
        payload={"path": str(tmp_path / "cleanup.tmp")},
        action_id="action-skip",
        now=now + timedelta(seconds=2),
    )

    coordinator_with_skip = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={
            "cleanup_temp": _skip_handler,
        },
    )
    context = journal.read_recovery_context(batch.batch_id)
    results = coordinator_with_skip.run_recovery_actions(context, lease=lease, now=now + timedelta(seconds=3))

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [record.status for record in results] == [
        "succeeded",
        "skipped",
    ]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "skipped",
    ]


@requires_backup_restore_support
def test_file_recovery_planning_restores_overwritten_file_from_succeeded_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    result = coordinator.write_text_file(
        target,
        "new\n",
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="write:config",
        now=now,
    )
    _start_recovery_for_succeeded_batch(
        journal,
        lease=lease,
        batch_id=result.batch.batch_id,
        now=now + timedelta(seconds=1),
    )

    context = journal.read_recovery_context(result.batch.batch_id)
    plans = file_recovery_action_plans(context)

    assert len(plans) == 1
    planned_payload = plans[0]["payload"]
    assert planned_payload["expected_current"]["content_hash"] == snapshot_resource(target).content_hash
    assert Path(str(planned_payload["backup"]["content_path"])).is_file()

    recovery_result = coordinator.recover_batch(
        result.batch.batch_id,
        lease=lease,
        recover=lambda recovery_context: coordinator.run_recovery_actions(
            recovery_context,
            lease=lease,
            now=now + timedelta(seconds=2),
        ),
        now=now + timedelta(seconds=2),
    )

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [record.status for record in recovery_result.callback_result] == ["succeeded"]
    assert recovery_result.batch.phase == BatchPhase.RECOVERY_SUCCEEDED


@requires_backup_restore_support
def test_file_recovery_planning_restores_deleted_file_from_succeeded_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    result = coordinator.delete_file(
        target,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="delete:config",
        now=now,
    )
    assert target.exists() is False
    _start_recovery_for_succeeded_batch(
        journal,
        lease=lease,
        batch_id=result.batch.batch_id,
        now=now + timedelta(seconds=1),
    )

    recovery_result = coordinator.recover_batch(
        result.batch.batch_id,
        lease=lease,
        recover=lambda recovery_context: coordinator.run_recovery_actions(
            recovery_context,
            lease=lease,
            now=now + timedelta(seconds=2),
        ),
        now=now + timedelta(seconds=2),
    )

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [record.status for record in recovery_result.callback_result] == ["succeeded"]


@requires_backup_restore_support
def test_recovery_runner_restore_backup_retry_executes_with_original_planned_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    backup_content_file = tmp_path / "backup-content.bin"
    target.write_text("old\n", encoding="utf-8")
    backup = capture_backup(target, content_path=backup_content_file)
    target.write_text("mutated\n", encoding="utf-8")
    expected_current = snapshot_resource(target)

    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    planned_payload = _restore_backup_payload(backup, expected_current=expected_current)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="restore_backup",
        resource_key=file_resource_key(target),
        payload=planned_payload,
        action_id="action-restore",
        now=now + timedelta(seconds=1),
    )
    journal.mark_recovery_action_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_id="action-restore",
        payload={"event_type": "attempting"},
        now=now + timedelta(seconds=2),
    )

    coordinator = _coordinator(state_path, journal=journal)
    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=3),
    )

    assert target.read_text(encoding="utf-8") == "old\n"
    assert [record.status for record in results] == ["succeeded"]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


@requires_backup_restore_support
def test_recovery_runner_maps_restore_conflicts_to_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    backup_content_file = tmp_path / "backup-content.bin"
    target.write_text("old\n", encoding="utf-8")
    backup = capture_backup(target, content_path=backup_content_file)
    target.write_text("mutated\n", encoding="utf-8")
    expected_current = snapshot_resource(target)
    target.write_text("tampered\n", encoding="utf-8")

    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="restore_backup",
        resource_key=resource_key,
        payload=_restore_backup_payload(backup, expected_current=expected_current),
        action_id="action-restore",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
    exception_payload = recovery_actions[-1].payload["exception_payload"]
    assert exception_payload["reason_code"] == "restore_conflict"
    assert exception_payload["path"] == str(target)
    assert exception_payload["backup_content_path"] == str(backup_content_file)


@requires_backup_restore_support
@pytest.mark.parametrize("tamper_mode", ["missing", "tampered"])
def test_recovery_runner_maps_backup_proof_failures_to_manual_intervention(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    backup_content_file = tmp_path / "backup-content.bin"
    target.write_text("old\n", encoding="utf-8")
    backup = capture_backup(target, content_path=backup_content_file)
    target.write_text("mutated\n", encoding="utf-8")

    if tamper_mode == "missing":
        backup_content_file.unlink()
    else:
        backup_content_file.write_text("tampered\n", encoding="utf-8")

    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="restore_backup",
        resource_key=resource_key,
        payload=_restore_backup_payload(backup, expected_current=snapshot_resource(target)),
        action_id="action-restore",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
    exception_payload = recovery_actions[-1].payload["exception_payload"]
    assert exception_payload["reason_code"] == "backup_content_mismatch"
    assert exception_payload["path"] == str(target)
    assert exception_payload["backup_content_path"] == str(backup_content_file)


@requires_backup_restore_support
def test_recovery_runner_records_manual_intervention_for_file_batches_missing_after_proof(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    stage = "after_snapshot"

    def fail_after_mutation_snapshot(path: Path | str) -> ResourceSnapshot:
        if target.read_text(encoding="utf-8") == "new\n":
            raise UnsafePathError("injected after snapshot failure")
        return snapshot_resource(path)

    class FailingAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args, **kwargs):
            if stage == "after_checkpoint" and kwargs.get("checkpoint_type") == "after":
                raise RuntimeError("injected after checkpoint failure")
            return super().record_checkpoint(*args, **kwargs)

    journal = FailingAfterCheckpointJournal(state_path)
    coordinator = _coordinator(state_path, journal=journal, snapshot=fail_after_mutation_snapshot)

    expected_error = UnsafePathError if stage == "after_snapshot" else RuntimeError
    with pytest.raises(expected_error):
        coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key=f"write:{stage}",
            now=now,
        )

    batch = journal.list_batches()[0]
    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.recover_batch(
            batch.batch_id,
            lease=lease,
            recover=lambda recovery_context: coordinator.run_recovery_actions(
                recovery_context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_FAILED
    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == ["manual_intervention_required"]
    assert recovery_actions[0].payload["reason_code"] == "missing_expected_current_proof"
    assert recovery_actions[0].payload["failure_stage"] == stage


@requires_backup_restore_support
def test_recovery_runner_uses_observed_failure_after_as_expected_current_proof(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)

    class FailingAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args, **kwargs):
            if kwargs.get("checkpoint_type") == "after":
                raise RuntimeError("injected after checkpoint failure")
            return super().record_checkpoint(*args, **kwargs)

    journal = FailingAfterCheckpointJournal(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    with pytest.raises(RuntimeError, match="injected after checkpoint failure"):
        coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:after_checkpoint",
            now=now,
        )

    batch = journal.list_batches()[0]
    assert batch.status_payload["failure"]["after"]["content_hash"] == snapshot_resource(target).content_hash
    recovered = coordinator.recover_batch(
        batch.batch_id,
        lease=lease,
        recover=lambda recovery_context: coordinator.run_recovery_actions(
            recovery_context,
            lease=lease,
            now=now + timedelta(seconds=1),
        ),
        now=now + timedelta(seconds=1),
    )

    assert recovered.batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"


def test_recovery_runner_records_manual_intervention_for_file_batches_missing_backup_proof(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("new\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    payload = {
        "operation": "write_text",
        "path": str(target),
        "resource_key": resource_key,
    }
    batch = journal.create_batch(
        idempotency_key="write:missing-backup-proof",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload=payload,
        now=now,
    )
    batch, operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload=payload,
        now=now,
    )
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=resource_key,
        checkpoint_type="after",
        payload={
            "path": str(target),
            "exists": True,
            "file_type": "file",
            "content_hash": "missing-backup-proof",
            "size": 4,
            "mtime_ns": None,
            "symlink_target": None,
        },
        now=now,
    )
    journal.mark_failed(
        batch.batch_id,
        lease=lease,
        error="missing backup proof",
        observed_state={"resource_key": resource_key},
        now=now,
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="operator recovery",
        payload={"resource_key": resource_key},
        now=now,
    )

    with pytest.raises(JournaledFilesystemRecoveryError):
        coordinator.recover_batch(
            batch.batch_id,
            lease=lease,
            recover=lambda recovery_context: coordinator.run_recovery_actions(
                recovery_context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_FAILED
    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == ["planned", "manual_intervention_required"]
    assert recovery_actions[-1].payload["reason_code"] == "missing_backup_proof"
    assert target.read_text(encoding="utf-8") == "new\n"
