from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from recovery_runner_helpers import (
    _acquire_and_claim,
    _coordinator,
    fake_identity_remove_directory,
)
from recursive_mkdir_recovery_helpers import (
    _directory_identity_payload,
    _journaled_creation_proof_payload,
    _start_recursive_mkdir_recovery_batch,
    portable_snapshot,
)

from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemRecoveryError,
    OperationJournalStore,
    directory_resource_key,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import REMOVE_CREATED_DIRECTORY_ACTION

pytestmark = pytest.mark.safe_fs_ops


def _hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()


def test_recovery_runner_plans_recursive_mkdir_cleanup_and_deletes_owned_directories(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    journal, lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    leaf = created_parent / "cache"
    coordinator = _coordinator(
        tmp_path / "state.db",
        journal=journal,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=fake_identity_remove_directory,
        snapshot=portable_snapshot,
    )

    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch_id),
        lease=lease,
        now=now + timedelta(seconds=1),
    )

    planned_paths = [
        record.payload["path"] for record in journal.list_recovery_actions(batch_id) if record.status == "planned"
    ]
    assert planned_paths == [
        str(created_parent),
        str(created_root),
    ]
    assert {record.action_type for record in results} == {REMOVE_CREATED_DIRECTORY_ACTION}
    assert created_parent.exists() is False
    assert created_root.exists() is False
    assert leaf.exists() is False
    assert journal.get_batch(batch_id).phase == BatchPhase.RECOVERING  # type: ignore[union-attr]
    assert [record.status for record in journal.list_recovery_actions(batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "succeeded",
    ]


def test_recovery_runner_recursive_mkdir_cleanup_skips_missing_and_requires_manual_for_non_empty_directory(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    journal, lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    coordinator = _coordinator(
        tmp_path / "state.db",
        journal=journal,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=fake_identity_remove_directory,
        snapshot=portable_snapshot,
    )
    created_parent.rmdir()
    (created_root / "keep.txt").write_text("manual", encoding="utf-8")

    with pytest.raises(JournaledFilesystemRecoveryError, match="manual intervention"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch_id),
            lease=lease,
            now=now + timedelta(seconds=1),
        )

    assert created_parent.exists() is False
    assert created_root.is_dir()
    manual_record = journal.list_recovery_actions(batch_id)[-1]
    assert manual_record.payload["exception_payload"]["reason_code"] == "directory_not_empty"
    assert [record.status for record in journal.list_recovery_actions(batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "skipped",
        "attempting",
        "manual_intervention_required",
    ]


def test_recovery_runner_recursive_mkdir_cleanup_does_not_plan_foreign_batch_proof(
    tmp_path: Path,
) -> None:
    created_root = tmp_path / "config"
    created_root.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    state_path = tmp_path / "state.db"
    lease = _acquire_and_claim(state_path, directory_resource_key(created_root), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(
        state_path,
        journal=journal,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=fake_identity_remove_directory,
        snapshot=portable_snapshot,
    )

    batch = journal.create_batch(
        idempotency_key=f"recover:{directory_resource_key(created_root)}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=directory_resource_key(created_root),
        claim_owner="owner-a",
        payload={"operation": "make_directory"},
        batch_id="batch-1",
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(created_root),
        payload={"resource_key": directory_resource_key(created_root)},
        now=now,
    )
    journal.mark_failed(
        batch.batch_id,
        lease=lease,
        error="mkdir failed",
        observed_state={"resource_key": directory_resource_key(created_root)},
        now=now + timedelta(microseconds=1),
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="filesystem mutation failed",
        payload={
            "recursive_group": {
                "prior_created_steps": [
                    {
                        "step_index": 1,
                        "step_path": str(created_root),
                        "step_resource_key": directory_resource_key(created_root),
                        "ownership_class": "created_by_transaction",
                        "created_directory_identity": _directory_identity_payload(created_root),
                        "journaled_creation_proof": {
                            **_journaled_creation_proof_payload(created_root, sequence=1),
                            "batch_id": "foreign-batch",
                        },
                        "cleanup_policy": "owned_empty_directory_safe_ish",
                        "backend_guarantee": "identity_conditional_remove",
                    }
                ]
            },
        },
        recovery_id="recovery-desired",
        now=now + timedelta(microseconds=2),
    )
    journal.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-attempt",
        now=now + timedelta(microseconds=3),
    )

    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=1),
    )

    assert results == ()
    assert created_root.is_dir()
    assert journal.list_recovery_actions(batch.batch_id) == []
