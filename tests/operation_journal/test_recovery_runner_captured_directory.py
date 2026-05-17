from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from recovery_runner_helpers import _acquire_and_claim, _coordinator
from recursive_mkdir_recovery_helpers import _captured_directory_step_payload

from safe_fs_ops.filesystem_ops import capture_directory_to_quarantine
from safe_fs_ops.operation_journal import OperationJournalStore, directory_resource_key
from safe_fs_ops.operation_journal.captured_directory_recovery import RESTORE_CAPTURED_DIRECTORY_ACTION

pytestmark = pytest.mark.safe_fs_ops


def test_recovery_runner_plans_and_restores_captured_directory(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    original = tmp_path / "config"
    original.mkdir()
    (original / "settings.json").write_text("{}", encoding="utf-8")
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(original, quarantine_path=quarantine)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = directory_resource_key(original)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    batch = journal.create_batch(
        idempotency_key=f"recover:{resource_key}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload={"operation": "make_directory"},
        batch_id="batch-1",
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=resource_key,
        payload={
            "operation": "make_directory",
            "path": str(original),
            "resource_key": resource_key,
        },
        operation_id="operation-captured-1",
        now=now,
    )
    journal.mark_failed(
        batch.batch_id,
        lease=lease,
        error="mkdir failed",
        observed_state={"resource_key": resource_key},
        now=now + timedelta(microseconds=1),
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="filesystem mutation failed",
        payload={
            "recursive_group": {
                "prior_created_steps": [
                    _captured_directory_step_payload(captured, step_index=1, resource_key=resource_key),
                ]
            }
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

    assert [record.action_type for record in results] == [RESTORE_CAPTURED_DIRECTORY_ACTION]
    assert original.is_dir()
    assert quarantine.exists() is False
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]
