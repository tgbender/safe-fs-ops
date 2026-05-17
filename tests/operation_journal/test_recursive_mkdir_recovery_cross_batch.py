from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from recovery_runner_helpers import _acquire_and_claim, _coordinator, fake_identity_remove_directory

from safe_fs_ops.filesystem_ops import DirectoryIdentity, FileType, ResourceSnapshot
from safe_fs_ops.operation_journal import OperationBatchRecord, OperationJournalStore, directory_resource_key
from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import recursive_mkdir_recovery_action_plans
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def _hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()


def test_recursive_mkdir_planning_accepts_cross_batch_creation_proof(tmp_path: Path) -> None:
    journal, _lease, failed_batch_id, created_root, created_parent = _build_cross_batch_recursive_mkdir_failure(
        tmp_path
    )

    actions = recursive_mkdir_recovery_action_plans(journal.read_recovery_context(failed_batch_id))

    assert [cast(Mapping[str, Any], action["payload"])["path"] for action in actions] == [
        str(created_parent),
        str(created_root),
    ]


def test_recovery_runner_executes_cross_batch_recursive_mkdir_cleanup(tmp_path: Path) -> None:
    journal, lease, failed_batch_id, created_root, created_parent = _build_cross_batch_recursive_mkdir_failure(tmp_path)
    coordinator = _coordinator(
        tmp_path / "state.db",
        journal=journal,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=fake_identity_remove_directory,
        snapshot=portable_snapshot,
    )

    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(failed_batch_id),
        lease=lease,
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
    )

    assert [record.status for record in results] == ["succeeded", "succeeded"]
    assert created_parent.exists() is False
    assert created_root.exists() is False


def test_recursive_mkdir_planning_rejects_existing_foreign_batch_creation_proof(tmp_path: Path) -> None:
    journal, lease, failed_batch_id, created_root, created_parent = _build_cross_batch_recursive_mkdir_failure(tmp_path)
    foreign_batch, foreign_operation_id, foreign_checkpoint_id = _create_foreign_directory_batch(
        journal,
        lease=lease,
        path=created_root,
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(milliseconds=20),
    )
    context = journal.read_recovery_context(failed_batch_id)
    payload = _latest_recovery_payload(context)
    recursive_group = cast(dict[str, object], payload["recursive_group"])
    steps = cast(list[dict[str, object]], recursive_group["prior_created_steps"])
    steps[0]["journaled_creation_proof"] = {
        **cast(dict[str, object], steps[0]["journaled_creation_proof"]),
        "batch_id": foreign_batch.batch_id,
        "operation_id": foreign_operation_id,
        "checkpoint_id": foreign_checkpoint_id,
    }

    actions = recursive_mkdir_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == "manual_intervention_required"
    assert cast(Mapping[str, Any], actions[0]["payload"])["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"
    assert created_root.is_dir()


def _build_cross_batch_recursive_mkdir_failure(
    tmp_path: Path,
) -> tuple[OperationJournalStore, LeaseRecord, str, Path, Path]:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config" / "state" / "cache"
    created_parent = target.parent
    created_root = created_parent.parent
    created_parent.mkdir(parents=True)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, directory_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    operation_run = journal.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-1",
        now=now,
    )
    operation_phase = journal.create_operation_phase(
        operation_run_id=operation_run.operation_run_id,
        lease=lease,
        phase_name="prepare",
        status="active",
        phase_order=1,
        operation_phase_id="phase-1",
        now=now,
    )

    created_root_batch, created_root_operation_id, created_root_checkpoint_id = _create_created_step_batch(
        journal,
        lease=lease,
        batch_id="batch-created-1",
        path=created_root,
        target=target,
        step_index=1,
        step_count=3,
        operation_run_id=operation_run.operation_run_id,
        operation_phase_id=operation_phase.operation_phase_id,
        now=now,
    )
    created_parent_batch, created_parent_operation_id, created_parent_checkpoint_id = _create_created_step_batch(
        journal,
        lease=lease,
        batch_id="batch-created-2",
        path=created_parent,
        target=target,
        step_index=2,
        step_count=3,
        operation_run_id=operation_run.operation_run_id,
        operation_phase_id=operation_phase.operation_phase_id,
        now=now + timedelta(milliseconds=5),
    )

    failed_batch = journal.create_batch(
        idempotency_key="mkdir:recursive:3",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        operation_run_id=operation_run.operation_run_id,
        operation_phase_id=operation_phase.operation_phase_id,
        resource_key=directory_resource_key(target),
        claim_owner="owner-a",
        payload=_recursive_batch_payload(target=target, step_path=target, step_index=3, step_count=3),
        batch_id="batch-failed-3",
        now=now + timedelta(milliseconds=10),
    )
    journal.start_batch_operation(
        failed_batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(target),
        payload={
            "operation": "make_directory",
            "path": str(target),
            "resource_key": directory_resource_key(target),
        },
        operation_id="operation-failed-3",
        now=now + timedelta(milliseconds=11),
    )
    journal.mark_failed(
        failed_batch.batch_id,
        lease=lease,
        error="mkdir failed",
        observed_state={"resource_key": directory_resource_key(target)},
        now=now + timedelta(milliseconds=12),
    )
    journal.record_recovery_desired(
        failed_batch.batch_id,
        lease=lease,
        reason="filesystem mutation failed",
        payload={
            "recursive_mkdir": {
                "plan_target": str(target),
                "requested_resource_key": directory_resource_key(target),
                "step_index": 3,
                "step_count": 3,
                "step_path": str(target),
                "step_resource_key": directory_resource_key(target),
            },
            "recursive_group": {
                "plan_target": str(target),
                "requested_resource_key": directory_resource_key(target),
                "prior_created_steps": [
                    _created_step_payload(
                        created_root,
                        step_index=1,
                        batch_id=created_root_batch.batch_id,
                        operation_id=created_root_operation_id,
                        checkpoint_id=created_root_checkpoint_id,
                    ),
                    _created_step_payload(
                        created_parent,
                        step_index=2,
                        batch_id=created_parent_batch.batch_id,
                        operation_id=created_parent_operation_id,
                        checkpoint_id=created_parent_checkpoint_id,
                    ),
                ],
            },
        },
        recovery_id="recovery-desired",
        now=now + timedelta(milliseconds=13),
    )
    journal.start_recovery(
        failed_batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-attempt",
        now=now + timedelta(milliseconds=14),
    )
    return journal, lease, failed_batch.batch_id, created_root, created_parent


def _create_created_step_batch(
    journal: OperationJournalStore,
    *,
    lease: LeaseRecord,
    batch_id: str,
    path: Path,
    target: Path,
    step_index: int,
    step_count: int,
    operation_run_id: str,
    operation_phase_id: str,
    now: datetime,
) -> tuple[OperationBatchRecord, str, str]:
    batch = journal.create_batch(
        idempotency_key=f"mkdir:recursive:{step_index}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        operation_run_id=operation_run_id,
        operation_phase_id=operation_phase_id,
        resource_key=directory_resource_key(path),
        claim_owner="owner-a",
        payload=_recursive_batch_payload(target=target, step_path=path, step_index=step_index, step_count=step_count),
        batch_id=batch_id,
        now=now,
    )
    _, operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(path),
        payload={
            "operation": "make_directory",
            "path": str(path),
            "resource_key": directory_resource_key(path),
        },
        operation_id=f"operation-created-{step_index}",
        now=now + timedelta(microseconds=1),
    )
    checkpoint = journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=directory_resource_key(path),
        checkpoint_type="after",
        checkpoint_id=f"checkpoint-created-{step_index}",
        payload=_after_directory_checkpoint_payload(path),
        now=now + timedelta(microseconds=2),
    )
    journal.mark_succeeded(batch.batch_id, lease=lease, result={"changed": True}, now=now + timedelta(microseconds=3))
    return batch, operation.operation_id, checkpoint.checkpoint_id


def _create_foreign_directory_batch(
    journal: OperationJournalStore,
    *,
    lease: LeaseRecord,
    path: Path,
    now: datetime,
) -> tuple[OperationBatchRecord, str, str]:
    batch = journal.create_batch(
        idempotency_key="mkdir:foreign",
        lease=lease,
        owner="owner-a",
        run_id="run-2",
        resource_key=directory_resource_key(path),
        claim_owner="owner-a",
        payload={
            "operation": "make_directory",
            "recursive_mkdir": {
                "plan_target": str(path),
                "requested_resource_key": directory_resource_key(path),
                "step_index": 1,
                "step_count": 1,
                "step_path": str(path),
                "step_resource_key": directory_resource_key(path),
            },
        },
        batch_id="batch-foreign",
        now=now,
    )
    _, operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(path),
        payload={
            "operation": "make_directory",
            "path": str(path),
            "resource_key": directory_resource_key(path),
        },
        operation_id="operation-foreign-1",
        now=now + timedelta(microseconds=1),
    )
    checkpoint = journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=operation.operation_id,
        resource_key=directory_resource_key(path),
        checkpoint_type="after",
        checkpoint_id="checkpoint-foreign-1",
        payload=_after_directory_checkpoint_payload(path),
        now=now + timedelta(microseconds=2),
    )
    journal.mark_succeeded(batch.batch_id, lease=lease, result={"changed": True}, now=now + timedelta(microseconds=3))
    return batch, operation.operation_id, checkpoint.checkpoint_id


def _created_step_payload(
    path: Path,
    *,
    step_index: int,
    batch_id: str,
    operation_id: str,
    checkpoint_id: str,
) -> dict[str, object]:
    return {
        "step_index": step_index,
        "step_path": str(path),
        "step_resource_key": directory_resource_key(path),
        "ownership_class": "created_by_transaction",
        "created_directory_identity": _directory_identity_payload(path),
        "journaled_creation_proof": {
            "batch_id": batch_id,
            "operation_id": operation_id,
            "operation_type": "make_directory",
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": "after",
            "path": str(path),
            "resource_key": directory_resource_key(path),
        },
        "cleanup_policy": "owned_empty_directory_safe_ish",
        "backend_guarantee": "identity_conditional_remove",
    }


def _recursive_batch_payload(
    *,
    target: Path,
    step_path: Path,
    step_index: int,
    step_count: int,
) -> dict[str, object]:
    return {
        "operation": "make_directory",
        "recursive_mkdir": {
            "plan_target": str(target),
            "requested_resource_key": directory_resource_key(target),
            "step_index": step_index,
            "step_count": step_count,
            "step_path": str(step_path),
            "step_resource_key": directory_resource_key(step_path),
        },
    }


def _directory_identity_payload(path: Path) -> dict[str, object]:
    identity = DirectoryIdentity.from_stat(path.stat())
    return {
        "path": str(path),
        "resource_key": directory_resource_key(path),
        "file_type": "directory",
        "device": identity.device,
        "inode": identity.inode,
    }


def _after_directory_checkpoint_payload(path: Path) -> dict[str, object]:
    stat_result = path.stat()
    identity = DirectoryIdentity.from_stat(stat_result)
    return {
        "path": str(path),
        "exists": True,
        "file_type": "directory",
        "content_hash": None,
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "symlink_target": None,
        "device": identity.device,
        "inode": identity.inode,
    }


def _latest_recovery_payload(context: JournaledFilesystemRecoveryContext) -> dict[str, object]:
    for record in reversed(context.recovery_records):
        payload = cast(dict[str, object], record.payload)
        if "recursive_group" in payload:
            recursive_group = cast(dict[str, object], payload["recursive_group"])
            return {
                **payload,
                "recursive_group": {
                    **recursive_group,
                    "prior_created_steps": [
                        dict(step) for step in cast(list[Mapping[str, object]], recursive_group["prior_created_steps"])
                    ],
                },
            }
    raise AssertionError("recursive mkdir recovery payload not found")


def _context_with_recovery_payload(
    context: JournaledFilesystemRecoveryContext,
    payload: dict[str, object],
) -> JournaledFilesystemRecoveryContext:
    records = list(context.recovery_records)
    for index in range(len(records) - 1, -1, -1):
        if "recursive_group" in records[index].payload:
            records[index] = replace(records[index], payload=payload)
            return replace(context, recovery_records=tuple(records))
    raise AssertionError("recursive mkdir recovery payload not found")


def portable_snapshot(path: Path | str) -> ResourceSnapshot:
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
    content_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
    file_type = cast(FileType, "file" if target.is_file() else "directory" if target.is_dir() else "other")
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type=file_type,
        content_hash=content_hash,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
    )
