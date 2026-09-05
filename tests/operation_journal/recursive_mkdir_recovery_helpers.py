from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from safe_fs_ops.filesystem_ops import CapturedDirectoryRecord, DirectoryIdentity, ResourceSnapshot
from safe_fs_ops.operation_journal import OperationJournalStore, RecoveryActionRecord, directory_resource_key
from safe_fs_ops.workspace_state.models import LeaseRecord


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
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
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
    file_type = "file" if target.is_file() else "directory" if target.is_dir() else "other"
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type=file_type,
        content_hash=content_hash,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )


def _start_recursive_mkdir_recovery_batch(
    tmp_path: Path,
) -> tuple[OperationJournalStore, LeaseRecord, str, Path, Path]:
    state_path = tmp_path / "state.db"
    leaf = tmp_path / "config" / "state" / "cache"
    created_parent = leaf.parent
    created_root = created_parent.parent
    created_parent.mkdir(parents=True)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    from recovery_runner_helpers import _acquire_and_claim

    lease = _acquire_and_claim(state_path, directory_resource_key(leaf), now=now)
    journal = OperationJournalStore(state_path)
    batch = journal.create_batch(
        idempotency_key=f"recover:{directory_resource_key(leaf)}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=None,
        claim_owner="owner-a",
        payload={"operation": "make_directory"},
        batch_id="batch-1",
        now=now,
    )
    _, created_root_operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(created_root),
        payload={
            "operation": "make_directory",
            "path": str(created_root),
            "resource_key": directory_resource_key(created_root),
        },
        operation_id="operation-created-1",
        now=now,
    )
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=created_root_operation.operation_id,
        resource_key=directory_resource_key(created_root),
        checkpoint_type="after",
        checkpoint_id="checkpoint-created-1",
        payload=_after_directory_checkpoint_payload(created_root),
        now=now + timedelta(microseconds=1),
    )
    created_parent_operation = journal.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(created_parent),
        payload={
            "operation": "make_directory",
            "path": str(created_parent),
            "resource_key": directory_resource_key(created_parent),
        },
        operation_id="operation-created-2",
        now=now + timedelta(microseconds=2),
    )
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=created_parent_operation.operation_id,
        resource_key=directory_resource_key(created_parent),
        checkpoint_type="after",
        checkpoint_id="checkpoint-created-2",
        payload=_after_directory_checkpoint_payload(created_parent),
        now=now + timedelta(microseconds=3),
    )
    journal.append_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(leaf),
        payload={
            "operation": "make_directory",
            "path": str(leaf),
            "resource_key": directory_resource_key(leaf),
        },
        operation_id="operation-created-3",
        now=now + timedelta(microseconds=4),
    )
    journal.mark_failed(
        batch.batch_id,
        lease=lease,
        error="mkdir failed",
        observed_state={"resource_key": directory_resource_key(leaf)},
        now=now + timedelta(microseconds=5),
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="filesystem mutation failed",
        payload={
            "recursive_mkdir": {"step_index": 3, "step_path": str(leaf)},
            "recursive_group": {
                "prior_created_steps": [
                    {
                        "step_index": 1,
                        "step_path": str(created_root),
                        "step_resource_key": directory_resource_key(created_root),
                        "ownership_class": "created_by_transaction",
                        "created_directory_identity": _directory_identity_payload(created_root),
                        "journaled_creation_proof": _journaled_creation_proof_payload(created_root, sequence=1),
                        "cleanup_policy": "owned_empty_directory_safe_ish",
                        "backend_guarantee": "identity_conditional_remove",
                    },
                    {
                        "step_index": 2,
                        "step_path": str(created_parent),
                        "step_resource_key": directory_resource_key(created_parent),
                        "ownership_class": "created_by_transaction",
                        "created_directory_identity": _directory_identity_payload(created_parent),
                        "journaled_creation_proof": _journaled_creation_proof_payload(created_parent, sequence=2),
                        "cleanup_policy": "owned_empty_directory_safe_ish",
                        "backend_guarantee": "identity_conditional_remove",
                    },
                ]
            },
        },
        recovery_id="recovery-desired",
        now=now,
    )
    journal.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-attempt",
        now=now + timedelta(microseconds=1),
    )
    return journal, lease, batch.batch_id, created_root, created_parent


def _recovery_action(
    *,
    payload: Mapping[str, object],
    action_type: str | None = None,
) -> RecoveryActionRecord:
    from safe_fs_ops.operation_journal.recursive_mkdir_recovery_actions import REMOVE_CREATED_DIRECTORY_ACTION

    return RecoveryActionRecord(
        action_record_id="action-record-1",
        action_id="action-1",
        recovery_attempt_id="recovery-attempt",
        batch_id="batch-1",
        sequence=1,
        action_type=action_type or REMOVE_CREATED_DIRECTORY_ACTION,
        status="planned",
        resource_key=None,
        reason=None,
        payload=payload,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _directory_identity_payload(
    path: Path,
    *,
    resource_key: str | None = None,
    identity: DirectoryIdentity | None = None,
) -> dict[str, object]:
    current_identity = identity or DirectoryIdentity.from_stat(path.stat())
    return {
        "path": str(path),
        "resource_key": resource_key or directory_resource_key(path),
        "file_type": "directory",
        "device": current_identity.device,
        "inode": current_identity.inode,
    }


def _journaled_creation_proof_payload(path: Path, *, sequence: int) -> dict[str, object]:
    return {
        "batch_id": "batch-1",
        "operation_id": f"operation-created-{sequence}",
        "operation_type": "make_directory",
        "checkpoint_id": f"checkpoint-created-{sequence}",
        "checkpoint_type": "after",
        "path": str(path),
        "resource_key": directory_resource_key(path),
    }


def _after_directory_checkpoint_payload(path: Path) -> dict[str, object]:
    identity = DirectoryIdentity.from_stat(path.stat())
    return {
        "path": str(path),
        "exists": True,
        "file_type": "directory",
        "content_hash": None,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "symlink_target": None,
        "device": identity.device,
        "inode": identity.inode,
    }


def _captured_directory_step_payload(
    record: CapturedDirectoryRecord,
    *,
    step_index: int,
    resource_key: str | None = None,
) -> dict[str, object]:
    step_resource_key = resource_key or directory_resource_key(record.original_path)
    return {
        "step_index": step_index,
        "step_path": str(record.original_path),
        "step_resource_key": step_resource_key,
        "ownership_class": "captured_by_transaction",
        "captured_directory": {
            "original_path": str(record.original_path),
            "capture_token": record.capture_token,
            "quarantine_path": str(record.quarantine_path),
            "original_identity": _directory_identity_payload(
                record.original_path,
                resource_key=step_resource_key,
                identity=record.original_identity,
            ),
            "captured_identity": _directory_identity_payload(
                record.quarantine_path,
                resource_key=step_resource_key,
                identity=record.captured_identity,
            ),
            "ownership_class": "captured_by_transaction",
        },
    }


def _latest_recovery_payload(context: object) -> dict[str, object]:
    for record in reversed(cast(Any, context).recovery_records):
        payload = cast(dict[str, object], record.payload)
        if "recursive_group" in payload:
            return {
                **payload,
                "recursive_group": {
                    **cast(dict[str, object], payload["recursive_group"]),
                    "prior_created_steps": [
                        dict(cast(Mapping[str, object], step))
                        for step in cast(list[Mapping[str, object]], payload["recursive_group"]["prior_created_steps"])
                    ],
                },
            }
    raise AssertionError("recursive mkdir recovery payload not found")


def _context_with_recovery_payload(context: object, payload: dict[str, object]) -> object:
    records = list(cast(Any, context).recovery_records)
    for index in range(len(records) - 1, -1, -1):
        if "recursive_group" in records[index].payload:
            records[index] = replace(records[index], payload=payload)
            return replace(context, recovery_records=tuple(records))
    raise AssertionError("recursive mkdir recovery payload not found")


def _captured_directory_action_payload(
    record: CapturedDirectoryRecord,
    *,
    step_index: int = 1,
    resource_key: str | None = None,
) -> dict[str, object]:
    return {
        "path": str(record.original_path),
        "step_resource_key": resource_key or directory_resource_key(record.original_path),
        **_captured_directory_step_payload(
            record,
            step_index=step_index,
            resource_key=resource_key,
        ),
    }
