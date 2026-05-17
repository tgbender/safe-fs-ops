from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from safe_fs_ops.filesystem_ops import DirectoryIdentity
from safe_fs_ops.operation_journal import (
    CheckpointRecord,
    OperationBatchRecord,
    OperationRecord,
    RecoveryActionRecord,
    RecoveryAuthority,
    RecoveryRecord,
    directory_resource_key,
)
from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext


def recovery_action(*, payload: dict[str, object]) -> RecoveryActionRecord:
    from safe_fs_ops.operation_journal.recursive_mkdir_recovery_actions import REMOVE_CREATED_DIRECTORY_ACTION

    return RecoveryActionRecord(
        action_record_id="action-record-1",
        action_id="action-1",
        recovery_attempt_id="recovery-attempt",
        batch_id="batch-1",
        sequence=1,
        action_type=REMOVE_CREATED_DIRECTORY_ACTION,
        status="planned",
        resource_key=None,
        reason=None,
        payload=payload,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def directory_identity_payload(
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


def journaled_creation_proof_payload(path: Path, *, sequence: int = 1) -> dict[str, object]:
    return {
        "batch_id": "batch-1",
        "operation_id": f"operation-created-{sequence}",
        "operation_type": "make_directory",
        "checkpoint_id": f"checkpoint-created-{sequence}",
        "checkpoint_type": "after",
        "path": str(path),
        "resource_key": directory_resource_key(path),
    }


def verified_recovery_context(path: Path) -> JournaledFilesystemRecoveryContext:
    resource_key = directory_resource_key(path)
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    return JournaledFilesystemRecoveryContext(
        batch=OperationBatchRecord(
            batch_id="batch-1",
            idempotency_key="mkdir:state",
            lease_name="workspace",
            lease_fencing_token=1,
            owner="owner-a",
            run_id="run-1",
            operation_run_id=None,
            operation_phase_id=None,
            resource_key=resource_key,
            claim_owner="owner-a",
            claim_scope=None,
            phase="recovering",
            payload={"operation": "make_directory"},
            status_message=None,
            status_payload={},
            created_at=created_at,
            updated_at=created_at,
        ),
        operations=(
            OperationRecord(
                operation_id="operation-created-1",
                batch_id="batch-1",
                sequence=1,
                operation_type="make_directory",
                resource_key=resource_key,
                payload={
                    "operation": "make_directory",
                    "path": str(path),
                    "resource_key": resource_key,
                },
                created_at=created_at,
            ),
        ),
        checkpoints=(
            CheckpointRecord(
                checkpoint_id="checkpoint-created-1",
                batch_id="batch-1",
                sequence=2,
                operation_id="operation-created-1",
                resource_key=resource_key,
                checkpoint_type="after",
                payload={
                    "path": str(path),
                    "exists": True,
                    "file_type": "directory",
                    "device": DirectoryIdentity.from_stat(path.stat()).device,
                    "inode": DirectoryIdentity.from_stat(path.stat()).inode,
                },
                created_at=created_at,
            ),
        ),
        recovery_records=(),
        recovery_attempt=RecoveryRecord(
            recovery_id="recovery-attempt",
            batch_id="batch-1",
            sequence=3,
            phase="recovering",
            reason="recovery reserved",
            payload={},
            created_at=created_at,
        ),
        recovery_actions=(recovery_action(payload={}),),
        recovery_authority=RecoveryAuthority(lambda: None, recovery_attempt_id="recovery-attempt"),
    )
