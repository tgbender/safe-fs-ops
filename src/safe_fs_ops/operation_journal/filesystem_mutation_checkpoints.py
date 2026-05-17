from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from stat import S_IMODE, S_ISREG

from safe_fs_ops.filesystem_ops import (
    FileBackup,
    ResourceSnapshot,
    capture_backup,
)
from safe_fs_ops.operation_journal.filesystem_support import _snapshot_payload
from safe_fs_ops.operation_journal.journal import OperationJournalStore
from safe_fs_ops.operation_journal.models import ArtifactCleanupTrigger, CheckpointRecord
from safe_fs_ops.workspace_state.models import LeaseRecord


@dataclass(frozen=True, slots=True)
class MutationPreparation:
    before_checkpoint: CheckpointRecord
    recovery_payload: dict[str, object]


def prepare_mutation_checkpoints(
    journal_store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    batch_id: str,
    operation_id: str,
    resource_key: str,
    path: Path,
    before_snapshot: ResourceSnapshot,
    operation_type: str,
    recovery_payload: Mapping[str, object],
    capture_file_rollback_proof: bool,
    operation_time: Callable[[], datetime],
    capture_backup_operation: Callable[..., FileBackup] = capture_backup,
) -> MutationPreparation:
    before_checkpoint = journal_store.record_checkpoint(
        batch_id,
        lease=lease,
        operation_id=operation_id,
        resource_key=resource_key,
        checkpoint_type="before",
        payload=_snapshot_payload(before_snapshot),
        now=operation_time(),
    )
    updated_recovery_payload = dict(recovery_payload)
    if operation_type not in {"write_bytes", "write_text", "delete_file"} or not capture_file_rollback_proof:
        return MutationPreparation(
            before_checkpoint=before_checkpoint,
            recovery_payload=updated_recovery_payload,
        )

    backup_content = backup_content_path(
        journal_store,
        batch_id=batch_id,
        operation_id=operation_id,
        path=path,
    )
    backup_payload = planned_backup_payload(
        path,
        before_snapshot=before_snapshot,
        content_path=backup_content,
    )
    backup_phase_time = operation_time()
    backup_intent_checkpoint = journal_store.record_checkpoint(
        batch_id,
        lease=lease,
        operation_id=operation_id,
        resource_key=resource_key,
        checkpoint_type="backup_artifact_intent",
        payload=backup_payload,
        now=backup_phase_time,
    )
    from safe_fs_ops.operation_journal.file_backup_artifacts import plan_backup_artifact_cleanup_candidates

    for candidate in plan_backup_artifact_cleanup_candidates(
        (backup_intent_checkpoint,),
        state_path=Path(journal_store.path),
    ):
        journal_store.record_artifact_cleanup_planned(
            batch_id=batch_id,
            lease=lease,
            artifact_id=candidate.artifact_id,
            trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
            resource_key=candidate.resource_key,
            payload={
                "checkpoint_id": backup_intent_checkpoint.checkpoint_id,
                "content_path": str(candidate.content_path),
                **(
                    {}
                    if candidate.expected_content_path is None
                    else {"expected_content_path": str(candidate.expected_content_path)}
                ),
                **(
                    {}
                    if candidate.debt is None
                    else {
                        "reason_code": candidate.debt.reason_code,
                        "detail": candidate.debt.detail,
                    }
                ),
            },
            now=backup_phase_time,
        )
    captured_backup = capture_backup_artifact(
        journal_store,
        path,
        before_snapshot=before_snapshot,
        batch_id=batch_id,
        operation_id=operation_id,
        capture_backup_operation=capture_backup_operation,
    )
    if captured_backup_payload(captured_backup) != backup_payload:
        raise ValueError(f"backup rollback proof changed during capture: {path}")
    journal_store.record_checkpoint(
        batch_id,
        lease=lease,
        operation_id=operation_id,
        resource_key=resource_key,
        checkpoint_type="backup",
        payload=backup_payload,
        now=operation_time(),
    )
    expected_after_payload = expected_after_checkpoint_payload(path, recovery_payload)
    journal_store.record_checkpoint(
        batch_id,
        lease=lease,
        operation_id=operation_id,
        resource_key=resource_key,
        checkpoint_type="expected_after",
        payload=expected_after_payload,
        now=operation_time(),
    )
    updated_recovery_payload.update(
        {
            "backup": backup_payload,
            "expected_after": expected_after_payload,
        }
    )
    return MutationPreparation(
        before_checkpoint=before_checkpoint,
        recovery_payload=updated_recovery_payload,
    )


def record_after_mutation_checkpoint(
    journal_store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    batch_id: str,
    operation_id: str,
    resource_key: str,
    path: Path,
    after_snapshot: ResourceSnapshot,
    operation_type: str,
    operation_time: Callable[[], datetime],
) -> CheckpointRecord:
    payload = _snapshot_payload(after_snapshot)
    if operation_type == "make_directory":
        payload = make_directory_after_checkpoint_payload(
            path,
            snapshot=after_snapshot,
            payload=payload,
        )
    return journal_store.record_checkpoint(
        batch_id,
        lease=lease,
        operation_id=operation_id,
        resource_key=resource_key,
        checkpoint_type="after",
        payload=payload,
        now=operation_time(),
    )


def make_directory_after_checkpoint_payload(
    path: Path,
    *,
    snapshot: ResourceSnapshot,
    payload: dict[str, object],
) -> dict[str, object]:
    if not snapshot.exists or snapshot.file_type != "directory":
        return payload
    del path
    if snapshot.device is None or snapshot.inode is None:
        return payload
    return {**payload, "device": snapshot.device, "inode": snapshot.inode}


def expected_after_checkpoint_payload(
    path: Path,
    recovery_payload: Mapping[str, object],
) -> dict[str, object]:
    desired = recovery_payload.get("desired")
    payload = dict(desired) if isinstance(desired, Mapping) else {}
    payload["path"] = str(path)
    return payload


def captured_backup_payload(backup: FileBackup) -> dict[str, object]:
    return {
        "path": str(backup.path),
        "existed": backup.existed,
        "file_type": backup.file_type,
        "content_path": None if backup.content_path is None else str(backup.content_path),
        "content_hash": backup.content_hash,
        "size": backup.size,
        "permissions": backup.permissions,
        "snapshot": _snapshot_payload(backup.snapshot),
    }


def planned_backup_payload(
    path: Path,
    *,
    before_snapshot: ResourceSnapshot,
    content_path: Path,
) -> dict[str, object]:
    if before_snapshot.file_type == "missing":
        return captured_backup_payload(
            FileBackup(
                path=path,
                existed=False,
                file_type="missing",
                content_bytes=None,
                content_path=None,
                content_address=None,
                content_hash=None,
                size=None,
                permissions=None,
                snapshot=before_snapshot,
            )
        )
    permissions = _validated_backup_permissions(path, before_snapshot=before_snapshot)
    return captured_backup_payload(
        FileBackup(
            path=path,
            existed=True,
            file_type="file",
            content_bytes=None,
            content_path=content_path,
            content_address=None,
            content_hash=before_snapshot.content_hash,
            size=before_snapshot.size,
            permissions=permissions,
            snapshot=before_snapshot,
        )
    )


def capture_backup_artifact(
    journal_store: OperationJournalStore,
    path: Path,
    *,
    before_snapshot: ResourceSnapshot,
    batch_id: str,
    operation_id: str,
    capture_backup_operation: Callable[..., FileBackup] = capture_backup,
) -> FileBackup:
    content_path = backup_content_path(
        journal_store,
        batch_id=batch_id,
        operation_id=operation_id,
        path=path,
    )
    return capture_backup_operation(
        path,
        snapshot=before_snapshot,
        content_path=content_path,
    )


def _validated_backup_permissions(path: Path, *, before_snapshot: ResourceSnapshot) -> int:
    if before_snapshot.file_type != "file" or not before_snapshot.exists:
        raise ValueError("backup rollback proof can only be planned for missing or regular file snapshots")
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise ValueError(f"backup rollback proof refused because file metadata could not be captured: {path}") from exc
    if not S_ISREG(stat_result.st_mode):
        raise ValueError(f"backup rollback proof refused for non-regular file: {path}")
    if stat_result.st_size != before_snapshot.size or stat_result.st_mtime_ns != before_snapshot.mtime_ns:
        raise ValueError(f"backup rollback proof refused because file changed before backup planning: {path}")
    return S_IMODE(stat_result.st_mode)


def backup_content_path(
    journal_store: OperationJournalStore,
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
) -> Path:
    return backup_content_path_for_state_path(
        Path(journal_store.path),
        batch_id=batch_id,
        operation_id=operation_id,
        path=path,
    )


def backup_content_path_for_state_path(
    state_path: Path,
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
) -> Path:
    return (
        state_path.parent
        / f"{state_path.name}.artifacts"
        / "file-backups"
        / _backup_content_artifact_name(batch_id=batch_id, operation_id=operation_id, path=path)
    )


def _backup_content_artifact_name(
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
) -> str:
    artifact_key = f"safe-fs-ops:file-backup\0{batch_id}\0{operation_id}\0{path}"
    return f"{uuid.uuid5(uuid.NAMESPACE_URL, artifact_key).hex}.bak"
