from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from safe_fs_ops.filesystem_ops import RenameRecord, UnsafePathError, inspect_path, restore_inverse_rename
from safe_fs_ops.filesystem_ops.renames import RenamedFileType
from safe_fs_ops.operation_journal.models import (
    JournaledFilesystemRecoveryContext,
    RecoveryActionRecord,
    require_recovery_action_authority,
)
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery_actions import mapping_payload, optional_str

RESTORE_INVERSE_RENAME_ACTION = "restore_inverse_rename"


def rename_record_payload(record: RenameRecord) -> dict[str, object]:
    return {
        "source_path": str(record.source_path),
        "destination_path": str(record.destination_path),
        "file_type": record.file_type,
        "device": record.device,
        "inode": record.inode,
        "size": record.size,
        "mtime_ns": record.mtime_ns,
        "ctime_ns": record.ctime_ns,
    }


def restore_inverse_rename_recovery_action(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> dict[str, object]:
    payload = mapping_payload(action.payload)
    record = rename_record_from_payload(payload)
    try:
        require_recovery_action_authority(context, action)
        restored = restore_inverse_rename(record)
    except FileNotFoundError as exc:
        _skip_if_already_restored(record, cause=exc)
        raise _manual_intervention(record, reason_code="rename_destination_missing", detail=str(exc)) from exc
    except FileExistsError as exc:
        _skip_if_already_restored(record, cause=exc)
        raise _manual_intervention(record, reason_code="restore_destination_exists", detail=str(exc)) from exc
    except UnsafePathError as exc:
        raise _manual_intervention(record, reason_code="rename_identity_mismatch", detail=str(exc)) from exc
    return {
        "source_path": str(record.source_path),
        "destination_path": str(record.destination_path),
        "restored": True,
        "restored_device": restored.device,
        "restored_inode": restored.inode,
    }


def rename_record_from_payload(payload: Mapping[str, object]) -> RenameRecord:
    file_type = optional_str(payload.get("file_type"))
    if file_type not in {"file", "directory"}:
        raise RecoveryActionManualInterventionRequired(
            "inverse rename restore refused because payload file_type is invalid",
            payload={"reason_code": "invalid_payload"},
        )
    return RenameRecord(
        source_path=Path(str(payload.get("source_path"))),
        destination_path=Path(str(payload.get("destination_path"))),
        file_type=cast(RenamedFileType, file_type),
        device=_int_payload(payload, "device"),
        inode=_int_payload(payload, "inode"),
        size=_int_payload(payload, "size"),
        mtime_ns=_int_payload(payload, "mtime_ns"),
        ctime_ns=_optional_int_payload(payload, "ctime_ns"),
    )


def _int_payload(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise RecoveryActionManualInterventionRequired(
            f"inverse rename restore refused because payload {key!r} is invalid",
            payload={"reason_code": "invalid_payload"},
        )
    return value


def _optional_int_payload(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise RecoveryActionManualInterventionRequired(
            f"inverse rename restore refused because payload {key!r} is invalid",
            payload={"reason_code": "invalid_payload"},
        )
    return value


def _skip_if_already_restored(record: RenameRecord, *, cause: Exception) -> None:
    source_safety = inspect_path(record.source_path)
    destination_safety = inspect_path(record.destination_path)
    if destination_safety.exists:
        return
    if not source_safety.exists or source_safety.file_type != record.file_type:
        return
    current = record.source_path.lstat()
    if (
        current.st_dev != record.device
        or current.st_ino != record.inode
        or current.st_size != record.size
        or current.st_mtime_ns != record.mtime_ns
        or (record.ctime_ns is not None and current.st_ctime_ns != record.ctime_ns)
    ):
        return
    raise RecoveryActionSkipped(
        f"inverse rename restore skipped because path is already restored: {record.source_path}",
        payload={
            "source_path": str(record.source_path),
            "destination_path": str(record.destination_path),
            "reason_code": "already_restored",
            "detail": str(cause),
        },
    )


def _manual_intervention(
    record: RenameRecord,
    *,
    reason_code: str,
    detail: str,
) -> RecoveryActionManualInterventionRequired:
    return RecoveryActionManualInterventionRequired(
        f"inverse rename restore requires manual intervention: {record.source_path}",
        payload={
            "source_path": str(record.source_path),
            "destination_path": str(record.destination_path),
            "reason_code": reason_code,
            "detail": detail,
        },
    )


__all__ = [
    "RESTORE_INVERSE_RENAME_ACTION",
    "rename_record_from_payload",
    "rename_record_payload",
    "restore_inverse_rename_recovery_action",
]
