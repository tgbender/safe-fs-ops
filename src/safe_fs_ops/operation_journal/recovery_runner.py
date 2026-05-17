from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from safe_fs_ops.filesystem_ops import (
    BackupContentMismatchError,
    DurabilityMode,
    FileBackup,
    ResourceSnapshot,
    RestoreConflictError,
    UnsafePathError,
    restore_backup,
)
from safe_fs_ops.filesystem_ops.models import BackupFileType, FileType
from safe_fs_ops.operation_journal.models import (
    JournaledFilesystemRecoveryContext,
    RecoveryActionManualInterventionRequired,
    RecoveryActionRecord,
    require_recovery_action_authority,
)


class RecoveryActionSkipped(RuntimeError):
    def __init__(self, message: str, *, payload: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


RecoveryActionHandler = Callable[[JournaledFilesystemRecoveryContext, RecoveryActionRecord], object | None]

__all__ = [
    "RecoveryActionHandler",
    "RecoveryActionManualInterventionRequired",
    "RecoveryActionSkipped",
]


def _default_recovery_action_handlers() -> Mapping[str, RecoveryActionHandler]:
    from safe_fs_ops.operation_journal.tree_backup_recovery import (
        RESTORE_TREE_BACKUP_ACTION,
        restore_tree_backup_recovery_action,
    )

    return {
        "restore_backup": _restore_backup_recovery_action,
        RESTORE_TREE_BACKUP_ACTION: restore_tree_backup_recovery_action,
    }


def _latest_current_attempt_recovery_actions(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[RecoveryActionRecord, ...]:
    recovery_attempt_id = context.recovery_attempt_id
    if recovery_attempt_id is None:
        return ()
    latest_by_action_id: dict[str, RecoveryActionRecord] = {}
    action_order: list[str] = []
    for action in context.recovery_actions:
        if action.recovery_attempt_id != recovery_attempt_id:
            continue
        if action.action_id not in latest_by_action_id:
            action_order.append(action.action_id)
        latest_by_action_id[action.action_id] = action
    return tuple(latest_by_action_id[action_id] for action_id in action_order)


def _restore_backup_recovery_action(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> ResourceSnapshot:
    payload = action.payload
    backup = _file_backup_from_payload(payload)
    expected_current = _resource_snapshot_from_payload(payload.get("expected_current"))
    allow_overwrite = _bool_from_payload_value(payload.get("allow_overwrite"), default=False)
    try:
        require_recovery_action_authority(context, action)
        return restore_backup(
            backup,
            expected_current=expected_current,
            allow_overwrite=allow_overwrite,
            durability=DurabilityMode.FSYNC,
        )
    except (RestoreConflictError, BackupContentMismatchError, UnsafePathError) as exc:
        raise RecoveryActionManualInterventionRequired(
            f"backup restore requires manual intervention: {backup.path}",
            payload=_restore_backup_manual_intervention_payload(backup, exc),
        ) from exc


def _file_backup_from_payload(payload: Mapping[str, object]) -> FileBackup:
    backup_payload = payload.get("backup")
    if not isinstance(backup_payload, Mapping):
        raise ValueError("recovery action payload must include a backup mapping")
    snapshot_payload = backup_payload.get("snapshot")
    if not isinstance(snapshot_payload, Mapping):
        raise ValueError("recovery action payload must include a backup.snapshot mapping")
    snapshot = _resource_snapshot_from_payload(snapshot_payload)
    if snapshot is None:
        raise ValueError("recovery action payload must include a backup.snapshot mapping")
    return FileBackup(
        path=_path_from_payload_value(backup_payload.get("path")),
        existed=_bool_from_payload_value(backup_payload.get("existed")),
        file_type=_backup_file_type_from_payload_value(backup_payload.get("file_type")),
        content_bytes=None,
        content_path=_path_from_optional_payload_value(backup_payload.get("content_path")),
        content_hash=_optional_str_value(backup_payload.get("content_hash")),
        size=_optional_int_value(backup_payload.get("size")),
        permissions=_optional_int_value(backup_payload.get("permissions")),
        snapshot=snapshot,
    )


def _resource_snapshot_from_payload(value: object | None) -> ResourceSnapshot | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("recovery action payload snapshot must be a mapping")
    return ResourceSnapshot(
        path=_path_from_payload_value(value.get("path")),
        exists=_bool_from_payload_value(value.get("exists")),
        file_type=_file_type_from_payload_value(value.get("file_type")),
        content_hash=_optional_str_value(value.get("content_hash")),
        size=_optional_int_value(value.get("size")),
        mtime_ns=_optional_int_value(value.get("mtime_ns")),
        symlink_target=_optional_str_value(value.get("symlink_target")),
        device=_optional_int_value(value.get("device")),
        inode=_optional_int_value(value.get("inode")),
        ctime_ns=_optional_int_value(value.get("ctime_ns")),
    )


def _restore_backup_manual_intervention_payload(
    backup: FileBackup,
    error: RestoreConflictError | BackupContentMismatchError | UnsafePathError,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(backup.path),
        "reason_code": _restore_backup_reason_code(error),
        "detail": str(error),
    }
    if backup.content_path is not None:
        payload["backup_content_path"] = str(backup.content_path)
    return payload


def _restore_backup_reason_code(
    error: RestoreConflictError | BackupContentMismatchError | UnsafePathError,
) -> str:
    if isinstance(error, RestoreConflictError):
        return "restore_conflict"
    if isinstance(error, BackupContentMismatchError):
        return "backup_content_mismatch"
    return "unsafe_path"


def _recovery_action_status_payload(
    action: RecoveryActionRecord,
    *,
    event_type: str,
    reason: str | None = None,
    result_type: str | None = None,
    exception: Exception | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": event_type,
        "action_id": action.action_id,
        "action_type": action.action_type,
        "resource_key": action.resource_key,
        "recovery_attempt_id": action.recovery_attempt_id,
    }
    if reason is not None:
        payload["reason"] = reason
    if result_type is not None:
        payload["result_type"] = result_type
    for key in ("path", "step_resource_key", "recursive_mkdir_cleanup", "store_path"):
        value = action.payload.get(key)
        if value is not None:
            payload[key] = dict(value) if isinstance(value, Mapping) else value
    if exception is not None:
        payload["error_type"] = type(exception).__name__
        payload["error"] = str(exception)
        exception_payload = getattr(exception, "payload", None)
        if exception_payload is not None:
            payload["exception_payload"] = (
                dict(exception_payload) if isinstance(exception_payload, Mapping) else exception_payload
            )
    return payload


def _path_from_payload_value(value: object | None) -> Path:
    if value is None:
        raise ValueError("recovery action payload must include a path")
    return Path(str(value))


def _path_from_optional_payload_value(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_str_value(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _bool_from_payload_value(value: object | None, *, default: bool | None = None) -> bool:
    if value is None:
        if default is not None:
            return default
        raise ValueError("boolean payload value is required")
    if not isinstance(value, bool):
        raise ValueError(f"boolean payload value must be a bool, not {type(value).__name__}")
    return value


def _backup_file_type_from_payload_value(value: object | None) -> BackupFileType:
    file_type = str(value)
    if file_type not in {"missing", "file"}:
        raise ValueError(f"unsupported backup file_type: {file_type!r}")
    return cast("BackupFileType", file_type)


def _file_type_from_payload_value(value: object | None) -> FileType:
    file_type = str(value)
    if file_type not in {"missing", "file", "directory", "symlink", "other"}:
        raise ValueError(f"unsupported snapshot file_type: {file_type!r}")
    return cast("FileType", file_type)


def _optional_int_value(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("integer payload value must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"integer payload value must be int or str, not {type(value).__name__}")
