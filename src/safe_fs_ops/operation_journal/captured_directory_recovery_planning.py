from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from safe_fs_ops.filesystem_ops import DirectoryIdentity, inspect_path
from safe_fs_ops.filesystem_ops.capture_directories import require_directory_capture_token
from safe_fs_ops.filesystem_ops.paths import UnsafePathError
from safe_fs_ops.operation_journal.captured_directory_recovery_actions import (
    RESTORE_CAPTURED_DIRECTORY_ACTION,
    planned_captured_directory_restore_payload,
)
from safe_fs_ops.operation_journal.models import CheckpointRecord, JournaledFilesystemRecoveryContext


def captured_directory_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    direct_plans = _direct_captured_directory_recovery_action_plans(context)
    if direct_plans:
        return direct_plans
    payload = _latest_recursive_mkdir_failure_payload(context)
    if payload is None:
        manual_action = _latest_manual_intervention_action(context)
        if manual_action is not None:
            return (manual_action,)
        if _is_capture_directory_batch(context):
            fallback_plan = _observed_captured_directory_recovery_action_plan(context)
            if fallback_plan is not None:
                return (fallback_plan,)
            return (_missing_captured_directory_rollback_proof_action(context),)
        return ()
    recursive_group = payload.get("recursive_group")
    if not isinstance(recursive_group, Mapping):
        return ()
    prior_created_steps = recursive_group.get("prior_created_steps")
    if not isinstance(prior_created_steps, tuple | list):
        return ()
    plans: list[dict[str, object]] = []
    ordered_steps = sorted(
        (step for step in prior_created_steps if isinstance(step, Mapping)),
        key=lambda step: int(step.get("step_index", 0)),
        reverse=True,
    )
    for step in ordered_steps:
        planned_payload = planned_captured_directory_restore_payload(step)
        if planned_payload is None:
            continue
        plans.append(
            {
                "action_id": (
                    f"recursive-mkdir-restore-captured:{int(step.get('step_index', 0))}:"
                    f"{planned_payload['step_resource_key']}"
                ),
                "action_type": RESTORE_CAPTURED_DIRECTORY_ACTION,
                "resource_key": context.batch.resource_key,
                "payload": planned_payload,
            }
        )
    return tuple(plans)


def _direct_captured_directory_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type != "captured_directory":
            continue
        payload = checkpoint.payload
        if not isinstance(payload.get("captured_directory"), Mapping):
            continue
        return (
            {
                "action_id": f"restore-captured-directory:{payload.get('step_resource_key')}:{payload.get('path')}",
                "action_type": RESTORE_CAPTURED_DIRECTORY_ACTION,
                "resource_key": context.batch.resource_key,
                "payload": dict(payload),
            },
        )
    for record in reversed(context.recovery_records):
        record_payload = record.payload.get("captured_directory_restore")
        if not isinstance(record_payload, Mapping):
            continue
        return (
            {
                "action_id": (
                    f"restore-captured-directory:{record_payload.get('step_resource_key')}:{record_payload.get('path')}"
                ),
                "action_type": RESTORE_CAPTURED_DIRECTORY_ACTION,
                "resource_key": context.batch.resource_key,
                "payload": dict(record_payload),
            },
        )
    return ()


def _observed_captured_directory_recovery_action_plan(
    context: JournaledFilesystemRecoveryContext,
) -> dict[str, object] | None:
    payload = context.batch.payload if isinstance(context.batch.payload, Mapping) else {}
    resource_key = _resource_key_for_capture(context, payload)
    source_path = _path_payload(payload.get("path"))
    quarantine_path = _path_payload(payload.get("quarantine_path"))
    if resource_key is None or source_path is None or quarantine_path is None:
        return None
    before_checkpoint = _latest_directory_before_checkpoint(
        context,
        resource_key=resource_key,
        source_path=source_path,
    )
    if before_checkpoint is None:
        return None
    source_safety = inspect_path(source_path)
    if source_safety.exists:
        return None
    quarantine_safety = inspect_path(quarantine_path)
    if (
        not quarantine_safety.exists
        or not quarantine_safety.is_dir
        or quarantine_safety.is_symlink
        or quarantine_safety.is_windows_reparse_point
        or quarantine_safety.is_mount
    ):
        return None
    try:
        quarantine_identity = DirectoryIdentity.from_stat(quarantine_path.stat())
    except OSError:
        return None
    device = before_checkpoint.payload.get("device")
    inode = before_checkpoint.payload.get("inode")
    if not isinstance(device, int) or not isinstance(inode, int):
        return None
    if quarantine_identity != DirectoryIdentity(device=device, inode=inode):
        return None
    capture_token = before_checkpoint.payload.get("capture_token")
    if type(capture_token) is not str:
        return None
    try:
        require_directory_capture_token(quarantine_path, identity=quarantine_identity, capture_token=capture_token)
    except (OSError, UnsafePathError):
        return None
    restore_payload = _captured_directory_restore_payload(
        source_path=source_path,
        quarantine_path=quarantine_path,
        resource_key=resource_key,
        device=device,
        inode=inode,
        capture_token=capture_token,
    )
    return {
        "action_id": f"restore-captured-directory-from-before:{resource_key}:{source_path}",
        "action_type": RESTORE_CAPTURED_DIRECTORY_ACTION,
        "resource_key": context.batch.resource_key,
        "payload": restore_payload,
    }


def _latest_directory_before_checkpoint(
    context: JournaledFilesystemRecoveryContext,
    *,
    resource_key: str,
    source_path: Path,
) -> CheckpointRecord | None:
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type != "before":
            continue
        if checkpoint.resource_key != resource_key:
            continue
        if Path(str(checkpoint.payload.get("path"))) != source_path:
            continue
        if checkpoint.payload.get("exists") is not True or checkpoint.payload.get("file_type") != "directory":
            continue
        if not isinstance(checkpoint.payload.get("device"), int) or not isinstance(
            checkpoint.payload.get("inode"),
            int,
        ):
            continue
        return checkpoint
    return None


def _captured_directory_restore_payload(
    *,
    source_path: Path,
    quarantine_path: Path,
    resource_key: str,
    device: int,
    inode: int,
    capture_token: str,
) -> dict[str, object]:
    original_identity = {
        "file_type": "directory",
        "path": str(source_path),
        "resource_key": resource_key,
        "device": device,
        "inode": inode,
    }
    captured_identity = {
        "file_type": "directory",
        "path": str(quarantine_path),
        "resource_key": resource_key,
        "device": device,
        "inode": inode,
    }
    return {
        "path": str(source_path),
        "step_resource_key": resource_key,
        "ownership_class": "captured_by_transaction",
        "captured_directory": {
            "original_path": str(source_path),
            "quarantine_path": str(quarantine_path),
            "original_identity": original_identity,
            "captured_identity": captured_identity,
            "capture_token": capture_token,
            "ownership_class": "captured_by_transaction",
        },
    }


def _resource_key_for_capture(context: JournaledFilesystemRecoveryContext, payload: Mapping[str, object]) -> str | None:
    resource_key = payload.get("resource_key")
    if isinstance(resource_key, str) and resource_key:
        return resource_key
    if context.batch.resource_key is not None:
        return context.batch.resource_key
    return None


def _path_payload(value: object) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _latest_manual_intervention_action(context: JournaledFilesystemRecoveryContext) -> dict[str, object] | None:
    for record in reversed(context.recovery_records):
        action = record.payload.get("manual_intervention_action")
        if not isinstance(action, Mapping):
            continue
        payload = action.get("payload")
        if not isinstance(payload, Mapping) or payload.get("operation") != "capture_directory":
            continue
        action_id = action.get("action_id")
        action_type = action.get("action_type")
        if not isinstance(action_id, str) or action_type != "manual_intervention_required":
            continue
        return {
            "action_id": action_id,
            "action_type": action_type,
            "resource_key": context.batch.resource_key,
            "payload": dict(payload),
        }
    return None


def _is_capture_directory_batch(context: JournaledFilesystemRecoveryContext) -> bool:
    return isinstance(context.batch.payload, Mapping) and context.batch.payload.get("operation") == "capture_directory"


def _missing_captured_directory_rollback_proof_action(context: JournaledFilesystemRecoveryContext) -> dict[str, object]:
    payload = context.batch.payload if isinstance(context.batch.payload, Mapping) else {}
    return {
        "action_id": f"capture-directory-manual:{context.batch.batch_id}:{context.batch.resource_key}",
        "action_type": "manual_intervention_required",
        "resource_key": context.batch.resource_key,
        "payload": {
            "operation": "capture_directory",
            "batch_id": context.batch.batch_id,
            "resource_key": context.batch.resource_key,
            "path": str(payload.get("path")),
            "quarantine_path": str(payload.get("quarantine_path")),
            "reason_code": "missing_captured_directory_restore_proof",
            "detail": "capture_directory has no durable captured-directory restore proof",
        },
    }


def _latest_recursive_mkdir_failure_payload(
    context: JournaledFilesystemRecoveryContext,
) -> Mapping[str, object] | None:
    for record in reversed(context.recovery_records):
        payload = record.payload
        if not isinstance(payload, Mapping):
            continue
        recursive_group = payload.get("recursive_group")
        if not isinstance(recursive_group, Mapping):
            continue
        prior_created_steps = recursive_group.get("prior_created_steps")
        if isinstance(prior_created_steps, tuple | list):
            return payload
    return None


__all__ = ["captured_directory_recovery_action_plans"]
