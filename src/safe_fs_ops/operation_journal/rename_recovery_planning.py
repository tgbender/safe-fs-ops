from __future__ import annotations

from collections.abc import Mapping

from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext
from safe_fs_ops.operation_journal.rename_recovery_actions import RESTORE_INVERSE_RENAME_ACTION


def rename_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    payload = _latest_rename_payload(context)
    if payload is None:
        manual_action = _latest_manual_intervention_action(context)
        if manual_action is not None:
            return (manual_action,)
        if _is_rename_no_replace_batch(context):
            return (_missing_rename_rollback_proof_action(context),)
        return ()
    return (
        {
            "action_id": f"restore-inverse-rename:{payload['destination_path']}:{payload['source_path']}",
            "action_type": RESTORE_INVERSE_RENAME_ACTION,
            "resource_key": context.batch.resource_key,
            "payload": dict(payload),
        },
    )


def _latest_rename_payload(context: JournaledFilesystemRecoveryContext) -> Mapping[str, object] | None:
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type != "rename_record":
            continue
        payload = checkpoint.payload
        if _looks_like_rename_payload(payload):
            return payload
    for record in reversed(context.recovery_records):
        record_payload = record.payload.get("rename_record")
        if isinstance(record_payload, Mapping) and _looks_like_rename_payload(record_payload):
            return record_payload
    return None


def _latest_manual_intervention_action(context: JournaledFilesystemRecoveryContext) -> dict[str, object] | None:
    for record in reversed(context.recovery_records):
        action = record.payload.get("manual_intervention_action")
        if not isinstance(action, Mapping):
            continue
        payload = action.get("payload")
        if not isinstance(payload, Mapping) or payload.get("operation") != "rename_no_replace":
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


def _is_rename_no_replace_batch(context: JournaledFilesystemRecoveryContext) -> bool:
    return isinstance(context.batch.payload, Mapping) and context.batch.payload.get("operation") == "rename_no_replace"


def _missing_rename_rollback_proof_action(context: JournaledFilesystemRecoveryContext) -> dict[str, object]:
    payload = context.batch.payload if isinstance(context.batch.payload, Mapping) else {}
    return {
        "action_id": f"rename-no-replace-manual:{context.batch.batch_id}:{context.batch.resource_key}",
        "action_type": "manual_intervention_required",
        "resource_key": context.batch.resource_key,
        "payload": {
            "operation": "rename_no_replace",
            "batch_id": context.batch.batch_id,
            "resource_key": context.batch.resource_key,
            "source_path": str(payload.get("source_path")),
            "destination_path": str(payload.get("destination_path")),
            "reason_code": "missing_rename_rollback_proof",
            "detail": "rename_no_replace has no durable inverse rename proof",
        },
    }


def _looks_like_rename_payload(payload: Mapping[str, object]) -> bool:
    return all(key in payload for key in ("source_path", "destination_path", "file_type", "device", "inode"))


__all__ = ["rename_recovery_action_plans"]
