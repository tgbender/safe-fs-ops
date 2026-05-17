from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext
from safe_fs_ops.operation_journal.recursive_mkdir_recovery_actions import (
    ALLOWED_GENERIC_DIRECTORY_REMOVE_GUARANTEES,
    REMOVE_CREATED_DIRECTORY_ACTION,
    _journaled_creation_proof_matches_action,
    _payload_identity_matches_action,
    jsonable_mapping,
    jsonable_value,
    optional_str,
)


def recursive_mkdir_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    payload = _latest_recursive_mkdir_failure_payload(context)
    direct_cleanup = _direct_recursive_mkdir_cleanup_plan(context)
    if payload is None:
        return () if direct_cleanup is None else (direct_cleanup,)
    recursive_group = payload.get("recursive_group")
    if not isinstance(recursive_group, Mapping):
        return ()
    prior_created_steps = recursive_group.get("prior_created_steps")
    if not isinstance(prior_created_steps, tuple | list):
        return ()
    failed_step_path = _failed_recursive_mkdir_step_path(payload)
    plans: list[dict[str, object]] = []
    if direct_cleanup is not None:
        plans.append(direct_cleanup)
    prior_cleanup_count = 0
    ordered_steps = sorted(
        (step for step in prior_created_steps if isinstance(step, Mapping)),
        key=lambda step: int(step.get("step_index", 0)),
        reverse=True,
    )
    for step in ordered_steps:
        planned_payload = _planned_recursive_mkdir_cleanup_payload(context, step=step)
        if planned_payload is None:
            continue
        step_path = Path(str(planned_payload["path"]))
        if failed_step_path is not None and step_path == failed_step_path:
            continue
        plans.append(
            {
                "action_id": (
                    f"recursive-mkdir-cleanup:{int(step.get('step_index', 0))}:{planned_payload['step_resource_key']}"
                ),
                "action_type": REMOVE_CREATED_DIRECTORY_ACTION,
                "resource_key": context.batch.resource_key,
                "payload": planned_payload,
            }
        )
        prior_cleanup_count += 1
    expected_prior_count = _expected_prior_created_step_count(payload)
    if expected_prior_count is not None and prior_cleanup_count < expected_prior_count:
        return (
            _incomplete_recursive_mkdir_cleanup_proof_plan(
                context,
                payload=payload,
                expected_prior_count=expected_prior_count,
                planned_count=prior_cleanup_count,
            ),
        )
    return tuple(plans)


def _planned_recursive_mkdir_cleanup_payload(
    context: JournaledFilesystemRecoveryContext,
    *,
    step: Mapping[str, object],
) -> dict[str, object] | None:
    ownership_class = optional_str(step.get("ownership_class"))
    if ownership_class != "created_by_transaction":
        return None
    cleanup_policy = optional_str(step.get("cleanup_policy"))
    backend_guarantee = optional_str(step.get("backend_guarantee"))
    if cleanup_policy != "owned_empty_directory_safe_ish":
        return None
    if backend_guarantee not in ALLOWED_GENERIC_DIRECTORY_REMOVE_GUARANTEES:
        return None
    created_directory_identity = step.get("created_directory_identity")
    journaled_creation_proof = step.get("journaled_creation_proof")
    if not isinstance(created_directory_identity, Mapping) or not isinstance(journaled_creation_proof, Mapping):
        return None
    step_path = Path(str(step.get("step_path")))
    resource_key = str(step.get("step_resource_key"))
    if not _payload_identity_matches_action(created_directory_identity, path=step_path, resource_key=resource_key):
        return None
    if not _journaled_creation_proof_matches_action(
        journaled_creation_proof,
        path=step_path,
        resource_key=resource_key,
        created_directory_identity=created_directory_identity,
        context=context,
    ):
        return None
    return {
        "path": str(step_path),
        "step_resource_key": resource_key,
        "ownership_class": ownership_class,
        "created_directory_identity": jsonable_mapping(created_directory_identity),
        "journaled_creation_proof": jsonable_mapping(journaled_creation_proof),
        "cleanup_policy": cleanup_policy,
        "backend_guarantee": backend_guarantee,
        "cleanup": jsonable_value(step.get("cleanup")) if isinstance(step.get("cleanup"), Mapping) else {},
        "recursive_mkdir_cleanup": {
            "step_index": int(optional_str(step.get("step_index")) or "0"),
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


def _failed_recursive_mkdir_step_path(payload: Mapping[str, object]) -> Path | None:
    recursive_mkdir = payload.get("recursive_mkdir")
    if not isinstance(recursive_mkdir, Mapping):
        return None
    step_path = recursive_mkdir.get("step_path")
    if step_path is None:
        return None
    return Path(str(step_path))


def _expected_prior_created_step_count(payload: Mapping[str, object]) -> int | None:
    recursive_mkdir = payload.get("recursive_mkdir")
    if not isinstance(recursive_mkdir, Mapping):
        return None
    step_index = recursive_mkdir.get("step_index")
    if not isinstance(step_index, int):
        return None
    return max(step_index - 1, 0)


def _incomplete_recursive_mkdir_cleanup_proof_plan(
    context: JournaledFilesystemRecoveryContext,
    *,
    payload: Mapping[str, object],
    expected_prior_count: int,
    planned_count: int,
) -> dict[str, object]:
    recursive_mkdir = payload.get("recursive_mkdir")
    step_path = None
    if isinstance(recursive_mkdir, Mapping) and recursive_mkdir.get("step_path") is not None:
        step_path = str(recursive_mkdir["step_path"])
    return {
        "action_id": f"recursive-mkdir-cleanup:manual-incomplete-proof:{context.batch.batch_id}",
        "action_type": "manual_intervention_required",
        "resource_key": context.batch.resource_key,
        "payload": {
            "operation": "recursive_mkdir_cleanup",
            "reason_code": "incomplete_recursive_mkdir_cleanup_proof",
            "detail": "recursive mkdir cleanup proof is incomplete; automatic rollback refused",
            "batch_id": context.batch.batch_id,
            "step_path": step_path,
            "expected_prior_created_steps": expected_prior_count,
            "planned_cleanup_steps": planned_count,
        },
    }


def _direct_recursive_mkdir_cleanup_plan(
    context: JournaledFilesystemRecoveryContext,
) -> dict[str, object] | None:
    payload = context.batch.payload
    if payload.get("noop") is True:
        return None
    recursive_mkdir = payload.get("recursive_mkdir")
    rollback_diagnostic = payload.get("rollback_diagnostic")
    if not isinstance(rollback_diagnostic, Mapping):
        return None
    if optional_str(payload.get("operation")) != "make_directory":
        return None
    if isinstance(recursive_mkdir, Mapping):
        step_path = Path(str(recursive_mkdir.get("step_path")))
        step_resource_key = optional_str(recursive_mkdir.get("step_resource_key"))
        step_index = int(optional_str(recursive_mkdir.get("step_index")) or "0")
    else:
        path_value = payload.get("path")
        if path_value is None:
            return None
        step_path = Path(str(path_value))
        step_resource_key = optional_str(payload.get("resource_key"))
        step_index = 0
    if step_resource_key is None:
        return None
    before_checkpoint = next(
        (
            checkpoint
            for checkpoint in context.checkpoints
            if checkpoint.checkpoint_type == "before"
            and checkpoint.resource_key == step_resource_key
            and Path(str(checkpoint.payload.get("path"))) == step_path
            and checkpoint.payload.get("exists") is False
        ),
        None,
    )
    if before_checkpoint is None:
        return None
    operation = next(
        (
            operation
            for operation in context.operations
            if operation.operation_type == "make_directory"
            and operation.resource_key == step_resource_key
            and _payload_matches_step(operation.payload, path=step_path, resource_key=step_resource_key)
        ),
        None,
    )
    if operation is None:
        return None
    creation_checkpoint = next(
        (
            checkpoint
            for checkpoint in reversed(context.checkpoints)
            if checkpoint.checkpoint_type in {"after", "failure"}
            and checkpoint.resource_key == step_resource_key
            and Path(str(checkpoint.payload.get("path"))) == step_path
            and checkpoint.payload.get("file_type") == "directory"
            and checkpoint.payload.get("exists") is True
        ),
        None,
    )
    if creation_checkpoint is None:
        return None
    device = creation_checkpoint.payload.get("device")
    inode = creation_checkpoint.payload.get("inode")
    if not isinstance(device, int) or not isinstance(inode, int):
        return None
    planned_payload = {
        "path": str(step_path),
        "step_resource_key": step_resource_key,
        "ownership_class": "created_by_transaction",
        "created_directory_identity": {
            "path": str(step_path),
            "resource_key": step_resource_key,
            "file_type": "directory",
            "device": device,
            "inode": inode,
        },
        "journaled_creation_proof": {
            "batch_id": context.batch.batch_id,
            "operation_id": operation.operation_id,
            "operation_type": "make_directory",
            "checkpoint_id": creation_checkpoint.checkpoint_id,
            "checkpoint_type": creation_checkpoint.checkpoint_type,
            "path": str(step_path),
            "resource_key": step_resource_key,
        },
        "cleanup_policy": "owned_empty_directory_safe_ish",
        "backend_guarantee": "identity_conditional_remove",
        "cleanup": {
            "manual_intervention_possible": True,
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
            "remove_only_if": jsonable_value(rollback_diagnostic.get("remove_only_if")),
        },
        "recursive_mkdir_cleanup": {
            "step_index": step_index,
        },
    }
    return {
        "action_id": f"recursive-mkdir-cleanup:self:{step_resource_key}",
        "action_type": REMOVE_CREATED_DIRECTORY_ACTION,
        "resource_key": context.batch.resource_key,
        "payload": planned_payload,
    }


def _payload_matches_step(
    payload: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
) -> bool:
    payload_path = payload.get("path")
    payload_resource_key = payload.get("resource_key")
    return Path(str(payload_path)) == path and str(payload_resource_key) == resource_key
