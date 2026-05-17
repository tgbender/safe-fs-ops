from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from safe_fs_ops.filesystem_ops import (
    DirectoryIdentity,
    ResourceSnapshot,
    UnsupportedFilesystemMutationError,
    inspect_path,
)
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key
from safe_fs_ops.operation_journal.models import (
    JournaledFilesystemRecoveryContext,
    RecoveryActionRecord,
    require_recovery_action_authority,
)
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)

REMOVE_CREATED_DIRECTORY_ACTION = "remove_created_directory"
ALLOWED_GENERIC_DIRECTORY_REMOVE_GUARANTEES = frozenset({"identity_conditional_remove"})
# Journal ownership proof decides whether rollback is allowed; cleanup still
# requires an identity-conditional remover for the final name removal.


def remove_created_directory_recovery_action(
    remove_directory: Callable[..., None],
    identity_remove_directory: Callable[..., None] | None,
    snapshot: Callable[[Path | str], ResourceSnapshot],
    _context: JournaledFilesystemRecoveryContext | None,
    action: RecoveryActionRecord,
) -> dict[str, object]:
    del snapshot
    payload = mapping_payload(action.payload)
    path = Path(str(payload["path"]))
    expected_resource_key = str(payload.get("step_resource_key") or directory_resource_key(path))
    ownership_class = optional_str(payload.get("ownership_class"))
    if ownership_class != "created_by_transaction":
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because ownership proof is missing: {path}",
            payload={"path": str(path), "reason_code": "missing_ownership_proof"},
        )
    created_directory_identity_value = payload.get("created_directory_identity")
    if not isinstance(created_directory_identity_value, Mapping):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because ownership proof is invalid: {path}",
            payload={"path": str(path), "reason_code": "invalid_ownership_proof"},
        )
    journaled_creation_proof_value = payload.get("journaled_creation_proof")
    if not isinstance(journaled_creation_proof_value, Mapping):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because journaled ownership proof is missing: {path}",
            payload={"path": str(path), "reason_code": "missing_journaled_creation_proof"},
        )
    cleanup_policy = optional_str(payload.get("cleanup_policy"))
    if cleanup_policy != "owned_empty_directory_safe_ish":
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because cleanup policy is missing or unsupported: {path}",
            payload={"path": str(path), "reason_code": "missing_cleanup_policy"},
        )
    backend_guarantee = optional_str(payload.get("backend_guarantee"))
    if backend_guarantee not in ALLOWED_GENERIC_DIRECTORY_REMOVE_GUARANTEES:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because backend guarantee is missing or unsupported: {path}",
            payload={"path": str(path), "reason_code": "missing_backend_guarantee"},
        )
    created_directory_identity = jsonable_mapping(created_directory_identity_value)
    journaled_creation_proof = jsonable_mapping(journaled_creation_proof_value)
    if expected_resource_key != directory_resource_key(path):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because resource key did not match path: {path}",
            payload={"path": str(path), "expected_resource_key": expected_resource_key},
        )
    if not _payload_identity_matches_action(
        created_directory_identity,
        path=path,
        resource_key=expected_resource_key,
    ):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because ownership proof did not match action path: {path}",
            payload={"path": str(path), "reason_code": "ownership_proof_path_mismatch"},
        )
    if not _journaled_creation_proof_matches_action(
        journaled_creation_proof,
        path=path,
        resource_key=expected_resource_key,
        created_directory_identity=created_directory_identity,
        context=_context,
    ):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because journaled ownership proof did not match action path: {path}",
            payload={"path": str(path), "reason_code": "journaled_creation_proof_mismatch"},
        )
    safety = inspect_path(path)
    if not safety.exists:
        raise RecoveryActionSkipped(
            f"directory cleanup skipped because path is already absent: {path}",
            payload={"path": str(path)},
        )
    if not safety.is_dir or safety.is_symlink or safety.is_windows_reparse_point or safety.is_mount:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because path is unsafe: {path}",
            payload={"path": str(path), "file_type": safety.file_type},
        )
    expected_identity = _directory_identity_from_payload(created_directory_identity)
    if expected_identity is None:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because ownership proof is invalid: {path}",
            payload={"path": str(path), "reason_code": "invalid_ownership_proof"},
        )
    try:
        current_identity = DirectoryIdentity.from_stat(path.stat())
    except FileNotFoundError:
        raise RecoveryActionSkipped(
            f"directory cleanup skipped because path is already absent: {path}",
            payload={"path": str(path)},
        ) from None
    if current_identity != expected_identity:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because directory identity changed: {path}",
            payload={
                "path": str(path),
                "reason_code": "created_directory_identity_mismatch",
                "expected_device": expected_identity.device,
                "expected_inode": expected_identity.inode,
                "current_device": current_identity.device,
                "current_inode": current_identity.inode,
            },
        )
    if any(path.iterdir()):
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because directory is not empty: {path}",
            payload={"path": str(path), "reason_code": "directory_not_empty"},
        )
    if identity_remove_directory is None:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because no identity-conditional directory remover is available: {path}",
            payload={
                "path": str(path),
                "reason_code": "identity_safe_remove_directory_unavailable",
            },
        )
    try:
        if _context is None:
            raise RecoveryActionManualInterventionRequired(
                "recovery mutation requires an active recovery authority",
                payload={"reason_code": "missing_recovery_authority"},
            )
        require_recovery_action_authority(_context, action)
        identity_remove_directory(path, expected_identity=expected_identity)
    except UnsupportedFilesystemMutationError as exc:
        raise RecoveryActionManualInterventionRequired(
            f"directory cleanup refused because identity-conditional directory removal is unsupported: {path}",
            payload={
                "path": str(path),
                "reason_code": "identity_safe_remove_directory_unsupported",
                "detail": str(exc),
            },
        ) from exc
    return {
        "path": str(path),
        "step_resource_key": expected_resource_key,
        "removed": True,
        "cleanup_policy": cleanup_policy,
        "backend_guarantee": backend_guarantee,
        "effective_backend_guarantee": backend_guarantee,
    }


def remove_empty_directory_recovery_action(
    remove_directory: Callable[..., None],
    identity_remove_directory: Callable[..., None] | None,
    snapshot: Callable[[Path | str], ResourceSnapshot],
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> dict[str, object]:
    return remove_created_directory_recovery_action(
        remove_directory,
        identity_remove_directory,
        snapshot,
        context,
        action,
    )


def mapping_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("recovery payload must be a mapping")
    return value


def optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def jsonable_mapping(value: object) -> Mapping[str, object]:
    mapped = jsonable_value(value)
    if not isinstance(mapped, Mapping):
        raise ValueError("recovery payload must be a mapping")
    return mapped


def jsonable_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [jsonable_value(item) for item in value]
    return value


def _directory_identity_from_payload(value: Mapping[str, object]) -> DirectoryIdentity | None:
    path = value.get("path")
    file_type = value.get("file_type")
    resource_key = value.get("resource_key")
    device = value.get("device")
    inode = value.get("inode")
    if path is None or file_type != "directory" or resource_key is None:
        return None
    if not isinstance(device, int) or not isinstance(inode, int):
        return None
    return DirectoryIdentity(device=device, inode=inode)


def _payload_identity_matches_action(
    value: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
) -> bool:
    payload_path = value.get("path")
    payload_resource_key = value.get("resource_key")
    return Path(str(payload_path)) == path and str(payload_resource_key) == resource_key


def _journaled_creation_proof_matches_action(
    value: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
    created_directory_identity: Mapping[str, object],
    context: JournaledFilesystemRecoveryContext | None,
) -> bool:
    if Path(str(value.get("path"))) != path or str(value.get("resource_key")) != resource_key:
        return False
    if optional_str(value.get("operation_type")) != "make_directory":
        return False
    proof_checkpoint_type = optional_str(value.get("checkpoint_type"))
    if proof_checkpoint_type not in {"after", "failure"}:
        return False
    batch_id = optional_str(value.get("batch_id"))
    operation_id = optional_str(value.get("operation_id"))
    checkpoint_id = optional_str(value.get("checkpoint_id"))
    if batch_id is None or operation_id is None or checkpoint_id is None:
        return False
    if context is None:
        return False
    proof_context = _proof_context_for_batch(context, batch_id=batch_id)
    if proof_context is None:
        return False
    if not _proof_batch_matches_recursive_lineage(context, proof_context):
        return False
    operation = next((item for item in proof_context.operations if item.operation_id == operation_id), None)
    checkpoint = next((item for item in proof_context.checkpoints if item.checkpoint_id == checkpoint_id), None)
    if operation is None or checkpoint is None:
        return False
    if operation.batch_id != batch_id or checkpoint.batch_id != batch_id:
        return False
    if operation.operation_type != "make_directory":
        return False
    if operation.resource_key != resource_key:
        return False
    if not _payload_matches_action_path_and_resource_key(operation.payload, path=path, resource_key=resource_key):
        return False
    if checkpoint.operation_id not in {operation.operation_id, None}:
        return False
    if checkpoint.checkpoint_type != proof_checkpoint_type:
        return False
    if checkpoint.resource_key != resource_key:
        return False
    if Path(str(checkpoint.payload.get("path"))) != path:
        return False
    if checkpoint.payload.get("file_type") != "directory":
        return False
    if checkpoint.payload.get("exists") is not True:
        return False
    return _checkpoint_payload_matches_owned_directory_identity(
        checkpoint.payload,
        created_directory_identity=created_directory_identity,
    )


def _proof_context_for_batch(
    context: JournaledFilesystemRecoveryContext,
    *,
    batch_id: str,
) -> JournaledFilesystemRecoveryContext | None:
    if batch_id == context.batch.batch_id:
        return context
    return context.proof_contexts.get(batch_id)


def _proof_batch_matches_recursive_lineage(
    context: JournaledFilesystemRecoveryContext,
    proof_context: JournaledFilesystemRecoveryContext,
) -> bool:
    failed_batch = context.batch
    proof_batch = proof_context.batch
    if proof_batch.owner != failed_batch.owner or proof_batch.run_id != failed_batch.run_id:
        return False
    if not _same_optional_value(failed_batch.operation_run_id, proof_batch.operation_run_id):
        return False
    if not _same_optional_value(failed_batch.operation_phase_id, proof_batch.operation_phase_id):
        return False
    return _same_recursive_batch_intent(failed_batch.payload, proof_batch.payload)


def _same_recursive_batch_intent(
    failed_payload: Mapping[str, object],
    proof_payload: Mapping[str, object],
) -> bool:
    failed_recursive = failed_payload.get("recursive_mkdir")
    proof_recursive = proof_payload.get("recursive_mkdir")
    if not isinstance(failed_recursive, Mapping) or not isinstance(proof_recursive, Mapping):
        return failed_recursive is proof_recursive
    if optional_str(failed_payload.get("operation")) != optional_str(proof_payload.get("operation")):
        return False
    for key, required in (
        ("requested_resource_key", True),
        ("plan_target", True),
        ("step_count", False),
    ):
        failed_value = failed_recursive.get(key)
        proof_value = proof_recursive.get(key)
        if failed_value is None or proof_value is None:
            if required:
                return False
            continue
        if str(failed_value) != str(proof_value):
            return False
    return True


def _same_optional_value(current: str | None, other: str | None) -> bool:
    if current is None and other is None:
        return True
    if current is None or other is None:
        return False
    return current == other


def _payload_matches_action_path_and_resource_key(
    value: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
) -> bool:
    payload_path = value.get("path")
    payload_resource_key = value.get("resource_key")
    if payload_path is not None and Path(str(payload_path)) != path:
        return False
    return not (payload_resource_key is not None and str(payload_resource_key) != resource_key)


def _checkpoint_payload_matches_owned_directory_identity(
    payload: Mapping[str, object],
    *,
    created_directory_identity: Mapping[str, object],
) -> bool:
    checkpoint_device = payload.get("device")
    checkpoint_inode = payload.get("inode")
    expected_device = created_directory_identity.get("device")
    expected_inode = created_directory_identity.get("inode")
    return (
        isinstance(checkpoint_device, int)
        and isinstance(checkpoint_inode, int)
        and checkpoint_device == expected_device
        and checkpoint_inode == expected_inode
    )
