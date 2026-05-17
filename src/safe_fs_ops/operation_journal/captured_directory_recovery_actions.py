from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    cleanup_captured_directory,
    inspect_path,
    restore_captured_directory,
)
from safe_fs_ops.filesystem_ops.remove_directories import IdentitySafeRemoveDirectoryUnavailableError
from safe_fs_ops.operation_journal.models import (
    CheckpointRecord,
    JournaledFilesystemRecoveryContext,
    RecoveryActionRecord,
    require_recovery_action_authority,
)
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery_actions import (
    jsonable_mapping,
    mapping_payload,
    optional_str,
)

RESTORE_CAPTURED_DIRECTORY_ACTION = "restore_captured_directory"
CLEANUP_CAPTURED_BACKUP_ON_COMMIT_ACTION = "cleanup_captured_backup_on_commit"


@dataclass(frozen=True, slots=True)
class CapturedDirectoryCleanupDebt:
    batch_id: str
    quarantine_path: Path
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class CapturedDirectoryCleanupCandidate:
    artifact_id: str
    batch_id: str
    checkpoint_id: str
    resource_key: str
    quarantine_path: Path
    record: CapturedDirectoryRecord | None
    captured_directory_cleanup: str | None = None
    debt: CapturedDirectoryCleanupDebt | None = None


@dataclass(frozen=True, slots=True)
class CapturedDirectoryCleanupResult:
    artifact_id: str
    batch_id: str
    quarantine_path: Path
    status: str
    reason_code: str | None = None
    detail: str | None = None


class CapturedDirectoryCleanupError(RuntimeError):
    def __init__(self, debts: Iterable[CapturedDirectoryCleanupDebt]) -> None:
        self.debts = tuple(debts)
        super().__init__(_captured_directory_cleanup_error_message(self.debts))


def restore_captured_directory_recovery_action(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> dict[str, object]:
    payload = mapping_payload(action.payload)
    record = _captured_directory_record_from_payload(payload)
    try:
        require_recovery_action_authority(context, action)
        restored_identity = restore_captured_directory(record)
    except FileNotFoundError as exc:
        _skip_if_already_restored(record, payload=payload, cause=exc)
        raise _manual_intervention_required(
            record,
            reason_code="captured_directory_missing",
            detail=str(exc),
        ) from exc
    except FileExistsError as exc:
        _skip_if_already_restored(record, payload=payload, cause=exc)
        raise _manual_intervention_required(
            record,
            reason_code="restore_destination_exists",
            detail=str(exc),
        ) from exc
    except (UnsafePathError, UnsupportedFilesystemMutationError) as exc:
        raise _manual_intervention_required(
            record,
            reason_code="captured_directory_tampered",
            detail=str(exc),
        ) from exc
    return {
        "path": str(record.original_path),
        "step_resource_key": str(payload["step_resource_key"]),
        "ownership_class": record.ownership_class,
        "restored": True,
        "restored_device": restored_identity.device,
        "restored_inode": restored_identity.inode,
    }


def plan_captured_directory_cleanup_candidates(
    checkpoints: Iterable[CheckpointRecord],
) -> tuple[CapturedDirectoryCleanupCandidate, ...]:
    seen_artifact_ids: set[str] = set()
    candidates: list[CapturedDirectoryCleanupCandidate] = []
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_type != "captured_directory":
            continue
        candidate = _captured_directory_cleanup_candidate(checkpoint)
        if candidate.artifact_id in seen_artifact_ids:
            continue
        seen_artifact_ids.add(candidate.artifact_id)
        candidates.append(candidate)
    return tuple(candidates)


def execute_captured_directory_cleanup_candidate(
    candidate: CapturedDirectoryCleanupCandidate,
    *,
    cleanup_operation: Callable[[CapturedDirectoryRecord], None] | None = None,
    identity_cleanup_operation: Callable[..., None] | None = None,
    before_delete: Callable[[], None] | None = None,
) -> CapturedDirectoryCleanupResult:
    if candidate.debt is not None:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="manual_intervention_required",
            reason_code=candidate.debt.reason_code,
            detail=candidate.debt.detail,
        )
    if candidate.record is None:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="manual_intervention_required",
            reason_code="malformed_captured_directory_checkpoint",
            detail="captured-directory cleanup candidate is missing a captured directory record",
        )
    delete_tree: Callable[[CapturedDirectoryRecord], None]
    if cleanup_operation is None:

        def delete_tree(record: CapturedDirectoryRecord) -> None:
            cleanup_captured_directory(
                record,
                _identity_remove_directory=identity_cleanup_operation,
            )
    else:
        delete_tree = cleanup_operation
    if before_delete is not None:
        before_delete()
    try:
        delete_tree(candidate.record)
    except FileNotFoundError:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="skipped",
            reason_code="captured_directory_missing",
            detail=f"captured directory already missing at {candidate.quarantine_path}",
        )
    except IdentitySafeRemoveDirectoryUnavailableError as exc:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="manual_intervention_required",
            reason_code="identity_safe_remove_directory_unavailable",
            detail=str(exc),
        )
    except UnsupportedFilesystemMutationError as exc:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="manual_intervention_required",
            reason_code="captured_directory_cleanup_unsafe",
            detail=str(exc),
        )
    except UnsafePathError as exc:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="manual_intervention_required",
            reason_code="captured_directory_cleanup_unsafe",
            detail=str(exc),
        )
    except OSError as exc:
        return CapturedDirectoryCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            quarantine_path=candidate.quarantine_path,
            status="failed",
            reason_code="captured_directory_cleanup_failed",
            detail=str(exc),
        )
    return CapturedDirectoryCleanupResult(
        artifact_id=candidate.artifact_id,
        batch_id=candidate.batch_id,
        quarantine_path=candidate.quarantine_path,
        status="succeeded",
    )


def planned_captured_directory_restore_payload(
    step: Mapping[str, object],
) -> dict[str, object] | None:
    ownership_class = optional_str(step.get("ownership_class"))
    if ownership_class != "captured_by_transaction":
        return None
    step_path = Path(str(step.get("step_path")))
    resource_key = str(step.get("step_resource_key"))
    captured_directory = step.get("captured_directory")
    if not isinstance(captured_directory, Mapping):
        return None
    quarantine_path = captured_directory.get("quarantine_path")
    original_identity = captured_directory.get("original_identity")
    captured_identity = captured_directory.get("captured_identity")
    if quarantine_path is None:
        return None
    if not isinstance(original_identity, Mapping) or not isinstance(captured_identity, Mapping):
        return None
    if not _identity_payload_matches(original_identity, path=step_path, resource_key=resource_key):
        return None
    if not _identity_payload_matches(captured_identity, path=Path(str(quarantine_path)), resource_key=resource_key):
        return None
    return {
        "path": str(step_path),
        "step_resource_key": resource_key,
        "ownership_class": ownership_class,
        "captured_directory": {
            "original_path": str(step_path),
            "quarantine_path": str(quarantine_path),
            "original_identity": jsonable_mapping(original_identity),
            "captured_identity": jsonable_mapping(captured_identity),
            "ownership_class": ownership_class,
        },
        "recursive_mkdir_cleanup": {
            "step_index": int(optional_str(step.get("step_index")) or "0"),
        },
    }


def _captured_directory_record_from_payload(payload: Mapping[str, object]) -> CapturedDirectoryRecord:
    ownership_class = optional_str(payload.get("ownership_class"))
    if ownership_class != "captured_by_transaction":
        raise RecoveryActionManualInterventionRequired(
            "captured-directory restore refused because ownership proof is missing",
            payload={"reason_code": "missing_ownership_proof"},
        )
    step_resource_key = optional_str(payload.get("step_resource_key"))
    path = Path(str(payload.get("path")))
    captured_directory = payload.get("captured_directory")
    if step_resource_key is None or not isinstance(captured_directory, Mapping):
        raise RecoveryActionManualInterventionRequired(
            f"captured-directory restore refused because payload is incomplete: {path}",
            payload={"path": str(path), "reason_code": "invalid_payload"},
        )
    original_path = Path(str(captured_directory.get("original_path")))
    quarantine_path = Path(str(captured_directory.get("quarantine_path")))
    nested_ownership = optional_str(captured_directory.get("ownership_class"))
    if original_path != path or nested_ownership != ownership_class:
        raise RecoveryActionManualInterventionRequired(
            f"captured-directory restore refused because payload did not match action path: {path}",
            payload={"path": str(path), "reason_code": "payload_path_mismatch"},
        )
    original_identity_payload = captured_directory.get("original_identity")
    captured_identity_payload = captured_directory.get("captured_identity")
    if not isinstance(original_identity_payload, Mapping) or not isinstance(captured_identity_payload, Mapping):
        raise RecoveryActionManualInterventionRequired(
            f"captured-directory restore refused because payload is incomplete: {path}",
            payload={"path": str(path), "reason_code": "invalid_payload"},
        )
    original_identity = _directory_identity_from_payload(
        original_identity_payload,
        path=original_path,
        resource_key=step_resource_key,
    )
    captured_identity = _directory_identity_from_payload(
        captured_identity_payload,
        path=quarantine_path,
        resource_key=step_resource_key,
    )
    return CapturedDirectoryRecord(
        original_path=original_path,
        quarantine_path=quarantine_path,
        original_identity=original_identity,
        captured_identity=captured_identity,
        ownership_class="captured_by_transaction",
    )


def _captured_directory_cleanup_candidate(checkpoint: CheckpointRecord) -> CapturedDirectoryCleanupCandidate:
    try:
        record = _captured_directory_record_from_payload(checkpoint.payload)
    except RecoveryActionManualInterventionRequired as exc:
        quarantine_path = _quarantine_path_from_payload(checkpoint.payload)
        reason_code = str(exc.payload.get("reason_code") or "malformed_captured_directory_checkpoint")
        detail = str(exc)
        return CapturedDirectoryCleanupCandidate(
            artifact_id=f"checkpoint:{checkpoint.checkpoint_id}",
            batch_id=checkpoint.batch_id,
            checkpoint_id=checkpoint.checkpoint_id,
            resource_key=checkpoint.resource_key,
            quarantine_path=quarantine_path,
            record=None,
            captured_directory_cleanup=_captured_directory_cleanup_from_payload(checkpoint.payload),
            debt=CapturedDirectoryCleanupDebt(
                batch_id=checkpoint.batch_id,
                quarantine_path=quarantine_path,
                reason_code=reason_code,
                detail=detail,
            ),
        )
    return CapturedDirectoryCleanupCandidate(
        artifact_id=f"captured-directory:{_normalized_path(record.quarantine_path)}",
        batch_id=checkpoint.batch_id,
        checkpoint_id=checkpoint.checkpoint_id,
        resource_key=checkpoint.resource_key,
        quarantine_path=record.quarantine_path,
        record=record,
        captured_directory_cleanup=_captured_directory_cleanup_from_payload(checkpoint.payload),
    )


def _quarantine_path_from_payload(payload: Mapping[str, object]) -> Path:
    captured_directory = payload.get("captured_directory")
    if isinstance(captured_directory, Mapping) and captured_directory.get("quarantine_path") is not None:
        return Path(str(captured_directory.get("quarantine_path")))
    return Path(str(payload.get("path")))


def _captured_directory_cleanup_from_payload(payload: Mapping[str, object]) -> str | None:
    cleanup_policy = optional_str(payload.get("captured_directory_cleanup"))
    if cleanup_policy in {"automatic", "retain"}:
        return cleanup_policy
    return None


def _directory_identity_from_payload(
    value: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
) -> DirectoryIdentity:
    if not _identity_payload_matches(value, path=path, resource_key=resource_key):
        raise RecoveryActionManualInterventionRequired(
            f"captured-directory restore refused because identity payload did not match expected path: {path}",
            payload={"path": str(path), "reason_code": "identity_payload_mismatch"},
        )
    device = value.get("device")
    inode = value.get("inode")
    if not isinstance(device, int) or not isinstance(inode, int):
        raise RecoveryActionManualInterventionRequired(
            f"captured-directory restore refused because identity payload is invalid: {path}",
            payload={"path": str(path), "reason_code": "invalid_identity_payload"},
        )
    return DirectoryIdentity(device=device, inode=inode)


def _identity_payload_matches(
    value: Mapping[str, object],
    *,
    path: Path,
    resource_key: str,
) -> bool:
    return (
        value.get("file_type") == "directory"
        and Path(str(value.get("path"))) == path
        and str(value.get("resource_key")) == resource_key
    )


def _skip_if_already_restored(
    record: CapturedDirectoryRecord,
    *,
    payload: Mapping[str, object],
    cause: Exception,
) -> None:
    quarantine_safety = inspect_path(record.quarantine_path)
    if quarantine_safety.exists:
        return
    original_safety = inspect_path(record.original_path)
    if (
        not original_safety.exists
        or not original_safety.is_dir
        or original_safety.is_symlink
        or original_safety.is_windows_reparse_point
        or original_safety.is_mount
    ):
        return
    try:
        original_identity = DirectoryIdentity.from_stat(record.original_path.stat())
    except FileNotFoundError:
        return
    if original_identity != record.captured_identity:
        return
    raise RecoveryActionSkipped(
        f"captured-directory restore skipped because directory is already restored: {record.original_path}",
        payload={
            "path": str(record.original_path),
            "step_resource_key": str(payload["step_resource_key"]),
            "restored_device": original_identity.device,
            "restored_inode": original_identity.inode,
            "reason_code": "already_restored",
            "detail": str(cause),
        },
    )


def _manual_intervention_required(
    record: CapturedDirectoryRecord,
    *,
    reason_code: str,
    detail: str,
) -> RecoveryActionManualInterventionRequired:
    return RecoveryActionManualInterventionRequired(
        f"captured-directory restore requires manual intervention: {record.original_path}",
        payload={
            "path": str(record.original_path),
            "quarantine_path": str(record.quarantine_path),
            "reason_code": reason_code,
            "detail": detail,
        },
    )


def _captured_directory_cleanup_error_message(debts: tuple[CapturedDirectoryCleanupDebt, ...]) -> str:
    if len(debts) == 1:
        debt = debts[0]
        return (
            f"captured directory cleanup debt for batch {debt.batch_id!r}: "
            f"{debt.reason_code} at {debt.quarantine_path} ({debt.detail})"
        )
    details = ", ".join(f"{debt.batch_id!r}:{debt.reason_code}:{debt.quarantine_path}" for debt in debts)
    return f"captured directory cleanup debt for {len(debts)} batches: {details}"


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


__all__ = [
    "CLEANUP_CAPTURED_BACKUP_ON_COMMIT_ACTION",
    "CapturedDirectoryCleanupCandidate",
    "CapturedDirectoryCleanupDebt",
    "CapturedDirectoryCleanupError",
    "CapturedDirectoryCleanupResult",
    "RESTORE_CAPTURED_DIRECTORY_ACTION",
    "execute_captured_directory_cleanup_candidate",
    "planned_captured_directory_restore_payload",
    "plan_captured_directory_cleanup_candidates",
    "restore_captured_directory_recovery_action",
]
