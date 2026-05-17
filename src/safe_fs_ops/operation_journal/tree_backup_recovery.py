from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

from safe_fs_ops.filesystem_ops import (
    BackupContentMismatchError,
    ContentAddressedStore,
    ContentRef,
    DurabilityMode,
    ResourceSnapshot,
    RestoreConflictError,
    UnsafePathError,
    delete_file,
)
from safe_fs_ops.filesystem_ops._windows_directory_operations import delete_file_windows
from safe_fs_ops.filesystem_ops.mutations import DeleteHooks
from safe_fs_ops.operation_journal.filesystem_support import tree_resource_key
from safe_fs_ops.operation_journal.models import (
    CheckpointRecord,
    JournaledFilesystemRecoveryContext,
    RecoveryActionRecord,
    RecoveryAuthority,
    require_recovery_action_authority,
)
from safe_fs_ops.operation_journal.recovery_runner import RecoveryActionManualInterventionRequired

RESTORE_TREE_BACKUP_ACTION = "restore_tree_backup"
TREE_BACKUP_CHECKPOINT = "tree_backup"
TREE_BACKUP_PARTIAL_ARTIFACTS_CHECKPOINT = "tree_backup_partial_artifacts"


TreeBackupOperation = Callable[..., object]
RestoreTreeBackupOperation = Callable[..., object]


@dataclass(frozen=True, slots=True)
class TreeBackupArtifactCleanupDebt:
    batch_id: str
    content_path: Path
    reason_code: str
    detail: str


@dataclass(frozen=True, slots=True)
class TreeBackupArtifactCleanupCandidate:
    artifact_id: str
    batch_id: str
    checkpoint_id: str
    resource_key: str
    store_path: Path
    content_path: Path
    digest: str
    size: int
    debt: TreeBackupArtifactCleanupDebt | None = None


@dataclass(frozen=True, slots=True)
class TreeBackupArtifactCleanupResult:
    artifact_id: str
    batch_id: str
    content_path: Path
    status: str
    reason_code: str | None = None
    detail: str | None = None


class TreeBackupArtifactCleanupError(RuntimeError):
    def __init__(self, debts: Sequence[TreeBackupArtifactCleanupDebt]) -> None:
        self.debts = tuple(debts)
        if len(self.debts) == 1:
            debt = self.debts[0]
            message = (
                f"tree backup artifact cleanup debt for batch {debt.batch_id!r}: "
                f"{debt.reason_code} at {debt.content_path} ({debt.detail})"
            )
        else:
            details = ", ".join(f"{debt.batch_id!r}:{debt.reason_code}:{debt.content_path}" for debt in self.debts)
            message = f"tree backup artifact cleanup debt for {len(self.debts)} batches: {details}"
        super().__init__(message)


def default_tree_backup_store_path(state_path: Path | str) -> Path:
    journal_path = Path(state_path)
    return journal_path.parent / f"{journal_path.name}.artifacts" / "tree-backups" / "cas"


def tree_backup_checkpoint_payload(backup: object, *, store_path: Path | str | None = None) -> dict[str, object]:
    payload = _jsonable_tree_backup_mapping(backup)
    payload.setdefault("format", "safe_fs_ops.tree_backup.v1")
    if store_path is not None:
        payload["store_path"] = str(store_path)
    elif "store_path" not in payload:
        backup_store_path = _store_path_from_backup_object(backup)
        if backup_store_path is not None:
            payload["store_path"] = str(backup_store_path)
    return payload


def tree_backup_checkpoint_fingerprint(backup: object, *, store_path: Path | str | None = None) -> str:
    payload = tree_backup_checkpoint_payload(backup, store_path=store_path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_tree_backup_payload_resource_key(payload: Mapping[str, object], resource_key: str) -> Path:
    root = _tree_backup_root_from_payload(payload)
    if root is None:
        raise ValueError("tree backup payload is missing a root path")
    expected = tree_resource_key(root)
    if expected != resource_key:
        raise ValueError(f"tree backup root {root} does not match resource_key {resource_key!r}")
    return root


def tree_backup_from_payload(payload: Mapping[str, object], *, require_known_backup: bool = False) -> object:
    backup_payload = payload.get("backup")
    source_payload = backup_payload if isinstance(backup_payload, Mapping) else payload
    reconstructed = _known_tree_backup_from_payload(source_payload)
    if reconstructed is not None:
        return reconstructed
    if require_known_backup and _looks_like_known_tree_backup_payload(source_payload):
        raise ValueError("malformed tree backup payload")
    backup_type = _tree_backup_type()
    converted = _tree_backup_value_from_payload(source_payload)
    if backup_type is None or not isinstance(converted, Mapping):
        return converted
    try:
        return backup_type(**dict(converted))
    except TypeError:
        return converted


def tree_backup_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    if isinstance(context.batch.payload, Mapping) and context.batch.payload.get("operation") == "backup_tree":
        return ()
    resource_key = context.batch.resource_key
    if resource_key is None:
        return ()
    checkpoint = _latest_tree_backup_checkpoint(context)
    if checkpoint is None:
        return _missing_tree_restore_proof_action_plans(context, resource_key)
    payload = dict(checkpoint.payload)
    try:
        destination_root = require_tree_backup_payload_resource_key(payload, resource_key)
    except ValueError as exc:
        return (
            _manual_intervention_action_plan(
                action_id=f"tree-backup-root-mismatch:{context.batch.batch_id}:{resource_key}",
                resource_key=resource_key,
                reason_code="tree_backup_resource_mismatch",
                detail=str(exc),
                payload={
                    "operation": "restore_tree_backup",
                    "batch_id": context.batch.batch_id,
                    "resource_key": resource_key,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "backup": payload,
                },
            ),
        )
    store_path = _store_path_from_payload(payload)
    action_payload: dict[str, object] = {
        "backup": payload,
        "allow_overwrite": False,
        "destination_root": str(destination_root),
    }
    if store_path is not None:
        action_payload["store_path"] = str(store_path)
    return (
        {
            "action_id": f"restore-tree-backup:self:{resource_key}",
            "action_type": RESTORE_TREE_BACKUP_ACTION,
            "resource_key": resource_key,
            "payload": action_payload,
        },
    )


def _missing_tree_restore_proof_action_plans(
    context: JournaledFilesystemRecoveryContext,
    resource_key: str,
) -> tuple[dict[str, object], ...]:
    payload = context.batch.payload
    if not isinstance(payload, Mapping) or payload.get("operation") != "restore_tree_backup":
        return ()
    return (
        _manual_intervention_action_plan(
            action_id=f"restore-tree-backup-manual:{context.batch.batch_id}:{resource_key}",
            resource_key=resource_key,
            reason_code="missing_tree_restore_rollback_proof",
            detail="restore_tree_backup has no durable per-entry rollback proof",
            payload={
                "operation": "restore_tree_backup",
                "batch_id": context.batch.batch_id,
                "resource_key": resource_key,
                "destination_root": str(payload.get("destination_root")),
                "store_path": str(payload.get("store_path")),
            },
        ),
    )


def _manual_intervention_action_plan(
    *,
    action_id: str,
    resource_key: str,
    reason_code: str,
    detail: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_type": "manual_intervention_required",
        "resource_key": resource_key,
        "payload": {
            "reason_code": reason_code,
            "detail": detail,
            **dict(payload),
        },
    }


def restore_tree_backup_recovery_action(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
    *,
    restore_tree_backup_operation: RestoreTreeBackupOperation | None = None,
) -> object:
    payload = action.payload
    try:
        backup = tree_backup_from_payload(payload, require_known_backup=restore_tree_backup_operation is None)
        store_path = _store_path_from_payload(payload)
        if store_path is None:
            backup_payload = payload.get("backup")
            if isinstance(backup_payload, Mapping):
                store_path = _default_store_path_from_tree_backup_payload(backup_payload)
        if store_path is None:
            raise RecoveryActionManualInterventionRequired(
                "tree backup restore requires manual intervention: missing CAS store path",
                payload={
                    "reason_code": "missing_tree_backup_store_path",
                    "detail": "restore_tree_backup recovery action payload did not include store_path",
                },
            )
        allow_overwrite = _bool_from_payload(payload.get("allow_overwrite"), default=False)
        restore_operation = restore_tree_backup_operation or _default_restore_tree_backup_operation()
        store = ContentAddressedStore(store_path)
        destination_root = _effective_restore_destination(payload)
        if action.resource_key is not None:
            if destination_root is None:
                raise RecoveryActionManualInterventionRequired(
                    "tree backup restore requires manual intervention: missing restore destination",
                    payload={
                        "reason_code": "missing_tree_backup_destination",
                        "detail": "restore_tree_backup recovery action payload did not include a destination root",
                        "store_path": str(store_path),
                    },
                )
            expected_resource_key = tree_resource_key(destination_root)
            if expected_resource_key != action.resource_key:
                raise RecoveryActionManualInterventionRequired(
                    "tree backup restore requires manual intervention: destination does not match resource",
                    payload={
                        "reason_code": "tree_backup_resource_mismatch",
                        "detail": (
                            f"restore destination {destination_root} maps to {expected_resource_key!r}, "
                            f"not action resource {action.resource_key!r}"
                        ),
                        "store_path": str(store_path),
                        **_tree_backup_target_payload(payload),
                    },
                )
    except RecoveryActionManualInterventionRequired:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryActionManualInterventionRequired(
            "tree backup restore requires manual intervention: malformed recovery payload",
            payload={
                "reason_code": "malformed_tree_backup_recovery_payload",
                "detail": str(exc),
            },
        ) from exc
    try:
        require_recovery_action_authority(context, action)
        return _call_restore_tree_backup(
            restore_operation,
            backup,
            store=store,
            allow_overwrite=allow_overwrite,
            destination_root=destination_root,
            recovery_authority=context.recovery_authority,
            recovery_action=action,
        )
    except (RestoreConflictError, BackupContentMismatchError, UnsafePathError, FileNotFoundError, ValueError) as exc:
        raise RecoveryActionManualInterventionRequired(
            "tree backup restore requires manual intervention",
            payload=_restore_tree_backup_manual_intervention_payload(
                payload,
                store_path=store_path,
                error=exc,
            ),
        ) from exc


def run_backup_tree_operation(
    operation: TreeBackupOperation,
    path: Path,
    *,
    relative_paths: Sequence[Path | str],
    store: ContentAddressedStore,
) -> object:
    return _call_backup_tree(operation, path, relative_paths=relative_paths, store=store)


def run_restore_tree_backup_operation(
    operation: RestoreTreeBackupOperation,
    backup: object,
    *,
    destination_root: Path | None,
    store: ContentAddressedStore,
    conflict_policy: str,
) -> object:
    return _call_restore_tree_backup(
        operation,
        backup,
        store=store,
        allow_overwrite=conflict_policy == "replace",
        destination_root=destination_root,
    )


def tree_backup_partial_artifacts_checkpoint_payload(
    *,
    root: Path,
    relative_paths: Sequence[Path | str],
    store_path: Path,
    content_refs: Sequence[ContentRef],
) -> dict[str, object]:
    return {
        "format": "safe_fs_ops.tree_backup_partial_artifacts.v1",
        "root": str(root),
        "relative_paths": [str(relative_path) for relative_path in relative_paths],
        "store_path": str(store_path),
        "content_refs": [_content_ref_payload(ref) for ref in content_refs],
    }


def plan_tree_backup_artifact_cleanup_candidates(
    checkpoints: Sequence[CheckpointRecord],
) -> tuple[TreeBackupArtifactCleanupCandidate, ...]:
    candidates: list[TreeBackupArtifactCleanupCandidate] = []
    seen: set[str] = set()
    completed_tree_backup_batches = {
        checkpoint.batch_id for checkpoint in checkpoints if checkpoint.checkpoint_type == TREE_BACKUP_CHECKPOINT
    }
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_type != TREE_BACKUP_PARTIAL_ARTIFACTS_CHECKPOINT:
            continue
        if checkpoint.batch_id in completed_tree_backup_batches:
            continue
        store_path = _store_path_from_payload(checkpoint.payload)
        refs = checkpoint.payload.get("content_refs")
        if store_path is None or not isinstance(refs, Sequence) or isinstance(refs, str | bytes | bytearray):
            debt_path = store_path or Path(str(checkpoint.payload.get("store_path", ".")))
            artifact_id = f"checkpoint:{checkpoint.checkpoint_id}"
            if artifact_id in seen:
                continue
            seen.add(artifact_id)
            candidates.append(
                TreeBackupArtifactCleanupCandidate(
                    artifact_id=artifact_id,
                    batch_id=checkpoint.batch_id,
                    checkpoint_id=checkpoint.checkpoint_id,
                    resource_key=checkpoint.resource_key,
                    store_path=debt_path,
                    content_path=debt_path,
                    digest="",
                    size=0,
                    debt=TreeBackupArtifactCleanupDebt(
                        batch_id=checkpoint.batch_id,
                        content_path=debt_path,
                        reason_code="malformed_tree_backup_artifact_checkpoint",
                        detail="tree backup partial artifact checkpoint is missing store_path or content_refs",
                    ),
                )
            )
            continue
        for ref_payload in refs:
            candidate = _tree_backup_artifact_candidate_from_payload(
                checkpoint,
                store_path=store_path,
                ref_payload=ref_payload,
            )
            if candidate.artifact_id in seen:
                continue
            seen.add(candidate.artifact_id)
            candidates.append(candidate)
    return tuple(candidates)


def execute_tree_backup_artifact_cleanup_candidate(
    candidate: TreeBackupArtifactCleanupCandidate,
    *,
    delete_operation: Callable[[Path], None] | None = None,
    before_delete: Callable[[], None] | None = None,
) -> TreeBackupArtifactCleanupResult:
    delete_path = delete_operation
    if candidate.debt is not None:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code=candidate.debt.reason_code,
            detail=candidate.debt.detail,
        )
    expected_path = _content_ref_path(candidate.store_path, candidate.digest)
    if _normalized_path(candidate.content_path) != _normalized_path(expected_path):
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="unsafe_tree_backup_artifact_path",
            detail=f"refusing to remove tree backup artifact {candidate.content_path}; expected {expected_path}",
        )
    symlink_detail = _symlink_chain_violation(candidate.content_path, candidate.store_path)
    if symlink_detail is not None:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="unsafe_tree_backup_artifact_path",
            detail=symlink_detail,
        )
    verification = _verify_tree_backup_artifact_cleanup_candidate(candidate)
    if verification is not None:
        return verification
    if before_delete is not None:
        before_delete()
        verification = _verify_tree_backup_artifact_cleanup_candidate(candidate)
        if verification is not None:
            return verification
    try:
        if delete_path is None:
            _default_delete_verified_operation(
                candidate.content_path,
                verify=lambda: _verify_tree_backup_artifact_cleanup_candidate(candidate),
            )
        else:
            delete_path(candidate.content_path)
    except _TreeBackupArtifactVerificationChanged as exc:
        return exc.result
    except FileNotFoundError:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="skipped",
            reason_code="artifact_missing",
            detail=f"tree backup artifact already missing at {candidate.content_path}",
        )
    except OSError as exc:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="failed",
            reason_code="artifact_cleanup_failed",
            detail=str(exc),
        )
    return TreeBackupArtifactCleanupResult(
        artifact_id=candidate.artifact_id,
        batch_id=candidate.batch_id,
        content_path=candidate.content_path,
        status="succeeded",
    )


def _verify_tree_backup_artifact_cleanup_candidate(
    candidate: TreeBackupArtifactCleanupCandidate,
) -> TreeBackupArtifactCleanupResult | None:
    try:
        digest, size = _hash_regular_file(candidate.content_path)
    except FileNotFoundError:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="skipped",
            reason_code="artifact_missing",
            detail=f"tree backup artifact already missing at {candidate.content_path}",
        )
    except OSError as exc:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="tree_backup_artifact_verification_failed",
            detail=str(exc),
        )
    if digest != candidate.digest or size != candidate.size:
        return TreeBackupArtifactCleanupResult(
            artifact_id=candidate.artifact_id,
            batch_id=candidate.batch_id,
            content_path=candidate.content_path,
            status="manual_intervention_required",
            reason_code="tree_backup_artifact_content_mismatch",
            detail=f"tree backup artifact content changed at {candidate.content_path}",
        )
    return None


def _call_backup_tree(
    operation: TreeBackupOperation,
    path: Path,
    *,
    relative_paths: Sequence[Path | str],
    store: ContentAddressedStore,
) -> object:
    kwargs = _supported_store_kwargs(operation, store=store)
    if relative_paths or _callable_accepts_positional_argument(operation, position=1):
        return operation(path, tuple(relative_paths), **kwargs)
    return operation(path, **kwargs)


def _call_restore_tree_backup(
    operation: RestoreTreeBackupOperation,
    backup: object,
    *,
    store: ContentAddressedStore,
    allow_overwrite: bool,
    destination_root: Path | None,
    recovery_authority: RecoveryAuthority | None = None,
    recovery_action: RecoveryActionRecord | None = None,
) -> object:
    kwargs = _supported_store_kwargs(operation, store=store)
    if recovery_authority is not None:
        effective_authority = recovery_authority
        if recovery_action is not None:
            effective_authority = RecoveryAuthority(
                lambda: recovery_authority.require_action_current(recovery_action),
                recovery_attempt_id=recovery_authority.recovery_attempt_id,
                _require_action_current=recovery_authority._require_action_current,
            )
        effective_authority.require_current()
        if _callable_accepts_keyword(operation, "recovery_authority"):
            kwargs["recovery_authority"] = effective_authority
        elif _callable_accepts_keyword(operation, "check_authority"):
            kwargs["check_authority"] = effective_authority.require_current
    if _callable_accepts_keyword(operation, "durability"):
        kwargs["durability"] = DurabilityMode.FSYNC
    if _callable_accepts_keyword(operation, "allow_overwrite"):
        kwargs["allow_overwrite"] = allow_overwrite
    if _callable_accepts_keyword(operation, "conflict_policy"):
        kwargs["conflict_policy"] = "replace" if allow_overwrite else "no_replace"
    if destination_root is not None:
        if _callable_accepts_keyword(operation, "destination_root"):
            kwargs["destination_root"] = destination_root
            return operation(backup, **kwargs)
        if _callable_accepts_positional_argument(operation, position=1):
            return operation(backup, destination_root, **kwargs)
        raise TypeError("restore_tree_backup operation must accept destination_root")
    return operation(backup, **kwargs)


class _TreeBackupArtifactVerificationChanged(RuntimeError):
    def __init__(self, result: TreeBackupArtifactCleanupResult) -> None:
        super().__init__(result.detail)
        self.result = result


def _supported_store_kwargs(operation: Callable[..., object], *, store: ContentAddressedStore) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for keyword in ("artifact_store", "content_store", "store"):
        if _callable_accepts_keyword(operation, keyword):
            kwargs[keyword] = store
            break
    return kwargs


def _callable_accepts_keyword(operation: Callable[..., object], keyword: str) -> bool:
    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            return True
    return False


def _callable_accepts_positional_argument(operation: Callable[..., object], *, position: int) -> bool:
    try:
        signature = inspect.signature(operation)
    except (TypeError, ValueError):
        return True
    positional_count = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            if positional_count == position:
                return True
            positional_count += 1
    return False


def _restore_tree_backup_manual_intervention_payload(
    payload: Mapping[str, object],
    *,
    store_path: Path,
    error: Exception,
) -> dict[str, object]:
    return {
        "reason_code": _restore_tree_backup_reason_code(error),
        "detail": str(error),
        "store_path": str(store_path),
        **_tree_backup_target_payload(payload),
    }


def _restore_tree_backup_reason_code(error: Exception) -> str:
    if isinstance(error, RestoreConflictError):
        return "restore_conflict"
    if isinstance(error, FileNotFoundError):
        return "tree_backup_content_missing"
    if isinstance(error, BackupContentMismatchError | ValueError):
        return "tree_backup_content_mismatch"
    return "unsafe_path"


def _tree_backup_target_payload(payload: Mapping[str, object]) -> dict[str, object]:
    backup_payload = payload.get("backup")
    source = backup_payload if isinstance(backup_payload, Mapping) else payload
    result: dict[str, object] = {}
    for key in ("path", "root", "root_path", "source_path"):
        value = source.get(key)
        if value is not None:
            result[key] = str(value)
    return result


def _effective_restore_destination(payload: Mapping[str, object]) -> Path | None:
    destination_root = _optional_path_from_payload(payload.get("destination_root"))
    if destination_root is not None:
        return destination_root
    return _tree_backup_root_from_payload(payload)


def _tree_backup_root_from_payload(payload: Mapping[str, object]) -> Path | None:
    backup_payload = payload.get("backup")
    source = backup_payload if isinstance(backup_payload, Mapping) else payload
    root = source.get("root") or source.get("root_path") or source.get("path") or source.get("source_path")
    if root is None:
        return None
    return Path(str(root))


def _known_tree_backup_from_payload(payload: Mapping[str, object]) -> object | None:
    tree_backup_type = _tree_backup_type()
    tree_backup_entry_type = _tree_backup_entry_type()
    if tree_backup_type is None or tree_backup_entry_type is None:
        return None
    root = payload.get("root") or payload.get("root_path")
    entries = payload.get("entries")
    if root is None or not isinstance(entries, Sequence) or isinstance(entries, str | bytes | bytearray):
        return None
    try:
        return cast(Callable[..., object], tree_backup_type)(
            root=Path(str(root)),
            entries=tuple(_known_tree_backup_entry_from_payload(entry, tree_backup_entry_type) for entry in entries),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _looks_like_known_tree_backup_payload(payload: Mapping[str, object]) -> bool:
    return ("root" in payload or "root_path" in payload) and "entries" in payload


def _known_tree_backup_entry_from_payload(payload: object, tree_backup_entry_type: type[object]) -> object:
    if not isinstance(payload, Mapping):
        raise ValueError("tree backup entry payload must be a mapping")
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("tree backup entry payload must include a snapshot mapping")
    return cast(Callable[..., object], tree_backup_entry_type)(
        relative_path=Path(str(payload["relative_path"])),
        kind=str(payload["kind"]),
        snapshot=_resource_snapshot_from_payload(snapshot),
        content_address=_optional_content_ref_from_payload(payload.get("content_address")),
        content_hash=_optional_str_from_payload(payload.get("content_hash")),
        size=_optional_int_from_payload(payload.get("size")),
        permissions=_optional_int_from_payload(payload.get("permissions")),
        symlink_target=_optional_str_from_payload(payload.get("symlink_target")),
    )


def _resource_snapshot_from_payload(payload: Mapping[str, object]) -> ResourceSnapshot:
    return ResourceSnapshot(
        path=Path(str(payload["path"])),
        exists=_bool_from_required_payload(payload.get("exists")),
        file_type=cast(Any, str(payload["file_type"])),
        content_hash=_optional_str_from_payload(payload.get("content_hash")),
        size=_optional_int_from_payload(payload.get("size")),
        mtime_ns=_optional_int_from_payload(payload.get("mtime_ns")),
        symlink_target=_optional_str_from_payload(payload.get("symlink_target")),
        device=_optional_int_from_payload(payload.get("device")),
        inode=_optional_int_from_payload(payload.get("inode")),
        ctime_ns=_optional_int_from_payload(payload.get("ctime_ns")),
    )


def _latest_tree_backup_checkpoint(context: JournaledFilesystemRecoveryContext) -> CheckpointRecord | None:
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type != TREE_BACKUP_CHECKPOINT:
            continue
        if checkpoint.resource_key != context.batch.resource_key:
            continue
        return checkpoint
    return None


def _store_path_from_payload(payload: Mapping[str, object]) -> Path | None:
    store_path = payload.get("store_path")
    if store_path is not None:
        return Path(str(store_path))
    backup_payload = payload.get("backup")
    if isinstance(backup_payload, Mapping):
        nested = backup_payload.get("store_path")
        if nested is not None:
            return Path(str(nested))
    return None


def _default_store_path_from_tree_backup_payload(payload: Mapping[str, object]) -> Path | None:
    root = payload.get("path") or payload.get("root") or payload.get("root_path") or payload.get("source_path")
    if root is None:
        return None
    return Path(str(root)).parent / ".safe-fs-ops-tree-backups" / "cas"


def _jsonable_tree_backup_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return cast(dict[str, object], _jsonable_tree_backup_value(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable_tree_backup_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    payload: dict[str, object] = {}
    for key in (
        "path",
        "root_path",
        "source_path",
        "entries",
        "root",
        "snapshot",
        "created_at",
        "store_path",
        "content_refs",
    ):
        if hasattr(value, key):
            payload[key] = _jsonable_tree_backup_value(getattr(value, key))
    if payload:
        return payload
    raise TypeError(f"cannot serialize tree backup object of type {type(value).__name__}")


def _jsonable_tree_backup_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ContentRef):
        return _content_ref_payload(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_tree_backup_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable_tree_backup_value(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable_tree_backup_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"cannot serialize tree backup value of type {type(value).__name__}")


def _content_ref_payload(ref: ContentRef) -> dict[str, object]:
    return {
        "algo": ref.algo,
        "digest": ref.digest,
        "size": ref.size,
        "path": str(ref.path),
    }


def _tree_backup_artifact_candidate_from_payload(
    checkpoint: CheckpointRecord,
    *,
    store_path: Path,
    ref_payload: object,
) -> TreeBackupArtifactCleanupCandidate:
    debt: TreeBackupArtifactCleanupDebt | None = None
    digest = ""
    size = 0
    content_path = store_path
    if not isinstance(ref_payload, Mapping):
        debt = TreeBackupArtifactCleanupDebt(
            batch_id=checkpoint.batch_id,
            content_path=content_path,
            reason_code="malformed_tree_backup_artifact_checkpoint",
            detail="tree backup content reference is not a mapping",
        )
    else:
        try:
            ref = _content_ref_from_payload(ref_payload)
            digest = ref.digest
            size = ref.size
            content_path = ref.path
        except (KeyError, TypeError, ValueError) as exc:
            debt = TreeBackupArtifactCleanupDebt(
                batch_id=checkpoint.batch_id,
                content_path=content_path,
                reason_code="malformed_tree_backup_artifact_checkpoint",
                detail=str(exc),
            )
    artifact_id = (
        str(_normalized_path(_content_ref_path(store_path, digest)))
        if digest
        else f"checkpoint:{checkpoint.checkpoint_id}:{len(str(ref_payload))}"
    )
    return TreeBackupArtifactCleanupCandidate(
        artifact_id=artifact_id,
        batch_id=checkpoint.batch_id,
        checkpoint_id=checkpoint.checkpoint_id,
        resource_key=checkpoint.resource_key,
        store_path=store_path,
        content_path=content_path,
        digest=digest,
        size=size,
        debt=debt,
    )


def _content_ref_path(store_path: Path, digest: str) -> Path:
    return store_path / "sha256" / digest[:2] / digest


def _hash_regular_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _default_delete_operation(path: Path) -> None:
    if os.name == "nt":
        delete_file_windows(path, durability=DurabilityMode.FSYNC)
        return
    delete_file(path, durability=DurabilityMode.FSYNC)


def _default_delete_verified_operation(
    path: Path,
    *,
    verify: Callable[[], TreeBackupArtifactCleanupResult | None],
) -> None:
    def after_unlink_validation(_path: Path) -> None:
        result = verify()
        if result is not None:
            raise _TreeBackupArtifactVerificationChanged(result)

    hooks = DeleteHooks(after_unlink_validation=after_unlink_validation)
    if os.name == "nt":
        delete_file_windows(path, durability=DurabilityMode.FSYNC, hooks=hooks)
        return
    delete_file(path, durability=DurabilityMode.FSYNC, hooks=hooks)


def _symlink_chain_violation(path: Path, root: Path) -> str | None:
    normalized_path = _normalized_path(path)
    normalized_root = _normalized_path(root)
    try:
        normalized_path.relative_to(normalized_root)
    except ValueError:
        return f"refusing to remove tree backup artifact outside {root}"
    for candidate in _path_chain(normalized_root, normalized_path):
        try:
            if candidate.is_symlink():
                return f"refusing to remove tree backup artifact beneath symlinked path {candidate}"
        except OSError as exc:
            return f"refusing to inspect tree backup artifact path {candidate}: {exc}"
    return None


def _path_chain(root: Path, path: Path) -> tuple[Path, ...]:
    chain: list[Path] = [path]
    chain.extend(path.parents)
    included: list[Path] = []
    for candidate in chain:
        included.append(candidate)
        if candidate == root:
            break
    return tuple(reversed(included))


def _normalized_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _tree_backup_value_from_payload(value: object, *, key: str | None = None) -> object:
    if isinstance(value, Mapping):
        if _looks_like_content_ref(value):
            return _content_ref_from_payload(value)
        return {
            str(item_key): _tree_backup_value_from_payload(item, key=str(item_key)) for item_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_tree_backup_value_from_payload(item) for item in value)
    if key is not None and _looks_like_path_key(key) and value is not None:
        return Path(str(value))
    return value


def _looks_like_content_ref(value: Mapping[str, object]) -> bool:
    return {"digest", "size", "path"}.issubset(value.keys()) and str(value.get("algo", "sha256")) == "sha256"


def _content_ref_from_payload(payload: Mapping[str, object]) -> ContentRef:
    return ContentRef(
        algo="sha256",
        digest=str(payload["digest"]),
        size=_int_from_payload(payload["size"]),
        path=Path(str(payload["path"])),
    )


def _looks_like_path_key(key: str) -> bool:
    return key in {"path", "root_path", "source_path", "store_path"} or key.endswith("_path")


def _store_path_from_backup_object(backup: object) -> Path | None:
    store_path = getattr(backup, "store_path", None)
    if store_path is not None:
        return Path(str(store_path))
    store = getattr(backup, "store", None)
    root = getattr(store, "root", None)
    if root is not None:
        return Path(str(root))
    return None


def _tree_backup_type() -> type[object] | None:
    import safe_fs_ops.filesystem_ops as filesystem_ops

    tree_backup = filesystem_ops.__dict__.get("TreeBackup")
    if tree_backup is None:
        return None
    return cast(type[object], tree_backup)


def _tree_backup_entry_type() -> type[object] | None:
    import safe_fs_ops.filesystem_ops as filesystem_ops

    tree_backup_entry = filesystem_ops.__dict__.get("TreeBackupEntry")
    if tree_backup_entry is None:
        return None
    return cast(type[object], tree_backup_entry)


def _default_restore_tree_backup_operation() -> RestoreTreeBackupOperation:
    import safe_fs_ops.filesystem_ops as filesystem_ops

    restore_tree_backup = filesystem_ops.__dict__.get("restore_tree_backup")
    if restore_tree_backup is None:
        raise RuntimeError("restore_tree_backup primitive is not available")
    return cast(RestoreTreeBackupOperation, restore_tree_backup)


def _bool_from_payload(value: object | None, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"boolean payload value must be a bool, not {type(value).__name__}")
    return value


def _bool_from_required_payload(value: object | None) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"boolean payload value must be a bool, not {type(value).__name__}")
    return value


def _optional_path_from_payload(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value))


def _optional_content_ref_from_payload(value: object | None) -> ContentRef | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("content address payload must be a mapping")
    return _content_ref_from_payload(value)


def _optional_str_from_payload(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int_from_payload(value: object | None) -> int | None:
    if value is None:
        return None
    return _int_from_payload(value)


def _int_from_payload(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("integer payload value must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"integer payload value must be int or str, not {type(value).__name__}")


__all__ = [
    "RESTORE_TREE_BACKUP_ACTION",
    "TREE_BACKUP_CHECKPOINT",
    "TREE_BACKUP_PARTIAL_ARTIFACTS_CHECKPOINT",
    "TreeBackupArtifactCleanupCandidate",
    "TreeBackupArtifactCleanupDebt",
    "TreeBackupArtifactCleanupError",
    "TreeBackupArtifactCleanupResult",
    "default_tree_backup_store_path",
    "execute_tree_backup_artifact_cleanup_candidate",
    "plan_tree_backup_artifact_cleanup_candidates",
    "require_tree_backup_payload_resource_key",
    "restore_tree_backup_recovery_action",
    "run_restore_tree_backup_operation",
    "run_backup_tree_operation",
    "tree_backup_partial_artifacts_checkpoint_payload",
    "tree_backup_checkpoint_fingerprint",
    "tree_backup_checkpoint_payload",
    "tree_backup_from_payload",
    "tree_backup_recovery_action_plans",
]
