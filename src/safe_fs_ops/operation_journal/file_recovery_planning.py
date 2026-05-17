from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from safe_fs_ops.operation_journal.models import CheckpointRecord, JournaledFilesystemRecoveryContext


def file_recovery_action_plans(
    context: JournaledFilesystemRecoveryContext,
) -> tuple[dict[str, object], ...]:
    proof = _file_recovery_proof(context)
    if proof is None or proof["kind"] != "restore":
        return ()
    proof_payload = _mapping_payload_copy(proof["payload"])
    return (
        {
            "action_id": str(proof["action_id"]),
            "action_type": "restore_backup",
            "resource_key": str(proof["resource_key"]),
            "payload": proof_payload,
        },
    )


def automatic_file_rollback_supported(context: JournaledFilesystemRecoveryContext) -> bool:
    proof = _file_recovery_proof(context)
    return proof is not None and proof["kind"] in {"restore", "noop"}


def file_rollback_manual_intervention_payload(
    context: JournaledFilesystemRecoveryContext,
) -> dict[str, object] | None:
    proof = _file_recovery_proof(context)
    if proof is None or proof["kind"] != "manual":
        return None
    proof_payload = _mapping_payload_copy(proof["payload"])
    return {
        "action_id": str(proof["action_id"]),
        "resource_key": str(proof["resource_key"]),
        "payload": proof_payload,
    }


def batch_has_automatic_file_rollback_proof(
    *,
    payload: Mapping[str, object],
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    resource_key: str | None,
    failure_payload: object | None = None,
) -> bool:
    proof = _file_recovery_proof_from_records(
        payload=payload,
        checkpoints=checkpoints,
        resource_key=resource_key,
        failure_payload=failure_payload,
    )
    return proof is not None and proof["kind"] in {"restore", "noop"}


def batch_has_file_rollback_backup_proof(
    *,
    payload: Mapping[str, object],
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    resource_key: str | None,
    failure_payload: object | None = None,
) -> bool:
    proof = _file_recovery_proof_from_records(
        payload=payload,
        checkpoints=checkpoints,
        resource_key=resource_key,
        failure_payload=failure_payload,
    )
    return proof is not None and (proof["kind"] == "noop" or _proof_payload_has_backup(proof))


def _file_recovery_proof(
    context: JournaledFilesystemRecoveryContext,
) -> dict[str, object] | None:
    return _file_recovery_proof_from_records(
        payload=context.batch.payload,
        checkpoints=context.checkpoints,
        resource_key=context.batch.resource_key,
        failure_payload=_latest_failure_payload(context),
    )


def _file_recovery_proof_from_records(
    *,
    payload: Mapping[str, object],
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    resource_key: str | None,
    failure_payload: object | None = None,
) -> dict[str, object] | None:
    operation = payload.get("operation")
    if operation not in {"write_bytes", "write_text", "delete_file"}:
        return None
    path_value = payload.get("path")
    if path_value is None or resource_key is None:
        return None
    path = Path(str(path_value))
    backup_checkpoint = _latest_checkpoint_from_records(
        checkpoints,
        checkpoint_type="backup",
        resource_key=resource_key,
        path=path,
    )
    action_id = f"restore-backup:self:{resource_key}"
    if backup_checkpoint is None or not _is_backup_payload(backup_checkpoint.payload, path=path):
        noop_proof = _before_mutation_noop_proof(
            checkpoints,
            resource_key=resource_key,
            path=path,
            action_id=action_id,
            failure_payload=failure_payload,
        )
        if noop_proof is not None:
            return noop_proof
        return {
            "kind": "manual",
            "action_id": f"manual:{action_id}",
            "resource_key": resource_key,
            "payload": {
                "path": str(path),
                "reason_code": "missing_backup_proof",
                "failure_stage": _failure_stage(failure_payload),
            },
        }
    after_checkpoint = _latest_checkpoint_from_records(
        checkpoints,
        checkpoint_type="after",
        resource_key=resource_key,
        path=path,
    )
    expected_current_payload: Mapping[str, object] | None = None
    if after_checkpoint is not None and _is_snapshot_payload(after_checkpoint.payload, path=path):
        expected_current_payload = after_checkpoint.payload
    else:
        failure_snapshot = _failure_observed_after_payload(failure_payload)
        expected_after_checkpoint = _latest_checkpoint_from_records(
            checkpoints,
            checkpoint_type="expected_after",
            resource_key=resource_key,
            path=path,
        )
        if (
            failure_snapshot is not None
            and expected_after_checkpoint is not None
            and _failure_snapshot_matches_expected_after(
                failure_snapshot,
                expected_after_checkpoint.payload,
                path=path,
            )
        ):
            expected_current_payload = failure_snapshot
        if expected_current_payload is None and expected_after_checkpoint is not None:
            failure_checkpoint = _latest_checkpoint_from_records(
                checkpoints,
                checkpoint_type="failure",
                resource_key=resource_key,
                path=path,
            )
            if failure_checkpoint is not None and _failure_snapshot_matches_expected_after(
                failure_checkpoint.payload,
                expected_after_checkpoint.payload,
                path=path,
            ):
                expected_current_payload = failure_checkpoint.payload
    if expected_current_payload is None and _failure_stage_is_before_mutation(failure_payload):
        failure_snapshot = _failure_snapshot_payload(failure_payload)
        if failure_snapshot is not None and _is_snapshot_payload(failure_snapshot, path=path):
            expected_current_payload = failure_snapshot
    if expected_current_payload is not None:
        return {
            "kind": "restore",
            "action_id": action_id,
            "resource_key": resource_key,
            "payload": {
                "path": str(path),
                "backup": _mapping_payload_copy(backup_checkpoint.payload),
                "expected_current": _mapping_payload_copy(expected_current_payload),
                "allow_overwrite": False,
            },
        }
    no_side_effect_proof = _no_side_effect_noop_proof(
        checkpoints,
        resource_key=resource_key,
        path=path,
        action_id=action_id,
        backup_checkpoint=backup_checkpoint,
        failure_payload=failure_payload,
    )
    if no_side_effect_proof is not None:
        return no_side_effect_proof
    return {
        "kind": "manual",
        "action_id": f"manual:{action_id}",
        "resource_key": resource_key,
        "payload": {
            "path": str(path),
            "backup": _mapping_payload_copy(backup_checkpoint.payload),
            "reason_code": "missing_expected_current_proof",
            "failure_stage": _failure_stage(failure_payload),
        },
    }


def _no_side_effect_noop_proof(
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    *,
    resource_key: str,
    path: Path,
    action_id: str,
    backup_checkpoint: CheckpointRecord,
    failure_payload: object | None,
) -> dict[str, object] | None:
    failure_snapshot = _failure_snapshot_payload(failure_payload)
    if failure_snapshot is None:
        return None
    expected_after_checkpoint = _latest_checkpoint_from_records(
        checkpoints,
        checkpoint_type="expected_after",
        resource_key=resource_key,
        path=path,
    )
    if expected_after_checkpoint is None:
        return None
    before_checkpoint = _latest_checkpoint_from_records(
        checkpoints,
        checkpoint_type="before",
        resource_key=resource_key,
        path=path,
    )
    before_payload: Mapping[str, object] | None = before_checkpoint.payload if before_checkpoint is not None else None
    backup_snapshot = backup_checkpoint.payload.get("snapshot")
    matches_before = before_payload is not None and _same_snapshot_payload(before_payload, failure_snapshot, path=path)
    matches_backup = isinstance(backup_snapshot, Mapping) and _same_snapshot_payload(
        backup_snapshot,
        failure_snapshot,
        path=path,
    )
    if not matches_before and not matches_backup:
        return None
    return {
        "kind": "noop",
        "action_id": f"noop:{action_id}",
        "resource_key": resource_key,
        "payload": {
            "path": str(path),
            "reason_code": "mutation_not_started",
            "failure_stage": _failure_stage(failure_payload),
        },
    }


def _latest_checkpoint(
    context: JournaledFilesystemRecoveryContext,
    *,
    checkpoint_type: str,
    resource_key: str,
    path: Path,
) -> CheckpointRecord | None:
    return _latest_checkpoint_from_records(
        context.checkpoints,
        checkpoint_type=checkpoint_type,
        resource_key=resource_key,
        path=path,
    )


def _latest_checkpoint_from_records(
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    *,
    checkpoint_type: str,
    resource_key: str,
    path: Path,
) -> CheckpointRecord | None:
    for checkpoint in reversed(checkpoints):
        if checkpoint.checkpoint_type != checkpoint_type:
            continue
        if checkpoint.resource_key != resource_key:
            continue
        if Path(str(checkpoint.payload.get("path"))) != path:
            continue
        return checkpoint
    return None


def _is_backup_payload(payload: Mapping[str, object], *, path: Path) -> bool:
    snapshot = payload.get("snapshot")
    file_type = payload.get("file_type")
    existed = payload.get("existed")
    if Path(str(payload.get("path"))) != path:
        return False
    if file_type not in {"missing", "file"}:
        return False
    if not isinstance(existed, bool):
        return False
    if not isinstance(snapshot, Mapping):
        return False
    if Path(str(snapshot.get("path"))) != path:
        return False
    return snapshot.get("file_type") == file_type and snapshot.get("exists") == existed


def _is_snapshot_payload(payload: Mapping[str, object], *, path: Path) -> bool:
    if Path(str(payload.get("path"))) != path:
        return False
    exists = payload.get("exists")
    file_type = payload.get("file_type")
    return isinstance(exists, bool) and file_type in {"missing", "file", "directory", "symlink", "other"}


def _failure_stage_is_before_mutation(failure_payload: object | None) -> bool:
    if not isinstance(failure_payload, Mapping):
        return False
    return failure_payload.get("stage") == "prepare_checkpoints"


def _before_mutation_noop_proof(
    checkpoints: tuple[CheckpointRecord, ...] | list[CheckpointRecord],
    *,
    resource_key: str,
    path: Path,
    action_id: str,
    failure_payload: object | None,
) -> dict[str, object] | None:
    if not _failure_stage_is_before_mutation(failure_payload):
        return None
    before_checkpoint = _latest_checkpoint_from_records(
        checkpoints,
        checkpoint_type="before",
        resource_key=resource_key,
        path=path,
    )
    failure_snapshot = _failure_snapshot_payload(failure_payload)
    if before_checkpoint is None or failure_snapshot is None:
        return None
    if not _same_snapshot_payload(before_checkpoint.payload, failure_snapshot, path=path):
        return None
    return {
        "kind": "noop",
        "action_id": f"noop:{action_id}",
        "resource_key": resource_key,
        "payload": {
            "path": str(path),
            "reason_code": "mutation_not_started",
            "failure_stage": _failure_stage(failure_payload),
        },
    }


def _same_snapshot_payload(left: Mapping[str, object], right: Mapping[str, object], *, path: Path) -> bool:
    if not _is_snapshot_payload(left, path=path) or not _is_snapshot_payload(right, path=path):
        return False
    return all(
        left.get(key) == right.get(key)
        for key in (
            "exists",
            "file_type",
            "content_hash",
            "size",
            "mtime_ns",
            "symlink_target",
            "device",
            "inode",
        )
    )


def _failure_snapshot_matches_expected_after(
    snapshot: Mapping[str, object],
    expected_after: Mapping[str, object],
    *,
    path: Path,
) -> bool:
    if not _is_snapshot_payload(snapshot, path=path):
        return False
    if Path(str(expected_after.get("path"))) != path:
        return False
    expected_exists = expected_after.get("exists")
    expected_file_type = expected_after.get("file_type")
    if expected_exists is False:
        return snapshot.get("exists") is False and snapshot.get("file_type") == "missing"
    if expected_exists is True and expected_file_type == "file":
        expected_hash = expected_after.get("sha256", expected_after.get("content_hash"))
        return (
            snapshot.get("exists") is True
            and snapshot.get("file_type") == "file"
            and snapshot.get("size") == expected_after.get("size")
            and snapshot.get("content_hash") == expected_hash
        )
    return False


def _failure_stage(failure_payload: object | None) -> str | None:
    if not isinstance(failure_payload, Mapping):
        return None
    stage = failure_payload.get("stage")
    if stage is None:
        return None
    return str(stage)


def _failure_snapshot_payload(failure_payload: object | None) -> Mapping[str, object] | None:
    if not isinstance(failure_payload, Mapping):
        return None
    snapshot = failure_payload.get("snapshot")
    if not isinstance(snapshot, Mapping):
        return None
    return snapshot


def _failure_observed_after_payload(failure_payload: object | None) -> Mapping[str, object] | None:
    if not isinstance(failure_payload, Mapping):
        return None
    after = failure_payload.get("after")
    if isinstance(after, Mapping):
        return after
    return _failure_snapshot_payload(failure_payload)


def _latest_failure_payload(context: JournaledFilesystemRecoveryContext) -> object | None:
    failure_payload: object | None = context.batch.status_payload.get("failure")
    if failure_payload is not None:
        return failure_payload
    for record in reversed(context.recovery_records):
        payload = record.payload
        if not isinstance(payload, Mapping):
            continue
        record_failure_payload: object | None = payload.get("failure")
        if record_failure_payload is not None:
            return record_failure_payload
    return None


def _proof_payload_has_backup(proof: Mapping[str, object]) -> bool:
    payload = proof.get("payload")
    return isinstance(payload, Mapping) and isinstance(payload.get("backup"), Mapping)


def _mapping_payload_copy(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("recovery payload must be a mapping")
    return dict(value)
