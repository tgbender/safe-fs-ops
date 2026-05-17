from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from safe_fs_ops.filesystem_ops import FileType, PathSafety, ResourceSnapshot
from safe_fs_ops.operation_journal import JournaledFilesystemResult
from safe_fs_ops.operation_journal.filesystem_support import _utcnow
from safe_fs_ops.recursive_mkdir import (
    RecursiveMkdirPlan,
    RecursiveMkdirStep,
    _require_safe_existing_directory,
    plan_directory_creation,
)
from safe_fs_ops.resources import DirectoryResource

if TYPE_CHECKING:
    from safe_fs_ops.workspace_operation import SafePhase
    from safe_fs_ops.workspace_transaction import SafeTransaction


@dataclass(frozen=True, slots=True)
class RecursiveMkdirContext:
    transaction: SafeTransaction
    operation_run_id: str | None
    operation_phase_id: str | None


def execute_phase_recursive_mkdir(
    phase: SafePhase,
    resource: DirectoryResource,
    *,
    exist_ok: bool,
    idempotency_key: str,
    now: datetime | None,
) -> JournaledFilesystemResult:
    return execute_transaction_recursive_mkdir(
        RecursiveMkdirContext(
            transaction=phase.operation._transaction,
            operation_run_id=phase._operation_run_id(),
            operation_phase_id=phase._phase_id(),
        ),
        resource,
        exist_ok=exist_ok,
        idempotency_key=idempotency_key,
        now=now,
    )


def execute_transaction_recursive_mkdir(
    context: RecursiveMkdirContext,
    resource: DirectoryResource,
    *,
    exist_ok: bool,
    idempotency_key: str,
    now: datetime | None,
) -> JournaledFilesystemResult:
    assert resource.path is not None
    path_inspector = context.transaction.workspace.path_inspector
    snapshot = context.transaction.workspace.snapshot
    plan = plan_directory_creation(resource.path, parents=True, exist_ok=exist_ok, path_inspector=path_inspector)
    if not plan.steps:
        _require_runtime_safe_existing_chain(plan, path_inspector=path_inspector)
        return _record_durable_noop_result(context, resource=resource, idempotency_key=idempotency_key, now=now)
    last_result: JournaledFilesystemResult | None = None
    prior_created_steps: list[dict[str, object]] = []
    step_count = len(plan.steps)
    for step_index, step in enumerate(plan.steps, start=1):
        _require_runtime_safe_descent(plan, step=step, path_inspector=path_inspector)
        step_metadata = _step_metadata(
            plan,
            resource=resource,
            step=step,
            step_index=step_index,
            step_count=step_count,
            prior_created_steps=_refresh_created_steps(prior_created_steps, snapshot=snapshot),
        )
        context.transaction._claim_directory_resource(step.resource_key, now=now)
        last_result = context.transaction._make_directory_path(
            step.path,
            resource_key=step.resource_key,
            idempotency_key=_step_idempotency_key(
                base=idempotency_key,
                step=step,
                step_index=step_index,
                step_count=step_count,
            ),
            operation_run_id=context.operation_run_id,
            operation_phase_id=context.operation_phase_id,
            parents=False,
            exist_ok=False,
            intent_metadata=step_metadata,
            recovery_metadata=step_metadata,
            now=now,
        )
        created_step = _created_step_recovery_metadata(
            step=step,
            step_index=step_index,
            result=last_result,
        )
        if created_step is not None:
            prior_created_steps.append(created_step)
    assert last_result is not None
    return last_result


def _step_idempotency_key(
    *,
    base: str,
    step: RecursiveMkdirStep,
    step_index: int,
    step_count: int,
) -> str:
    resource_digest = hashlib.sha256(step.resource_key.encode("utf-8")).hexdigest()[:16]
    return f"{base}:recursive:{step_index}-of-{step_count}:{resource_digest}"


def _step_metadata(
    plan: RecursiveMkdirPlan,
    *,
    resource: DirectoryResource,
    step: RecursiveMkdirStep,
    step_index: int,
    step_count: int,
    prior_created_steps: list[dict[str, object]],
) -> dict[str, object]:
    rollback_diagnostic = _rollback_diagnostic()
    return {
        "recursive_mkdir": {
            "plan_target": str(plan.target),
            "requested_resource_key": resource.resource_key,
            "step_index": step_index,
            "step_count": step_count,
            "step_path": str(step.path),
            "step_resource_key": step.resource_key,
        },
        "recursive_group": {
            "plan_target": str(plan.target),
            "requested_resource_key": resource.resource_key,
            "prior_created_steps": list(prior_created_steps),
        },
        "rollback_diagnostic": rollback_diagnostic,
    }


def _record_durable_noop_result(
    context: RecursiveMkdirContext,
    *,
    resource: DirectoryResource,
    idempotency_key: str,
    now: datetime | None,
) -> JournaledFilesystemResult:
    assert resource.path is not None
    transaction = context.transaction
    workspace = transaction.workspace
    coordinator = workspace._coordinator
    intent_payload = {
        "operation": "make_directory",
        "path": str(resource.path),
        "resource_key": resource.resource_key,
        "parents": True,
        "exist_ok": True,
        "recursive_mkdir": {
            "plan_target": str(resource.path),
            "requested_resource_key": resource.resource_key,
            "step_index": 0,
            "step_count": 0,
        },
        "noop": True,
    }
    recovery_payload = {
        "desired": {"exists": True, "file_type": "directory"},
        "recursive_mkdir": intent_payload["recursive_mkdir"],
        "noop": True,
        "skipped": True,
    }
    return coordinator.record_directory_noop(
        resource.path,
        resource_key=resource.resource_key,
        lease=transaction.lease,
        owner=workspace.owner,
        run_id=transaction.run_id,
        idempotency_key=idempotency_key,
        operation_run_id=context.operation_run_id,
        operation_phase_id=context.operation_phase_id,
        claim_scope=transaction.claim_scope,
        intent_payload=intent_payload,
        recovery_payload=recovery_payload,
        now=_utcnow(now or transaction.now),
    )


def _rollback_diagnostic() -> dict[str, object]:
    return {
        "automatic_recursive_delete": False,
        "manual_intervention_possible": True,
        "remove_only_if": {
            "created_by_exact_step": True,
            "directory_is_empty": True,
            "path_is_still_safe_directory": True,
            "resource_key_still_matches": True,
        },
    }


def _created_step_recovery_metadata(
    *,
    step: RecursiveMkdirStep,
    step_index: int,
    result: JournaledFilesystemResult,
) -> dict[str, object] | None:
    if result.operation is None or result.after_checkpoint is None:
        return None
    created_directory_identity = _created_directory_identity(
        step=step,
        after_checkpoint_payload=result.after_checkpoint.payload,
    )
    if created_directory_identity is None:
        return None
    return {
        "step_index": step_index,
        "step_path": str(step.path),
        "step_resource_key": step.resource_key,
        "ownership_class": "created_by_transaction",
        "created_directory_identity": created_directory_identity,
        "journaled_creation_proof": {
            "batch_id": result.batch.batch_id,
            "operation_id": result.operation.operation_id,
            "operation_type": result.operation.operation_type,
            "checkpoint_id": result.after_checkpoint.checkpoint_id,
            "checkpoint_type": result.after_checkpoint.checkpoint_type,
            "path": str(step.path),
            "resource_key": step.resource_key,
        },
        "cleanup_policy": "owned_empty_directory_safe_ish",
        "backend_guarantee": "identity_conditional_remove",
        "cleanup": {
            "manual_intervention_possible": True,
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
            "remove_only_if": _rollback_diagnostic()["remove_only_if"],
        },
    }


def _refresh_created_steps(
    created_steps: list[dict[str, object]],
    *,
    snapshot: Callable[[Path | str], ResourceSnapshot],
) -> list[dict[str, object]]:
    del snapshot
    return [dict(step) for step in created_steps]


def _created_directory_identity(
    *,
    step: RecursiveMkdirStep,
    after_checkpoint_payload: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if after_checkpoint_payload is None:
        return None
    snapshot = _resource_snapshot_from_payload(after_checkpoint_payload)
    if snapshot is None or snapshot.path != step.path or snapshot.file_type != "directory" or not snapshot.exists:
        return None
    if snapshot.device is None or snapshot.inode is None:
        return None
    return {
        "path": str(step.path),
        "resource_key": step.resource_key,
        "file_type": "directory",
        "device": snapshot.device,
        "inode": snapshot.inode,
    }


def _resource_snapshot_from_payload(payload: Mapping[str, object]) -> ResourceSnapshot | None:
    try:
        path = payload["path"]
        exists = payload["exists"]
        file_type = payload["file_type"]
    except KeyError:
        return None
    if not isinstance(exists, bool):
        return None
    size = payload.get("size")
    mtime_ns = payload.get("mtime_ns")
    return ResourceSnapshot(
        path=Path(str(path)),
        exists=exists,
        file_type=cast(FileType, str(file_type)),
        content_hash=None if payload.get("content_hash") is None else str(payload.get("content_hash")),
        size=size if isinstance(size, int) else None,
        mtime_ns=mtime_ns if isinstance(mtime_ns, int) else None,
        symlink_target=None if payload.get("symlink_target") is None else str(payload.get("symlink_target")),
        device=device if isinstance((device := payload.get("device")), int) else None,
        inode=inode if isinstance((inode := payload.get("inode")), int) else None,
    )


def _require_runtime_safe_descent(
    plan: RecursiveMkdirPlan,
    *,
    step: RecursiveMkdirStep,
    path_inspector: Callable[[Path], PathSafety],
) -> None:
    current = step.path
    while True:
        parent = current.parent
        if parent == current:
            return
        safety = path_inspector(current)
        if safety.exists:
            role = "step target" if current == step.path else "ancestor"
            _require_safe_existing_directory(
                current,
                role=role,
                operation="recursive mkdir runtime",
                path_inspector=path_inspector,
            )
        if plan.root_boundary is not None and current == plan.root_boundary:
            return
        current = parent


def _require_runtime_safe_existing_chain(
    plan: RecursiveMkdirPlan,
    *,
    path_inspector: Callable[[Path], PathSafety],
) -> None:
    current = plan.target
    while True:
        parent = current.parent
        if parent == current:
            return
        safety = path_inspector(current)
        if not safety.exists:
            raise FileNotFoundError(current)
        role = "target" if current == plan.target else "ancestor"
        _require_safe_existing_directory(
            current,
            role=role,
            operation="recursive mkdir runtime",
            path_inspector=path_inspector,
        )
        if plan.root_boundary is not None and current == plan.root_boundary:
            return
        current = parent
