from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, inspect_path
from safe_fs_ops.filesystem_ops.pinned_directory import pin_directory
from safe_fs_ops.operation_journal import directory_resource_key
from safe_fs_ops.operation_journal.legacy_captures import (
    ADOPTION,
    APPROVAL,
    LegacyCapture,
    adoption_payload,
    inventory_legacy_captures,
    observed_directory,
)
from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext
from safe_fs_ops.workspace_state.models import LeaseRecord

if TYPE_CHECKING:
    from safe_fs_ops.workspace import SafeWorkspace


def list_legacy_captures(workspace: SafeWorkspace, *, run_id: str | None = None) -> tuple[LegacyCapture, ...]:
    return tuple(
        candidate
        for batch in workspace.journal_store.list_batches(run_id=run_id)
        for candidate in inventory_legacy_captures(workspace.journal_store.read_recovery_context(batch.batch_id))
    )


def recover_legacy_capture(
    workspace: SafeWorkspace,
    candidate: LegacyCapture,
    *,
    confirm_ownership: bool,
    reason: str,
    after_stage: Callable[[str], None] | None = None,
) -> None:
    """Explicitly adopt one legacy capture and resume its batch, without cleanup authorization."""
    from safe_fs_ops.workspace_rollback import recover_pending_workspace_batches

    if confirm_ownership is not True or not reason.strip():
        raise ValueError("legacy capture recovery requires confirm_ownership=True and a non-empty reason")
    initial = workspace.journal_store.read_recovery_context(candidate.batch_id)
    current = _find_candidate(initial, candidate)
    if current.status not in {"needs_confirmation", "adopted", "restored"}:
        raise UnsafePathError(f"legacy capture cannot be adopted: {current.status} at {current.quarantine_path}")

    def adopt(context: JournaledFilesystemRecoveryContext, lease: LeaseRecord) -> JournaledFilesystemRecoveryContext:
        authority = context.recovery_authority
        if authority is None:
            raise RuntimeError("legacy adoption requires an active recovery attempt")
        authority.require_current()
        selected = _find_candidate(context, candidate)
        for key in {candidate.resource_key, directory_resource_key(candidate.quarantine_path)}:
            claim = workspace.claim_store.get(key)
            if claim is not None and (claim.owner, claim.scope) != (
                context.batch.claim_owner,
                context.batch.claim_scope,
            ):
                raise UnsafePathError(f"legacy capture resource has a conflicting claim: {key}")
        if selected.status == "restored":
            return context
        destination = inspect_path(candidate.original_path)
        if destination.exists or destination.is_symlink or destination.is_windows_reparse_point:
            raise FileExistsError(candidate.original_path)
        if selected.observed is None or selected.status not in {"needs_confirmation", "adopted"}:
            raise UnsafePathError(f"legacy capture is unavailable or changed: {candidate.quarantine_path}")
        device, inode = selected.observed[:2]
        with pin_directory(candidate.quarantine_path, device=device, inode=inode):
            observed = observed_directory(candidate.quarantine_path)
            existing = directory_capture_token(candidate.quarantine_path, device=device, inode=inode)
            approval = adoption_payload(context, candidate.capture_id, checkpoint_type=APPROVAL)
            # A matching durable approval and tag make retries safe after a tag write changed ctime.
            retry = approval is not None and existing == approval["capture_token"]
            if observed != candidate.observed and not retry:
                raise UnsafePathError("legacy capture changed since inspection; inspect it again before confirming")
            token = str(approval["capture_token"]) if approval is not None else existing or uuid4().hex
            payload = {
                "version": 1,
                "capture_id": candidate.capture_id,
                "legacy_payload": json.loads(candidate.record_json),
                "capture_token": token,
                "approved_by": workspace.owner,
                "reason": reason,
                "observed": list(observed),
            }

            def checkpoint(kind: str) -> None:
                authority.require_current()
                workspace.journal_store.record_checkpoint(
                    candidate.batch_id,
                    lease=lease,
                    resource_key=context.batch.resource_key or candidate.resource_key,
                    checkpoint_type=kind,
                    payload=payload,
                    recovery_attempt_id=context.recovery_attempt_id,
                )

            if approval is None:
                checkpoint(APPROVAL)
                if after_stage:
                    after_stage("approval")
            authority.require_current()
            actual = directory_capture_token(
                candidate.quarantine_path, device=device, inode=inode, create=True, _token=token
            )
            if actual != token:
                raise UnsafePathError("legacy capture token is unavailable, malformed, or belongs to another adoption")
            if after_stage:
                after_stage("tag")
            authority.require_current()
            if adoption_payload(context, candidate.capture_id) is None:
                checkpoint(ADOPTION)
            if after_stage:
                after_stage("proof")
        return replace(workspace.journal_store.read_recovery_context(candidate.batch_id), recovery_authority=authority)

    error = recover_pending_workspace_batches(
        workspace, run_id=initial.batch.run_id, _batch_id=candidate.batch_id, _before_recovery=adopt
    )
    if error is not None:
        raise error


def _find_candidate(context: JournaledFilesystemRecoveryContext, selected: LegacyCapture) -> LegacyCapture:
    for current in inventory_legacy_captures(context):
        if current.capture_id == selected.capture_id and current.record_json == selected.record_json:
            if (current.original_path, current.quarantine_path, current.resource_key) != (
                selected.original_path,
                selected.quarantine_path,
                selected.resource_key,
            ):
                raise ValueError("legacy capture paths and resource key must match the inspected journal record")
            return current
    raise ValueError("legacy capture no longer matches the journal or belongs to another workspace")
