from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from safe_fs_ops.operation_journal import BatchPhase
from safe_fs_ops.operation_journal.captured_directory_recovery import captured_directory_recovery_action_plans
from safe_fs_ops.operation_journal.captured_directory_recovery_actions import (
    CapturedDirectoryCleanupCandidate,
    CapturedDirectoryCleanupDebt,
    CapturedDirectoryCleanupError,
    CapturedDirectoryCleanupResult,
    execute_captured_directory_cleanup_candidate,
    plan_captured_directory_cleanup_candidates,
)
from safe_fs_ops.operation_journal.file_backup_artifacts import (
    BackupArtifactCleanupCandidate,
    BackupArtifactCleanupDebt,
    BackupArtifactCleanupError,
    BackupArtifactCleanupResult,
    execute_backup_artifact_cleanup_candidate,
    plan_backup_artifact_cleanup_candidates,
)
from safe_fs_ops.operation_journal.file_recovery_planning import batch_has_file_rollback_backup_proof
from safe_fs_ops.operation_journal.filesystem_support import _snapshot_payload
from safe_fs_ops.operation_journal.journal_payloads import _payload_to_json
from safe_fs_ops.operation_journal.journal_rows import _batch_row, _next_sequence
from safe_fs_ops.operation_journal.journal_schema import (
    _BATCH_TABLE,
    _RECOVERY_TABLE,
    _BatchColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import (
    ArtifactCleanupRecord,
    ArtifactCleanupTrigger,
    CheckpointRecord,
    JournaledFilesystemRecoveryContext,
    OperationBatchRecord,
)
from safe_fs_ops.operation_journal.tree_backup_recovery import (
    TreeBackupArtifactCleanupCandidate,
    TreeBackupArtifactCleanupDebt,
    TreeBackupArtifactCleanupError,
    TreeBackupArtifactCleanupResult,
    execute_tree_backup_artifact_cleanup_candidate,
    plan_tree_backup_artifact_cleanup_candidates,
)
from safe_fs_ops.workspace import SafeWorkspaceError
from safe_fs_ops.workspace_state.lease_heartbeat import maintain_lease
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord

if TYPE_CHECKING:
    from safe_fs_ops.workspace import SafeWorkspace
    from safe_fs_ops.workspace_transaction import SafeTransaction

_ArtifactCleanupCandidate = (
    BackupArtifactCleanupCandidate | CapturedDirectoryCleanupCandidate | TreeBackupArtifactCleanupCandidate
)
_ArtifactCleanupDebt = BackupArtifactCleanupDebt | CapturedDirectoryCleanupDebt | TreeBackupArtifactCleanupDebt
_ArtifactCleanupResult = BackupArtifactCleanupResult | CapturedDirectoryCleanupResult | TreeBackupArtifactCleanupResult


class _CombinedArtifactCleanupError(RuntimeError):
    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        self.errors = errors
        super().__init__("; ".join(str(error) for error in errors))


_ArtifactCleanupError = (
    BackupArtifactCleanupError
    | CapturedDirectoryCleanupError
    | TreeBackupArtifactCleanupError
    | _CombinedArtifactCleanupError
)


@dataclass(frozen=True, slots=True)
class _PreparedArtifactCleanupAttempt:
    artifact_id: str
    should_mark_attempting: bool
    should_execute: bool = True


@dataclass(slots=True)
class _WorkspaceArtifactCleanupContext:
    workspace: SafeWorkspace
    cleanup_clock: Callable[[], datetime] | None = None
    _lease: LeaseRecord | None = None


def _workspace_batch_recovery_runner(
    workspace: SafeWorkspace,
    lease: LeaseRecord,
    before_recovery: Callable[[JournaledFilesystemRecoveryContext, LeaseRecord], JournaledFilesystemRecoveryContext]
    | None = None,
) -> Callable[[JournaledFilesystemRecoveryContext], object | None]:
    def recover(context: JournaledFilesystemRecoveryContext) -> object | None:
        with maintain_lease(workspace.lease_store, lease, enabled=before_recovery is not None):
            if before_recovery is not None:
                context = before_recovery(context, lease)
            return workspace._coordinator.run_recovery_actions(context, lease=lease, now=datetime.now(UTC))

    return recover


def run_automatic_transaction_rollback(
    transaction: SafeTransaction,
    *,
    body_error: BaseException | None,
    now: datetime,
) -> bool:
    if transaction.rollback != "automatic":
        return True

    failures: list[tuple[OperationBatchRecord, BaseException]] = []
    cleanup_debts: list[tuple[OperationBatchRecord, _ArtifactCleanupError]] = []
    for batch in _eligible_recovery_batches(transaction, now=now, body_error=body_error):
        if _is_unsupported_automatic_rollback_batch(transaction, batch):
            failures.append((batch, SafeWorkspaceError(_unsupported_automatic_rollback_message(batch))))
            continue
        try:
            recovery_result = transaction.workspace._coordinator.recover_batch(
                batch.batch_id,
                lease=transaction.lease,
                recover=lambda context: transaction.workspace._coordinator.run_recovery_actions(
                    context,
                    lease=transaction.lease,
                    now=now,
                ),
                now=now,
            )
            _cleanup_batch_artifacts(
                transaction,
                recovery_result.batch,
                trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
            )
        except BaseException as exc:
            if isinstance(
                exc,
                BackupArtifactCleanupError
                | CapturedDirectoryCleanupError
                | TreeBackupArtifactCleanupError
                | _CombinedArtifactCleanupError,
            ):
                cleanup_debts.append((batch, exc))
                continue
            failures.append((batch, exc))

    if not failures:
        if cleanup_debts:
            _handle_artifact_cleanup_debt(transaction, cleanup_debts, body_error=body_error)
        return True

    message = _automatic_rollback_failure_message(failures)
    error = SafeWorkspaceError(message)
    error.__cause__ = failures[0][1]
    for failed_batch, failed_exc in failures[1:]:
        error.add_note(f"rollback batch {failed_batch.batch_id} also failed: {failed_exc}")
    for debt_batch, debt_error in cleanup_debts:
        error.add_note(f"rollback batch {debt_batch.batch_id} cleanup debt: {debt_error}")
    if body_error is not None:
        body_error.add_note(f"SafeWorkspace automatic rollback also failed: {message}")
        return False
    raise error


def run_pending_transaction_rollback(
    transaction: SafeTransaction,
    *,
    now: datetime,
) -> None:
    if transaction.rollback != "automatic":
        return
    failures: list[tuple[OperationBatchRecord, BaseException]] = []
    cleanup_debts: list[tuple[OperationBatchRecord, _ArtifactCleanupError]] = []
    for batch in _pending_recovery_batches(transaction, now=now):
        if _is_unsupported_automatic_rollback_batch(transaction, batch):
            failures.append((batch, SafeWorkspaceError(_unsupported_automatic_rollback_message(batch))))
            continue
        try:
            recovery_result = transaction.workspace._coordinator.recover_batch(
                batch.batch_id,
                lease=transaction.lease,
                recover=lambda context: transaction.workspace._coordinator.run_recovery_actions(
                    context,
                    lease=transaction.lease,
                    now=now,
                ),
                now=now,
            )
            _cleanup_batch_artifacts(
                transaction,
                recovery_result.batch,
                trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
            )
        except BaseException as exc:
            if isinstance(
                exc,
                BackupArtifactCleanupError
                | CapturedDirectoryCleanupError
                | TreeBackupArtifactCleanupError
                | _CombinedArtifactCleanupError,
            ):
                cleanup_debts.append((batch, exc))
                continue
            failures.append((batch, exc))
    if failures:
        error = SafeWorkspaceError(_automatic_rollback_failure_message(failures))
        error.__cause__ = failures[0][1]
        for failed_batch, failed_exc in failures[1:]:
            error.add_note(f"rollback batch {failed_batch.batch_id} also failed: {failed_exc}")
        for debt_batch, debt_error in cleanup_debts:
            error.add_note(f"rollback batch {debt_batch.batch_id} cleanup debt: {debt_error}")
        raise error
    if cleanup_debts:
        _handle_artifact_cleanup_debt(transaction, cleanup_debts, body_error=None)


def cleanup_committed_transaction_artifacts(
    transaction: SafeTransaction,
) -> SafeWorkspaceError | None:
    matching_batches = [
        batch
        for batch in transaction.workspace.journal_store.list_batches(run_id=transaction.run_id)
        if _belongs_to_transaction_instance(transaction, batch) and batch.phase == BatchPhase.SUCCEEDED
    ]
    if not matching_batches:
        return None
    return _cleanup_artifacts_for_batches(
        transaction,
        matching_batches,
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
    )


def cleanup_committed_transaction_backup_artifacts(
    transaction: SafeTransaction,
) -> SafeWorkspaceError | None:
    return cleanup_committed_transaction_artifacts(transaction)


def cleanup_outstanding_workspace_artifacts(
    workspace: SafeWorkspace,
    *,
    run_id: str | None = None,
) -> SafeWorkspaceError | None:
    matching_batches_by_phase: dict[str, list[OperationBatchRecord]] = {
        BatchPhase.SUCCEEDED: [],
        BatchPhase.RECOVERY_SUCCEEDED: [],
    }
    seen_batch_ids: set[str] = set()
    for batch in workspace.journal_store.list_batches(run_id=run_id):
        if batch.phase not in matching_batches_by_phase:
            continue
        if not workspace.journal_store.list_checkpoints(batch.batch_id):
            continue
        seen_batch_ids.add(batch.batch_id)
        matching_batches_by_phase[batch.phase].append(batch)
    for record in workspace.journal_store.list_outstanding_artifact_cleanup_records():
        if record.batch_id in seen_batch_ids:
            continue
        record_batch = workspace.journal_store.get_batch(record.batch_id)
        if record_batch is None:
            continue
        if run_id is not None and record_batch.run_id != run_id:
            continue
        if record_batch.phase not in matching_batches_by_phase:
            continue
        seen_batch_ids.add(record_batch.batch_id)
        matching_batches_by_phase[record_batch.phase].append(record_batch)

    cleanup_context = _WorkspaceArtifactCleanupContext(workspace=workspace)
    commit_error = _cleanup_artifacts_for_batches(
        cleanup_context,  # type: ignore[arg-type]
        matching_batches_by_phase[BatchPhase.SUCCEEDED],
        trigger=ArtifactCleanupTrigger.COMMIT_CLEANUP,
    )
    recovery_error = _cleanup_artifacts_for_batches(
        cleanup_context,  # type: ignore[arg-type]
        matching_batches_by_phase[BatchPhase.RECOVERY_SUCCEEDED],
        trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
    )
    if commit_error is None:
        return recovery_error
    if recovery_error is not None:
        commit_error.add_note(str(recovery_error))
    return commit_error


def recover_pending_workspace_batches(
    workspace: SafeWorkspace,
    *,
    run_id: str | None = None,
    _batch_id: str | None = None,
    _before_recovery: Callable[[JournaledFilesystemRecoveryContext, LeaseRecord], JournaledFilesystemRecoveryContext]
    | None = None,
) -> SafeWorkspaceError | None:
    cleanup_context = _WorkspaceArtifactCleanupContext(workspace=workspace)
    failures: list[tuple[OperationBatchRecord, BaseException]] = []
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=workspace.lease_ttl,
        now=datetime.now(UTC),
    )
    if not lease.acquired:
        return SafeWorkspaceError(f"workspace lease {workspace.lease_name!r} is held by {lease.owner!r}")
    cleanup_context._lease = lease
    try:
        all_batches = workspace.journal_store.list_batches(run_id=run_id)
        all_batch_owner_scopes = _batch_owner_scopes(workspace.journal_store.list_batches())
        batches = [
            batch
            for batch in all_batches
            if _is_pending_workspace_recovery_batch(workspace, batch)
            or (batch.batch_id == _batch_id and batch.phase == BatchPhase.SUCCEEDED)
        ]
        batches.sort(key=lambda batch: (batch.storage_order, batch.created_at, batch.batch_id), reverse=True)
        pending_batch_ids_by_owner_scope = _pending_recovery_batch_ids_by_owner_scope(batches)
        pending_batch_ids_by_claim = _pending_recovery_batch_ids_by_claim(batches)
        recovered_batch_ids: set[str] = {
            batch.batch_id
            for batch in batches
            if _batch_id is not None and batch.phase == BatchPhase.RECOVERY_SUCCEEDED
        }
        for batch in batches:
            if _batch_id is not None and batch.batch_id != _batch_id:
                continue
            try:
                batch_now = datetime.now(UTC)
                refreshed_lease = workspace.lease_store.heartbeat(lease, ttl=workspace.lease_ttl, now=batch_now)
                if refreshed_lease is None:
                    raise SafeWorkspaceError(f"workspace lease {workspace.lease_name!r} expired during recovery replay")
                lease = refreshed_lease
                cleanup_context._lease = lease
                if batch.phase == BatchPhase.RECOVERY_SUCCEEDED:
                    _record_recovered_batch_operation_terminal_diagnostics(
                        workspace,
                        batch,
                        lease=lease,
                        now=batch_now,
                    )
                    recovered_batch_ids.add(batch.batch_id)
                    _release_recovered_batch_claims(
                        workspace,
                        batch,
                        lease=lease,
                        pending_batch_ids_by_owner_scope=pending_batch_ids_by_owner_scope,
                        pending_batch_ids_by_claim=pending_batch_ids_by_claim,
                        recovered_batch_ids=recovered_batch_ids,
                    )
                    cleanup_error = _cleanup_artifacts_for_batches(
                        cleanup_context,  # type: ignore[arg-type]
                        (batch,),
                        trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
                    )
                    if cleanup_error is not None:
                        raise cleanup_error
                    continue
                batch = _prepare_workspace_recovery_batch(workspace, batch, lease=lease, now=batch_now)
                recovery_result = workspace._coordinator.recover_batch(
                    batch.batch_id,
                    lease=lease,
                    recover=_workspace_batch_recovery_runner(workspace, lease, _before_recovery),
                    now=batch_now,
                )
                recovered_batch_ids.add(recovery_result.batch.batch_id)
                _record_recovered_batch_operation_terminal_diagnostics(
                    workspace,
                    recovery_result.batch,
                    lease=lease,
                    now=batch_now,
                )
                _release_recovered_batch_claims(
                    workspace,
                    recovery_result.batch,
                    lease=lease,
                    pending_batch_ids_by_owner_scope=pending_batch_ids_by_owner_scope,
                    pending_batch_ids_by_claim=pending_batch_ids_by_claim,
                    recovered_batch_ids=recovered_batch_ids,
                )
                cleanup_error = _cleanup_artifacts_for_batches(
                    cleanup_context,  # type: ignore[arg-type]
                    (recovery_result.batch,),
                    trigger=ArtifactCleanupTrigger.RECOVERY_CLEANUP,
                )
                if cleanup_error is not None:
                    raise cleanup_error
            except BaseException as exc:
                failures.append((batch, exc))
        _release_orphaned_transaction_claims_without_batches(
            workspace,
            lease=lease,
            run_id=run_id,
            batch_owner_scopes=all_batch_owner_scopes,
        )
    finally:
        workspace.lease_store.release(lease, now=datetime.now(UTC))
        cleanup_context._lease = None
    if not failures:
        return None
    error = SafeWorkspaceError(_automatic_rollback_failure_message(failures))
    error.__cause__ = failures[0][1]
    for failed_batch, failed_exc in failures[1:]:
        error.add_note(f"recovery batch {failed_batch.batch_id} also failed: {failed_exc}")
    return error


def _prepare_workspace_recovery_batch(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> OperationBatchRecord:
    if batch.phase in {BatchPhase.ATTEMPTING, BatchPhase.RECOVERING}:
        payload: dict[str, object] = {
            "batch_id": batch.batch_id,
            "run_id": batch.run_id,
            "previous_phase": batch.phase,
        }
        if isinstance(batch.payload, Mapping) and batch.payload.get("path") is not None:
            path = str(batch.payload.get("path"))
            try:
                payload["failure"] = {
                    "path": path,
                    "resource_key": batch.resource_key,
                    "stage": "workspace_recovery_replay",
                    "snapshot": _snapshot_payload(workspace._coordinator._snapshot(path)),
                }
            except Exception as exc:
                payload["failure"] = {
                    "path": path,
                    "resource_key": batch.resource_key,
                    "stage": "workspace_recovery_replay",
                    "snapshot_error": {"error_type": type(exc).__name__, "error": str(exc)},
                }
        if batch.phase == BatchPhase.ATTEMPTING:
            workspace.journal_store.record_interrupted_recovery_desired(
                batch.batch_id,
                lease=lease,
                resource_key=batch.resource_key,
                reason="workspace recovery replay found abandoned batch",
                payload=payload,
                now=now,
            )
        else:
            workspace.journal_store.record_recovery_desired(
                batch.batch_id,
                lease=lease,
                reason="workspace recovery replay",
                payload=payload,
                now=now,
            )
    elif batch.phase in {BatchPhase.FAILED, BatchPhase.RECOVERY_FAILED}:
        workspace.journal_store.record_recovery_desired(
            batch.batch_id,
            lease=lease,
            reason="workspace recovery replay",
            payload={"batch_id": batch.batch_id, "run_id": batch.run_id, "previous_phase": batch.phase},
            now=now,
        )
    elif batch.phase == BatchPhase.SUCCEEDED:
        _record_abandoned_succeeded_batch_recovery_desired(
            workspace,
            batch,
            lease=lease,
            now=now,
        )
    updated = workspace.journal_store.get_batch(batch.batch_id)
    if updated is None:
        raise SafeWorkspaceError(f"recovery batch {batch.batch_id!r} disappeared")
    return updated


def _is_pending_workspace_recovery_batch(workspace: SafeWorkspace, batch: OperationBatchRecord) -> bool:
    if batch.phase in {
        BatchPhase.ATTEMPTING,
        BatchPhase.FAILED,
        BatchPhase.RECOVERING,
        BatchPhase.RECOVERY_DESIRED,
        BatchPhase.RECOVERY_FAILED,
    }:
        return True
    if _is_recovered_workspace_batch_with_unreleased_claim(workspace, batch):
        return True
    return _is_abandoned_succeeded_workspace_recovery_batch(workspace, batch)


def _is_abandoned_succeeded_workspace_recovery_batch(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
) -> bool:
    if batch.phase != BatchPhase.SUCCEEDED:
        return False
    if batch.resource_key is None or batch.claim_owner is None:
        return False
    claim = workspace.claim_store.get(batch.resource_key)
    if claim is None:
        return False
    return claim.owner == batch.claim_owner and claim.scope == batch.claim_scope


def _batch_owner_scopes(
    batches: Sequence[OperationBatchRecord],
) -> set[tuple[str, str | None]]:
    return {
        (batch.claim_owner, batch.claim_scope)
        for batch in batches
        if batch.claim_owner is not None and batch.claim_scope is not None
    }


def _release_orphaned_transaction_claims_without_batches(
    workspace: SafeWorkspace,
    *,
    lease: LeaseRecord,
    run_id: str | None,
    batch_owner_scopes: set[tuple[str, str | None]],
) -> None:
    for claim in workspace.claim_store.list_claims(owner=workspace.owner):
        if claim.scope is None or not claim.scope.startswith("transaction-instance:"):
            continue
        if (claim.owner, claim.scope) in batch_owner_scopes:
            continue
        claim_details = _claim_details_payload(claim.details)
        if claim_details is None:
            continue
        if run_id is not None and claim_details.get("run_id") != run_id:
            continue
        claim_fencing_token = claim_details.get("lease_fencing_token")
        if not isinstance(claim_fencing_token, int) or claim_fencing_token >= lease.fencing_token:
            continue
        workspace.claim_store.release_if_owner_scope_matches(
            claim.resource_key,
            lease=lease,
            owner=claim.owner,
            scope=claim.scope,
            expected_claim=claim,
        )


def _claim_details_payload(details: str | None) -> Mapping[str, object] | None:
    if details is None:
        return None
    try:
        payload = json.loads(details)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _is_recovered_workspace_batch_with_unreleased_claim(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
) -> bool:
    if batch.phase != BatchPhase.RECOVERY_SUCCEEDED:
        return False
    if batch.claim_owner is None:
        return False
    if batch.resource_key is not None:
        claim = workspace.claim_store.get(batch.resource_key)
        if claim is not None and claim.owner == batch.claim_owner and claim.scope == batch.claim_scope:
            return True
    if batch.claim_scope is None:
        return False
    return any(claim.scope == batch.claim_scope for claim in workspace.claim_store.list_claims(owner=batch.claim_owner))


def _record_abandoned_succeeded_batch_recovery_desired(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> None:
    payload = {
        "batch_id": batch.batch_id,
        "run_id": batch.run_id,
        "previous_phase": batch.phase,
    }
    payload_text = _payload_to_json(payload)
    workspace.journal_store.initialize()
    with workspace.journal_store.sqlite_store.transaction() as connection:
        require_current_lease(connection, lease, now=now)
        row = _batch_row(connection, batch.batch_id)
        if row is None:
            raise SafeWorkspaceError(f"recovery batch {batch.batch_id!r} disappeared")
        current_phase = str(row[_BatchColumn.PHASE])
        if current_phase != BatchPhase.SUCCEEDED:
            return
        if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
            raise SafeWorkspaceError(f"batch {batch.batch_id!r} is bound to a different lease")
        if int(row[_BatchColumn.LEASE_FENCING_TOKEN]) > lease.fencing_token:
            raise SafeWorkspaceError(f"batch {batch.batch_id!r} is bound to a newer lease")
        sequence = _next_sequence(connection, batch.batch_id)
        recovery_id = uuid.uuid4().hex
        connection.execute(
            f"""
            UPDATE {_BATCH_TABLE}
            SET {_BatchColumn.PHASE} = ?,
                {_BatchColumn.LEASE_FENCING_TOKEN} = ?,
                {_BatchColumn.STATUS_MESSAGE} = ?,
                {_BatchColumn.STATUS_PAYLOAD} = ?,
                {_BatchColumn.UPDATED_AT} = ?
            WHERE {_BatchColumn.BATCH_ID} = ?
            """,
            (
                BatchPhase.RECOVERY_DESIRED,
                lease.fencing_token,
                "workspace recovery replay found abandoned succeeded batch",
                payload_text,
                now.isoformat(),
                batch.batch_id,
            ),
        )
        connection.execute(
            f"""
            INSERT INTO {_RECOVERY_TABLE} (
                {_RecoveryColumn.RECOVERY_ID},
                {_RecoveryColumn.BATCH_ID},
                {_RecoveryColumn.SEQUENCE},
                {_RecoveryColumn.PHASE},
                {_RecoveryColumn.REASON},
                {_RecoveryColumn.PAYLOAD},
                {_RecoveryColumn.CREATED_AT}
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recovery_id,
                batch.batch_id,
                sequence,
                BatchPhase.RECOVERY_DESIRED,
                "workspace recovery replay found abandoned succeeded batch",
                payload_text,
                now.isoformat(),
            ),
        )


def _record_recovered_batch_operation_terminal_diagnostics(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> None:
    if batch.operation_phase_id is not None:
        operation_phase = workspace.journal_store.get_operation_phase(batch.operation_phase_id)
        if operation_phase is not None and operation_phase.status in {"active", "succeeded"}:
            workspace.journal_store.record_operation_phase_terminal_diagnostic(
                operation_phase.operation_phase_id,
                lease=lease,
                status="finalization_failed",
                now=now,
            )
    if batch.operation_run_id is not None:
        operation_run = workspace.journal_store.get_operation_run(batch.operation_run_id)
        if operation_run is not None and operation_run.status in {"active", "succeeded"}:
            workspace.journal_store.record_operation_run_terminal_diagnostic(
                operation_run.operation_run_id,
                lease=lease,
                status="finalization_failed",
                now=now,
            )


def _pending_recovery_batch_ids_by_owner_scope(
    batches: Sequence[OperationBatchRecord],
) -> dict[tuple[str, str | None], set[str]]:
    pending: dict[tuple[str, str | None], set[str]] = {}
    for batch in batches:
        if batch.claim_owner is None:
            continue
        pending.setdefault((batch.claim_owner, batch.claim_scope), set()).add(batch.batch_id)
    return pending


def _pending_recovery_batch_ids_by_claim(
    batches: Sequence[OperationBatchRecord],
) -> dict[tuple[str, str | None, str], set[str]]:
    pending: dict[tuple[str, str | None, str], set[str]] = {}
    for batch in batches:
        if batch.claim_owner is None or batch.resource_key is None:
            continue
        pending.setdefault((batch.claim_owner, batch.claim_scope, batch.resource_key), set()).add(batch.batch_id)
    return pending


def _release_recovered_batch_claims(
    workspace: SafeWorkspace,
    batch: OperationBatchRecord,
    *,
    lease: LeaseRecord,
    pending_batch_ids_by_owner_scope: Mapping[tuple[str, str | None], set[str]],
    pending_batch_ids_by_claim: Mapping[tuple[str, str | None, str], set[str]],
    recovered_batch_ids: set[str],
) -> None:
    if batch.claim_owner is None:
        return
    if batch.resource_key is not None:
        pending_claim_group = pending_batch_ids_by_claim.get(
            (batch.claim_owner, batch.claim_scope, batch.resource_key),
            set(),
        )
        if pending_claim_group.issubset(recovered_batch_ids):
            claim = workspace.claim_store.get(batch.resource_key)
            if claim is not None and claim.owner == batch.claim_owner and claim.scope == batch.claim_scope:
                workspace.claim_store.release_if_owner_scope_matches(
                    claim.resource_key,
                    lease=lease,
                    owner=batch.claim_owner,
                    scope=batch.claim_scope,
                    expected_claim=claim,
                )
    if batch.claim_scope is None:
        return
    pending_group = pending_batch_ids_by_owner_scope.get((batch.claim_owner, batch.claim_scope), set())
    if not pending_group.issubset(recovered_batch_ids):
        return
    for claim in workspace.claim_store.list_claims(owner=batch.claim_owner):
        if claim.scope != batch.claim_scope:
            continue
        workspace.claim_store.release_if_owner_scope_matches(
            claim.resource_key,
            lease=lease,
            owner=batch.claim_owner,
            scope=batch.claim_scope,
            expected_claim=claim,
        )


def _eligible_recovery_batches(
    transaction: SafeTransaction,
    *,
    now: datetime,
    body_error: BaseException | None,
) -> tuple[OperationBatchRecord, ...]:
    del body_error
    candidates = transaction.workspace.journal_store.list_batches(run_id=transaction.run_id)
    matching: list[OperationBatchRecord] = []
    for batch in candidates:
        if not _belongs_to_transaction_instance(transaction, batch):
            continue
        if _is_read_only_automatic_rollback_batch(batch):
            continue
        if batch.phase in {BatchPhase.RECOVERY_DESIRED, BatchPhase.RECOVERY_FAILED}:
            if batch.phase == BatchPhase.RECOVERY_FAILED:
                batch = _mark_batch_recovery_desired(transaction, batch, now=now)
            matching.append(batch)
            continue
        if batch.phase == BatchPhase.ATTEMPTING:
            matching.append(_mark_batch_interrupted_recovery_desired(transaction, batch, now=now))
            continue
        if batch.phase == BatchPhase.FAILED:
            matching.append(_mark_batch_recovery_desired(transaction, batch, now=now))
            continue
        if batch.phase == BatchPhase.SUCCEEDED:
            matching.append(_mark_batch_recovery_desired(transaction, batch, now=now))
    # Recover newest-first so the most recent transaction side effects are handled
    # before older siblings with the same run/scope.
    matching.sort(key=lambda batch: (batch.storage_order, batch.created_at, batch.batch_id), reverse=True)
    return tuple(matching)


def _pending_recovery_batches(
    transaction: SafeTransaction,
    *,
    now: datetime,
) -> tuple[OperationBatchRecord, ...]:
    candidates = transaction.workspace.journal_store.list_batches(run_id=transaction.run_id)
    matching: list[OperationBatchRecord] = []
    for batch in candidates:
        if not _belongs_to_transaction_instance(transaction, batch):
            continue
        if _is_read_only_automatic_rollback_batch(batch):
            continue
        if batch.phase == BatchPhase.RECOVERY_DESIRED:
            matching.append(batch)
            continue
        if batch.phase in {BatchPhase.FAILED, BatchPhase.RECOVERY_FAILED}:
            matching.append(_mark_batch_recovery_desired(transaction, batch, now=now))
    matching.sort(key=lambda batch: (batch.storage_order, batch.created_at, batch.batch_id), reverse=True)
    return tuple(matching)


def _belongs_to_transaction_instance(transaction: SafeTransaction, batch: OperationBatchRecord) -> bool:
    return (
        batch.run_id == transaction.run_id
        and batch.owner == transaction.workspace.owner
        and batch.claim_owner == transaction.workspace.owner
        and batch.claim_scope == transaction.claim_scope
    )


def _supports_automatic_transaction_rollback(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
) -> bool:
    payload = batch.payload
    if not isinstance(payload, Mapping):
        return False
    operation = payload.get("operation")
    if operation == "make_directory":
        if payload.get("noop") is True:
            return True
        checkpoints = transaction.workspace.journal_store.list_checkpoints(batch.batch_id)
        if isinstance(payload.get("rollback_diagnostic"), Mapping):
            if isinstance(payload.get("recursive_mkdir"), Mapping):
                return _has_recursive_mkdir_rollback_proof(
                    transaction,
                    batch,
                    checkpoints,
                )
            return _has_direct_mkdir_success_rollback_proof(
                batch,
                checkpoints,
            ) or _has_direct_failed_mkdir_noop_proof(batch, checkpoints)
        return _has_direct_failed_mkdir_noop_proof(
            batch,
            checkpoints,
        )
    checkpoints = transaction.workspace.journal_store.list_checkpoints(batch.batch_id)
    if operation == "rename_no_replace":
        return _has_rename_no_replace_rollback_proof(transaction, batch, checkpoints)
    if operation == "capture_directory":
        return _has_capture_directory_rollback_proof(transaction, batch, checkpoints)
    return batch_has_file_rollback_backup_proof(
        payload=payload,
        checkpoints=checkpoints,
        resource_key=batch.resource_key,
        failure_payload=_latest_failure_payload(transaction, batch),
    )


def _has_recursive_mkdir_rollback_proof(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    if _has_direct_mkdir_success_rollback_proof(batch, checkpoints):
        return True
    if _has_direct_mkdir_failure_created_rollback_proof(batch, checkpoints):
        return True
    if not _has_direct_failed_mkdir_noop_proof(batch, checkpoints):
        return False
    payload = _latest_recursive_mkdir_recovery_payload(transaction, batch)
    if payload is None:
        return _expected_prior_created_step_count(batch.payload) == 0
    expected_prior_count = _expected_prior_created_step_count(payload)
    if expected_prior_count is None:
        return False
    return _rollback_capable_recursive_mkdir_step_count(payload) >= expected_prior_count


def _latest_recursive_mkdir_recovery_payload(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
) -> Mapping[str, object] | None:
    for record in reversed(transaction.workspace.journal_store.list_recovery_records(batch.batch_id)):
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


def _expected_prior_created_step_count(payload: Mapping[str, object]) -> int | None:
    recursive_mkdir = payload.get("recursive_mkdir")
    if not isinstance(recursive_mkdir, Mapping):
        return None
    step_index = recursive_mkdir.get("step_index")
    if not isinstance(step_index, int):
        return None
    return max(step_index - 1, 0)


def _rollback_capable_recursive_mkdir_step_count(payload: Mapping[str, object]) -> int:
    recursive_group = payload.get("recursive_group")
    if not isinstance(recursive_group, Mapping):
        return 0
    prior_created_steps = recursive_group.get("prior_created_steps")
    if not isinstance(prior_created_steps, tuple | list):
        return 0
    return sum(
        1 for step in prior_created_steps if isinstance(step, Mapping) and _is_rollback_capable_recursive_step(step)
    )


def _is_rollback_capable_recursive_step(step: Mapping[str, object]) -> bool:
    if step.get("ownership_class") != "created_by_transaction":
        return False
    if step.get("cleanup_policy") != "owned_empty_directory_safe_ish":
        return False
    if step.get("backend_guarantee") != "identity_conditional_remove":
        return False
    created_directory_identity = step.get("created_directory_identity")
    journaled_creation_proof = step.get("journaled_creation_proof")
    if not isinstance(created_directory_identity, Mapping) or not isinstance(journaled_creation_proof, Mapping):
        return False
    return isinstance(created_directory_identity.get("device"), int) and isinstance(
        created_directory_identity.get("inode"),
        int,
    )


def _has_direct_mkdir_success_rollback_proof(
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    before_missing = False
    after_created = False
    for checkpoint in checkpoints:
        if checkpoint.resource_key != batch.resource_key:
            continue
        if checkpoint.checkpoint_type == "before" and checkpoint.payload.get("exists") is False:
            before_missing = True
        if (
            checkpoint.checkpoint_type == "after"
            and checkpoint.payload.get("exists") is True
            and checkpoint.payload.get("file_type") == "directory"
            and isinstance(checkpoint.payload.get("device"), int)
            and isinstance(checkpoint.payload.get("inode"), int)
        ):
            after_created = True
    return before_missing and after_created


def _has_direct_mkdir_failure_created_rollback_proof(
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    before_missing = False
    failure_created = False
    for checkpoint in checkpoints:
        if checkpoint.resource_key != batch.resource_key:
            continue
        if checkpoint.checkpoint_type == "before" and checkpoint.payload.get("exists") is False:
            before_missing = True
        if (
            checkpoint.checkpoint_type == "failure"
            and checkpoint.payload.get("exists") is True
            and checkpoint.payload.get("file_type") == "directory"
            and isinstance(checkpoint.payload.get("device"), int)
            and isinstance(checkpoint.payload.get("inode"), int)
        ):
            failure_created = True
    return before_missing and failure_created


def _has_direct_failed_mkdir_noop_proof(
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    before_missing = False
    failure_missing = False
    for checkpoint in checkpoints:
        if checkpoint.resource_key != batch.resource_key:
            continue
        if checkpoint.checkpoint_type == "before" and checkpoint.payload.get("exists") is False:
            before_missing = True
        if checkpoint.checkpoint_type == "failure" and checkpoint.payload.get("exists") is False:
            failure_missing = True
    return before_missing and failure_missing


def _has_rename_no_replace_rollback_proof(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_type == "rename_record" and _looks_like_rename_payload(checkpoint.payload):
            return True
    for record in transaction.workspace.journal_store.list_recovery_records(batch.batch_id):
        payload = record.payload
        if not isinstance(payload, Mapping):
            continue
        rename_record = payload.get("rename_record")
        if isinstance(rename_record, Mapping) and _looks_like_rename_payload(rename_record):
            return True
        if _looks_like_manual_intervention_action(
            payload.get("manual_intervention_action"),
            operation="rename_no_replace",
        ):
            return True
    return False


def _has_capture_directory_rollback_proof(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    checkpoints: Sequence[CheckpointRecord],
) -> bool:
    for checkpoint in checkpoints:
        if checkpoint.checkpoint_type == "captured_directory" and _looks_like_captured_directory_payload(
            checkpoint.payload,
        ):
            return True
    for record in transaction.workspace.journal_store.list_recovery_records(batch.batch_id):
        payload = record.payload
        if not isinstance(payload, Mapping):
            continue
        captured_restore = payload.get("captured_directory_restore")
        if isinstance(captured_restore, Mapping) and _looks_like_captured_directory_payload(captured_restore):
            return True
        if _looks_like_manual_intervention_action(
            payload.get("manual_intervention_action"),
            operation="capture_directory",
        ):
            return True
    return any(
        action.get("action_type") == "restore_captured_directory"
        for action in captured_directory_recovery_action_plans(
            JournaledFilesystemRecoveryContext(
                batch=batch,
                operations=tuple(transaction.workspace.journal_store.list_operations(batch.batch_id)),
                checkpoints=tuple(checkpoints),
                recovery_records=tuple(transaction.workspace.journal_store.list_recovery_records(batch.batch_id)),
            )
        )
    )


def _latest_failure_payload(transaction: SafeTransaction, batch: OperationBatchRecord) -> object | None:
    failure_payload: object | None = batch.status_payload.get("failure")
    if failure_payload is not None:
        return failure_payload
    for record in reversed(transaction.workspace.journal_store.list_recovery_records(batch.batch_id)):
        record_failure: object | None = record.payload.get("failure")
        if record_failure is not None:
            return record_failure
    return None


def _looks_like_rename_payload(payload: Mapping[str, object]) -> bool:
    return all(key in payload for key in ("source_path", "destination_path", "file_type", "device", "inode"))


def _looks_like_captured_directory_payload(payload: Mapping[str, object]) -> bool:
    captured_directory = payload.get("captured_directory")
    return (
        isinstance(captured_directory, Mapping)
        and captured_directory.get("original_path") is not None
        and captured_directory.get("quarantine_path") is not None
        and isinstance(captured_directory.get("original_identity"), Mapping)
        and isinstance(captured_directory.get("captured_identity"), Mapping)
    )


def _looks_like_manual_intervention_action(value: object, *, operation: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("action_type") != "manual_intervention_required":
        return False
    payload = value.get("payload")
    return isinstance(payload, Mapping) and payload.get("operation") == operation


def _is_read_only_automatic_rollback_batch(batch: OperationBatchRecord) -> bool:
    payload = batch.payload
    if not isinstance(payload, Mapping):
        return False
    operation = payload.get("operation")
    if operation == "snapshot_bundle":
        return True
    return operation == "backup_tree" and batch.phase == BatchPhase.SUCCEEDED


def _is_unsupported_automatic_rollback_batch(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
) -> bool:
    payload = batch.payload
    if not isinstance(payload, Mapping):
        return True
    operation = payload.get("operation")
    if operation == "restore_tree_backup":
        return True
    if operation == "backup_tree":
        return False
    if operation in {
        "write_bytes",
        "write_text",
        "delete_file",
        "make_directory",
        "rename_no_replace",
        "capture_directory",
    }:
        return not _supports_automatic_transaction_rollback(transaction, batch)
    return True


def _unsupported_automatic_rollback_message(batch: OperationBatchRecord) -> str:
    payload = batch.payload
    operation = payload.get("operation") if isinstance(payload, Mapping) else None
    return f"batch {batch.batch_id!r} operation {operation!r} does not support automatic rollback"


def _automatic_rollback_payload(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
) -> dict[str, object]:
    return {
        "automatic_transaction_rollback": {
            "claim_scope": transaction.claim_scope,
            "run_id": transaction.run_id,
            "batch_id": batch.batch_id,
        }
    }


def _mark_batch_recovery_desired(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    *,
    now: datetime,
) -> OperationBatchRecord:
    transaction.workspace.journal_store.record_recovery_desired(
        batch.batch_id,
        lease=transaction.lease,
        reason="automatic transaction rollback",
        payload=_automatic_rollback_payload(transaction, batch),
        now=now,
    )
    updated = transaction.workspace.journal_store.get_batch(batch.batch_id)
    if updated is None:
        raise SafeWorkspaceError(f"automatic rollback batch {batch.batch_id!r} disappeared")
    return updated


def _mark_batch_interrupted_recovery_desired(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    *,
    now: datetime,
) -> OperationBatchRecord:
    transaction.workspace.journal_store.record_interrupted_recovery_desired(
        batch.batch_id,
        lease=transaction.lease,
        resource_key=batch.resource_key,
        reason="automatic transaction rollback found interrupted batch",
        payload=_automatic_rollback_payload(transaction, batch),
        now=now,
    )
    updated = transaction.workspace.journal_store.get_batch(batch.batch_id)
    if updated is None:
        raise SafeWorkspaceError(f"automatic rollback batch {batch.batch_id!r} disappeared")
    return updated


def _automatic_rollback_failure_message(
    failures: list[tuple[OperationBatchRecord, BaseException]],
) -> str:
    details = ", ".join(f"{batch.batch_id} ({type(exc).__name__}: {exc})" for batch, exc in failures)
    return f"automatic rollback failed for {len(failures)} batch(es): {details}"


def _cleanup_batch_artifacts(
    transaction: SafeTransaction,
    batch: OperationBatchRecord,
    *,
    trigger: str,
) -> None:
    cleanup_error = _cleanup_artifacts_for_batches(transaction, (batch,), trigger=trigger)
    if cleanup_error is not None:
        raise cleanup_error.__cause__ if cleanup_error.__cause__ is not None else cleanup_error


def _cleanup_artifacts_for_batches(
    transaction: SafeTransaction,
    batches: tuple[OperationBatchRecord, ...] | list[OperationBatchRecord],
    *,
    trigger: str,
) -> SafeWorkspaceError | None:
    debts: list[_ArtifactCleanupDebt] = []
    with _artifact_cleanup_lease(transaction) as cleanup_lease:
        for batch in batches:
            checkpoints = transaction.workspace.journal_store.list_checkpoints(batch.batch_id)
            if not checkpoints:
                continue
            for backup_candidate in plan_backup_artifact_cleanup_candidates(
                checkpoints,
                state_path=transaction.workspace.state_path,
            ):
                prepared = _prepare_artifact_cleanup_attempt(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    candidate=backup_candidate,
                    trigger=trigger,
                )
                if not prepared.should_execute:
                    continue

                def mark_backup_attempting(
                    *,
                    batch: OperationBatchRecord = batch,
                    prepared: _PreparedArtifactCleanupAttempt = prepared,
                    backup_candidate: BackupArtifactCleanupCandidate = backup_candidate,
                ) -> None:
                    transaction.workspace.journal_store.mark_artifact_cleanup_attempting(
                        batch_id=batch.batch_id,
                        lease=cleanup_lease,
                        artifact_id=prepared.artifact_id,
                        reason="deleting backup artifact",
                        payload=_artifact_cleanup_payload(backup_candidate),
                    )

                with maintain_lease(
                    transaction.workspace.lease_store,
                    cleanup_lease,
                    now=lambda: _cleanup_heartbeat_time(transaction),
                ):
                    backup_result = execute_backup_artifact_cleanup_candidate(
                        backup_candidate,
                        state_path=transaction.workspace.state_path,
                        delete_operation=transaction.workspace.backup_artifact_cleanup_operation,
                        before_delete=mark_backup_attempting if prepared.should_mark_attempting else None,
                    )
                _record_artifact_cleanup_terminal_result(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    artifact_id=prepared.artifact_id,
                    candidate_payload=_artifact_cleanup_payload(backup_candidate),
                    result=backup_result,
                )
                debt = _artifact_cleanup_result_debt(backup_result)
                if debt is not None:
                    debts.append(debt)
            legacy_context = None
            if any(
                checkpoint.checkpoint_type == "captured_directory"
                and isinstance(checkpoint.payload.get("captured_directory"), Mapping)
                and checkpoint.payload["captured_directory"].get("capture_token") is None
                for checkpoint in checkpoints
            ):
                legacy_context = transaction.workspace.journal_store.read_recovery_context(batch.batch_id)
            for captured_candidate in plan_captured_directory_cleanup_candidates(checkpoints, context=legacy_context):
                prepared = _prepare_artifact_cleanup_attempt(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    candidate=captured_candidate,
                    trigger=trigger,
                )
                if not prepared.should_execute:
                    continue

                def mark_captured_attempting(
                    *,
                    batch: OperationBatchRecord = batch,
                    prepared: _PreparedArtifactCleanupAttempt = prepared,
                    captured_candidate: CapturedDirectoryCleanupCandidate = captured_candidate,
                ) -> None:
                    transaction.workspace.journal_store.mark_artifact_cleanup_attempting(
                        batch_id=batch.batch_id,
                        lease=cleanup_lease,
                        artifact_id=prepared.artifact_id,
                        reason="deleting captured directory artifact",
                        payload=_artifact_cleanup_payload(captured_candidate),
                    )

                with maintain_lease(
                    transaction.workspace.lease_store,
                    cleanup_lease,
                    now=lambda: _cleanup_heartbeat_time(transaction),
                ):
                    captured_result = execute_captured_directory_cleanup_candidate(
                        captured_candidate,
                        identity_cleanup_operation=transaction.workspace._coordinator._identity_remove_directory,
                        before_delete=mark_captured_attempting if prepared.should_mark_attempting else None,
                    )
                _record_artifact_cleanup_terminal_result(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    artifact_id=prepared.artifact_id,
                    candidate_payload=_artifact_cleanup_payload(captured_candidate),
                    result=captured_result,
                )
                debt = _artifact_cleanup_result_debt(captured_result)
                if debt is not None:
                    debts.append(debt)
            for tree_candidate in plan_tree_backup_artifact_cleanup_candidates(checkpoints):
                prepared = _prepare_artifact_cleanup_attempt(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    candidate=tree_candidate,
                    trigger=trigger,
                )
                if not prepared.should_execute:
                    continue

                def mark_tree_attempting(
                    *,
                    batch: OperationBatchRecord = batch,
                    prepared: _PreparedArtifactCleanupAttempt = prepared,
                    tree_candidate: TreeBackupArtifactCleanupCandidate = tree_candidate,
                ) -> None:
                    transaction.workspace.journal_store.mark_artifact_cleanup_attempting(
                        batch_id=batch.batch_id,
                        lease=cleanup_lease,
                        artifact_id=prepared.artifact_id,
                        reason="deleting tree backup artifact",
                        payload=_artifact_cleanup_payload(tree_candidate),
                    )

                with maintain_lease(
                    transaction.workspace.lease_store,
                    cleanup_lease,
                    now=lambda: _cleanup_heartbeat_time(transaction),
                ):
                    tree_result = execute_tree_backup_artifact_cleanup_candidate(
                        tree_candidate,
                        delete_operation=transaction.workspace.backup_artifact_cleanup_operation,
                        before_delete=mark_tree_attempting if prepared.should_mark_attempting else None,
                    )
                _record_artifact_cleanup_terminal_result(
                    transaction,
                    batch=batch,
                    lease=cleanup_lease,
                    artifact_id=prepared.artifact_id,
                    candidate_payload=_artifact_cleanup_payload(tree_candidate),
                    result=tree_result,
                )
                debt = _artifact_cleanup_result_debt(tree_result)
                if debt is not None:
                    debts.append(debt)
    if debts:
        exc = _artifact_cleanup_error(debts)
        error = SafeWorkspaceError(str(exc))
        error.__cause__ = exc
        return error
    return None


def _prepare_artifact_cleanup_attempt(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    lease: LeaseRecord,
    candidate: _ArtifactCleanupCandidate,
    trigger: str,
) -> _PreparedArtifactCleanupAttempt:
    records = _artifact_cleanup_records_for_candidate(
        transaction,
        batch=batch,
        candidate=candidate,
    )
    if not records:
        if (
            isinstance(candidate, CapturedDirectoryCleanupCandidate)
            and _captured_directory_cleanup_policy(
                transaction,
                batch=batch,
                candidate=candidate,
            )
            == "retain"
        ):
            return _record_retained_captured_directory_cleanup(
                transaction,
                batch=batch,
                lease=lease,
                candidate=candidate,
                trigger=trigger,
            )
        return _backfill_legacy_artifact_cleanup_plan(
            transaction,
            batch=batch,
            lease=lease,
            candidate=candidate,
            trigger=trigger,
        )
    latest = records[-1]
    if latest.status == "planned":
        return _PreparedArtifactCleanupAttempt(
            artifact_id=latest.artifact_id,
            should_mark_attempting=True,
        )
    if latest.status in {"failed", "manual_intervention_required"}:
        transaction.workspace.journal_store.record_artifact_cleanup_planned(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id=latest.artifact_id,
            trigger=trigger,
            resource_key=candidate.resource_key,
            payload=_artifact_cleanup_payload(candidate),
        )
        return _PreparedArtifactCleanupAttempt(
            artifact_id=latest.artifact_id,
            should_mark_attempting=True,
        )
    if latest.status in {"succeeded", "skipped"}:
        if latest.resource_key != candidate.resource_key:
            raise SafeWorkspaceError(
                f"batch {batch.batch_id!r} artifact cleanup resource mismatch for {candidate.artifact_id!r}"
            )
        if latest.trigger not in {
            ArtifactCleanupTrigger.DEFERRED_CLEANUP,
            ArtifactCleanupTrigger.COMMIT_CLEANUP,
            ArtifactCleanupTrigger.RECOVERY_CLEANUP,
        }:
            raise SafeWorkspaceError(
                f"batch {batch.batch_id!r} artifact cleanup trigger mismatch for {candidate.artifact_id!r}: "
                f"{latest.trigger!r}"
            )
        return _PreparedArtifactCleanupAttempt(
            artifact_id=latest.artifact_id,
            should_mark_attempting=False,
            should_execute=False,
        )
    if latest.status not in {"attempting"}:
        raise SafeWorkspaceError(
            f"batch {batch.batch_id!r} cannot run artifact cleanup for {candidate.artifact_id!r} "
            f"from status {latest.status!r}"
        )
    if latest.resource_key != candidate.resource_key:
        raise SafeWorkspaceError(
            f"batch {batch.batch_id!r} artifact cleanup resource mismatch for {candidate.artifact_id!r}"
        )
    if latest.trigger not in {
        ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        ArtifactCleanupTrigger.COMMIT_CLEANUP,
        ArtifactCleanupTrigger.RECOVERY_CLEANUP,
    }:
        raise SafeWorkspaceError(
            f"batch {batch.batch_id!r} artifact cleanup trigger mismatch for {candidate.artifact_id!r}: "
            f"{latest.trigger!r}"
        )
    return _PreparedArtifactCleanupAttempt(
        artifact_id=latest.artifact_id,
        should_mark_attempting=False,
    )


def _record_retained_captured_directory_cleanup(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    lease: LeaseRecord,
    candidate: CapturedDirectoryCleanupCandidate,
    trigger: str,
) -> _PreparedArtifactCleanupAttempt:
    payload = _artifact_cleanup_payload(candidate)
    transaction.workspace.journal_store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=candidate.artifact_id,
        trigger=trigger,
        resource_key=candidate.resource_key,
        payload=payload,
    )
    retained_payload = dict(payload)
    retained_payload["reason_code"] = "captured_directory_retained"
    transaction.workspace.journal_store.record_artifact_cleanup_skipped(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=candidate.artifact_id,
        reason="captured_directory_retained",
        payload=retained_payload,
    )
    return _PreparedArtifactCleanupAttempt(
        artifact_id=candidate.artifact_id,
        should_mark_attempting=False,
        should_execute=False,
    )


def _backfill_legacy_artifact_cleanup_plan(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    lease: LeaseRecord,
    candidate: _ArtifactCleanupCandidate,
    trigger: str,
) -> _PreparedArtifactCleanupAttempt:
    if not _allows_legacy_artifact_cleanup_backfill(batch, trigger=trigger):
        raise SafeWorkspaceError(
            f"batch {batch.batch_id!r} is missing durable artifact cleanup intent for {candidate.artifact_id!r}"
        )
    transaction.workspace.journal_store.record_artifact_cleanup_planned(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=candidate.artifact_id,
        trigger=trigger,
        resource_key=candidate.resource_key,
        payload=_artifact_cleanup_payload(candidate),
    )
    return _PreparedArtifactCleanupAttempt(
        artifact_id=candidate.artifact_id,
        should_mark_attempting=True,
    )


def _allows_legacy_artifact_cleanup_backfill(batch: OperationBatchRecord, *, trigger: str) -> bool:
    if trigger == ArtifactCleanupTrigger.COMMIT_CLEANUP:
        return batch.phase == BatchPhase.SUCCEEDED
    if trigger == ArtifactCleanupTrigger.RECOVERY_CLEANUP:
        return batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    return False


def _artifact_cleanup_records_for_candidate(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    candidate: _ArtifactCleanupCandidate,
) -> list[ArtifactCleanupRecord]:
    records = transaction.workspace.journal_store.list_artifact_cleanup_records(
        batch.batch_id,
        artifact_id=candidate.artifact_id,
    )
    if records:
        return records
    checkpoint_id = candidate.checkpoint_id
    for record in transaction.workspace.journal_store.list_artifact_cleanup_records(batch.batch_id):
        if record.payload.get("checkpoint_id") == checkpoint_id:
            return transaction.workspace.journal_store.list_artifact_cleanup_records(
                batch.batch_id,
                artifact_id=record.artifact_id,
            )
    return []


def _record_artifact_cleanup_terminal_result(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    lease: LeaseRecord,
    artifact_id: str,
    candidate_payload: dict[str, object],
    result: _ArtifactCleanupResult,
) -> None:
    payload = dict(candidate_payload)
    payload["status"] = result.status
    if result.reason_code is not None:
        payload["reason_code"] = result.reason_code
    if result.detail is not None:
        payload["detail"] = result.detail
    if result.status == "succeeded":
        transaction.workspace.journal_store.record_artifact_cleanup_succeeded(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id=artifact_id,
            payload=payload,
        )
        return
    if result.status == "skipped":
        transaction.workspace.journal_store.record_artifact_cleanup_skipped(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id=artifact_id,
            reason=result.reason_code,
            payload=payload,
        )
        return
    if result.status == "failed":
        transaction.workspace.journal_store.record_artifact_cleanup_failed(
            batch_id=batch.batch_id,
            lease=lease,
            artifact_id=artifact_id,
            reason=result.reason_code,
            payload=payload,
        )
        return
    transaction.workspace.journal_store.record_artifact_cleanup_manual_intervention_required(
        batch_id=batch.batch_id,
        lease=lease,
        artifact_id=artifact_id,
        reason=result.reason_code,
        payload=payload,
    )


def _artifact_cleanup_payload(candidate: _ArtifactCleanupCandidate) -> dict[str, object]:
    payload: dict[str, object] = {"checkpoint_id": candidate.checkpoint_id}
    if isinstance(candidate, BackupArtifactCleanupCandidate):
        payload["artifact_type"] = "backup_file"
        payload["content_path"] = str(candidate.content_path)
        if candidate.expected_content_path is not None:
            payload["expected_content_path"] = str(candidate.expected_content_path)
    elif isinstance(candidate, CapturedDirectoryCleanupCandidate):
        payload["artifact_type"] = "captured_directory"
        payload["quarantine_path"] = str(candidate.quarantine_path)
        if candidate.captured_directory_cleanup is not None:
            payload["captured_directory_cleanup"] = candidate.captured_directory_cleanup
        if candidate.record is not None:
            payload["original_path"] = str(candidate.record.original_path)
            payload["captured_identity"] = {
                "device": candidate.record.captured_identity.device,
                "inode": candidate.record.captured_identity.inode,
            }
    else:
        payload["artifact_type"] = "tree_backup_object"
        payload["content_path"] = str(candidate.content_path)
        payload["store_path"] = str(candidate.store_path)
        payload["digest"] = candidate.digest
        payload["size"] = candidate.size
    if candidate.debt is not None:
        payload["detail"] = candidate.debt.detail
        payload["reason_code"] = candidate.debt.reason_code
    return payload


def _artifact_cleanup_result_debt(result: _ArtifactCleanupResult) -> _ArtifactCleanupDebt | None:
    if result.status not in {"failed", "manual_intervention_required"}:
        return None
    assert result.reason_code is not None
    assert result.detail is not None
    if isinstance(result, BackupArtifactCleanupResult):
        return BackupArtifactCleanupDebt(
            batch_id=result.batch_id,
            content_path=result.content_path,
            reason_code=result.reason_code,
            detail=result.detail,
        )
    if isinstance(result, TreeBackupArtifactCleanupResult):
        return TreeBackupArtifactCleanupDebt(
            batch_id=result.batch_id,
            content_path=result.content_path,
            reason_code=result.reason_code,
            detail=result.detail,
        )
    return CapturedDirectoryCleanupDebt(
        batch_id=result.batch_id,
        quarantine_path=result.quarantine_path,
        reason_code=result.reason_code,
        detail=result.detail,
    )


def _captured_directory_cleanup_policy(
    transaction: SafeTransaction,
    *,
    batch: OperationBatchRecord,
    candidate: CapturedDirectoryCleanupCandidate,
) -> str:
    if candidate.captured_directory_cleanup in {"automatic", "retain"}:
        return candidate.captured_directory_cleanup
    payload = batch.payload if isinstance(batch.payload, Mapping) else {}
    batch_policy = payload.get("captured_directory_cleanup")
    if batch_policy in {"automatic", "retain"}:
        return str(batch_policy)
    return transaction.workspace.captured_directory_cleanup


def _artifact_cleanup_error(debts: list[_ArtifactCleanupDebt]) -> _ArtifactCleanupError:
    backup_debts = [debt for debt in debts if isinstance(debt, BackupArtifactCleanupDebt)]
    captured_debts = [debt for debt in debts if isinstance(debt, CapturedDirectoryCleanupDebt)]
    tree_debts = [debt for debt in debts if isinstance(debt, TreeBackupArtifactCleanupDebt)]
    errors: list[BaseException] = []
    if backup_debts:
        errors.append(BackupArtifactCleanupError(backup_debts))
    if captured_debts:
        errors.append(CapturedDirectoryCleanupError(captured_debts))
    if tree_debts:
        errors.append(TreeBackupArtifactCleanupError(tree_debts))
    if len(errors) == 1:
        return errors[0]  # type: ignore[return-value]
    return _CombinedArtifactCleanupError(tuple(errors))


def _cleanup_heartbeat_time(transaction: SafeTransaction) -> datetime:
    if transaction.cleanup_clock is None:
        return datetime.now(UTC)
    value = transaction.cleanup_clock()
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@contextmanager
def _artifact_cleanup_lease(transaction: SafeTransaction) -> Iterator[LeaseRecord]:
    cleanup_now = transaction.cleanup_clock() if transaction.cleanup_clock is not None else None
    existing_lease = getattr(transaction, "_lease", None)
    if existing_lease is not None and transaction.workspace.lease_store.is_current(existing_lease, now=cleanup_now):
        yield existing_lease
        return
    cleanup_lease = transaction.workspace.lease_store.acquire(
        transaction.workspace.lease_name,
        owner=transaction.workspace.owner,
        ttl=transaction.workspace.lease_ttl,
        now=cleanup_now,
    )
    if not cleanup_lease.acquired:
        raise SafeWorkspaceError(
            f"workspace lease {transaction.workspace.lease_name!r} is held by {cleanup_lease.owner!r}"
        )
    try:
        yield cleanup_lease
    finally:
        transaction.workspace.lease_store.release(cleanup_lease, now=cleanup_now)


def _cleanup_backup_artifacts_for_batches(
    transaction: SafeTransaction,
    batches: tuple[OperationBatchRecord, ...] | list[OperationBatchRecord],
    *,
    trigger: str,
) -> SafeWorkspaceError | None:
    return _cleanup_artifacts_for_batches(transaction, batches, trigger=trigger)


def _handle_artifact_cleanup_debt(
    transaction: SafeTransaction,
    cleanup_debts: list[tuple[OperationBatchRecord, _ArtifactCleanupError]],
    *,
    body_error: BaseException | None,
) -> None:
    message = _artifact_cleanup_debt_message(cleanup_debts)
    error = SafeWorkspaceError(message)
    error.__cause__ = cleanup_debts[0][1]
    if body_error is not None:
        body_error.add_note(f"SafeWorkspace artifact cleanup also failed: {message}")
        return
    raise error


def _artifact_cleanup_debt_message(
    cleanup_debts: list[tuple[OperationBatchRecord, _ArtifactCleanupError]],
) -> str:
    details = ", ".join(f"{batch.batch_id} ({error})" for batch, error in cleanup_debts)
    return f"artifact cleanup debt for {len(cleanup_debts)} batch(es): {details}"


__all__ = [
    "cleanup_committed_transaction_artifacts",
    "cleanup_committed_transaction_backup_artifacts",
    "cleanup_outstanding_workspace_artifacts",
    "recover_pending_workspace_batches",
    "run_automatic_transaction_rollback",
    "run_pending_transaction_rollback",
]
