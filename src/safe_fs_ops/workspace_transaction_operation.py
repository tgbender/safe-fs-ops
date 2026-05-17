from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from safe_fs_ops.operation_journal import JournalLeaseMismatchError
from safe_fs_ops.operation_journal.journal_operation_runs import (
    _update_operation_phase_status_in_connection,
    _update_operation_run_status_in_connection,
)
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.workspace import SafeWorkspaceError, _ClaimReleaseError
from safe_fs_ops.workspace_rollback import (
    cleanup_committed_transaction_artifacts,
    run_automatic_transaction_rollback,
    run_pending_transaction_rollback,
)
from safe_fs_ops.workspace_state import LeaseLostError
from safe_fs_ops.workspace_state.claims import _release_claim_in_connection, lease_claim_details_payload
from safe_fs_ops.workspace_state.leases import _release_lease_in_connection, require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord
from safe_fs_ops.workspace_transaction_bookkeeping import (
    abandon_transaction_for_recovery,
    cleanup_now,
    cleanup_transaction,
    record_cleanup_error,
)
from safe_fs_ops.workspace_transaction_state import forget_claim

if TYPE_CHECKING:
    from safe_fs_ops.workspace_transaction import SafeTransaction


def finalize_operation_success(transaction: SafeTransaction, operation_run_id: str) -> None:
    with transaction._state_lock:
        lease = transaction.lease
        transaction._stop_lease_heartbeat()
        cleanup_now_value = cleanup_now(transaction)
        released_resource_keys: list[str] = []
        transaction.workspace.claim_store.initialize()
        transaction.workspace.journal_store.initialize()
        transaction.workspace.lease_store.initialize()
        with transaction.workspace.journal_store.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=cleanup_now_value)
            if transaction._operation_phase is not None:
                transaction._operation_phase = _update_operation_phase_status_in_connection(
                    connection,
                    transaction._operation_phase.operation_phase_id,
                    lease=lease,
                    status="succeeded",
                    now=cleanup_now_value,
                    require_current=False,
                )
            for resource_key in reversed(tuple(transaction._claim_release_order)):
                expected_claim = transaction._claims_by_resource_key.get(resource_key)
                if expected_claim is None:
                    continue
                released = _release_claim_in_connection(
                    connection,
                    resource_key,
                    owner=transaction.workspace.owner,
                    scope=transaction.claim_scope,
                    expected_claim=expected_claim,
                )
                if not released:
                    raise _ClaimReleaseError(
                        f"claim {resource_key!r} could not be released for owner {transaction.workspace.owner!r}"
                    )
                released_resource_keys.append(resource_key)
            _update_operation_run_status_in_connection(
                connection,
                operation_run_id,
                lease=lease,
                status="succeeded",
                now=cleanup_now_value,
                require_current=False,
            )
            if not _release_lease_in_connection(connection, lease, now=cleanup_now_value):
                raise SafeWorkspaceError("failed to release transaction lease")
        for resource_key in released_resource_keys:
            forget_claim(transaction, resource_key)
        transaction._lease = None
        transaction._cleanup_error = None


def finalize_touched_operation_success(transaction: SafeTransaction) -> None:
    with transaction._state_lock:
        lease = transaction.lease
        transaction._stop_lease_heartbeat()
        cleanup_now_value = cleanup_now(transaction)
        released_resource_keys: list[str] = []
        touched_operation_phases = _touched_operation_phases(transaction)
        touched_operation_runs = _touched_operation_runs(transaction)
        transaction.workspace.claim_store.initialize()
        transaction.workspace.journal_store.initialize()
        transaction.workspace.lease_store.initialize()
        with transaction.workspace.lease_store.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=cleanup_now_value)
            for operation_phase in touched_operation_phases:
                updated_phase = _update_operation_phase_status_in_connection(
                    connection,
                    operation_phase.operation_phase_id,
                    lease=lease,
                    status="succeeded",
                    now=cleanup_now_value,
                    require_current=False,
                )
                if (
                    transaction._operation_phase is not None
                    and updated_phase.operation_phase_id == transaction._operation_phase.operation_phase_id
                ):
                    transaction._operation_phase = updated_phase
            for resource_key in reversed(tuple(transaction._claim_release_order)):
                expected_claim = transaction._claims_by_resource_key.get(resource_key)
                if expected_claim is None:
                    continue
                released = _release_claim_in_connection(
                    connection,
                    resource_key,
                    owner=transaction.workspace.owner,
                    scope=transaction.claim_scope,
                    expected_claim=expected_claim,
                )
                if not released:
                    raise _ClaimReleaseError(
                        f"claim {resource_key!r} could not be released for owner {transaction.workspace.owner!r}"
                    )
                released_resource_keys.append(resource_key)
            for operation_run in touched_operation_runs:
                updated_run = _update_operation_run_status_in_connection(
                    connection,
                    operation_run.operation_run_id,
                    lease=lease,
                    status="succeeded",
                    now=cleanup_now_value,
                    require_current=False,
                )
                if (
                    transaction._operation_run is not None
                    and updated_run.operation_run_id == transaction._operation_run.operation_run_id
                ):
                    transaction._operation_run = updated_run
            if not _release_lease_in_connection(connection, lease, now=cleanup_now_value):
                raise SafeWorkspaceError("failed to release transaction lease")
        for resource_key in released_resource_keys:
            forget_claim(transaction, resource_key)
        transaction._lease = None
        transaction._cleanup_error = None


def ensure_implicit_operation_phase(
    transaction: SafeTransaction,
    *,
    now: datetime | None = None,
) -> tuple[OperationRunRecord, OperationPhaseRecord]:
    transaction._require_entered()
    with transaction._state_lock:
        if transaction._operation_run is None:
            transaction._operation_run = transaction.workspace.journal_store.create_operation_run(
                run_id=transaction.run_id,
                lease=transaction.lease,
                owner=transaction.workspace.owner,
                status="active",
                payload={
                    "name": transaction.name,
                    "rollback": transaction.rollback,
                    "api": "transaction",
                },
                now=now or transaction.now,
            )
        if transaction._operation_phase is None:
            transaction._operation_phase = transaction.workspace.journal_store.create_operation_phase(
                operation_run_id=transaction._operation_run.operation_run_id,
                lease=transaction.lease,
                phase_name="transaction",
                status="active",
                phase_order=1,
                payload={"implicit": True, "rollback": transaction.rollback},
                now=now or transaction.now,
            )
        return transaction._operation_run, transaction._operation_phase


def exit_with_implicit_operation(
    transaction: SafeTransaction,
    exc_type: object,
    exc: object,
    traceback: object,
) -> Literal[False]:
    with transaction._state_lock:
        del exc_type, traceback
        body_error = exc if isinstance(exc, BaseException) else None
        status_error: BaseException | None = None
        finalization_now = cleanup_now(transaction)
        lease = transaction.lease
        unresolved_rollback_error: BaseException | None = None
        try:
            if body_error is not None:
                mark_touched_operation_phases_failed(
                    transaction,
                    now=finalization_now,
                )
        except BaseException as transition_exc:
            status_error = transition_exc
            if body_error is not None:
                body_error.add_note(f"transaction phase status update also failed: {transition_exc}")

        if body_error is not None:
            try:
                rollback_resolved = run_automatic_transaction_rollback(
                    transaction,
                    body_error=body_error,
                    now=finalization_now,
                )
                if not rollback_resolved:
                    unresolved_rollback_error = SafeWorkspaceError("automatic rollback did not resolve all batches")
            except BaseException as rollback_exc:
                unresolved_rollback_error = rollback_exc
                body_error.add_note(f"SafeWorkspace automatic rollback also failed: {rollback_exc}")

        if body_error is None and status_error is None and transaction._operation_run is not None:
            success_failure_stage = "pending_rollback"
            try:
                run_pending_transaction_rollback(transaction, now=finalization_now)
                success_failure_stage = "artifact_cleanup"
                artifact_cleanup_error = cleanup_committed_transaction_artifacts(transaction)
                transaction._cleanup_error = artifact_cleanup_error
                if artifact_cleanup_error is not None:
                    raise artifact_cleanup_error
                success_failure_stage = "operation_success"
                finalize_operation_success(transaction, transaction._operation_run.operation_run_id)
                transaction._operation_run = transaction.workspace.journal_store.get_operation_run(
                    transaction._operation_run.operation_run_id
                )
                return False
            except BaseException as success_exc:
                assert transaction._operation_run is not None
                terminal_status_error: BaseException | None = None
                try:
                    mark_touched_operation_phases(
                        transaction,
                        lease=lease,
                        status="finalization_failed",
                        now=finalization_now,
                    )
                except BaseException as transition_exc:
                    terminal_status_error = transition_exc
                try:
                    mark_touched_operation_runs(
                        transaction,
                        lease=lease,
                        status="finalization_failed",
                        now=finalization_now,
                    )
                except BaseException as transition_exc:
                    if terminal_status_error is None:
                        terminal_status_error = transition_exc
                    else:
                        terminal_status_error.add_note(
                            f"transaction operation status update also failed: {transition_exc}"
                        )
                if success_failure_stage == "pending_rollback":
                    abandon_error = abandon_transaction_for_recovery(transaction)
                    if abandon_error is not None:
                        success_exc.add_note(f"SafeWorkspace recovery handoff also failed: {abandon_error}")
                    if terminal_status_error is not None:
                        success_exc.add_note(f"transaction terminal status update also failed: {terminal_status_error}")
                    raise success_exc
                cleanup_error = cleanup_transaction(transaction)
                if cleanup_error is not None:
                    transaction._cleanup_error = cleanup_error
                primary_error: BaseException = cleanup_error or success_exc
                if terminal_status_error is not None:
                    primary_error.add_note(f"transaction terminal status update also failed: {terminal_status_error}")
                if primary_error is success_exc:
                    raise
                raise primary_error from success_exc

        try:
            if transaction._touched_operation_run_ids:
                mark_touched_operation_runs(
                    transaction,
                    lease=lease,
                    status="failed" if body_error is not None else "finalization_failed",
                    now=finalization_now,
                )
        except BaseException as transition_exc:
            status_error = transition_exc
            if body_error is not None:
                body_error.add_note(f"transaction operation status update also failed: {transition_exc}")

        if unresolved_rollback_error is not None:
            abandon_error = abandon_transaction_for_recovery(transaction)
            if abandon_error is not None and body_error is not None:
                body_error.add_note(f"SafeWorkspace recovery handoff also failed: {abandon_error}")
            return False

        cleanup_error = cleanup_transaction(transaction)
        transaction._cleanup_error = cleanup_error
        if cleanup_error is not None and body_error is not None:
            record_cleanup_error(transaction, cleanup_error, body_error)
        if body_error is not None:
            return False
        if cleanup_error is not None:
            if status_error is not None:
                cleanup_error.add_note(f"transaction operation status update also failed: {status_error}")
            raise cleanup_error
        if status_error is not None:
            raise status_error
        return False


def workspace_error(message: str, cause: BaseException) -> SafeWorkspaceError:
    error = SafeWorkspaceError(message)
    error.__cause__ = cause
    return error


def normalized_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def claim_details(*, resource_kind: str, run_id: str, lease: LeaseRecord) -> str:
    return json.dumps(
        lease_claim_details_payload(
            lease,
            claim_id=uuid.uuid4().hex,
            extra={
                "resource_kind": resource_kind,
                "run_id": run_id,
            },
        ),
        separators=(",", ":"),
        sort_keys=True,
    )


def update_operation_run_terminal_status(
    transaction: SafeTransaction,
    operation_run: OperationRunRecord,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> OperationRunRecord:
    try:
        return transaction.workspace.journal_store.update_operation_run_status(
            operation_run.operation_run_id,
            lease=lease,
            status=status,
            now=now,
        )
    except (JournalLeaseMismatchError, LeaseLostError):
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        return transaction.workspace.journal_store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=lease,
            status=diagnostic_status,
            now=now,
        )
    except BaseException as exc:
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        try:
            transaction._operation_run = transaction.workspace.journal_store.record_operation_run_terminal_diagnostic(
                operation_run.operation_run_id,
                lease=lease,
                status=diagnostic_status,
                now=now,
            )
        except BaseException as diagnostic_exc:
            exc.add_note(f"transaction operation terminal diagnostic also failed: {diagnostic_exc}")
        raise


def record_operation_run_terminal_diagnostic(
    transaction: SafeTransaction,
    operation_run: OperationRunRecord,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> BaseException | None:
    try:
        transaction._operation_run = transaction.workspace.journal_store.record_operation_run_terminal_diagnostic(
            operation_run.operation_run_id,
            lease=lease,
            status=status,
            now=now,
        )
    except BaseException as exc:
        return exc
    return None


def update_operation_phase_terminal_status(
    transaction: SafeTransaction,
    operation_phase: OperationPhaseRecord,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> OperationPhaseRecord:
    try:
        return transaction.workspace.journal_store.update_operation_phase_status(
            operation_phase.operation_phase_id,
            lease=lease,
            status=status,
            now=now,
        )
    except (JournalLeaseMismatchError, LeaseLostError):
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        return transaction.workspace.journal_store.record_operation_phase_terminal_diagnostic(
            operation_phase.operation_phase_id,
            lease=lease,
            status=diagnostic_status,
            now=now,
        )
    except BaseException as exc:
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        try:
            transaction._operation_phase = (
                transaction.workspace.journal_store.record_operation_phase_terminal_diagnostic(
                    operation_phase.operation_phase_id,
                    lease=lease,
                    status=diagnostic_status,
                    now=now,
                )
            )
        except BaseException as diagnostic_exc:
            exc.add_note(f"transaction phase terminal diagnostic also failed: {diagnostic_exc}")
        raise


def mark_touched_operation_phases_failed(
    transaction: SafeTransaction,
    *,
    now: datetime,
) -> None:
    mark_touched_operation_phases(
        transaction,
        lease=transaction.lease,
        status="failed",
        now=now,
    )


def mark_touched_operation_runs_failed(
    transaction: SafeTransaction,
    *,
    now: datetime,
) -> None:
    mark_touched_operation_runs(
        transaction,
        lease=transaction.lease,
        status="failed",
        now=now,
    )


def mark_touched_operation_phases(
    transaction: SafeTransaction,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> None:
    for operation_phase in _touched_operation_phases(transaction):
        updated_phase = update_operation_phase_terminal_status(
            transaction,
            operation_phase,
            lease=lease,
            status=status,
            now=now,
        )
        if (
            transaction._operation_phase is not None
            and updated_phase.operation_phase_id == transaction._operation_phase.operation_phase_id
        ):
            transaction._operation_phase = updated_phase


def mark_touched_operation_runs(
    transaction: SafeTransaction,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> None:
    for operation_run in _touched_operation_runs(transaction):
        updated_run = update_operation_run_terminal_status(
            transaction,
            operation_run,
            lease=lease,
            status=status,
            now=now,
        )
        if (
            transaction._operation_run is not None
            and updated_run.operation_run_id == transaction._operation_run.operation_run_id
        ):
            transaction._operation_run = updated_run


def _touched_operation_phases(transaction: SafeTransaction) -> tuple[OperationPhaseRecord, ...]:
    records: list[OperationPhaseRecord] = []
    seen_ids: set[str] = set()
    implicit_phase = transaction._operation_phase
    if implicit_phase is not None:
        seen_ids.add(implicit_phase.operation_phase_id)
        records.append(implicit_phase)
    for operation_phase_id in transaction._touched_operation_phase_ids:
        if operation_phase_id in seen_ids:
            continue
        record = transaction.workspace.journal_store.get_operation_phase(operation_phase_id)
        if record is None:
            continue
        seen_ids.add(operation_phase_id)
        records.append(record)
    return tuple(records)


def _touched_operation_runs(transaction: SafeTransaction) -> tuple[OperationRunRecord, ...]:
    records: list[OperationRunRecord] = []
    seen_ids: set[str] = set()
    implicit_run = transaction._operation_run
    if implicit_run is not None:
        seen_ids.add(implicit_run.operation_run_id)
        records.append(implicit_run)
    for operation_run_id in transaction._touched_operation_run_ids:
        if operation_run_id in seen_ids:
            continue
        record = transaction.workspace.journal_store.get_operation_run(operation_run_id)
        if record is None:
            continue
        seen_ids.add(operation_run_id)
        records.append(record)
    return tuple(records)


__all__ = [
    "claim_details",
    "ensure_implicit_operation_phase",
    "exit_with_implicit_operation",
    "finalize_operation_success",
    "finalize_touched_operation_success",
    "mark_touched_operation_phases_failed",
    "mark_touched_operation_runs_failed",
    "normalized_utc",
    "record_operation_run_terminal_diagnostic",
    "update_operation_phase_terminal_status",
    "update_operation_run_terminal_status",
    "workspace_error",
]
