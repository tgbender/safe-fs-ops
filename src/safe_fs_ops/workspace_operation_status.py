from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from safe_fs_ops.operation_journal import JournalLeaseMismatchError
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.workspace import SafeWorkspaceError
from safe_fs_ops.workspace_state import LeaseLostError
from safe_fs_ops.workspace_state.models import LeaseRecord

if TYPE_CHECKING:
    from safe_fs_ops.workspace_operation import SafeOperation, SafePhase
    from safe_fs_ops.workspace_transaction import SafeTransaction


def reconcile_active_phase_on_operation_exit(
    operation: SafeOperation,
    *,
    phase: SafePhase,
    now: datetime,
) -> tuple[SafeWorkspaceError, BaseException | None]:
    phase_error = SafeWorkspaceError(f"operation cannot exit while phase {phase.name!r} is still active")
    status_error: BaseException | None = None
    try:
        if phase.phase_record is not None:
            phase._set_phase_record(update_phase_terminal_status(phase, status="failed", now=now))
    except BaseException as transition_exc:
        status_error = transition_exc
        phase_error.add_note(f"phase status update also failed: {transition_exc}")
    phase._close_due_to_operation_exit()
    operation._exit_phase(phase)
    return phase_error, status_error


def update_operation_run_terminal_status(
    operation: SafeOperation,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> OperationRunRecord:
    assert operation.operation_run is not None
    try:
        return operation._transaction.workspace.journal_store.update_operation_run_status(
            operation.operation_run.operation_run_id,
            lease=lease,
            status=status,
            now=now,
        )
    except (JournalLeaseMismatchError, LeaseLostError):
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        return operation._transaction.workspace.journal_store.record_operation_run_terminal_diagnostic(
            operation.operation_run.operation_run_id,
            lease=lease,
            status=diagnostic_status,
            now=now,
        )
    except BaseException as exc:
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        try:
            operation._operation_run = (
                operation._transaction.workspace.journal_store.record_operation_run_terminal_diagnostic(
                    operation.operation_run.operation_run_id,
                    lease=lease,
                    status=diagnostic_status,
                    now=now,
                )
            )
        except BaseException as diagnostic_exc:
            exc.add_note(f"operation terminal diagnostic also failed: {diagnostic_exc}")
        raise


def record_operation_finalization_failure(
    transaction: SafeTransaction,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> BaseException | None:
    from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_phases, mark_touched_operation_runs

    first_error: BaseException | None = None
    try:
        mark_touched_operation_phases(transaction, lease=lease, status=status, now=now)
    except BaseException as exc:
        first_error = exc
    try:
        mark_touched_operation_runs(transaction, lease=lease, status=status, now=now)
    except BaseException as exc:
        if first_error is None:
            first_error = exc
        else:
            first_error.add_note(f"operation run terminal diagnostic also failed: {exc}")
    return first_error


def update_phase_terminal_status(
    phase: SafePhase,
    *,
    status: str,
    now: datetime,
) -> OperationPhaseRecord:
    assert phase.phase_record is not None
    try:
        return phase.operation._transaction.workspace.journal_store.update_operation_phase_status(
            phase.phase_record.operation_phase_id,
            lease=phase.operation.lease,
            status=status,
            now=now,
        )
    except (JournalLeaseMismatchError, LeaseLostError):
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        return phase.operation._transaction.workspace.journal_store.record_operation_phase_terminal_diagnostic(
            phase.phase_record.operation_phase_id,
            lease=phase.operation.lease,
            status=diagnostic_status,
            now=now,
        )
    except BaseException as exc:
        diagnostic_status = "failed" if status == "failed" else "finalization_failed"
        try:
            phase._set_phase_record(
                phase.operation._transaction.workspace.journal_store.record_operation_phase_terminal_diagnostic(
                    phase.phase_record.operation_phase_id,
                    lease=phase.operation.lease,
                    status=diagnostic_status,
                    now=now,
                )
            )
        except BaseException as diagnostic_exc:
            exc.add_note(f"phase terminal diagnostic also failed: {diagnostic_exc}")
        raise


__all__ = [
    "reconcile_active_phase_on_operation_exit",
    "record_operation_finalization_failure",
    "update_operation_run_terminal_status",
    "update_phase_terminal_status",
]
