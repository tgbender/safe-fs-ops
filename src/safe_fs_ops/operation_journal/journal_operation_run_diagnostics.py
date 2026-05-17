from __future__ import annotations

import sqlite3
from datetime import datetime

from safe_fs_ops.operation_journal.journal_errors import JournalLeaseMismatchError
from safe_fs_ops.operation_journal.journal_operation_run_rows import (
    _operation_phase_from_row,
    _operation_phase_row,
    _operation_run_from_row,
    _operation_run_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _OPERATION_PHASE_TABLE,
    _OPERATION_RUN_TABLE,
    _OperationPhaseColumn,
    _OperationRunColumn,
)
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord


def record_operation_run_terminal_diagnostic_in_connection(
    connection: sqlite3.Connection,
    operation_run_id: str,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> OperationRunRecord:
    require_current_lease(connection, lease, now=now)
    row = _require_operation_run_for_terminal_diagnostic(connection, operation_run_id, lease=lease)
    if row is None:
        raise ValueError(f"operation_run_id {operation_run_id!r} does not exist")
    _require_operation_terminal_diagnostic_transition(
        current_status=str(row[_OperationRunColumn.STATUS]),
        next_status=status,
        kind="operation run",
    )
    if str(row[_OperationRunColumn.STATUS]) == status:
        return _operation_run_from_row(row)
    connection.execute(
        f"""
        UPDATE {_OPERATION_RUN_TABLE}
        SET {_OperationRunColumn.STATUS} = ?,
            {_OperationRunColumn.UPDATED_AT} = ?
        WHERE {_OperationRunColumn.OPERATION_RUN_ID} = ?
        """,
        (status, now.isoformat(), operation_run_id),
    )
    updated = _operation_run_row(connection, operation_run_id)
    if updated is None:
        raise RuntimeError("operation run disappeared during terminal diagnostic update")
    return _operation_run_from_row(updated)


def record_operation_phase_terminal_diagnostic_in_connection(
    connection: sqlite3.Connection,
    operation_phase_id: str,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
) -> OperationPhaseRecord:
    require_current_lease(connection, lease, now=now)
    row = _operation_phase_row(connection, operation_phase_id)
    if row is None:
        raise ValueError(f"operation_phase_id {operation_phase_id!r} does not exist")
    _require_operation_run_for_terminal_diagnostic(
        connection,
        str(row[_OperationPhaseColumn.OPERATION_RUN_ID]),
        lease=lease,
    )
    _require_operation_terminal_diagnostic_transition(
        current_status=str(row[_OperationPhaseColumn.STATUS]),
        next_status=status,
        kind="operation phase",
    )
    if str(row[_OperationPhaseColumn.STATUS]) == status:
        return _operation_phase_from_row(row)
    connection.execute(
        f"""
        UPDATE {_OPERATION_PHASE_TABLE}
        SET {_OperationPhaseColumn.STATUS} = ?,
            {_OperationPhaseColumn.UPDATED_AT} = ?
        WHERE {_OperationPhaseColumn.OPERATION_PHASE_ID} = ?
        """,
        (status, now.isoformat(), operation_phase_id),
    )
    updated = _operation_phase_row(connection, operation_phase_id)
    if updated is None:
        raise RuntimeError("operation phase disappeared during terminal diagnostic update")
    return _operation_phase_from_row(updated)


def require_terminal_diagnostic_status(status: str, *, kind: str) -> None:
    if status not in {"failed", "finalization_failed"}:
        raise ValueError(f"{kind} terminal diagnostic status must be 'failed' or 'finalization_failed'")


def _require_operation_terminal_diagnostic_transition(*, current_status: str, next_status: str, kind: str) -> None:
    if current_status == next_status:
        return
    if next_status == "failed" and current_status == "active":
        return
    if next_status == "finalization_failed" and current_status in {"active", "succeeded"}:
        return
    raise ValueError(f"cannot record terminal diagnostic for {kind} from {current_status!r} to {next_status!r}")


def _require_operation_run_for_terminal_diagnostic(
    connection: sqlite3.Connection,
    operation_run_id: str,
    *,
    lease: LeaseRecord,
) -> sqlite3.Row:
    row = _operation_run_row(connection, operation_run_id)
    if row is None:
        raise ValueError(f"operation_run_id {operation_run_id!r} does not exist")
    if str(row[_OperationRunColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(
            f"operation run {operation_run_id!r} uses lease {row[_OperationRunColumn.LEASE_NAME]!r}, not {lease.name!r}"
        )
    run_fencing_token = int(row[_OperationRunColumn.LEASE_FENCING_TOKEN])
    if run_fencing_token > lease.fencing_token:
        raise JournalLeaseMismatchError(
            f"operation run {operation_run_id!r} uses newer fencing token {run_fencing_token!r}"
        )
    return row
