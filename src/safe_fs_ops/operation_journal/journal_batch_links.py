from __future__ import annotations

import sqlite3

from safe_fs_ops.operation_journal.journal_operation_run_rows import (
    _operation_phase_row,
    _operation_run_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _OperationPhaseColumn,
    _OperationRunColumn,
)


def _require_valid_batch_operation_links(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    owner: str,
    operation_run_id: str | None,
    operation_phase_id: str | None,
) -> None:
    if operation_run_id is None:
        if operation_phase_id is not None:
            raise ValueError("operation_phase_id requires operation_run_id")
        return
    operation_run_row = _operation_run_row(connection, operation_run_id)
    if operation_run_row is None:
        raise ValueError(f"operation_run_id {operation_run_id!r} does not exist")
    stored_run_id = str(operation_run_row[_OperationRunColumn.RUN_ID])
    if stored_run_id != run_id:
        raise ValueError(f"operation_run_id {operation_run_id!r} belongs to run_id {stored_run_id!r}")
    stored_owner = str(operation_run_row[_OperationRunColumn.OWNER])
    if stored_owner != owner:
        raise ValueError(f"operation_run_id {operation_run_id!r} belongs to owner {stored_owner!r}")
    _require_active_operation_link_status(
        "operation_run_id",
        operation_run_id,
        status=str(operation_run_row[_OperationRunColumn.STATUS]),
    )
    if operation_phase_id is None:
        return
    operation_phase_row = _operation_phase_row(connection, operation_phase_id)
    if operation_phase_row is None:
        raise ValueError(f"operation_phase_id {operation_phase_id!r} does not exist")
    phase_run_id = str(operation_phase_row[_OperationPhaseColumn.OPERATION_RUN_ID])
    if phase_run_id != operation_run_id:
        raise ValueError(f"operation_phase_id {operation_phase_id!r} belongs to operation_run_id {phase_run_id!r}")
    _require_active_operation_link_status(
        "operation_phase_id",
        operation_phase_id,
        status=str(operation_phase_row[_OperationPhaseColumn.STATUS]),
    )


def _require_active_operation_link_status(kind: str, record_id: str, *, status: str) -> None:
    if status == "active":
        return
    raise ValueError(f"{kind} {record_id!r} status {status!r} is terminal")
