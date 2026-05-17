from __future__ import annotations

import sqlite3
from typing import cast

from safe_fs_ops.operation_journal.journal_payloads import _frozen_payload, _parse_datetime
from safe_fs_ops.operation_journal.journal_schema import (
    _OPERATION_PHASE_SELECT_LIST,
    _OPERATION_PHASE_TABLE,
    _OPERATION_RUN_SELECT_LIST,
    _OPERATION_RUN_TABLE,
    _OperationPhaseColumn,
    _OperationRunColumn,
)
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord


def _list_operation_runs_in_connection(
    connection: sqlite3.Connection,
    *,
    run_id: str | None = None,
    owner: str | None = None,
    status: str | None = None,
) -> list[OperationRunRecord]:
    where_parts: list[str] = []
    parameters: list[str] = []
    if run_id is not None:
        where_parts.append(f"{_OperationRunColumn.RUN_ID} = ?")
        parameters.append(run_id)
    if owner is not None:
        where_parts.append(f"{_OperationRunColumn.OWNER} = ?")
        parameters.append(owner)
    if status is not None:
        where_parts.append(f"{_OperationRunColumn.STATUS} = ?")
        parameters.append(status)
    where_clause = "" if not where_parts else f"WHERE {' AND '.join(where_parts)}"
    rows = connection.execute(
        f"""
        SELECT {_OPERATION_RUN_SELECT_LIST}
        FROM {_OPERATION_RUN_TABLE}
        {where_clause}
        ORDER BY {_OperationRunColumn.CREATED_AT}, {_OperationRunColumn.OPERATION_RUN_ID}
        """,
        tuple(parameters),
    ).fetchall()
    return [_operation_run_from_row(row) for row in rows]


def _list_operation_phases_in_connection(
    connection: sqlite3.Connection,
    operation_run_id: str,
) -> list[OperationPhaseRecord]:
    rows = connection.execute(
        f"""
        SELECT {_OPERATION_PHASE_SELECT_LIST}
        FROM {_OPERATION_PHASE_TABLE}
        WHERE {_OperationPhaseColumn.OPERATION_RUN_ID} = ?
        ORDER BY {_OperationPhaseColumn.PHASE_ORDER}, {_OperationPhaseColumn.OPERATION_PHASE_ID}
        """,
        (operation_run_id,),
    ).fetchall()
    return [_operation_phase_from_row(row) for row in rows]


def _operation_run_row(connection: sqlite3.Connection, operation_run_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_OPERATION_RUN_SELECT_LIST}
            FROM {_OPERATION_RUN_TABLE}
            WHERE {_OperationRunColumn.OPERATION_RUN_ID} = ?
            """,
            (operation_run_id,),
        ).fetchone(),
    )


def _operation_phase_row(connection: sqlite3.Connection, operation_phase_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_OPERATION_PHASE_SELECT_LIST}
            FROM {_OPERATION_PHASE_TABLE}
            WHERE {_OperationPhaseColumn.OPERATION_PHASE_ID} = ?
            """,
            (operation_phase_id,),
        ).fetchone(),
    )


def _operation_phase_row_by_identity(
    connection: sqlite3.Connection,
    operation_run_id: str,
    phase_name: str,
) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_OPERATION_PHASE_SELECT_LIST}
            FROM {_OPERATION_PHASE_TABLE}
            WHERE {_OperationPhaseColumn.OPERATION_RUN_ID} = ?
              AND {_OperationPhaseColumn.PHASE_NAME} = ?
            """,
            (operation_run_id, phase_name),
        ).fetchone(),
    )


def _operation_run_from_row(row: sqlite3.Row) -> OperationRunRecord:
    return OperationRunRecord(
        operation_run_id=str(row[_OperationRunColumn.OPERATION_RUN_ID]),
        run_id=str(row[_OperationRunColumn.RUN_ID]),
        owner=str(row[_OperationRunColumn.OWNER]),
        lease_name=str(row[_OperationRunColumn.LEASE_NAME]),
        lease_fencing_token=int(row[_OperationRunColumn.LEASE_FENCING_TOKEN]),
        status=str(row[_OperationRunColumn.STATUS]),
        payload=_frozen_payload(str(row[_OperationRunColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_OperationRunColumn.CREATED_AT])),
        updated_at=_parse_datetime(str(row[_OperationRunColumn.UPDATED_AT])),
    )


def _operation_phase_from_row(row: sqlite3.Row) -> OperationPhaseRecord:
    return OperationPhaseRecord(
        operation_phase_id=str(row[_OperationPhaseColumn.OPERATION_PHASE_ID]),
        operation_run_id=str(row[_OperationPhaseColumn.OPERATION_RUN_ID]),
        phase_name=str(row[_OperationPhaseColumn.PHASE_NAME]),
        status=str(row[_OperationPhaseColumn.STATUS]),
        phase_order=int(row[_OperationPhaseColumn.PHASE_ORDER]),
        payload=_frozen_payload(str(row[_OperationPhaseColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_OperationPhaseColumn.CREATED_AT])),
        updated_at=_parse_datetime(str(row[_OperationPhaseColumn.UPDATED_AT])),
    )
