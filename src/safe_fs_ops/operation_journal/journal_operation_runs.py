from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from safe_fs_ops.operation_journal.journal_errors import BatchIdempotencyMismatchError, JournalLeaseMismatchError
from safe_fs_ops.operation_journal.journal_operation_run_diagnostics import (
    record_operation_phase_terminal_diagnostic_in_connection,
    record_operation_run_terminal_diagnostic_in_connection,
    require_terminal_diagnostic_status,
)
from safe_fs_ops.operation_journal.journal_operation_run_rows import (
    _list_operation_phases_in_connection,
    _list_operation_runs_in_connection,
    _operation_phase_from_row,
    _operation_phase_row,
    _operation_phase_row_by_identity,
    _operation_run_from_row,
    _operation_run_row,
)
from safe_fs_ops.operation_journal.journal_payloads import _payload_to_json, _utcnow, _validate_required
from safe_fs_ops.operation_journal.journal_schema import (
    _OPERATION_PHASE_TABLE,
    _OPERATION_RUN_TABLE,
    _OperationPhaseColumn,
    _OperationRunColumn,
)
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord


class JournalOperationRunMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def create_operation_run(
        self,
        *,
        run_id: str,
        lease: LeaseRecord,
        owner: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
        operation_run_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationRunRecord:
        _validate_required("run_id", run_id)
        _validate_required("owner", owner)
        _validate_required("status", status)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            if operation_run_id is not None:
                existing = _operation_run_row(connection, operation_run_id)
                if existing is not None:
                    _require_matching_operation_run(
                        existing,
                        run_id=run_id,
                        owner=owner,
                        lease=lease,
                        status=status,
                        payload_text=payload_text,
                    )
                    return _operation_run_from_row(existing)
            new_operation_run_id = operation_run_id or uuid.uuid4().hex
            try:
                connection.execute(
                    f"""
                    INSERT INTO {_OPERATION_RUN_TABLE} (
                        {_OperationRunColumn.OPERATION_RUN_ID},
                        {_OperationRunColumn.RUN_ID},
                        {_OperationRunColumn.OWNER},
                        {_OperationRunColumn.LEASE_NAME},
                        {_OperationRunColumn.LEASE_FENCING_TOKEN},
                        {_OperationRunColumn.STATUS},
                        {_OperationRunColumn.PAYLOAD},
                        {_OperationRunColumn.CREATED_AT},
                        {_OperationRunColumn.UPDATED_AT}
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_operation_run_id,
                        run_id,
                        owner,
                        lease.name,
                        lease.fencing_token,
                        status,
                        payload_text,
                        current_time.isoformat(),
                        current_time.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                _raise_operation_run_integrity_error(
                    connection,
                    exc,
                    operation_run_id=new_operation_run_id,
                    run_id=run_id,
                    owner=owner,
                )
            row = _operation_run_row(connection, new_operation_run_id)
            if row is None:
                raise RuntimeError("operation run creation did not produce a row")
            return _operation_run_from_row(row)

    def get_operation_run(self, operation_run_id: str) -> OperationRunRecord | None:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            row = _operation_run_row(connection, operation_run_id)
        return None if row is None else _operation_run_from_row(row)

    def list_operation_runs(
        self,
        *,
        run_id: str | None = None,
        owner: str | None = None,
        status: str | None = None,
    ) -> list[OperationRunRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_operation_runs_in_connection(connection, run_id=run_id, owner=owner, status=status)

    def create_operation_phase(
        self,
        *,
        operation_run_id: str,
        lease: LeaseRecord,
        phase_name: str,
        status: str,
        phase_order: int,
        payload: Mapping[str, Any] | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationPhaseRecord:
        _validate_required("operation_run_id", operation_run_id)
        _validate_required("phase_name", phase_name)
        _validate_required("status", status)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            operation_run_row = _require_operation_run_for_write(connection, operation_run_id, lease=lease)
            existing = _operation_phase_row_by_identity(
                connection,
                str(operation_run_row[_OperationRunColumn.OPERATION_RUN_ID]),
                phase_name,
            )
            if existing is not None:
                _require_matching_operation_phase(
                    existing,
                    status=status,
                    phase_order=phase_order,
                    payload_text=payload_text,
                    operation_phase_id=operation_phase_id,
                )
                return _operation_phase_from_row(existing)
            new_operation_phase_id = operation_phase_id or uuid.uuid4().hex
            try:
                connection.execute(
                    f"""
                    INSERT INTO {_OPERATION_PHASE_TABLE} (
                        {_OperationPhaseColumn.OPERATION_PHASE_ID},
                        {_OperationPhaseColumn.OPERATION_RUN_ID},
                        {_OperationPhaseColumn.PHASE_NAME},
                        {_OperationPhaseColumn.STATUS},
                        {_OperationPhaseColumn.PHASE_ORDER},
                        {_OperationPhaseColumn.PAYLOAD},
                        {_OperationPhaseColumn.CREATED_AT},
                        {_OperationPhaseColumn.UPDATED_AT}
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_operation_phase_id,
                        str(operation_run_row[_OperationRunColumn.OPERATION_RUN_ID]),
                        phase_name,
                        status,
                        phase_order,
                        payload_text,
                        current_time.isoformat(),
                        current_time.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                _raise_operation_phase_integrity_error(
                    connection,
                    exc,
                    operation_phase_id=new_operation_phase_id,
                    operation_run_id=operation_run_id,
                    phase_name=phase_name,
                )
            row = _operation_phase_row(connection, new_operation_phase_id)
            if row is None:
                raise RuntimeError("operation phase creation did not produce a row")
            return _operation_phase_from_row(row)

    def get_operation_phase(self, operation_phase_id: str) -> OperationPhaseRecord | None:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            row = _operation_phase_row(connection, operation_phase_id)
        return None if row is None else _operation_phase_from_row(row)

    def list_operation_phases(self, operation_run_id: str) -> list[OperationPhaseRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_operation_phases_in_connection(connection, operation_run_id)

    def update_operation_run_status(
        self,
        operation_run_id: str,
        *,
        lease: LeaseRecord,
        status: str,
        now: datetime | None = None,
    ) -> OperationRunRecord:
        _validate_required("operation_run_id", operation_run_id)
        _validate_required("status", status)
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            return _update_operation_run_status_in_connection(
                connection,
                operation_run_id,
                lease=lease,
                status=status,
                now=current_time,
                require_current=True,
            )

    def update_operation_phase_status(
        self,
        operation_phase_id: str,
        *,
        lease: LeaseRecord,
        status: str,
        now: datetime | None = None,
    ) -> OperationPhaseRecord:
        _validate_required("operation_phase_id", operation_phase_id)
        _validate_required("status", status)
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            return _update_operation_phase_status_in_connection(
                connection,
                operation_phase_id,
                lease=lease,
                status=status,
                now=current_time,
                require_current=True,
            )

    def record_operation_run_terminal_diagnostic(
        self,
        operation_run_id: str,
        *,
        lease: LeaseRecord,
        status: str,
        now: datetime | None = None,
    ) -> OperationRunRecord:
        require_terminal_diagnostic_status(status, kind="operation run")
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            return record_operation_run_terminal_diagnostic_in_connection(
                connection,
                operation_run_id,
                lease=lease,
                status=status,
                now=current_time,
            )

    def record_operation_phase_terminal_diagnostic(
        self,
        operation_phase_id: str,
        *,
        lease: LeaseRecord,
        status: str,
        now: datetime | None = None,
    ) -> OperationPhaseRecord:
        require_terminal_diagnostic_status(status, kind="operation phase")
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            return record_operation_phase_terminal_diagnostic_in_connection(
                connection,
                operation_phase_id,
                lease=lease,
                status=status,
                now=current_time,
            )


def _require_matching_operation_run(
    row: sqlite3.Row,
    *,
    run_id: str,
    owner: str,
    lease: LeaseRecord,
    status: str,
    payload_text: str,
) -> None:
    if str(row[_OperationRunColumn.RUN_ID]) != run_id:
        raise BatchIdempotencyMismatchError("existing operation run has different run_id")
    if str(row[_OperationRunColumn.OWNER]) != owner:
        raise BatchIdempotencyMismatchError("existing operation run has different owner")
    if str(row[_OperationRunColumn.LEASE_NAME]) != lease.name:
        raise BatchIdempotencyMismatchError("existing operation run has different lease_name")
    if int(row[_OperationRunColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token:
        raise BatchIdempotencyMismatchError("existing operation run has different lease_fencing_token")
    if str(row[_OperationRunColumn.STATUS]) != status:
        raise BatchIdempotencyMismatchError("existing operation run has different status")
    if str(row[_OperationRunColumn.PAYLOAD]) != payload_text:
        raise BatchIdempotencyMismatchError("existing operation run has different payload")


def _require_matching_operation_phase(
    row: sqlite3.Row,
    *,
    status: str,
    phase_order: int,
    payload_text: str,
    operation_phase_id: str | None,
) -> None:
    if str(row[_OperationPhaseColumn.STATUS]) != status:
        raise BatchIdempotencyMismatchError("existing operation phase has different status")
    if int(row[_OperationPhaseColumn.PHASE_ORDER]) != phase_order:
        raise BatchIdempotencyMismatchError("existing operation phase has different phase_order")
    if str(row[_OperationPhaseColumn.PAYLOAD]) != payload_text:
        raise BatchIdempotencyMismatchError("existing operation phase has different payload")
    if operation_phase_id is not None and str(row[_OperationPhaseColumn.OPERATION_PHASE_ID]) != operation_phase_id:
        raise BatchIdempotencyMismatchError("existing operation phase has different operation_phase_id")


def _raise_operation_run_integrity_error(
    connection: sqlite3.Connection,
    exc: sqlite3.IntegrityError,
    *,
    operation_run_id: str,
    run_id: str,
    owner: str,
) -> None:
    message = str(exc)
    if _is_unique_constraint_for(
        message,
        f"{_OPERATION_RUN_TABLE}.{_OperationRunColumn.OPERATION_RUN_ID}",
    ):
        existing = _operation_run_row(connection, operation_run_id)
        if existing is not None and (
            str(existing[_OperationRunColumn.RUN_ID]) != run_id or str(existing[_OperationRunColumn.OWNER]) != owner
        ):
            raise BatchIdempotencyMismatchError("existing operation run has different operation_run_id") from exc
    raise exc


def _raise_operation_phase_integrity_error(
    connection: sqlite3.Connection,
    exc: sqlite3.IntegrityError,
    *,
    operation_phase_id: str,
    operation_run_id: str,
    phase_name: str,
) -> None:
    message = str(exc)
    if _is_unique_constraint_for(
        message,
        f"{_OPERATION_PHASE_TABLE}.{_OperationPhaseColumn.OPERATION_PHASE_ID}",
    ):
        existing = _operation_phase_row(connection, operation_phase_id)
        if existing is not None and (
            str(existing[_OperationPhaseColumn.OPERATION_RUN_ID]) != operation_run_id
            or str(existing[_OperationPhaseColumn.PHASE_NAME]) != phase_name
        ):
            raise BatchIdempotencyMismatchError("existing operation phase has different operation_phase_id") from exc
    if _is_unique_constraint_for(
        message,
        f"{_OPERATION_PHASE_TABLE}.{_OperationPhaseColumn.OPERATION_RUN_ID}",
        f"{_OPERATION_PHASE_TABLE}.{_OperationPhaseColumn.PHASE_NAME}",
    ):
        raise ValueError(f"phase_name {phase_name!r} already exists for operation_run_id {operation_run_id!r}") from exc
    raise exc


def _is_unique_constraint_for(message: str, *columns: str) -> bool:
    return message == f"UNIQUE constraint failed: {', '.join(columns)}"


def _require_operation_run_for_write(
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
    if int(row[_OperationRunColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token:
        raise JournalLeaseMismatchError(
            f"operation run {operation_run_id!r} uses fencing token "
            f"{row[_OperationRunColumn.LEASE_FENCING_TOKEN]!r}, not {lease.fencing_token!r}"
        )
    return row


def _require_operation_phase_for_write(
    connection: sqlite3.Connection,
    operation_phase_id: str,
    *,
    lease: LeaseRecord,
) -> sqlite3.Row:
    row = _operation_phase_row(connection, operation_phase_id)
    if row is None:
        raise ValueError(f"operation_phase_id {operation_phase_id!r} does not exist")
    _require_operation_run_for_write(
        connection,
        str(row[_OperationPhaseColumn.OPERATION_RUN_ID]),
        lease=lease,
    )
    return row


def _require_operation_status_transition(*, current_status: str, next_status: str, kind: str) -> None:
    if current_status == next_status:
        return
    if current_status != "active" or next_status not in {"succeeded", "failed", "finalization_failed"}:
        raise ValueError(f"cannot transition {kind} from {current_status!r} to {next_status!r}")


def _update_operation_run_status_in_connection(
    connection: sqlite3.Connection,
    operation_run_id: str,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
    require_current: bool,
) -> OperationRunRecord:
    if require_current:
        require_current_lease(connection, lease, now=now)
    row = _require_operation_run_for_write(connection, operation_run_id, lease=lease)
    _require_operation_status_transition(
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
        raise RuntimeError("operation run disappeared during status update")
    return _operation_run_from_row(updated)


def _update_operation_phase_status_in_connection(
    connection: sqlite3.Connection,
    operation_phase_id: str,
    *,
    lease: LeaseRecord,
    status: str,
    now: datetime,
    require_current: bool,
) -> OperationPhaseRecord:
    if require_current:
        require_current_lease(connection, lease, now=now)
    row = _require_operation_phase_for_write(connection, operation_phase_id, lease=lease)
    _require_operation_status_transition(
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
        raise RuntimeError("operation phase disappeared during status update")
    return _operation_phase_from_row(updated)
