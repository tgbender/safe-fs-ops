from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypeVar

from safe_fs_ops.operation_journal.journal_batch_links import _require_valid_batch_operation_links
from safe_fs_ops.operation_journal.journal_errors import (
    BatchNotFoundError,
)
from safe_fs_ops.operation_journal.journal_guards import (
    _lease_token_digest,
    _require_appendable_phase,
    _require_batch_for_write,
    _require_batch_resource_key,
    _require_matching_idempotent_batch,
    _require_startable_batch,
    _require_valid_transition,
)
from safe_fs_ops.operation_journal.journal_payloads import (
    _optional_str,
    _payload_to_json,
    _utcnow,
    _validate_required,
)
from safe_fs_ops.operation_journal.journal_rows import (
    _batch_from_row,
    _batch_row,
    _batch_row_by_idempotency_key,
    _checkpoint_from_row,
    _checkpoint_row,
    _next_sequence,
    _operation_from_row,
    _operation_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _BATCH_SELECT_LIST,
    _BATCH_TABLE,
    _CHECKPOINT_SELECT_LIST,
    _CHECKPOINT_TABLE,
    _OPERATION_SELECT_LIST,
    _OPERATION_TABLE,
    BatchPhase,
    _BatchColumn,
    _CheckpointColumn,
    _OperationColumn,
)
from safe_fs_ops.operation_journal.models import (
    CheckpointRecord,
    OperationBatchRecord,
    OperationRecord,
)
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord

_T = TypeVar("_T")


class JournalBatchMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def create_batch(
        self,
        *,
        idempotency_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        resource_key: str | None = None,
        claim_owner: str | None = None,
        claim_scope: str | None = None,
        payload: Mapping[str, Any] | None = None,
        batch_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationBatchRecord:
        _validate_required("idempotency_key", idempotency_key)
        _validate_required("owner", owner)
        _validate_required("run_id", run_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            existing = _batch_row_by_idempotency_key(connection, idempotency_key)
            if existing is not None:
                _require_matching_idempotent_batch(
                    existing,
                    lease=lease,
                    owner=owner,
                    run_id=run_id,
                    operation_run_id=operation_run_id,
                    operation_phase_id=operation_phase_id,
                    resource_key=resource_key,
                    claim_owner=claim_owner,
                    claim_scope=claim_scope,
                    payload_text=payload_text,
                )
                return _batch_from_row(existing)
            _require_valid_batch_operation_links(
                connection,
                run_id=run_id,
                owner=owner,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            new_batch_id = batch_id or uuid.uuid4().hex
            connection.execute(
                f"""
                INSERT INTO {_BATCH_TABLE} (
                    {_BatchColumn.BATCH_ID},
                    {_BatchColumn.IDEMPOTENCY_KEY},
                    {_BatchColumn.LEASE_NAME},
                    {_BatchColumn.LEASE_OWNER},
                    {_BatchColumn.LEASE_FENCING_TOKEN},
                    {_BatchColumn.LEASE_TOKEN_DIGEST},
                    {_BatchColumn.OWNER},
                    {_BatchColumn.RUN_ID},
                    {_BatchColumn.OPERATION_RUN_ID},
                    {_BatchColumn.OPERATION_PHASE_ID},
                    {_BatchColumn.RESOURCE_KEY},
                    {_BatchColumn.CLAIM_OWNER},
                    {_BatchColumn.CLAIM_SCOPE},
                    {_BatchColumn.PHASE},
                    {_BatchColumn.PAYLOAD},
                    {_BatchColumn.STATUS_MESSAGE},
                    {_BatchColumn.STATUS_PAYLOAD},
                    {_BatchColumn.CREATED_AT},
                    {_BatchColumn.UPDATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_batch_id,
                    idempotency_key,
                    lease.name,
                    lease.owner,
                    lease.fencing_token,
                    _lease_token_digest(lease),
                    owner,
                    run_id,
                    operation_run_id,
                    operation_phase_id,
                    resource_key,
                    claim_owner,
                    claim_scope,
                    BatchPhase.PLANNED,
                    payload_text,
                    None,
                    _payload_to_json(None),
                    current_time.isoformat(),
                    current_time.isoformat(),
                ),
            )
            row = _batch_row(connection, new_batch_id)
            if row is None:
                raise RuntimeError("batch creation did not produce a row")
            return _batch_from_row(row)

    def get_batch(self, batch_id: str) -> OperationBatchRecord | None:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            row = _batch_row(connection, batch_id)
        return None if row is None else _batch_from_row(row)

    def list_batches(self, *, run_id: str | None = None, phase: str | None = None) -> list[OperationBatchRecord]:
        self.initialize()
        where_parts: list[str] = []
        parameters: list[str] = []
        if run_id is not None:
            where_parts.append(f"{_BatchColumn.RUN_ID} = ?")
            parameters.append(run_id)
        if phase is not None:
            where_parts.append(f"{_BatchColumn.PHASE} = ?")
            parameters.append(phase)
        where_clause = "" if not where_parts else f"WHERE {' AND '.join(where_parts)}"
        with self.sqlite_store.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_BATCH_SELECT_LIST}
                FROM {_BATCH_TABLE}
                {where_clause}
                ORDER BY {_BatchColumn.CREATED_AT}, {_BatchColumn.BATCH_ID}
                """,
                tuple(parameters),
            ).fetchall()
        return [_batch_from_row(row) for row in rows]

    def append_operation(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        operation_type: str,
        resource_key: str | None = None,
        payload: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        now: datetime | None = None,
    ) -> OperationRecord:
        _validate_required("batch_id", batch_id)
        _validate_required("operation_type", operation_type)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row = _require_batch_for_write(connection, batch_id, lease=lease, now=current_time)
            _require_batch_resource_key(row, resource_key, action="operation append")
            _require_appendable_phase(str(row[_BatchColumn.PHASE]))
            sequence = _next_sequence(connection, batch_id)
            new_operation_id = operation_id or uuid.uuid4().hex
            connection.execute(
                f"""
                INSERT INTO {_OPERATION_TABLE} (
                    {_OperationColumn.OPERATION_ID},
                    {_OperationColumn.BATCH_ID},
                    {_OperationColumn.SEQUENCE},
                    {_OperationColumn.OPERATION_TYPE},
                    {_OperationColumn.RESOURCE_KEY},
                    {_OperationColumn.PAYLOAD},
                    {_OperationColumn.CREATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_operation_id,
                    row[_BatchColumn.BATCH_ID],
                    sequence,
                    operation_type,
                    resource_key,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            operation_row = _operation_row(connection, new_operation_id)
            if operation_row is None:
                raise RuntimeError("operation append did not produce a row")
            return _operation_from_row(operation_row)

    def start_batch_operation(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        operation_type: str,
        resource_key: str | None = None,
        payload: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        status_payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> tuple[OperationBatchRecord, OperationRecord]:
        _validate_required("batch_id", batch_id)
        _validate_required("operation_type", operation_type)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            row = _batch_row(connection, batch_id)
            if row is None:
                raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
            _require_startable_batch(connection, row, lease=lease, resource_key=resource_key)
            if int(row[_BatchColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token:
                connection.execute(
                    f"""
                    UPDATE {_BATCH_TABLE}
                    SET {_BatchColumn.LEASE_FENCING_TOKEN} = ?,
                        {_BatchColumn.LEASE_OWNER} = ?,
                        {_BatchColumn.LEASE_TOKEN_DIGEST} = ?,
                        {_BatchColumn.UPDATED_AT} = ?
                    WHERE {_BatchColumn.BATCH_ID} = ?
                    """,
                    (lease.fencing_token, lease.owner, _lease_token_digest(lease), current_time.isoformat(), batch_id),
                )
                row = _batch_row(connection, batch_id)
                if row is None:
                    raise RuntimeError("batch disappeared during start")
            sequence = _next_sequence(connection, batch_id)
            new_operation_id = operation_id or uuid.uuid4().hex
            status_payload_text = _payload_to_json(
                {"operation_id": new_operation_id} if status_payload is None else status_payload
            )
            connection.execute(
                f"""
                INSERT INTO {_OPERATION_TABLE} (
                    {_OperationColumn.OPERATION_ID},
                    {_OperationColumn.BATCH_ID},
                    {_OperationColumn.SEQUENCE},
                    {_OperationColumn.OPERATION_TYPE},
                    {_OperationColumn.RESOURCE_KEY},
                    {_OperationColumn.PAYLOAD},
                    {_OperationColumn.CREATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_operation_id,
                    batch_id,
                    sequence,
                    operation_type,
                    resource_key,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (BatchPhase.ATTEMPTING, None, status_payload_text, current_time.isoformat(), batch_id),
            )
            updated = _batch_row(connection, batch_id)
            operation_row = _operation_row(connection, new_operation_id)
            if updated is None or operation_row is None:
                raise RuntimeError("batch start did not produce expected rows")
            return _batch_from_row(updated), _operation_from_row(operation_row)

    def record_checkpoint(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        resource_key: str,
        checkpoint_type: str,
        payload: Mapping[str, Any] | None = None,
        operation_id: str | None = None,
        checkpoint_id: str | None = None,
        now: datetime | None = None,
    ) -> CheckpointRecord:
        _validate_required("batch_id", batch_id)
        _validate_required("resource_key", resource_key)
        _validate_required("checkpoint_type", checkpoint_type)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row = _require_batch_for_write(connection, batch_id, lease=lease, now=current_time)
            _require_batch_resource_key(row, resource_key, action="checkpoint record")
            _require_appendable_phase(str(row[_BatchColumn.PHASE]))
            if operation_id is not None:
                operation_row = _operation_row(connection, operation_id)
                if operation_row is None:
                    raise ValueError(f"operation_id {operation_id!r} does not exist")
                if str(operation_row[_OperationColumn.BATCH_ID]) != batch_id:
                    raise ValueError(f"operation_id {operation_id!r} belongs to a different batch")
                operation_resource_key = _optional_str(operation_row[_OperationColumn.RESOURCE_KEY])
                if operation_resource_key != resource_key:
                    raise ValueError(
                        f"operation_id {operation_id!r} uses resource_key {operation_resource_key!r}, "
                        f"not {resource_key!r}"
                    )
            sequence = _next_sequence(connection, batch_id)
            new_checkpoint_id = checkpoint_id or uuid.uuid4().hex
            connection.execute(
                f"""
                INSERT INTO {_CHECKPOINT_TABLE} (
                    {_CheckpointColumn.CHECKPOINT_ID},
                    {_CheckpointColumn.BATCH_ID},
                    {_CheckpointColumn.SEQUENCE},
                    {_CheckpointColumn.OPERATION_ID},
                    {_CheckpointColumn.RESOURCE_KEY},
                    {_CheckpointColumn.CHECKPOINT_TYPE},
                    {_CheckpointColumn.PAYLOAD},
                    {_CheckpointColumn.CREATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_checkpoint_id,
                    row[_BatchColumn.BATCH_ID],
                    sequence,
                    operation_id,
                    resource_key,
                    checkpoint_type,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            checkpoint_row = _checkpoint_row(connection, new_checkpoint_id)
            if checkpoint_row is None:
                raise RuntimeError("checkpoint record did not produce a row")
            return _checkpoint_from_row(checkpoint_row)

    def mark_attempting(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        payload: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> OperationBatchRecord:
        return self._transition_batch(
            batch_id,
            lease=lease,
            phase=BatchPhase.ATTEMPTING,
            status_message=None,
            status_payload=payload,
            now=now,
        )

    def mark_succeeded(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        result: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> OperationBatchRecord:
        return self._transition_batch(
            batch_id,
            lease=lease,
            phase=BatchPhase.SUCCEEDED,
            status_message=None,
            status_payload=result,
            now=now,
        )

    def mark_failed(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        error: str,
        observed_state: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> OperationBatchRecord:
        _validate_required("error", error)
        return self._transition_batch(
            batch_id,
            lease=lease,
            phase=BatchPhase.FAILED,
            status_message=error,
            status_payload=observed_state,
            now=now,
        )

    def list_operations(self, batch_id: str) -> list[OperationRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_OPERATION_SELECT_LIST}
                FROM {_OPERATION_TABLE}
                WHERE {_OperationColumn.BATCH_ID} = ?
                ORDER BY {_OperationColumn.SEQUENCE}, {_OperationColumn.OPERATION_ID}
                """,
                (batch_id,),
            ).fetchall()
        return [_operation_from_row(row) for row in rows]

    def list_checkpoints(self, batch_id: str) -> list[CheckpointRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_CHECKPOINT_SELECT_LIST}
                FROM {_CHECKPOINT_TABLE}
                WHERE {_CheckpointColumn.BATCH_ID} = ?
                ORDER BY {_CheckpointColumn.SEQUENCE}, {_CheckpointColumn.CHECKPOINT_ID}
                """,
                (batch_id,),
            ).fetchall()
        return [_checkpoint_from_row(row) for row in rows]

    def _transition_batch(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        phase: str,
        status_message: str | None,
        status_payload: Mapping[str, Any] | None,
        now: datetime | None,
    ) -> OperationBatchRecord:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        status_payload_text = _payload_to_json(status_payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row = _require_batch_for_write(connection, batch_id, lease=lease, now=current_time)
            _require_valid_transition(str(row[_BatchColumn.PHASE]), phase)
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (phase, status_message, status_payload_text, current_time.isoformat(), batch_id),
            )
            updated = _batch_row(connection, batch_id)
            if updated is None:
                raise RuntimeError("batch disappeared during transition")
            return _batch_from_row(updated)
