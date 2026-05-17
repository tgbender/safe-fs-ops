from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, TypeVar

from safe_fs_ops.operation_journal.journal_errors import (
    BatchNotFoundError,
    JournalLeaseMismatchError,
)
from safe_fs_ops.operation_journal.journal_guards import (
    _lease_token_digest,
    _require_active_recovery_attempt,
    _require_batch_for_recovery_attempt_check,
    _require_batch_for_recovery_write,
    _require_lease_lost_recovery_marker,
    _require_recovery_takeover,
    _require_valid_transition,
)
from safe_fs_ops.operation_journal.journal_payloads import (
    _payload_to_json,
    _utcnow,
    _validate_required,
)
from safe_fs_ops.operation_journal.journal_rows import (
    _batch_from_row,
    _batch_row,
    _next_sequence,
    _recovery_from_row,
    _recovery_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _BATCH_TABLE,
    _RECOVERY_TABLE,
    BatchPhase,
    _BatchColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import (
    OperationBatchRecord,
    RecoveryRecord,
)
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import LeaseLostError, require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord

_T = TypeVar("_T")


class JournalRecoveryMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def record_recovery_desired(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        return self._record_recovery_phase(
            batch_id,
            lease=lease,
            phase=BatchPhase.RECOVERY_DESIRED,
            recovery_attempt_id=None,
            reason=reason,
            payload=payload,
            recovery_id=recovery_id,
            now=now,
        )

    def start_recovery(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> tuple[OperationBatchRecord, RecoveryRecord]:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row, _ = _require_batch_for_recovery_write(
                connection,
                batch_id,
                lease=lease,
                recovery_phase=BatchPhase.RECOVERING,
                now=current_time,
            )
            _require_valid_transition(str(row[_BatchColumn.PHASE]), BatchPhase.RECOVERING)
            sequence = _next_sequence(connection, batch_id)
            new_recovery_id = recovery_id or uuid.uuid4().hex
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (BatchPhase.RECOVERING, reason, payload_text, current_time.isoformat(), batch_id),
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
                    new_recovery_id,
                    batch_id,
                    sequence,
                    BatchPhase.RECOVERING,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            updated = _batch_row(connection, batch_id)
            recovery_row = _recovery_row(connection, new_recovery_id)
            if updated is None or recovery_row is None:
                raise RuntimeError("recovery start did not produce expected rows")
            return _batch_from_row(updated), _recovery_from_row(recovery_row)

    def require_active_recovery_attempt(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            _require_batch_for_recovery_attempt_check(
                connection,
                batch_id,
                lease=lease,
                now=current_time,
            )
            attempt_row = _require_active_recovery_attempt(
                connection,
                batch_id,
                recovery_attempt_id=recovery_attempt_id,
            )
            return _recovery_from_row(attempt_row)

    def run_with_active_recovery_attempt(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        run: Callable[[], _T],
        now: datetime | None = None,
    ) -> _T:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            _require_batch_for_recovery_attempt_check(
                connection,
                batch_id,
                lease=lease,
                now=current_time,
            )
            _require_active_recovery_attempt(
                connection,
                batch_id,
                recovery_attempt_id=recovery_attempt_id,
            )
        return run()

    def record_recovery_succeeded(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        recovery_attempt_id: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        return self._record_recovery_phase(
            batch_id,
            lease=lease,
            phase=BatchPhase.RECOVERY_SUCCEEDED,
            recovery_attempt_id=recovery_attempt_id,
            reason=reason,
            payload=payload,
            recovery_id=recovery_id,
            now=now,
        )

    def record_recovery_failed(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        recovery_attempt_id: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        return self._record_recovery_phase(
            batch_id,
            lease=lease,
            phase=BatchPhase.RECOVERY_FAILED,
            recovery_attempt_id=recovery_attempt_id,
            reason=reason,
            payload=payload,
            recovery_id=recovery_id,
            now=now,
        )

    def record_interrupted_recovery_desired(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        resource_key: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            row = _batch_row(connection, batch_id)
            if row is None:
                raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
            _require_recovery_takeover(connection, row, lease=lease, resource_key=resource_key)
            sequence = _next_sequence(connection, batch_id)
            new_recovery_id = recovery_id or uuid.uuid4().hex
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.LEASE_FENCING_TOKEN} = ?,
                    {_BatchColumn.LEASE_OWNER} = ?,
                    {_BatchColumn.LEASE_TOKEN_DIGEST} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (
                    BatchPhase.RECOVERY_DESIRED,
                    lease.fencing_token,
                    lease.owner,
                    _lease_token_digest(lease),
                    reason,
                    payload_text,
                    current_time.isoformat(),
                    batch_id,
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
                    new_recovery_id,
                    batch_id,
                    sequence,
                    BatchPhase.RECOVERY_DESIRED,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            recovery_row = _recovery_row(connection, new_recovery_id)
            if recovery_row is None:
                raise RuntimeError("recovery record did not produce a row")
            return _recovery_from_row(recovery_row)

    def record_lease_lost_recovery_desired(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        resource_key: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        recovery_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryRecord:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            try:
                require_current_lease(connection, lease, now=current_time)
            except LeaseLostError:
                pass
            else:
                raise JournalLeaseMismatchError("lease-lost recovery marker requires an expired or superseded lease")
            row = _batch_row(connection, batch_id)
            if row is None:
                raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
            _require_lease_lost_recovery_marker(row, lease=lease, resource_key=resource_key)
            sequence = _next_sequence(connection, batch_id)
            new_recovery_id = recovery_id or uuid.uuid4().hex
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (
                    BatchPhase.RECOVERY_DESIRED,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                    batch_id,
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
                    new_recovery_id,
                    batch_id,
                    sequence,
                    BatchPhase.RECOVERY_DESIRED,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            recovery_row = _recovery_row(connection, new_recovery_id)
            if recovery_row is None:
                raise RuntimeError("recovery record did not produce a row")
            return _recovery_from_row(recovery_row)

    def _record_recovery_phase(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        phase: str,
        recovery_attempt_id: str | None,
        reason: str | None,
        payload: Mapping[str, Any] | None,
        recovery_id: str | None,
        now: datetime | None,
    ) -> RecoveryRecord:
        _validate_required("batch_id", batch_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row, takeover_recorded = _require_batch_for_recovery_write(
                connection,
                batch_id,
                lease=lease,
                recovery_phase=phase,
                now=current_time,
            )
            current_phase = str(row[_BatchColumn.PHASE])
            if not (
                (takeover_recorded and current_phase == BatchPhase.RECOVERING and phase == BatchPhase.RECOVERY_DESIRED)
                or (current_phase == BatchPhase.SUCCEEDED and phase == BatchPhase.RECOVERY_DESIRED)
            ):
                _require_valid_transition(current_phase, phase)
            if phase in {BatchPhase.RECOVERY_SUCCEEDED, BatchPhase.RECOVERY_FAILED}:
                _require_active_recovery_attempt(
                    connection,
                    batch_id,
                    recovery_attempt_id=recovery_attempt_id,
                )
            sequence = _next_sequence(connection, batch_id)
            new_recovery_id = recovery_id or uuid.uuid4().hex
            connection.execute(
                f"""
                UPDATE {_BATCH_TABLE}
                SET {_BatchColumn.PHASE} = ?,
                    {_BatchColumn.STATUS_MESSAGE} = ?,
                    {_BatchColumn.STATUS_PAYLOAD} = ?,
                    {_BatchColumn.UPDATED_AT} = ?
                WHERE {_BatchColumn.BATCH_ID} = ?
                """,
                (phase, reason, payload_text, current_time.isoformat(), batch_id),
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
                (new_recovery_id, batch_id, sequence, phase, reason, payload_text, current_time.isoformat()),
            )
            recovery_row = _recovery_row(connection, new_recovery_id)
            if recovery_row is None:
                raise RuntimeError("recovery record did not produce a row")
            return _recovery_from_row(recovery_row)
