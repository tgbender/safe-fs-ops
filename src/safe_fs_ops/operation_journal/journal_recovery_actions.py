from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import Any, TypeVar, cast

from safe_fs_ops.operation_journal.journal_errors import (
    RecoveryAttemptMismatchError,
)
from safe_fs_ops.operation_journal.journal_guards import (
    _prepare_recovery_action_write,
    _require_active_recovery_attempt,
    _require_batch_for_recovery_action_write,
    _require_batch_resource_key,
)
from safe_fs_ops.operation_journal.journal_payloads import (
    _payload_to_json,
    _utcnow,
    _validate_required,
)
from safe_fs_ops.operation_journal.journal_rows import (
    _list_recovery_actions_in_connection,
    _next_sequence,
    _recovery_action_from_row,
    _recovery_action_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _RECOVERY_ACTION_TABLE,
    _RecoveryActionColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import (
    RecoveryActionRecord,
    RecoveryActionStatus,
)
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.models import LeaseRecord

_T = TypeVar("_T")


class JournalRecoveryActionMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def record_recovery_action_planned(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_type: str,
        resource_key: str | None = None,
        payload: Mapping[str, Any] | None = None,
        action_id: str | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        _validate_required("action_type", action_type)
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.PLANNED,
            action_type=action_type,
            resource_key=resource_key,
            reason=None,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def mark_recovery_action_attempting(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.ATTEMPTING,
            action_type=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def record_recovery_action_succeeded(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_id: str,
        payload: Mapping[str, Any] | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.SUCCEEDED,
            action_type=None,
            resource_key=None,
            reason=None,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def record_recovery_action_failed(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.FAILED,
            action_type=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def record_recovery_action_skipped(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.SKIPPED,
            action_type=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def record_recovery_action_manual_intervention_required(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        action_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        action_record_id: str | None = None,
        now: datetime | None = None,
    ) -> RecoveryActionRecord:
        return self._record_recovery_action(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            status=RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED,
            action_type=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            action_id=action_id,
            action_record_id=action_record_id,
            now=now,
        )

    def _record_recovery_action(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        status: str,
        action_type: str | None,
        resource_key: str | None,
        reason: str | None,
        payload: Mapping[str, Any] | None,
        action_id: str | None,
        action_record_id: str | None,
        now: datetime | None,
    ) -> RecoveryActionRecord:
        _validate_required("batch_id", batch_id)
        if not recovery_attempt_id:
            raise RecoveryAttemptMismatchError(
                f"batch {batch_id!r} recovery action writes require the active recovery_attempt_id"
            )
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row = _require_batch_for_recovery_action_write(
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
            existing_action = _prepare_recovery_action_write(
                connection,
                batch_id=batch_id,
                recovery_attempt_id=recovery_attempt_id,
                status=status,
                action_id=action_id,
                action_type=action_type,
                resource_key=resource_key,
            )
            sequence = _next_sequence(connection, batch_id)
            new_action_id = (
                existing_action.action_id if existing_action is not None else (action_id or uuid.uuid4().hex)
            )
            new_action_type = existing_action.action_type if existing_action is not None else cast(str, action_type)
            new_resource_key = existing_action.resource_key if existing_action is not None else resource_key
            _require_batch_resource_key(row, new_resource_key, action="recovery action append")
            new_action_record_id = action_record_id or uuid.uuid4().hex
            connection.execute(
                f"""
                INSERT INTO {_RECOVERY_ACTION_TABLE} (
                    {_RecoveryActionColumn.ACTION_RECORD_ID},
                    {_RecoveryActionColumn.ACTION_ID},
                    {_RecoveryActionColumn.RECOVERY_ATTEMPT_ID},
                    {_RecoveryActionColumn.BATCH_ID},
                    {_RecoveryActionColumn.SEQUENCE},
                    {_RecoveryActionColumn.ACTION_TYPE},
                    {_RecoveryActionColumn.STATUS},
                    {_RecoveryActionColumn.RESOURCE_KEY},
                    {_RecoveryActionColumn.REASON},
                    {_RecoveryActionColumn.PAYLOAD},
                    {_RecoveryActionColumn.CREATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_action_record_id,
                    new_action_id,
                    str(attempt_row[_RecoveryColumn.RECOVERY_ID]),
                    batch_id,
                    sequence,
                    new_action_type,
                    status,
                    new_resource_key,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            action_row = _recovery_action_row(connection, new_action_record_id)
            if action_row is None:
                raise RuntimeError("recovery action record did not produce a row")
            return _recovery_action_from_row(action_row)

    def list_recovery_actions(
        self,
        batch_id: str,
        *,
        recovery_attempt_id: str | None = None,
    ) -> list[RecoveryActionRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_recovery_actions_in_connection(
                connection,
                batch_id,
                recovery_attempt_id=recovery_attempt_id,
            )
