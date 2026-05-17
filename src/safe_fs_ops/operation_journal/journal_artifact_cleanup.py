from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeVar, cast

from safe_fs_ops.operation_journal.journal_errors import (
    BatchIdempotencyMismatchError,
    BatchNotFoundError,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
)
from safe_fs_ops.operation_journal.journal_payloads import (
    _optional_str,
    _payload_to_json,
    _utcnow,
    _validate_required,
)
from safe_fs_ops.operation_journal.journal_rows import (
    _artifact_cleanup_from_row,
    _artifact_cleanup_row,
    _batch_row,
    _latest_artifact_cleanup_row,
    _list_artifact_cleanup_records_in_connection,
    _list_outstanding_artifact_cleanup_records_in_connection,
    _list_unresolved_artifact_cleanup_debt_in_connection,
    _next_sequence,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _ARTIFACT_CLEANUP_TABLE,
    BatchPhase,
    _ArtifactCleanupColumn,
    _BatchColumn,
)
from safe_fs_ops.operation_journal.models import (
    ARTIFACT_CLEANUP_ACTIVE_STATUSES,
    ArtifactCleanupRecord,
    ArtifactCleanupStatus,
    ArtifactCleanupTrigger,
)
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _ExistingArtifactCleanup:
    artifact_id: str
    trigger: str
    resource_key: str | None
    status: str


_ARTIFACT_CLEANUP_TRANSITIONS = {
    ArtifactCleanupStatus.PLANNED: frozenset(
        {
            ArtifactCleanupStatus.ATTEMPTING,
            ArtifactCleanupStatus.SUCCEEDED,
            ArtifactCleanupStatus.SKIPPED,
            ArtifactCleanupStatus.FAILED,
            ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
        }
    ),
    ArtifactCleanupStatus.ATTEMPTING: frozenset(
        {
            ArtifactCleanupStatus.SUCCEEDED,
            ArtifactCleanupStatus.SKIPPED,
            ArtifactCleanupStatus.FAILED,
            ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
        }
    ),
    ArtifactCleanupStatus.SUCCEEDED: frozenset[str](),
    ArtifactCleanupStatus.SKIPPED: frozenset[str](),
    ArtifactCleanupStatus.FAILED: frozenset[str](),
    ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED: frozenset[str](),
}

_BATCH_PHASE_BY_TRIGGER = {
    ArtifactCleanupTrigger.DEFERRED_CLEANUP: BatchPhase.ATTEMPTING,
    ArtifactCleanupTrigger.COMMIT_CLEANUP: BatchPhase.SUCCEEDED,
    ArtifactCleanupTrigger.RECOVERY_CLEANUP: BatchPhase.RECOVERY_SUCCEEDED,
}

_ARTIFACT_CLEANUP_RETRYABLE_STATUSES = frozenset(
    {
        ArtifactCleanupStatus.FAILED,
        ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
    }
)


class JournalArtifactCleanupMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def record_artifact_cleanup_planned(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        trigger: str,
        resource_key: str | None = None,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        _validate_required("trigger", trigger)
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.PLANNED,
            trigger=trigger,
            resource_key=resource_key,
            reason=None,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def mark_artifact_cleanup_attempting(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.ATTEMPTING,
            trigger=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def record_artifact_cleanup_succeeded(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.SUCCEEDED,
            trigger=None,
            resource_key=None,
            reason=None,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def record_artifact_cleanup_skipped(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.SKIPPED,
            trigger=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def record_artifact_cleanup_failed(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.FAILED,
            trigger=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def record_artifact_cleanup_manual_intervention_required(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        cleanup_record_id: str | None = None,
        now: datetime | None = None,
    ) -> ArtifactCleanupRecord:
        return self._record_artifact_cleanup(
            batch_id=batch_id,
            lease=lease,
            artifact_id=artifact_id,
            status=ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
            trigger=None,
            resource_key=None,
            reason=reason,
            payload=payload,
            cleanup_record_id=cleanup_record_id,
            now=now,
        )

    def list_artifact_cleanup_records(
        self,
        batch_id: str,
        *,
        artifact_id: str | None = None,
    ) -> list[ArtifactCleanupRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_artifact_cleanup_records_in_connection(
                connection,
                batch_id,
                artifact_id=artifact_id,
            )

    def list_unresolved_artifact_cleanup_debt(self) -> list[ArtifactCleanupRecord]:
        """Return latest artifact cleanup rows whose current state is failed/manual debt."""
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_unresolved_artifact_cleanup_debt_in_connection(connection)

    def list_outstanding_artifact_cleanup_records(self) -> list[ArtifactCleanupRecord]:
        """Return latest artifact cleanup rows that still need attention or execution."""
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            return _list_outstanding_artifact_cleanup_records_in_connection(connection)

    def _record_artifact_cleanup(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        artifact_id: str,
        status: str,
        trigger: str | None,
        resource_key: str | None,
        reason: str | None,
        payload: Mapping[str, Any] | None,
        cleanup_record_id: str | None,
        now: datetime | None,
    ) -> ArtifactCleanupRecord:
        _validate_required("batch_id", batch_id)
        _validate_required("artifact_id", artifact_id)
        current_time = _utcnow(now)
        payload_text = _payload_to_json(payload)
        self.initialize()
        with self.sqlite_store.transaction() as connection:
            row = _require_batch_for_artifact_cleanup_write(
                connection,
                batch_id=batch_id,
                lease=lease,
                now=current_time,
            )
            existing = _prepare_artifact_cleanup_write(
                connection,
                batch_id=batch_id,
                artifact_id=artifact_id,
                status=status,
                trigger=trigger,
                resource_key=resource_key,
            )
            sequence = _next_sequence(connection, batch_id)
            new_cleanup_record_id = cleanup_record_id or uuid.uuid4().hex
            new_trigger = (
                cast(str, trigger)
                if (
                    status == ArtifactCleanupStatus.PLANNED
                    and existing is not None
                    and existing.status in _ARTIFACT_CLEANUP_RETRYABLE_STATUSES
                )
                else existing.trigger
                if existing is not None
                else cast(str, trigger)
            )
            new_resource_key = existing.resource_key if existing is not None else resource_key
            _require_batch_phase_for_trigger(
                row,
                batch_id=batch_id,
                trigger=new_trigger,
                status=status,
            )
            _require_batch_resource_key(row, resource_key=new_resource_key, action="artifact cleanup append")
            connection.execute(
                f"""
                INSERT INTO {_ARTIFACT_CLEANUP_TABLE} (
                    {_ArtifactCleanupColumn.CLEANUP_RECORD_ID},
                    {_ArtifactCleanupColumn.ARTIFACT_ID},
                    {_ArtifactCleanupColumn.BATCH_ID},
                    {_ArtifactCleanupColumn.SEQUENCE},
                    {_ArtifactCleanupColumn.TRIGGER},
                    {_ArtifactCleanupColumn.STATUS},
                    {_ArtifactCleanupColumn.RESOURCE_KEY},
                    {_ArtifactCleanupColumn.REASON},
                    {_ArtifactCleanupColumn.PAYLOAD},
                    {_ArtifactCleanupColumn.CREATED_AT}
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_cleanup_record_id,
                    artifact_id,
                    batch_id,
                    sequence,
                    new_trigger,
                    status,
                    new_resource_key,
                    reason,
                    payload_text,
                    current_time.isoformat(),
                ),
            )
            cleanup_row = _artifact_cleanup_row(connection, new_cleanup_record_id)
            if cleanup_row is None:
                raise RuntimeError("artifact cleanup record did not produce a row")
            return _artifact_cleanup_from_row(cleanup_row)


def _require_batch_for_artifact_cleanup_write(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    lease: LeaseRecord,
    now: datetime,
) -> sqlite3.Row:
    require_current_lease(connection, lease, now=now)
    row = _batch_row(connection, batch_id)
    if row is None:
        raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    batch_fencing_token = int(row[_BatchColumn.LEASE_FENCING_TOKEN])
    if batch_fencing_token == lease.fencing_token:
        return row
    if batch_fencing_token < lease.fencing_token and str(row[_BatchColumn.PHASE]) in {
        BatchPhase.SUCCEEDED,
        BatchPhase.RECOVERY_SUCCEEDED,
    }:
        return row
    if batch_fencing_token > lease.fencing_token:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a newer lease")
    raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")


def _prepare_artifact_cleanup_write(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    artifact_id: str,
    status: str,
    trigger: str | None,
    resource_key: str | None,
) -> _ExistingArtifactCleanup | None:
    _require_valid_artifact_cleanup_status(status)
    existing_row = _latest_artifact_cleanup_row(connection, batch_id, artifact_id)
    if status == ArtifactCleanupStatus.PLANNED:
        _require_valid_artifact_cleanup_trigger(cast(str, trigger))
        if existing_row is None:
            return None
        existing = _existing_artifact_cleanup_from_row(existing_row)
        if resource_key is not None and resource_key != existing.resource_key:
            raise BatchIdempotencyMismatchError("artifact cleanup transition requested for a different resource")
        if existing.status in ARTIFACT_CLEANUP_ACTIVE_STATUSES:
            raise InvalidBatchPhaseTransitionError(
                f"artifact cleanup {artifact_id!r} already has active work for batch {batch_id!r}"
            )
        if existing.status in {ArtifactCleanupStatus.SUCCEEDED, ArtifactCleanupStatus.SKIPPED}:
            raise InvalidBatchPhaseTransitionError(
                f"cannot transition artifact cleanup {artifact_id!r} from {existing.status!r} to {status!r}"
            )
        if existing.status in _ARTIFACT_CLEANUP_RETRYABLE_STATUSES:
            return existing
        if trigger != existing.trigger:
            raise BatchIdempotencyMismatchError("artifact cleanup transition requested for a different trigger")
        return existing
    if existing_row is None:
        raise InvalidBatchPhaseTransitionError(
            f"artifact cleanup {artifact_id!r} does not exist for batch {batch_id!r}"
        )
    existing = _existing_artifact_cleanup_from_row(existing_row)
    if trigger is not None and trigger != existing.trigger:
        raise BatchIdempotencyMismatchError("artifact cleanup transition requested for a different trigger")
    if resource_key is not None and resource_key != existing.resource_key:
        raise BatchIdempotencyMismatchError("artifact cleanup transition requested for a different resource")
    allowed = _ARTIFACT_CLEANUP_TRANSITIONS.get(existing.status)
    if allowed is None or status not in allowed:
        raise InvalidBatchPhaseTransitionError(
            f"cannot transition artifact cleanup {artifact_id!r} from {existing.status!r} to {status!r}"
        )
    return existing


def _require_batch_phase_for_trigger(
    row: sqlite3.Row,
    *,
    batch_id: str,
    trigger: str,
    status: str,
) -> None:
    if status != ArtifactCleanupStatus.PLANNED:
        return
    expected_phase = _BATCH_PHASE_BY_TRIGGER.get(trigger)
    if expected_phase is None:
        raise ValueError(f"unsupported artifact cleanup trigger {trigger!r}")
    phase = str(row[_BatchColumn.PHASE])
    if phase != expected_phase:
        raise InvalidBatchPhaseTransitionError(
            f"cannot append artifact cleanup records for trigger {trigger!r} while batch {batch_id!r} "
            f"is in phase {phase!r}"
        )


def _require_batch_resource_key(row: sqlite3.Row, *, resource_key: str | None, action: str) -> None:
    stored_resource_key = _optional_str(row[_BatchColumn.RESOURCE_KEY])
    if stored_resource_key is not None and resource_key != stored_resource_key:
        raise BatchIdempotencyMismatchError(f"{action} requested for a different resource")


def _require_valid_artifact_cleanup_status(status: str) -> None:
    if status not in _ARTIFACT_CLEANUP_TRANSITIONS:
        raise ValueError(f"unsupported artifact cleanup status {status!r}")


def _require_valid_artifact_cleanup_trigger(trigger: str) -> None:
    if trigger not in _BATCH_PHASE_BY_TRIGGER:
        raise ValueError(f"unsupported artifact cleanup trigger {trigger!r}")


def _existing_artifact_cleanup_from_row(row: sqlite3.Row) -> _ExistingArtifactCleanup:
    return _ExistingArtifactCleanup(
        artifact_id=str(row[_ArtifactCleanupColumn.ARTIFACT_ID]),
        trigger=str(row[_ArtifactCleanupColumn.TRIGGER]),
        resource_key=_optional_str(row[_ArtifactCleanupColumn.RESOURCE_KEY]),
        status=str(row[_ArtifactCleanupColumn.STATUS]),
    )
