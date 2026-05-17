from __future__ import annotations

import hashlib
import hmac
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

from safe_fs_ops.operation_journal.journal_errors import (
    BatchIdempotencyMismatchError,
    BatchNotFoundError,
    BatchStartConflictError,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
    RecoveryAttemptMismatchError,
)
from safe_fs_ops.operation_journal.journal_payloads import _optional_str, _payload_to_json
from safe_fs_ops.operation_journal.journal_rows import (
    _batch_has_journal_records,
    _batch_row,
    _latest_recovery_action_for_attempt,
    _latest_recovery_row,
    _next_sequence,
    _recovery_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _BATCH_TABLE,
    _RECOVERY_TABLE,
    _VALID_TRANSITIONS,
    BatchPhase,
    _BatchColumn,
    _RecoveryActionColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import RecoveryActionStatus
from safe_fs_ops.workspace_state.leases import require_current_lease
from safe_fs_ops.workspace_state.models import LeaseRecord


@dataclass(frozen=True, slots=True)
class _ExistingRecoveryAction:
    action_id: str
    action_type: str
    resource_key: str | None
    status: str


_RECOVERY_ACTION_TRANSITIONS = {
    RecoveryActionStatus.PLANNED: frozenset(
        {
            RecoveryActionStatus.ATTEMPTING,
            RecoveryActionStatus.SKIPPED,
            RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED,
        }
    ),
    RecoveryActionStatus.ATTEMPTING: frozenset(
        {
            RecoveryActionStatus.SUCCEEDED,
            RecoveryActionStatus.FAILED,
            RecoveryActionStatus.SKIPPED,
            RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED,
        }
    ),
    RecoveryActionStatus.SUCCEEDED: frozenset[str](),
    RecoveryActionStatus.FAILED: frozenset[str](),
    RecoveryActionStatus.SKIPPED: frozenset[str](),
    RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED: frozenset[str](),
}


def _require_batch_for_write(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> sqlite3.Row:
    require_current_lease(connection, lease, now=now)
    row = _batch_row(connection, batch_id)
    if row is None:
        raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
    if (
        str(row[_BatchColumn.LEASE_NAME]) != lease.name
        or int(row[_BatchColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token
    ):
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    return row


def _require_batch_for_recovery_write(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    lease: LeaseRecord,
    recovery_phase: str,
    now: datetime,
) -> tuple[sqlite3.Row, bool]:
    require_current_lease(connection, lease, now=now)
    row = _batch_row(connection, batch_id)
    if row is None:
        raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    batch_fencing_token = int(row[_BatchColumn.LEASE_FENCING_TOKEN])
    if batch_fencing_token == lease.fencing_token:
        return row, False
    if batch_fencing_token < lease.fencing_token and (
        (
            recovery_phase in {BatchPhase.RECOVERING, BatchPhase.RECOVERY_SUCCEEDED, BatchPhase.RECOVERY_FAILED}
            and str(row[_BatchColumn.PHASE]) == BatchPhase.RECOVERY_DESIRED
        )
        or (
            recovery_phase == BatchPhase.RECOVERY_DESIRED
            and str(row[_BatchColumn.PHASE]) in {BatchPhase.FAILED, BatchPhase.RECOVERING, BatchPhase.RECOVERY_FAILED}
        )
    ):
        _record_recovery_takeover_handoff(connection, row, lease=lease, now=now)
        updated = _batch_row(connection, batch_id)
        if updated is None:
            raise RuntimeError("batch disappeared during recovery takeover")
        return updated, True
    if batch_fencing_token > lease.fencing_token:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a newer lease")
    raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")


def _require_batch_for_recovery_action_write(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> sqlite3.Row:
    require_current_lease(connection, lease, now=now)
    row = _batch_row(connection, batch_id)
    if row is None:
        raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    _require_recovering_batch(row, batch_id=batch_id)
    batch_fencing_token = int(row[_BatchColumn.LEASE_FENCING_TOKEN])
    if batch_fencing_token == lease.fencing_token:
        return row
    if batch_fencing_token < lease.fencing_token:
        _record_recovery_takeover_handoff(connection, row, lease=lease, now=now)
        updated = _batch_row(connection, batch_id)
        if updated is None:
            raise RuntimeError("batch disappeared during recovery takeover")
        return updated
    raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a newer lease")


def _require_batch_for_recovery_attempt_check(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> sqlite3.Row:
    require_current_lease(connection, lease, now=now)
    row = _batch_row(connection, batch_id)
    if row is None:
        raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
    if (
        str(row[_BatchColumn.LEASE_NAME]) != lease.name
        or int(row[_BatchColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token
    ):
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    phase = str(row[_BatchColumn.PHASE])
    if phase != BatchPhase.RECOVERING:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} has no active recovery attempt; current phase is {phase!r}"
        )
    return row


def _record_recovery_takeover_handoff(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    lease: LeaseRecord,
    now: datetime,
) -> None:
    batch_id = str(row[_BatchColumn.BATCH_ID])
    previous_fencing_token = int(row[_BatchColumn.LEASE_FENCING_TOKEN])
    sequence = _next_sequence(connection, batch_id)
    payload_text = _payload_to_json(
        {
            "event_type": "recovery_takeover",
            "previous_lease": {
                "name": str(row[_BatchColumn.LEASE_NAME]),
                "fencing_token": previous_fencing_token,
            },
            "current_lease": {
                "name": lease.name,
                "fencing_token": lease.fencing_token,
            },
            "previous_phase": str(row[_BatchColumn.PHASE]),
        }
    )
    connection.execute(
        f"""
        UPDATE {_BATCH_TABLE}
        SET {_BatchColumn.LEASE_FENCING_TOKEN} = ?,
            {_BatchColumn.LEASE_OWNER} = ?,
            {_BatchColumn.LEASE_TOKEN_DIGEST} = ?,
            {_BatchColumn.UPDATED_AT} = ?
        WHERE {_BatchColumn.BATCH_ID} = ?
        """,
        (lease.fencing_token, lease.owner, _lease_token_digest(lease), now.isoformat(), batch_id),
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
            uuid.uuid4().hex,
            batch_id,
            sequence,
            BatchPhase.RECOVERY_DESIRED,
            "recovery takeover",
            payload_text,
            now.isoformat(),
        ),
    )


def _require_active_recovery_attempt(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    recovery_attempt_id: str | None,
) -> sqlite3.Row:
    if not recovery_attempt_id:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery completion requires the active recovery_attempt_id"
        )
    attempt_row = _recovery_row(connection, recovery_attempt_id)
    if attempt_row is None or str(attempt_row[_RecoveryColumn.BATCH_ID]) != batch_id:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery attempt {recovery_attempt_id!r} does not belong to the batch"
        )
    if str(attempt_row[_RecoveryColumn.PHASE]) != BatchPhase.RECOVERING:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery attempt {recovery_attempt_id!r} is not recoverable"
        )
    latest_attempt_row = _latest_recovery_row(connection, batch_id)
    if (
        latest_attempt_row is None
        or str(latest_attempt_row[_RecoveryColumn.RECOVERY_ID]) != recovery_attempt_id
        or str(latest_attempt_row[_RecoveryColumn.PHASE]) != BatchPhase.RECOVERING
    ):
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery attempt {recovery_attempt_id!r} is not the active recovery attempt"
        )
    return attempt_row


def _prepare_recovery_action_write(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    recovery_attempt_id: str,
    status: str,
    action_id: str | None,
    action_type: str | None,
    resource_key: str | None,
) -> _ExistingRecoveryAction | None:
    if status == RecoveryActionStatus.PLANNED:
        if action_id is not None:
            existing_row = _latest_recovery_action_for_attempt(connection, recovery_attempt_id, action_id)
            if existing_row is not None:
                raise InvalidBatchPhaseTransitionError(
                    f"recovery action {action_id!r} already exists for batch {batch_id!r} "
                    f"attempt {recovery_attempt_id!r}"
                )
        if action_type is None:
            raise ValueError("action_type must be provided for planned recovery actions")
        return None
    if not action_id:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery action transition to {status!r} requires a planned action_id"
        )
    existing_row = _latest_recovery_action_for_attempt(connection, recovery_attempt_id, action_id)
    if existing_row is None or str(existing_row[_RecoveryActionColumn.BATCH_ID]) != batch_id:
        raise RecoveryAttemptMismatchError(
            f"batch {batch_id!r} recovery action {action_id!r} does not belong to the active recovery attempt"
        )
    existing = _ExistingRecoveryAction(
        action_id=str(existing_row[_RecoveryActionColumn.ACTION_ID]),
        action_type=str(existing_row[_RecoveryActionColumn.ACTION_TYPE]),
        resource_key=_optional_str(existing_row[_RecoveryActionColumn.RESOURCE_KEY]),
        status=str(existing_row[_RecoveryActionColumn.STATUS]),
    )
    if action_type is not None and action_type != existing.action_type:
        raise BatchIdempotencyMismatchError("recovery action transition requested for a different action type")
    if resource_key is not None and resource_key != existing.resource_key:
        raise BatchIdempotencyMismatchError("recovery action transition requested for a different resource")
    allowed = _RECOVERY_ACTION_TRANSITIONS.get(existing.status)
    if allowed is None or status not in allowed:
        raise InvalidBatchPhaseTransitionError(
            f"cannot transition recovery action {action_id!r} from {existing.status!r} to {status!r}"
        )
    return existing


def _require_recovering_batch(row: sqlite3.Row, *, batch_id: str) -> None:
    phase = str(row[_BatchColumn.PHASE])
    if phase != BatchPhase.RECOVERING:
        raise InvalidBatchPhaseTransitionError(
            f"cannot append recovery actions while batch {batch_id!r} is in phase {phase!r}"
        )


def _require_matching_idempotent_batch(
    row: sqlite3.Row,
    *,
    lease: LeaseRecord,
    owner: str,
    run_id: str,
    operation_run_id: str | None,
    operation_phase_id: str | None,
    resource_key: str | None,
    claim_owner: str | None,
    claim_scope: str | None,
    payload_text: str,
) -> None:
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise BatchIdempotencyMismatchError("idempotency key is bound to a different lease name")
    comparisons = {
        "owner": (_optional_str(row[_BatchColumn.OWNER]), owner),
        "run_id": (_optional_str(row[_BatchColumn.RUN_ID]), run_id),
        "operation_run_id": (_optional_str(row[_BatchColumn.OPERATION_RUN_ID]), operation_run_id),
        "operation_phase_id": (_optional_str(row[_BatchColumn.OPERATION_PHASE_ID]), operation_phase_id),
        "resource_key": (_optional_str(row[_BatchColumn.RESOURCE_KEY]), resource_key),
        "claim_owner": (_optional_str(row[_BatchColumn.CLAIM_OWNER]), claim_owner),
        "claim_scope": (_optional_str(row[_BatchColumn.CLAIM_SCOPE]), claim_scope),
        "payload": (str(row[_BatchColumn.PAYLOAD]), payload_text),
    }
    mismatches = [name for name, (existing, requested) in comparisons.items() if existing != requested]
    if mismatches:
        joined = ", ".join(mismatches)
        raise BatchIdempotencyMismatchError(f"idempotency key is already bound to a different request: {joined}")


def _require_appendable_phase(phase: str) -> None:
    if phase not in {BatchPhase.PLANNED, BatchPhase.ATTEMPTING}:
        raise InvalidBatchPhaseTransitionError(f"cannot append journal records while batch is in phase {phase!r}")


def _require_startable_batch(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    lease: LeaseRecord,
    resource_key: str | None,
) -> None:
    batch_id = str(row[_BatchColumn.BATCH_ID])
    phase = str(row[_BatchColumn.PHASE])
    if phase != BatchPhase.PLANNED:
        raise BatchStartConflictError(f"batch {batch_id!r} is already in phase {phase!r}")
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    if int(row[_BatchColumn.LEASE_FENCING_TOKEN]) > lease.fencing_token:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a newer lease")
    stored_resource_key = _optional_str(row[_BatchColumn.RESOURCE_KEY])
    if stored_resource_key is not None and resource_key != stored_resource_key:
        raise BatchIdempotencyMismatchError("batch start requested for a different resource")
    if _batch_has_journal_records(connection, batch_id):
        raise BatchStartConflictError(f"batch {batch_id!r} already has journal records")


def _require_batch_resource_key(row: sqlite3.Row, resource_key: str | None, *, action: str) -> None:
    stored_resource_key = _optional_str(row[_BatchColumn.RESOURCE_KEY])
    if stored_resource_key is not None and resource_key != stored_resource_key:
        raise BatchIdempotencyMismatchError(f"{action} requested for a different resource")


def _require_recovery_takeover(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    lease: LeaseRecord,
    resource_key: str | None,
) -> None:
    phase = str(row[_BatchColumn.PHASE])
    if phase not in {BatchPhase.PLANNED, BatchPhase.ATTEMPTING, BatchPhase.FAILED, BatchPhase.RECOVERY_FAILED}:
        raise InvalidBatchPhaseTransitionError(f"cannot interrupt batch from phase {phase!r}")
    if str(row[_BatchColumn.LEASE_NAME]) != lease.name:
        raise JournalLeaseMismatchError(f"batch {row[_BatchColumn.BATCH_ID]!r} is bound to a different lease")
    if int(row[_BatchColumn.LEASE_FENCING_TOKEN]) >= lease.fencing_token:
        raise JournalLeaseMismatchError(f"batch {row[_BatchColumn.BATCH_ID]!r} requires a newer lease to interrupt")
    stored_resource_key = _optional_str(row[_BatchColumn.RESOURCE_KEY])
    if stored_resource_key is not None and resource_key != stored_resource_key:
        raise BatchIdempotencyMismatchError("interrupted recovery requested for a different resource")
    if phase == BatchPhase.PLANNED and not _batch_has_journal_records(connection, str(row[_BatchColumn.BATCH_ID])):
        raise InvalidBatchPhaseTransitionError("empty planned batch can be restarted without recovery")


def _require_lease_lost_recovery_marker(
    row: sqlite3.Row,
    *,
    lease: LeaseRecord,
    resource_key: str | None,
) -> None:
    batch_id = str(row[_BatchColumn.BATCH_ID])
    if (
        str(row[_BatchColumn.LEASE_NAME]) != lease.name
        or int(row[_BatchColumn.LEASE_FENCING_TOKEN]) != lease.fencing_token
    ):
        raise JournalLeaseMismatchError(f"batch {batch_id!r} is bound to a different lease")
    stored_lease_owner = _optional_str(row[_BatchColumn.LEASE_OWNER])
    stored_token_digest = _optional_str(row[_BatchColumn.LEASE_TOKEN_DIGEST])
    if not lease.token or stored_lease_owner is None or stored_token_digest is None:
        raise JournalLeaseMismatchError(f"batch {batch_id!r} has no lease-lost marker authority")
    if stored_lease_owner != lease.owner or not hmac.compare_digest(stored_token_digest, _lease_token_digest(lease)):
        raise JournalLeaseMismatchError(f"batch {batch_id!r} lease-lost marker authority does not match")
    stored_resource_key = _optional_str(row[_BatchColumn.RESOURCE_KEY])
    if stored_resource_key is not None and resource_key != stored_resource_key:
        raise BatchIdempotencyMismatchError("lease-lost recovery marker requested for a different resource")
    phase = str(row[_BatchColumn.PHASE])
    if phase not in {BatchPhase.ATTEMPTING, BatchPhase.FAILED, BatchPhase.RECOVERY_FAILED}:
        raise InvalidBatchPhaseTransitionError(f"cannot record lease-lost recovery marker from phase {phase!r}")


def _lease_token_digest(lease: LeaseRecord) -> str:
    return hashlib.sha256(lease.token.encode("utf-8")).hexdigest()


def _require_valid_transition(current_phase: str, next_phase: str) -> None:
    allowed = _VALID_TRANSITIONS.get(current_phase)
    if allowed is None or next_phase not in allowed:
        raise InvalidBatchPhaseTransitionError(f"cannot transition batch from {current_phase!r} to {next_phase!r}")
