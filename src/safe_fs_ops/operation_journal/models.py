from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class JournalTarget:
    target_type: str
    subject: str
    address: str
    owner_id: str
    target_id: str | None = None


@dataclass(frozen=True, slots=True)
class FileStateCommit:
    path: Path
    content_text: str
    content_hash: bytes
    size: int
    mtime_ns: int | None
    format: str | None
    original_exists: bool
    operations: tuple[Mapping[str, Any], ...] | Sequence[Mapping[str, Any]]
    event_type: str
    summary: str
    spec_hash: bytes | None = None
    original_text: str | None = None
    record_baseline: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(_freeze_json(operation) for operation in self.operations))


@dataclass(frozen=True, slots=True)
class OperationBatchRecord:
    batch_id: str
    idempotency_key: str
    lease_name: str
    lease_fencing_token: int
    owner: str
    run_id: str
    operation_run_id: str | None
    operation_phase_id: str | None
    resource_key: str | None
    claim_owner: str | None
    claim_scope: str | None
    phase: str
    payload: Mapping[str, Any]
    status_message: str | None
    status_payload: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime
    storage_order: int = 0


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    batch_id: str
    sequence: int
    operation_type: str
    resource_key: str | None
    payload: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class OperationRunRecord:
    operation_run_id: str
    run_id: str
    owner: str
    lease_name: str
    lease_fencing_token: int
    status: str
    payload: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class OperationPhaseRecord:
    operation_phase_id: str
    operation_run_id: str
    phase_name: str
    status: str
    phase_order: int
    payload: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_id: str
    batch_id: str
    sequence: int
    operation_id: str | None
    resource_key: str
    checkpoint_type: str
    payload: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    recovery_id: str
    batch_id: str
    sequence: int
    phase: str
    reason: str | None
    payload: Mapping[str, Any]
    created_at: datetime


class RecoveryActionStatus:
    PLANNED = "planned"
    ATTEMPTING = "attempting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


@dataclass(frozen=True, slots=True)
class RecoveryActionRecord:
    action_record_id: str
    action_id: str
    recovery_attempt_id: str
    batch_id: str
    sequence: int
    action_type: str
    status: str
    resource_key: str | None
    reason: str | None
    payload: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RecoveryAuthority:
    _require_current: Callable[[], None]
    recovery_attempt_id: str | None = None
    _require_action_current: Callable[[RecoveryActionRecord], None] | None = None

    def require_current(self) -> None:
        self._require_current()

    def require_action_current(self, action: RecoveryActionRecord) -> None:
        self.require_current()
        if self._require_action_current is not None:
            self._require_action_current(action)

    def check(self) -> None:
        self.require_current()

    def cancelled(self) -> bool:
        try:
            self.require_current()
        except Exception:
            return True
        return False


class RecoveryActionManualInterventionRequired(RuntimeError):
    def __init__(self, message: str, *, payload: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.payload = payload or {}


class ArtifactCleanupStatus:
    PLANNED = "planned"
    ATTEMPTING = "attempting"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    MANUAL_INTERVENTION_REQUIRED = "manual_intervention_required"


class ArtifactCleanupTrigger:
    DEFERRED_CLEANUP = "deferred_cleanup"
    COMMIT_CLEANUP = "commit_cleanup"
    RECOVERY_CLEANUP = "recovery_cleanup"


ARTIFACT_CLEANUP_ACTIVE_STATUSES = frozenset(
    {
        ArtifactCleanupStatus.PLANNED,
        ArtifactCleanupStatus.ATTEMPTING,
    }
)
ARTIFACT_CLEANUP_DEBT_STATUSES = frozenset(
    {
        ArtifactCleanupStatus.FAILED,
        ArtifactCleanupStatus.MANUAL_INTERVENTION_REQUIRED,
    }
)
ARTIFACT_CLEANUP_OUTSTANDING_STATUSES = frozenset(
    {
        *ARTIFACT_CLEANUP_ACTIVE_STATUSES,
        *ARTIFACT_CLEANUP_DEBT_STATUSES,
    }
)


@dataclass(frozen=True, slots=True)
class ArtifactCleanupRecord:
    cleanup_record_id: str
    artifact_id: str
    batch_id: str
    sequence: int
    trigger: str
    status: str
    resource_key: str | None
    reason: str | None
    payload: Mapping[str, Any]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class JournaledFilesystemRecoveryContext:
    batch: OperationBatchRecord
    operations: tuple[OperationRecord, ...]
    checkpoints: tuple[CheckpointRecord, ...]
    recovery_records: tuple[RecoveryRecord, ...]
    recovery_attempt: RecoveryRecord | None = None
    recovery_actions: tuple[RecoveryActionRecord, ...] = ()
    proof_contexts: MappingABC[str, JournaledFilesystemRecoveryContext] = field(default_factory=dict)
    recovery_authority: RecoveryAuthority | None = None

    @property
    def recovery_attempt_id(self) -> str | None:
        if self.recovery_attempt is None:
            return None
        return self.recovery_attempt.recovery_id

    def require_recovery_authority(self) -> None:
        if self.recovery_authority is None:
            raise RecoveryActionManualInterventionRequired(
                "recovery mutation requires an active recovery authority",
                payload={"reason_code": "missing_recovery_authority"},
            )
        self.recovery_authority.require_current()


def require_recovery_authority(context: object) -> None:
    checker = getattr(context, "require_recovery_authority", None)
    if not callable(checker):
        raise RecoveryActionManualInterventionRequired(
            "recovery mutation requires an active recovery authority",
            payload={"reason_code": "missing_recovery_authority"},
        )
    checker()


def require_recovery_action_authority(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> None:
    context.require_recovery_authority()
    recovery_attempt_id = context.recovery_attempt_id
    if recovery_attempt_id is None:
        raise RecoveryActionManualInterventionRequired(
            "recovery mutation requires an active recovery action",
            payload={"reason_code": "missing_recovery_action_authority"},
        )
    authority = context.recovery_authority
    if authority is None or authority.recovery_attempt_id not in {None, recovery_attempt_id}:
        raise RecoveryActionManualInterventionRequired(
            "recovery mutation requires an active recovery action",
            payload={"reason_code": "recovery_authority_attempt_mismatch"},
        )
    if action.batch_id != context.batch.batch_id or action.recovery_attempt_id != recovery_attempt_id:
        raise RecoveryActionManualInterventionRequired(
            "recovery mutation requires an active recovery action",
            payload={"reason_code": "recovery_action_attempt_mismatch"},
        )
    if authority is not None and authority._require_action_current is not None:
        authority.require_action_current(action)
        return
    latest: RecoveryActionRecord | None = None
    for candidate in context.recovery_actions:
        if candidate.recovery_attempt_id != recovery_attempt_id:
            continue
        if candidate.batch_id != context.batch.batch_id:
            continue
        if candidate.action_id != action.action_id:
            continue
        latest = candidate
    if latest is None or latest.status not in {
        RecoveryActionStatus.PLANNED,
        RecoveryActionStatus.ATTEMPTING,
    }:
        raise RecoveryActionManualInterventionRequired(
            "recovery mutation requires an active recovery action",
            payload={"reason_code": "missing_recovery_action_authority"},
        )


@dataclass(frozen=True, slots=True)
class JournaledFilesystemRecoveryResult:
    context: JournaledFilesystemRecoveryContext
    batch: OperationBatchRecord
    recovery_record: RecoveryRecord
    callback_result: object | None = None


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen_items = {str(key): _freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen_items)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_json(item) for item in value)
    return value
