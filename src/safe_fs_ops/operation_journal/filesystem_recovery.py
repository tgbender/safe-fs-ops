from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal.captured_directory_recovery import (
    captured_directory_recovery_action_plans as _captured_directory_recovery_action_plans,
)
from safe_fs_ops.operation_journal.file_recovery_planning import (
    file_recovery_action_plans as _file_recovery_action_plans,
)
from safe_fs_ops.operation_journal.file_recovery_planning import (
    file_rollback_manual_intervention_payload as _file_rollback_manual_intervention_payload,
)
from safe_fs_ops.operation_journal.filesystem_support import (
    JournaledFilesystemBatchStateError,
    JournaledFilesystemRecoveryError,
)
from safe_fs_ops.operation_journal.journal import (
    BatchNotFoundError,
    BatchPhase,
    InvalidBatchPhaseTransitionError,
    JournalLeaseMismatchError,
    OperationJournalStore,
    RecoveryAttemptMismatchError,
)
from safe_fs_ops.operation_journal.models import (
    JournaledFilesystemRecoveryContext,
    JournaledFilesystemRecoveryResult,
    RecoveryActionRecord,
    RecoveryActionStatus,
    RecoveryAuthority,
    require_recovery_action_authority,
)
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionHandler,
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
    _latest_current_attempt_recovery_actions,
    _recovery_action_status_payload,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    jsonable_mapping as _jsonable_mapping,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    recursive_mkdir_recovery_action_plans as _recursive_mkdir_recovery_action_plans,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    remove_created_directory_recovery_action,
    remove_empty_directory_recovery_action,
)
from safe_fs_ops.operation_journal.rename_recovery_planning import (
    rename_recovery_action_plans as _rename_recovery_action_plans,
)
from safe_fs_ops.operation_journal.tree_backup_recovery import (
    tree_backup_recovery_action_plans as _tree_backup_recovery_action_plans,
)
from safe_fs_ops.workspace_state.lease_heartbeat import maintain_lease
from safe_fs_ops.workspace_state.leases import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord


class JournaledFilesystemRecoveryMixin:
    _lease_store: LeaseStore
    _journal_store: OperationJournalStore
    _recovery_action_handlers: Mapping[str, RecoveryActionHandler]
    _remove_directory: Callable[..., None]
    _identity_remove_directory: Callable[..., None] | None
    _snapshot: Callable[[Path | str], ResourceSnapshot]

    def recover_batch(
        self,
        batch_id: str,
        *,
        lease: LeaseRecord,
        recover: Callable[[JournaledFilesystemRecoveryContext], object | None],
        now: datetime | None = None,
    ) -> JournaledFilesystemRecoveryResult:
        entry_time = self._operation_time(now)
        self._lease_store.require_current(lease, now=entry_time)
        batch = self._journal_store.get_batch(batch_id)
        if batch is None:
            raise JournaledFilesystemBatchStateError(f"batch {batch_id!r} does not exist")
        if batch.phase != BatchPhase.RECOVERY_DESIRED:
            raise JournaledFilesystemBatchStateError(
                f"batch {batch_id!r} is in phase {batch.phase!r} and cannot be recovered"
            )
        try:
            _started_batch, started_recovery = self._journal_store.start_recovery(
                batch_id,
                lease=lease,
                reason="recovery callback started",
                payload={"batch_id": batch_id},
                now=entry_time,
            )
        except BatchNotFoundError as exc:
            raise JournaledFilesystemBatchStateError(f"batch {batch_id!r} does not exist") from exc
        except InvalidBatchPhaseTransitionError as exc:
            current_batch = self._journal_store.get_batch(batch_id)
            phase = batch.phase if current_batch is None else current_batch.phase
            raise JournaledFilesystemBatchStateError(
                f"batch {batch_id!r} is in phase {phase!r} and cannot be recovered"
            ) from exc
        context = self._journal_store.read_recovery_context(batch_id)
        authority = self._recovery_authority(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=started_recovery.recovery_id,
            now=now,
        )
        context = replace(context, recovery_authority=authority)
        callback_time = self._operation_time(now)
        self._journal_store.require_active_recovery_attempt(
            batch_id,
            lease=lease,
            recovery_attempt_id=started_recovery.recovery_id,
            now=callback_time,
        )
        try:
            authority.require_current()
            callback_result = recover(context)
        except Exception as exc:
            completion_time = self._operation_time(now)
            try:
                failure_record = self._journal_store.record_recovery_failed(
                    batch_id,
                    lease=lease,
                    recovery_attempt_id=started_recovery.recovery_id,
                    reason="recovery callback failed",
                    payload={
                        "batch_id": context.batch.batch_id,
                        "batch_phase": context.batch.phase,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "operations": [
                            {
                                "operation_id": operation.operation_id,
                                "sequence": operation.sequence,
                                "operation_type": operation.operation_type,
                            }
                            for operation in context.operations
                        ],
                        "checkpoints": [
                            {
                                "checkpoint_id": checkpoint.checkpoint_id,
                                "sequence": checkpoint.sequence,
                                "checkpoint_type": checkpoint.checkpoint_type,
                            }
                            for checkpoint in context.checkpoints
                        ],
                        "recovery_records": [
                            {
                                "recovery_id": recovery_record.recovery_id,
                                "sequence": recovery_record.sequence,
                                "phase": recovery_record.phase,
                                "reason": recovery_record.reason,
                            }
                            for recovery_record in context.recovery_records
                        ],
                    },
                    now=completion_time,
                )
            except Exception as recording_exc:
                raise JournaledFilesystemRecoveryError(
                    f"batch {batch_id!r} recovery callback failed",
                    batch_id=batch_id,
                    recording_error=recording_exc,
                ) from exc
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery callback failed",
                batch_id=batch_id,
                recovery_record=failure_record,
            ) from exc
        completion_time = self._operation_time(now)
        recovery_record = self._journal_store.record_recovery_succeeded(
            batch_id,
            lease=lease,
            recovery_attempt_id=started_recovery.recovery_id,
            reason="recovery callback succeeded",
            payload={
                "batch_id": context.batch.batch_id,
                "batch_phase": context.batch.phase,
                "callback_result_type": type(callback_result).__name__,
            },
            now=completion_time,
        )
        updated_batch = self._journal_store.get_batch(batch_id)
        if updated_batch is None:
            raise RuntimeError("batch disappeared during recovery completion")
        return JournaledFilesystemRecoveryResult(
            context=context,
            batch=updated_batch,
            recovery_record=recovery_record,
            callback_result=callback_result,
        )

    def run_recovery_actions(
        self,
        context: JournaledFilesystemRecoveryContext,
        *,
        lease: LeaseRecord,
        now: datetime | None = None,
    ) -> tuple[RecoveryActionRecord, ...]:
        if context.recovery_attempt is None:
            raise JournaledFilesystemRecoveryError(
                f"batch {context.batch.batch_id!r} has no active recovery attempt",
                batch_id=context.batch.batch_id,
            )
        self._lease_store.require_current(lease, now=self._operation_time(now))
        authority = self._recovery_authority(
            batch_id=context.batch.batch_id,
            lease=lease,
            recovery_attempt_id=context.recovery_attempt.recovery_id,
            now=now,
        )
        refreshed_context = replace(
            context,
            recovery_authority=authority,
            recovery_actions=tuple(
                self._journal_store.list_recovery_actions(
                    context.batch.batch_id,
                    recovery_attempt_id=context.recovery_attempt.recovery_id,
                ),
            ),
        )
        refreshed_context = self._plan_recovery_actions(refreshed_context, lease=lease, now=now)
        completed_actions: list[RecoveryActionRecord] = []
        for action in _latest_current_attempt_recovery_actions(refreshed_context):
            authority.require_current()
            completed_actions.append(
                self._run_recovery_action(
                    refreshed_context,
                    action,
                    lease=lease,
                    now=now,
                ),
            )
        return tuple(completed_actions)

    def _run_recovery_action(
        self,
        context: JournaledFilesystemRecoveryContext,
        action: RecoveryActionRecord,
        *,
        lease: LeaseRecord,
        now: datetime | None,
    ) -> RecoveryActionRecord:
        batch_id = context.batch.batch_id
        recovery_attempt_id = context.recovery_attempt_id
        if recovery_attempt_id is None:
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} has no active recovery attempt",
                batch_id=batch_id,
            )
        if action.status in {
            RecoveryActionStatus.SUCCEEDED,
            RecoveryActionStatus.FAILED,
            RecoveryActionStatus.SKIPPED,
            RecoveryActionStatus.MANUAL_INTERVENTION_REQUIRED,
        }:
            return action

        attempt_time = self._operation_time(now)
        self._lease_store.require_current(lease, now=attempt_time)
        current_action = action
        if action.status == RecoveryActionStatus.PLANNED:
            current_action = self._journal_store.mark_recovery_action_attempting(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=action.action_id,
                payload=_recovery_action_status_payload(action, event_type="attempting"),
                now=attempt_time,
            )
        handler_action = _recovery_action_payload_source(context, current_action)
        status_action = handler_action

        if current_action.action_type == "manual_intervention_required":
            reason = _manual_intervention_action_reason(handler_action)
            payload = _recovery_action_status_payload(
                status_action,
                event_type="manual_intervention_required",
                reason=reason,
            )
            payload["manual_intervention"] = _jsonable_mapping(handler_action.payload)
            self._journal_store.record_recovery_action_manual_intervention_required(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=current_action.action_id,
                reason=reason,
                payload=payload,
                now=self._operation_time(now),
            )
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery action {current_action.action_id!r} requires manual intervention",
                batch_id=batch_id,
            ) from RecoveryActionManualInterventionRequired(reason, payload=handler_action.payload)

        handler = self._recovery_action_handlers.get(current_action.action_type)
        if handler is None:
            self._journal_store.record_recovery_action_manual_intervention_required(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=current_action.action_id,
                reason=f"no built-in recovery handler for {current_action.action_type!r}",
                payload=_recovery_action_status_payload(
                    status_action,
                    event_type="manual_intervention_required",
                    reason=f"no built-in recovery handler for {current_action.action_type!r}",
                ),
                now=self._operation_time(now),
            )
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery action {current_action.action_id!r} requires manual intervention",
                batch_id=batch_id,
            ) from RecoveryActionManualInterventionRequired(
                f"no built-in recovery handler for {current_action.action_type!r}",
            )

        try:
            result = self._journal_store.run_with_active_recovery_attempt(
                batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                run=lambda: _run_recovery_handler_with_lease_heartbeat(
                    self._lease_store,
                    lease,
                    enabled=now is None and getattr(self, "_heartbeat_leases", True),
                    now=lambda: self._operation_time(None),
                    handler=lambda: _run_authorized_recovery_handler(
                        context,
                        action=current_action,
                        handler=lambda: handler(context, handler_action),
                    ),
                ),
                now=self._operation_time(now),
            )
        except RecoveryActionSkipped as exc:
            return self._journal_store.record_recovery_action_skipped(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=current_action.action_id,
                reason=str(exc),
                payload=_recovery_action_status_payload(
                    status_action,
                    event_type="skipped",
                    reason=str(exc),
                    exception=exc,
                ),
                now=self._operation_time(now),
            )
        except RecoveryActionManualInterventionRequired as exc:
            self._journal_store.record_recovery_action_manual_intervention_required(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=current_action.action_id,
                reason=str(exc),
                payload=_recovery_action_status_payload(
                    status_action,
                    event_type="manual_intervention_required",
                    reason=str(exc),
                    exception=exc,
                ),
                now=self._operation_time(now),
            )
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery action {current_action.action_id!r} requires manual intervention",
                batch_id=batch_id,
            ) from exc
        except (
            InvalidBatchPhaseTransitionError,
            JournalLeaseMismatchError,
            LeaseLostError,
            RecoveryAttemptMismatchError,
        ) as exc:
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery action {current_action.action_id!r} is no longer active",
                batch_id=batch_id,
            ) from exc
        except Exception as exc:
            self._journal_store.record_recovery_action_failed(
                batch_id=batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_id=current_action.action_id,
                reason=str(exc),
                payload=_recovery_action_status_payload(
                    status_action,
                    event_type="failed",
                    reason=str(exc),
                    exception=exc,
                ),
                now=self._operation_time(now),
            )
            raise JournaledFilesystemRecoveryError(
                f"batch {batch_id!r} recovery action {current_action.action_id!r} failed",
                batch_id=batch_id,
            ) from exc

        completion_time = self._operation_time(now)
        self._lease_store.require_current(lease, now=completion_time)
        return self._journal_store.record_recovery_action_succeeded(
            batch_id=batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            action_id=current_action.action_id,
            payload=_recovery_action_status_payload(
                status_action,
                event_type="succeeded",
                result_type=type(result).__name__,
            ),
            now=completion_time,
        )

    def _recovery_authority(
        self,
        *,
        batch_id: str,
        lease: LeaseRecord,
        recovery_attempt_id: str,
        now: datetime | None,
    ) -> RecoveryAuthority:
        def require_current() -> None:
            check_time = self._operation_time(now)
            self._lease_store.require_current(lease, now=check_time)
            self._journal_store.require_active_recovery_attempt(
                batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                now=check_time,
            )

        def require_action_current(action: RecoveryActionRecord) -> None:
            require_current()
            latest: RecoveryActionRecord | None = None
            for candidate in self._journal_store.list_recovery_actions(
                batch_id,
                recovery_attempt_id=recovery_attempt_id,
            ):
                if candidate.action_id == action.action_id:
                    latest = candidate
            if latest is None or latest.status != RecoveryActionStatus.ATTEMPTING:
                raise RecoveryAttemptMismatchError(
                    f"batch {batch_id!r} recovery action {action.action_id!r} is not actively attempting"
                )

        return RecoveryAuthority(
            require_current,
            recovery_attempt_id=recovery_attempt_id,
            _require_action_current=require_action_current,
        )

    def _plan_recovery_actions(
        self,
        context: JournaledFilesystemRecoveryContext,
        *,
        lease: LeaseRecord,
        now: datetime | None,
    ) -> JournaledFilesystemRecoveryContext:
        recovery_attempt_id = context.recovery_attempt_id
        if recovery_attempt_id is None:
            return context
        existing_action_ids = {
            action.action_id for action in context.recovery_actions if action.recovery_attempt_id == recovery_attempt_id
        }
        file_manual_action = _file_rollback_manual_intervention_payload(context)
        if file_manual_action is not None:
            action_id = str(file_manual_action["action_id"])
            if action_id not in existing_action_ids:
                self._journal_store.record_recovery_action_planned(
                    batch_id=context.batch.batch_id,
                    lease=lease,
                    recovery_attempt_id=recovery_attempt_id,
                    action_type="manual_intervention_required",
                    resource_key=str(file_manual_action["resource_key"]),
                    payload=_jsonable_mapping(file_manual_action["payload"]),
                    action_id=action_id,
                    now=self._operation_time(now),
                )
                self._journal_store.record_recovery_action_manual_intervention_required(
                    batch_id=context.batch.batch_id,
                    lease=lease,
                    recovery_attempt_id=recovery_attempt_id,
                    action_id=action_id,
                    reason="file rollback requires manual intervention",
                    payload=_jsonable_mapping(file_manual_action["payload"]),
                    now=self._operation_time(now),
                )
            raise JournaledFilesystemRecoveryError(
                f"batch {context.batch.batch_id!r} recovery action {action_id!r} requires manual intervention",
                batch_id=context.batch.batch_id,
            )
        planned_any = False
        for action_plan in (
            *_file_recovery_action_plans(context),
            *_tree_backup_recovery_action_plans(context),
            *_captured_directory_recovery_action_plans(context),
            *_rename_recovery_action_plans(context),
            *_recursive_mkdir_recovery_action_plans(context),
        ):
            if str(action_plan["action_id"]) in existing_action_ids:
                continue
            self._journal_store.record_recovery_action_planned(
                batch_id=context.batch.batch_id,
                lease=lease,
                recovery_attempt_id=recovery_attempt_id,
                action_type=str(action_plan["action_type"]),
                resource_key=None if action_plan.get("resource_key") is None else str(action_plan.get("resource_key")),
                payload=_jsonable_mapping(action_plan["payload"]),
                action_id=str(action_plan["action_id"]),
                now=self._operation_time(now),
            )
            planned_any = True
        if not planned_any:
            return context
        return replace(
            context,
            recovery_actions=tuple(
                self._journal_store.list_recovery_actions(
                    context.batch.batch_id,
                    recovery_attempt_id=recovery_attempt_id,
                ),
            ),
        )

    def _remove_empty_directory_recovery_action(
        self,
        context: JournaledFilesystemRecoveryContext,
        action: RecoveryActionRecord,
    ) -> dict[str, object]:
        return remove_empty_directory_recovery_action(
            self._remove_directory,
            self._identity_remove_directory,
            self._snapshot,
            context,
            action,
        )

    def _remove_created_directory_recovery_action(
        self,
        context: JournaledFilesystemRecoveryContext,
        action: RecoveryActionRecord,
    ) -> dict[str, object]:
        return remove_created_directory_recovery_action(
            self._remove_directory,
            self._identity_remove_directory,
            self._snapshot,
            context,
            action,
        )

    def _operation_time(self, now: datetime | None) -> datetime:
        raise NotImplementedError


def _run_recovery_handler_with_lease_heartbeat(
    lease_store: LeaseStore,
    lease: LeaseRecord,
    *,
    enabled: bool,
    now: Callable[[], datetime],
    handler: Callable[[], object],
) -> object:
    if not enabled:
        return handler()
    with maintain_lease(lease_store, lease, now=now):
        return handler()


def _run_authorized_recovery_handler(
    context: JournaledFilesystemRecoveryContext,
    *,
    action: RecoveryActionRecord,
    handler: Callable[[], object],
) -> object:
    require_recovery_action_authority(context, action)
    result = handler()
    require_recovery_action_authority(context, action)
    return result


def _recovery_action_payload_source(
    context: JournaledFilesystemRecoveryContext,
    action: RecoveryActionRecord,
) -> RecoveryActionRecord:
    fallback = action
    for candidate in context.recovery_actions:
        if candidate.recovery_attempt_id != action.recovery_attempt_id:
            continue
        if candidate.action_id != action.action_id:
            continue
        if candidate.status == RecoveryActionStatus.PLANNED:
            return candidate
        fallback = candidate
    return fallback


def _manual_intervention_action_reason(action: RecoveryActionRecord) -> str:
    reason = action.payload.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    detail = action.payload.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    return "recovery action requires manual intervention"
