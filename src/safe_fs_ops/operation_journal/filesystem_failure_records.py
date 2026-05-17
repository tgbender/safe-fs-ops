from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal.filesystem_support import _snapshot_payload
from safe_fs_ops.operation_journal.journal import BatchPhase, OperationJournalStore
from safe_fs_ops.operation_journal.models import OperationBatchRecord, RecoveryRecord
from safe_fs_ops.workspace_state.leases import LeaseLostError
from safe_fs_ops.workspace_state.models import LeaseRecord


class JournaledFilesystemFailureRecordingMixin:
    _journal_store: OperationJournalStore
    _snapshot: Callable[[Path | str], ResourceSnapshot]

    def _record_interrupted_if_taken_over(
        self,
        batch: OperationBatchRecord,
        *,
        lease: LeaseRecord,
        resource_key: str,
        reason: str,
        now: datetime | None,
    ) -> None:
        if batch.lease_name != lease.name or batch.lease_fencing_token >= lease.fencing_token:
            return
        if batch.phase not in {BatchPhase.PLANNED, BatchPhase.ATTEMPTING}:
            return
        self._journal_store.record_interrupted_recovery_desired(
            batch.batch_id,
            lease=lease,
            resource_key=resource_key,
            reason=reason,
            payload={
                "resource_key": resource_key,
                "previous_lease": {
                    "name": batch.lease_name,
                    "fencing_token": batch.lease_fencing_token,
                },
                "current_lease": {
                    "name": lease.name,
                    "fencing_token": lease.fencing_token,
                },
            },
            now=self._operation_time(now),
        )

    def _record_mutation_failure(
        self,
        batch_id: str,
        path: Path,
        *,
        resource_key: str,
        lease: LeaseRecord,
        before_snapshot: ResourceSnapshot,
        recovery_payload: Mapping[str, object],
        error: Exception,
        stage: str | None = None,
        now: datetime | None,
    ) -> RecoveryRecord:
        failure_payload: dict[str, object] = {
            "path": str(path),
            "resource_key": resource_key,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if stage is not None:
            failure_payload["stage"] = stage
        failure_snapshot_payload: dict[str, object] | None = None
        try:
            failure_snapshot = self._snapshot(path)
        except Exception as snapshot_error:
            failure_payload["snapshot_error"] = {
                "error_type": type(snapshot_error).__name__,
                "error": str(snapshot_error),
            }
        else:
            failure_snapshot_payload = _snapshot_payload(failure_snapshot)
            failure_payload["snapshot"] = failure_snapshot_payload
        recovery_details = {
            **dict(recovery_payload),
            "resource_key": resource_key,
            "before": _snapshot_payload(before_snapshot),
            "failure": failure_payload,
        }

        def record_lease_lost_recovery_desired() -> RecoveryRecord:
            return self._journal_store.record_lease_lost_recovery_desired(
                batch_id,
                lease=lease,
                resource_key=resource_key,
                reason="filesystem mutation failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )

        if failure_snapshot_payload is not None:
            try:
                self._journal_store.record_checkpoint(
                    batch_id,
                    lease=lease,
                    resource_key=resource_key,
                    checkpoint_type="failure",
                    payload=failure_snapshot_payload,
                    now=self._operation_time(now),
                )
            except LeaseLostError:
                return record_lease_lost_recovery_desired()
        try:
            self._journal_store.mark_failed(
                batch_id,
                lease=lease,
                error=str(error),
                observed_state=failure_payload,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return record_lease_lost_recovery_desired()
        try:
            return self._journal_store.record_recovery_desired(
                batch_id,
                lease=lease,
                reason="filesystem mutation failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return record_lease_lost_recovery_desired()

    def _record_before_snapshot_failure(
        self,
        batch_id: str,
        path: Path,
        *,
        resource_key: str,
        lease: LeaseRecord,
        recovery_payload: Mapping[str, object],
        error: Exception,
        now: datetime | None,
    ) -> RecoveryRecord:
        failure_payload: dict[str, object] = {
            "path": str(path),
            "resource_key": resource_key,
            "stage": "before_snapshot",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        recovery_details = {
            **dict(recovery_payload),
            "resource_key": resource_key,
            "failure": failure_payload,
        }

        def record_lease_lost_recovery_desired() -> RecoveryRecord:
            return self._journal_store.record_lease_lost_recovery_desired(
                batch_id,
                lease=lease,
                resource_key=resource_key,
                reason="filesystem before-snapshot failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )

        try:
            self._journal_store.mark_failed(
                batch_id,
                lease=lease,
                error=str(error),
                observed_state=failure_payload,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return record_lease_lost_recovery_desired()
        try:
            return self._journal_store.record_recovery_desired(
                batch_id,
                lease=lease,
                reason="filesystem before-snapshot failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return record_lease_lost_recovery_desired()

    def _record_post_mutation_failure(
        self,
        batch_id: str,
        path: Path,
        *,
        resource_key: str,
        lease: LeaseRecord,
        before_snapshot: ResourceSnapshot,
        after_snapshot: ResourceSnapshot | None,
        recovery_payload: Mapping[str, object],
        stage: str,
        error: Exception,
        now: datetime | None,
    ) -> RecoveryRecord:
        failure_payload: dict[str, object] = {
            "path": str(path),
            "resource_key": resource_key,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if after_snapshot is not None:
            failure_payload["after"] = _snapshot_payload(after_snapshot)
        else:
            try:
                observed_after = self._snapshot(path)
            except Exception as snapshot_error:
                failure_payload["snapshot_error"] = {
                    "error_type": type(snapshot_error).__name__,
                    "error": str(snapshot_error),
                }
            else:
                failure_payload["after"] = _snapshot_payload(observed_after)

        recovery_details = {
            **dict(recovery_payload),
            "resource_key": resource_key,
            "before": _snapshot_payload(before_snapshot),
            "failure": failure_payload,
        }
        try:
            self._journal_store.mark_failed(
                batch_id,
                lease=lease,
                error=f"filesystem post-mutation journal failed during {stage}",
                observed_state=failure_payload,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return self._journal_store.record_lease_lost_recovery_desired(
                batch_id,
                lease=lease,
                resource_key=resource_key,
                reason="filesystem post-mutation journal failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )
        try:
            return self._journal_store.record_recovery_desired(
                batch_id,
                lease=lease,
                reason="filesystem post-mutation journal failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            return self._journal_store.record_lease_lost_recovery_desired(
                batch_id,
                lease=lease,
                resource_key=resource_key,
                reason="filesystem post-mutation journal failed",
                payload=recovery_details,
                now=self._operation_time(now),
            )

    def _operation_time(self, now: datetime | None) -> datetime:
        raise NotImplementedError
