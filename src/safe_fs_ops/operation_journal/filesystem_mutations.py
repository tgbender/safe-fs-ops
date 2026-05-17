from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    ContentAddressedPutResult,
    ContentAddressedStore,
    ContentRef,
    RenameRecord,
    ResourceSnapshot,
    SnapshotBundle,
)
from safe_fs_ops.filesystem_ops.backend import UnsupportedFilesystemBackendError
from safe_fs_ops.filesystem_ops.paths import absolute_without_resolving
from safe_fs_ops.filesystem_ops.renames import RenamedFileType
from safe_fs_ops.operation_journal.filesystem_failure_records import (
    JournaledFilesystemFailureRecordingMixin,
)
from safe_fs_ops.operation_journal.filesystem_mutation_checkpoints import (
    prepare_mutation_checkpoints,
    record_after_mutation_checkpoint,
)
from safe_fs_ops.operation_journal.filesystem_support import (
    JournaledFilesystemBatchStateError,
    JournaledFilesystemMutationError,
    JournaledFilesystemResult,
    MissingResourceClaimError,
    ResourceClaimAuthorityError,
    _canonical_directory_path,
    _canonical_file_path,
    _normalized_text,
    _require_matching_directory_resource_key,
    _require_matching_file_resource_key,
    _require_matching_tree_resource_key,
    _snapshot_payload,
)
from safe_fs_ops.operation_journal.journal import (
    BatchPhase,
    BatchStartConflictError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.models import ArtifactCleanupTrigger, CheckpointRecord
from safe_fs_ops.operation_journal.rename_recovery_actions import rename_record_payload
from safe_fs_ops.operation_journal.tree_backup_recovery import (
    RestoreTreeBackupOperation,
    TreeBackupOperation,
    default_tree_backup_store_path,
    require_tree_backup_payload_resource_key,
    run_backup_tree_operation,
    run_restore_tree_backup_operation,
    tree_backup_checkpoint_fingerprint,
    tree_backup_checkpoint_payload,
    tree_backup_partial_artifacts_checkpoint_payload,
)
from safe_fs_ops.workspace_state.claims import ClaimStore, _lease_token_digest
from safe_fs_ops.workspace_state.lease_heartbeat import maintain_lease
from safe_fs_ops.workspace_state.leases import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import ClaimRecord, LeaseRecord


class JournaledFilesystemMutationMixin(JournaledFilesystemFailureRecordingMixin):
    _lease_store: LeaseStore
    _claim_store: ClaimStore
    _journal_store: OperationJournalStore
    _snapshot: Callable[[Path | str], ResourceSnapshot]
    _write_bytes: Callable[..., None]
    _write_text: Callable[..., None]
    _delete_file: Callable[..., None]
    _make_directory: Callable[..., None]
    _rename_no_replace: Callable[..., RenameRecord]
    _capture_directory: Callable[..., CapturedDirectoryRecord]
    _preflight_operation: Callable[[str], None] | None
    _automatic_file_rollback: bool
    _write_bytes_max_bytes: int
    _write_bytes_large_policy: Literal["reject", "allow"]
    _captured_directory_cleanup: Literal["automatic", "retain"]

    def backup_tree(
        self,
        path: Path | str,
        relative_paths: Iterable[Path | str] = (),
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        artifact_store: object | None = None,
        store_path: Path | str | None = None,
        backup_tree_operation: TreeBackupOperation | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_directory_path(path)
        _require_matching_tree_resource_key(target, resource_key)
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        resolved_store_path = (
            default_tree_backup_store_path(Path(self._journal_store.path)) if store_path is None else Path(store_path)
        )
        resolved_artifact_store = _tree_artifact_store(
            artifact_store,
            default_store_path=resolved_store_path,
        )
        checkpoint_store_path = _require_artifact_store_path(resolved_artifact_store)
        requested_relative_paths = tuple(relative_paths)
        intent_payload = {
            "operation": "backup_tree",
            "path": str(target),
            "relative_paths": [str(relative_path) for relative_path in requested_relative_paths],
            "resource_key": resource_key,
            "store_path": str(checkpoint_store_path),
        }
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload=intent_payload,
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            self._record_interrupted_if_taken_over(
                batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch outside planned phase",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot record a tree backup"
            )
        try:
            batch, operation = self._journal_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type="backup_tree",
                resource_key=resource_key,
                payload=intent_payload,
                now=self._operation_time(now),
            )
        except BatchStartConflictError as exc:
            current_batch = self._journal_store.get_batch(batch.batch_id) or batch
            self._record_interrupted_if_taken_over(
                current_batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch with journal records",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} cannot record a tree backup: {exc}"
            ) from exc
        tree_backup_operation = backup_tree_operation or _default_backup_tree_operation()
        partial_artifacts = _TreeBackupPartialArtifactTracker(
            self._journal_store,
            batch_id=batch.batch_id,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            lease=lease,
            root=target,
            relative_paths=requested_relative_paths,
            store_path=checkpoint_store_path,
            operation_time=lambda: self._operation_time(now),
        )
        tracked_artifact_store = _TrackingContentAddressedStore(
            resolved_artifact_store,
            record_created_ref=partial_artifacts.record_created_ref,
        )
        try:
            with _maintain_operation_lease(self, lease, now=now):
                backup = run_backup_tree_operation(
                    tree_backup_operation,
                    target,
                    relative_paths=requested_relative_paths,
                    store=tracked_artifact_store,
                )
            backup_payload = tree_backup_checkpoint_payload(backup, store_path=checkpoint_store_path)
            require_tree_backup_payload_resource_key(backup_payload, resource_key)
            checkpoint = self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="tree_backup",
                payload=backup_payload,
                now=self._operation_time(now),
            )
        except Exception as exc:
            self._record_tree_backup_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                partial_artifacts=partial_artifacts,
                error=exc,
                now=now,
            )
            raise JournaledFilesystemMutationError(str(exc), batch_id=batch.batch_id) from exc
        succeeded = self._journal_store.mark_succeeded(
            batch.batch_id,
            lease=lease,
            result={
                "operation_id": operation.operation_id,
                "resource_key": resource_key,
                "tree_backup": backup_payload,
            },
            now=self._operation_time(now),
        )
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=checkpoint,
            after_checkpoint=None,
            recovery_record=None,
        )

    def _record_tree_backup_failure(
        self,
        batch_id: str,
        path: Path,
        *,
        resource_key: str,
        lease: LeaseRecord,
        partial_artifacts: _TreeBackupPartialArtifactTracker,
        error: Exception,
        now: datetime | None,
    ) -> None:
        failure_payload: dict[str, object] = {
            "operation": "backup_tree",
            "path": str(path),
            "resource_key": resource_key,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        recovery_payload: dict[str, object] = {
            "operation": "backup_tree",
            "path": str(path),
            "resource_key": resource_key,
            "failure": failure_payload,
        }
        partial_payload = partial_artifacts.payload()
        if partial_payload is not None:
            recovery_payload["partial_artifacts"] = partial_payload

        def record_lease_lost_recovery_desired() -> None:
            self._journal_store.record_lease_lost_recovery_desired(
                batch_id,
                lease=lease,
                resource_key=resource_key,
                reason="tree backup failed",
                payload=recovery_payload,
                now=self._operation_time(now),
            )

        try:
            partial_artifacts.record_checkpoint()
        except LeaseLostError:
            record_lease_lost_recovery_desired()
            return
        try:
            self._journal_store.mark_failed(
                batch_id,
                lease=lease,
                error=str(error),
                observed_state=failure_payload,
                now=self._operation_time(now),
            )
        except LeaseLostError:
            record_lease_lost_recovery_desired()

    def restore_tree_backup(
        self,
        backup: object,
        *,
        destination_root: Path | str,
        artifact_store: object,
        conflict_policy: str = "no_replace",
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        restore_tree_backup_operation: RestoreTreeBackupOperation | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        if conflict_policy not in {"no_replace", "replace"}:
            raise ValueError(f"unsupported conflict_policy: {conflict_policy!r}")
        target = _canonical_directory_path(destination_root)
        _require_matching_tree_resource_key(target, resource_key)
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        resolved_artifact_store = _tree_artifact_store(artifact_store, default_store_path=None)
        store_path = _require_artifact_store_path(resolved_artifact_store)
        backup_fingerprint = tree_backup_checkpoint_fingerprint(backup, store_path=store_path)
        intent_payload = {
            "operation": "restore_tree_backup",
            "destination_root": str(target),
            "resource_key": resource_key,
            "store_path": str(store_path),
            "conflict_policy": conflict_policy,
            "backup_fingerprint": backup_fingerprint,
        }
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload=intent_payload,
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            self._record_interrupted_if_taken_over(
                batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch outside planned phase",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot restore a tree backup"
            )
        try:
            batch, operation = self._journal_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type="restore_tree_backup",
                resource_key=resource_key,
                payload=intent_payload,
                now=self._operation_time(now),
            )
        except BatchStartConflictError as exc:
            current_batch = self._journal_store.get_batch(batch.batch_id) or batch
            self._record_interrupted_if_taken_over(
                current_batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch with journal records",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} cannot restore a tree backup: {exc}"
            ) from exc
        recovery_payload = {
            "operation": "restore_tree_backup",
            "store_path": str(store_path),
            "conflict_policy": conflict_policy,
        }
        try:
            before_snapshot = self._snapshot(target)
        except Exception as exc:
            self._record_before_snapshot_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise
        before_checkpoint = self._journal_store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            checkpoint_type="before",
            payload=_snapshot_payload(before_snapshot),
            now=self._operation_time(now),
        )
        restore_operation = restore_tree_backup_operation or _default_restore_tree_backup_operation()
        try:
            with _maintain_operation_lease(self, lease, now=now):
                result = run_restore_tree_backup_operation(
                    restore_operation,
                    backup,
                    destination_root=target,
                    store=resolved_artifact_store,
                    conflict_policy=conflict_policy,
                )
        except Exception as exc:
            self._record_mutation_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise JournaledFilesystemMutationError(str(exc), batch_id=batch.batch_id) from exc
        try:
            after_snapshot = self._snapshot(target)
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=None,
                recovery_payload=recovery_payload,
                stage="after_snapshot",
                error=exc,
                now=now,
            )
            raise
        try:
            after_checkpoint = self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="after",
                payload=_snapshot_payload(after_snapshot),
                now=self._operation_time(now),
            )
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                recovery_payload=recovery_payload,
                stage="after_checkpoint",
                error=exc,
                now=now,
            )
            raise
        try:
            succeeded = self._journal_store.mark_succeeded(
                batch.batch_id,
                lease=lease,
                result={
                    "operation_id": operation.operation_id,
                    "resource_key": resource_key,
                    "store_path": str(store_path),
                    "conflict_policy": conflict_policy,
                    "result_type": type(result).__name__,
                },
                now=self._operation_time(now),
            )
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                target,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                recovery_payload=recovery_payload,
                stage="mark_succeeded",
                error=exc,
                now=now,
            )
            raise
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=before_checkpoint,
            after_checkpoint=after_checkpoint,
            recovery_record=None,
        )

    def snapshot_bundle(
        self,
        bundle: SnapshotBundle,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        payload = _snapshot_bundle_payload(bundle)
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload={"operation": "snapshot_bundle", **payload},
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot record a snapshot bundle"
            )
        batch, operation = self._journal_store.start_batch_operation(
            batch.batch_id,
            lease=lease,
            operation_type="snapshot_bundle",
            resource_key=resource_key,
            payload={"operation": "snapshot_bundle"},
            now=self._operation_time(now),
        )
        checkpoint = self._journal_store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            checkpoint_type="snapshot_bundle",
            payload=payload,
            now=self._operation_time(now),
        )
        succeeded = self._journal_store.mark_succeeded(
            batch.batch_id,
            lease=lease,
            result={"operation_id": operation.operation_id, "entry_count": len(bundle.entries)},
            now=self._operation_time(now),
        )
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=checkpoint,
            after_checkpoint=None,
            recovery_record=None,
        )

    def rename_no_replace(
        self,
        source: Path | str,
        destination: Path | str,
        *,
        resource_key: str,
        destination_resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        source_path = _canonical_rename_path(source)
        destination_path = _canonical_rename_path(destination)
        _require_matching_rename_resource_key(source_path, resource_key)
        _require_matching_rename_resource_key(destination_path, destination_resource_key)
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        intent_payload = {
            "operation": "rename_no_replace",
            "source_path": str(source_path),
            "destination_path": str(destination_path),
            "resource_key": resource_key,
            "destination_resource_key": destination_resource_key,
        }
        return self._run_rename_mutation(
            source_path,
            destination_path,
            resource_key=resource_key,
            destination_resource_key=destination_resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            intent_payload=intent_payload,
            now=now,
        )

    def capture_directory(
        self,
        source: Path | str,
        *,
        quarantine_path: Path | str,
        resource_key: str,
        destination_resource_key: str | None = None,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        source_path = _canonical_directory_path(source)
        _require_matching_directory_resource_key(source_path, resource_key)
        quarantine = absolute_without_resolving(quarantine_path)
        if destination_resource_key is None:
            raise ValueError("capture_directory requires destination_resource_key for the quarantine path")
        if destination_resource_key is not None:
            _require_matching_directory_resource_key(quarantine, destination_resource_key)
            self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        intent_payload = {
            "operation": "capture_directory",
            "path": str(source_path),
            "quarantine_path": str(quarantine),
            "resource_key": resource_key,
            "captured_directory_cleanup": self._captured_directory_cleanup,
        }
        if destination_resource_key is not None:
            intent_payload["destination_resource_key"] = destination_resource_key
        return self._run_capture_directory_mutation(
            source_path,
            quarantine_path=quarantine,
            resource_key=resource_key,
            destination_resource_key=destination_resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            intent_payload=intent_payload,
            now=now,
        )

    def write_text_file(
        self,
        path: Path | str,
        content: str,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        encoding: str = "utf-8",
        newline: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_file_path(path)
        _require_matching_file_resource_key(target, resource_key)
        desired_bytes = _normalized_text(content, newline=newline).encode(encoding)
        intent_payload = {
            "operation": "write_text",
            "path": str(target),
            "resource_key": resource_key,
            "encoding": encoding,
            "newline": newline,
            "desired_size": len(desired_bytes),
            "desired_sha256": hashlib.sha256(desired_bytes).hexdigest(),
        }

        def mutate() -> None:
            self._write_text(target, content, encoding=encoding, newline=newline)

        return self._run_file_mutation(
            target,
            resource_key=resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            operation_type="write_text",
            preflight_operation="write_text",
            intent_payload=intent_payload,
            recovery_payload={
                "desired": {
                    "exists": True,
                    "file_type": "file",
                    "size": len(desired_bytes),
                    "sha256": hashlib.sha256(desired_bytes).hexdigest(),
                }
            },
            mutate=mutate,
            now=now,
        )

    def write_bytes_file(
        self,
        path: Path | str,
        content: bytes,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        permissions: int | None = None,
        max_bytes: int | None = None,
        allow_large: bool | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_file_path(path)
        _require_matching_file_resource_key(target, resource_key)
        effective_max_bytes = self._write_bytes_max_bytes if max_bytes is None else max_bytes
        if effective_max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        effective_allow_large = self._write_bytes_large_policy == "allow" if allow_large is None else allow_large
        if len(content) > effective_max_bytes and not effective_allow_large:
            raise ValueError(
                "write_bytes refused "
                f"{len(content)} byte(s); max_bytes is {effective_max_bytes}. "
                "Pass allow_large=True or configure write_bytes_large_policy='allow' to opt in."
            )
        desired_sha256 = hashlib.sha256(content).hexdigest()
        intent_payload = {
            "operation": "write_bytes",
            "path": str(target),
            "resource_key": resource_key,
            "desired_size": len(content),
            "desired_sha256": desired_sha256,
            "permissions": permissions,
            "max_bytes": effective_max_bytes,
            "large_write_allowed": effective_allow_large,
        }

        def mutate() -> None:
            if permissions is None:
                self._write_bytes(target, content)
                return
            self._write_bytes(target, content, permissions=permissions)

        return self._run_file_mutation(
            target,
            resource_key=resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            operation_type="write_bytes",
            preflight_operation="write_text",
            intent_payload=intent_payload,
            recovery_payload={
                "desired": {
                    "exists": True,
                    "file_type": "file",
                    "size": len(content),
                    "sha256": desired_sha256,
                }
            },
            mutate=mutate,
            now=now,
        )

    def delete_file(
        self,
        path: Path | str,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        missing_ok: bool = False,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_file_path(path)
        _require_matching_file_resource_key(target, resource_key)
        intent_payload = {
            "operation": "delete_file",
            "path": str(target),
            "resource_key": resource_key,
            "missing_ok": missing_ok,
        }

        def mutate() -> None:
            self._delete_file(target, missing_ok=missing_ok)

        return self._run_file_mutation(
            target,
            resource_key=resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            operation_type="delete_file",
            preflight_operation="delete_file",
            intent_payload=intent_payload,
            recovery_payload={"desired": {"exists": False, "file_type": "missing"}},
            mutate=mutate,
            now=now,
        )

    def make_directory(
        self,
        path: Path | str,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        parents: bool = False,
        exist_ok: bool = False,
        intent_metadata: Mapping[str, object] | None = None,
        recovery_metadata: Mapping[str, object] | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_directory_path(path)
        _require_matching_directory_resource_key(target, resource_key)
        if parents:
            raise NotImplementedError(
                "journaled make_directory(..., parents=True) is deferred until ancestor journaling exists"
            )
        intent_payload = {
            "operation": "make_directory",
            "path": str(target),
            "resource_key": resource_key,
            "parents": parents,
            "exist_ok": exist_ok,
            "rollback_diagnostic": _make_directory_rollback_diagnostic(),
        }
        if intent_metadata is not None:
            intent_payload = {**intent_payload, **intent_metadata}

        def mutate() -> None:
            self._make_directory(target, parents=parents, exist_ok=exist_ok)

        return self._run_file_mutation(
            target,
            resource_key=resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            operation_type="make_directory",
            preflight_operation="make_directory",
            intent_payload=intent_payload,
            recovery_payload={
                "desired": {"exists": True, "file_type": "directory"},
                "rollback_diagnostic": intent_payload.get("rollback_diagnostic"),
                **({} if recovery_metadata is None else dict(recovery_metadata)),
            },
            mutate=mutate,
            now=now,
        )

    def record_directory_noop(
        self,
        path: Path | str,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        intent_payload: Mapping[str, object],
        recovery_payload: Mapping[str, object],
        operation_type: str = "make_directory_noop",
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        target = _canonical_directory_path(path)
        _require_matching_directory_resource_key(target, resource_key)

        result = self._run_file_mutation(
            target,
            resource_key=resource_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            idempotency_key=idempotency_key,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            claim_scope=claim_scope,
            operation_type=operation_type,
            preflight_operation=None,
            intent_payload=intent_payload,
            recovery_payload=recovery_payload,
            mutate=lambda: None,
            now=now,
        )
        if result.skipped:
            return result
        return JournaledFilesystemResult(
            batch=result.batch,
            operation=result.operation,
            before_checkpoint=result.before_checkpoint,
            after_checkpoint=result.after_checkpoint,
            recovery_record=result.recovery_record,
            skipped=True,
        )

    def _run_rename_mutation(
        self,
        source_path: Path,
        destination_path: Path,
        *,
        resource_key: str,
        destination_resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None,
        operation_phase_id: str | None,
        claim_scope: str | None,
        intent_payload: Mapping[str, object],
        now: datetime | None,
    ) -> JournaledFilesystemResult:
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload=intent_payload,
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            self._record_interrupted_if_taken_over(
                batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch outside planned phase",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot start a rename mutation"
            )
        try:
            started_batch, operation = self._journal_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type="rename_no_replace",
                resource_key=resource_key,
                payload=intent_payload,
                now=self._operation_time(now),
            )
        except BatchStartConflictError as exc:
            current_batch = self._journal_store.get_batch(batch.batch_id) or batch
            self._record_interrupted_if_taken_over(
                current_batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch with journal records",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} cannot start a rename mutation: {exc}"
            ) from exc
        batch = started_batch
        before_snapshot = self._snapshot(source_path)
        before_checkpoint = self._journal_store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            checkpoint_type="before",
            payload=_snapshot_payload(before_snapshot),
            now=self._operation_time(now),
        )
        self._journal_store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            checkpoint_type="destination_before",
            payload=_snapshot_payload(self._snapshot(destination_path)),
            now=self._operation_time(now),
        )
        rename_record: RenameRecord | None = None
        try:
            self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
            self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
            with _maintain_operation_lease(self, lease, now=now):
                self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
                self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
                rename_record = self._rename_no_replace(source_path, destination_path)
        except Exception as exc:
            source_after = _snapshot_or_none(self._snapshot, source_path)
            destination_after = _snapshot_or_none(self._snapshot, destination_path)
            recovery_payload = _ambiguous_rename_recovery_payload(
                source_path=source_path,
                destination_path=destination_path,
                resource_key=resource_key,
                before_snapshot=before_snapshot,
                source_after=source_after,
                destination_after=destination_after,
                error=exc,
            )
            if recovery_payload is None:
                self._record_mutation_failure(
                    batch.batch_id,
                    source_path,
                    resource_key=resource_key,
                    lease=lease,
                    before_snapshot=before_snapshot,
                    recovery_payload={},
                    error=exc,
                    now=now,
                )
            else:
                self._record_post_mutation_failure(
                    batch.batch_id,
                    source_path,
                    resource_key=resource_key,
                    lease=lease,
                    before_snapshot=before_snapshot,
                    after_snapshot=source_after,
                    recovery_payload=recovery_payload,
                    stage="rename_ambiguous_success",
                    error=exc,
                    now=now,
                )
            raise JournaledFilesystemMutationError(str(exc), batch_id=batch.batch_id) from exc
        rename_payload = rename_record_payload(rename_record)
        try:
            self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="rename_record",
                payload=rename_payload,
                now=self._operation_time(now),
            )
            after_checkpoint = self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="after",
                payload={
                    "source": _snapshot_payload(self._snapshot(source_path)),
                    "destination": _snapshot_payload(self._snapshot(destination_path)),
                },
                now=self._operation_time(now),
            )
            succeeded = self._journal_store.mark_succeeded(
                batch.batch_id,
                lease=lease,
                result={"operation_id": operation.operation_id, "rename_record": rename_payload},
                now=self._operation_time(now),
            )
        except Exception as exc:
            recovery_payload = {"rename_record": rename_payload}
            self._record_post_mutation_failure(
                batch.batch_id,
                source_path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=None,
                recovery_payload=recovery_payload,
                stage="rename_journal",
                error=exc,
                now=now,
            )
            raise
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=before_checkpoint,
            after_checkpoint=after_checkpoint,
            recovery_record=None,
        )

    def _run_capture_directory_mutation(
        self,
        source_path: Path,
        *,
        quarantine_path: Path,
        resource_key: str,
        destination_resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None,
        operation_phase_id: str | None,
        claim_scope: str | None,
        intent_payload: Mapping[str, object],
        now: datetime | None,
    ) -> JournaledFilesystemResult:
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload=intent_payload,
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            self._record_interrupted_if_taken_over(
                batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch outside planned phase",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot start a capture-directory mutation"
            )
        try:
            started_batch, operation = self._journal_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type="capture_directory",
                resource_key=resource_key,
                payload=intent_payload,
                now=self._operation_time(now),
            )
        except BatchStartConflictError as exc:
            current_batch = self._journal_store.get_batch(batch.batch_id) or batch
            self._record_interrupted_if_taken_over(
                current_batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch with journal records",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} cannot start a capture-directory mutation: {exc}"
            ) from exc
        batch = started_batch
        before_snapshot = self._snapshot(source_path)
        before_checkpoint = self._journal_store.record_checkpoint(
            batch.batch_id,
            lease=lease,
            operation_id=operation.operation_id,
            resource_key=resource_key,
            checkpoint_type="before",
            payload=_snapshot_payload(before_snapshot),
            now=self._operation_time(now),
        )
        captured: CapturedDirectoryRecord | None = None
        try:
            self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
            self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
            with _maintain_operation_lease(self, lease, now=now):
                self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
                self._require_authority(destination_resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
                captured = self._capture_directory(source_path, quarantine_path=quarantine_path)
        except Exception as exc:
            source_after = _snapshot_or_none(self._snapshot, source_path)
            quarantine_after = _snapshot_or_none(self._snapshot, quarantine_path)
            recovery_payload = _ambiguous_capture_recovery_payload(
                source_path=source_path,
                quarantine_path=quarantine_path,
                resource_key=resource_key,
                before_snapshot=before_snapshot,
                source_after=source_after,
                quarantine_after=quarantine_after,
                error=exc,
            )
            if recovery_payload is None:
                self._record_mutation_failure(
                    batch.batch_id,
                    source_path,
                    resource_key=resource_key,
                    lease=lease,
                    before_snapshot=before_snapshot,
                    recovery_payload={},
                    error=exc,
                    now=now,
                )
            else:
                self._record_post_mutation_failure(
                    batch.batch_id,
                    source_path,
                    resource_key=resource_key,
                    lease=lease,
                    before_snapshot=before_snapshot,
                    after_snapshot=source_after,
                    recovery_payload=recovery_payload,
                    stage="capture_directory_ambiguous_success",
                    error=exc,
                    now=now,
                )
            raise JournaledFilesystemMutationError(str(exc), batch_id=batch.batch_id) from exc
        capture_payload = _captured_directory_payload(
            captured,
            resource_key=resource_key,
            captured_directory_cleanup=self._captured_directory_cleanup,
        )
        try:
            captured_checkpoint = self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="captured_directory",
                payload=capture_payload,
                now=self._operation_time(now),
            )
            if self._captured_directory_cleanup == "automatic":
                _record_captured_directory_cleanup_plan(
                    self._journal_store,
                    lease=lease,
                    batch_id=batch.batch_id,
                    checkpoint=captured_checkpoint,
                    captured=captured,
                    resource_key=resource_key,
                    now=self._operation_time(now),
                )
            else:
                _record_captured_directory_cleanup_retained(
                    self._journal_store,
                    lease=lease,
                    batch_id=batch.batch_id,
                    checkpoint=captured_checkpoint,
                    captured=captured,
                    resource_key=resource_key,
                    now=self._operation_time(now),
                )
            after_checkpoint = self._journal_store.record_checkpoint(
                batch.batch_id,
                lease=lease,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                checkpoint_type="after",
                payload={
                    "source": _snapshot_payload(self._snapshot(source_path)),
                    "quarantine": _snapshot_payload(self._snapshot(quarantine_path)),
                },
                now=self._operation_time(now),
            )
            succeeded = self._journal_store.mark_succeeded(
                batch.batch_id,
                lease=lease,
                result={"operation_id": operation.operation_id, "captured_directory": capture_payload},
                now=self._operation_time(now),
            )
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                source_path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=None,
                recovery_payload={"captured_directory_restore": capture_payload},
                stage="capture_directory_journal",
                error=exc,
                now=now,
            )
            raise
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=before_checkpoint,
            after_checkpoint=after_checkpoint,
            recovery_record=None,
        )

    def _run_file_mutation(
        self,
        path: Path,
        *,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None,
        operation_phase_id: str | None,
        claim_scope: str | None,
        operation_type: str,
        preflight_operation: str | None,
        intent_payload: Mapping[str, object],
        recovery_payload: Mapping[str, object],
        mutate: Callable[[], None],
        now: datetime | None,
    ) -> JournaledFilesystemResult:
        if preflight_operation is not None and self._preflight_operation is not None:
            self._preflight_operation(preflight_operation)
        self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        batch = self._journal_store.create_batch(
            idempotency_key=idempotency_key,
            lease=lease,
            owner=owner,
            run_id=run_id,
            operation_run_id=operation_run_id,
            operation_phase_id=operation_phase_id,
            resource_key=resource_key,
            claim_owner=owner,
            claim_scope=claim_scope,
            payload=intent_payload,
            now=self._operation_time(now),
        )
        if batch.phase == BatchPhase.SUCCEEDED:
            return JournaledFilesystemResult(
                batch=batch,
                operation=None,
                before_checkpoint=None,
                after_checkpoint=None,
                recovery_record=None,
                skipped=True,
            )
        if batch.phase != BatchPhase.PLANNED:
            self._record_interrupted_if_taken_over(
                batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch outside planned phase",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} is in phase {batch.phase!r} and cannot start a file mutation"
            )
        try:
            started_batch, operation = self._journal_store.start_batch_operation(
                batch.batch_id,
                lease=lease,
                operation_type=operation_type,
                resource_key=resource_key,
                payload=intent_payload,
                now=self._operation_time(now),
            )
        except BatchStartConflictError as exc:
            current_batch = self._journal_store.get_batch(batch.batch_id) or batch
            self._record_interrupted_if_taken_over(
                current_batch,
                lease=lease,
                resource_key=resource_key,
                reason="retry found abandoned batch with journal records",
                now=now,
            )
            raise JournaledFilesystemBatchStateError(
                f"batch {batch.batch_id!r} cannot start a file mutation: {exc}"
            ) from exc
        batch = started_batch
        try:
            before_snapshot = self._snapshot(path)
        except Exception as exc:
            self._record_before_snapshot_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise
        try:
            with _maintain_operation_lease(self, lease, now=now):
                prepared = prepare_mutation_checkpoints(
                    self._journal_store,
                    lease=lease,
                    batch_id=batch.batch_id,
                    operation_id=operation.operation_id,
                    resource_key=resource_key,
                    path=path,
                    before_snapshot=before_snapshot,
                    operation_type=operation_type,
                    recovery_payload=recovery_payload,
                    capture_file_rollback_proof=self._automatic_file_rollback,
                    operation_time=lambda: self._operation_time(now),
                )
        except Exception as exc:
            self._record_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                recovery_payload=recovery_payload,
                error=exc,
                stage="prepare_checkpoints",
                now=now,
            )
            raise
        before_checkpoint = prepared.before_checkpoint
        recovery_payload = prepared.recovery_payload

        try:
            self._require_authority(resource_key, lease=lease, owner=owner, scope=claim_scope, now=now)
        except Exception as exc:
            self._record_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise
        try:
            with _maintain_operation_lease(self, lease, now=now):
                mutate()
        except UnsupportedFilesystemBackendError as exc:
            self._record_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise
        except Exception as exc:
            self._record_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                recovery_payload=recovery_payload,
                error=exc,
                now=now,
            )
            raise JournaledFilesystemMutationError(str(exc), batch_id=batch.batch_id) from exc

        try:
            after_snapshot = self._snapshot(path)
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=None,
                recovery_payload=recovery_payload,
                stage="after_snapshot",
                error=exc,
                now=now,
            )
            raise
        try:
            after_checkpoint = record_after_mutation_checkpoint(
                self._journal_store,
                lease=lease,
                batch_id=batch.batch_id,
                operation_id=operation.operation_id,
                resource_key=resource_key,
                path=path,
                after_snapshot=after_snapshot,
                operation_type=operation_type,
                operation_time=lambda: self._operation_time(now),
            )
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                recovery_payload=recovery_payload,
                stage="after_checkpoint",
                error=exc,
                now=now,
            )
            raise
        try:
            succeeded = self._journal_store.mark_succeeded(
                batch.batch_id,
                lease=lease,
                result={
                    "operation_id": operation.operation_id,
                    "resource_key": resource_key,
                    "after": _snapshot_payload(after_snapshot),
                },
                now=self._operation_time(now),
            )
        except Exception as exc:
            self._record_post_mutation_failure(
                batch.batch_id,
                path,
                resource_key=resource_key,
                lease=lease,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
                recovery_payload=recovery_payload,
                stage="mark_succeeded",
                error=exc,
                now=now,
            )
            raise
        return JournaledFilesystemResult(
            batch=succeeded,
            operation=operation,
            before_checkpoint=before_checkpoint,
            after_checkpoint=after_checkpoint,
            recovery_record=None,
        )

    def _require_authority(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None,
        now: datetime | None,
    ) -> None:
        self._lease_store.require_current(lease, now=self._operation_time(now))
        claim = self._claim_store.get(resource_key)
        if claim is None:
            raise MissingResourceClaimError(f"resource {resource_key!r} is not claimed")
        if lease.owner != owner:
            raise ResourceClaimAuthorityError(
                f"resource {resource_key!r} cannot be used by operation owner {owner!r} "
                f"with lease owner {lease.owner!r}",
                existing=claim,
            )
        if claim.owner != owner or claim.scope != scope:
            raise ResourceClaimAuthorityError(
                f"resource {resource_key!r} is claimed by {claim.owner!r} with a different scope",
                existing=claim,
            )
        _require_lease_bound_claim_details(resource_key, claim, lease=lease)

    def _operation_time(self, now: datetime | None) -> datetime:
        raise NotImplementedError


def _require_lease_bound_claim_details(resource_key: str, claim: ClaimRecord, *, lease: LeaseRecord) -> None:
    details = _lease_bound_claim_details(claim.details)
    if details is None:
        raise ResourceClaimAuthorityError(
            f"resource {resource_key!r} has no lease-bound claim authority",
            existing=claim,
        )
    claim_id = details.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        raise ResourceClaimAuthorityError(
            f"resource {resource_key!r} has invalid lease-bound claim identity",
            existing=claim,
        )
    lease_fencing_token = details.get("lease_fencing_token")
    if not isinstance(lease_fencing_token, int) or lease_fencing_token != lease.fencing_token:
        raise ResourceClaimAuthorityError(
            f"resource {resource_key!r} is bound to a different lease fencing token",
            existing=claim,
        )
    if (
        details.get("lease_name") != lease.name
        or details.get("lease_owner") != lease.owner
        or details.get("lease_token_digest") != _lease_token_digest(lease)
    ):
        raise ResourceClaimAuthorityError(
            f"resource {resource_key!r} is bound to a different lease identity",
            existing=claim,
        )


def _lease_bound_claim_details(details: str | None) -> Mapping[str, object] | None:
    if details is None:
        return None
    try:
        payload = json.loads(details)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _snapshot_bundle_payload(bundle: SnapshotBundle) -> dict[str, object]:
    return {
        "requested_paths": [str(path) for path in bundle.requested_paths],
        "include_children": bundle.include_children,
        "hash_policy": bundle.hash_policy,
        "small_file_max_bytes": bundle.small_file_max_bytes,
        "entries": [
            {
                "path": str(entry.path),
                "requested": entry.requested,
                "snapshot": _snapshot_payload(entry.snapshot),
            }
            for entry in bundle.entries
        ],
    }


def _make_directory_rollback_diagnostic() -> dict[str, object]:
    return {
        "automatic_recursive_delete": False,
        "manual_intervention_possible": True,
        "remove_only_if": {
            "created_by_exact_step": True,
            "directory_is_empty": True,
            "path_is_still_safe_directory": True,
            "resource_key_still_matches": True,
        },
    }


def _captured_directory_payload(
    record: CapturedDirectoryRecord,
    *,
    resource_key: str,
    captured_directory_cleanup: Literal["automatic", "retain"],
) -> dict[str, object]:
    original_identity = {
        "file_type": "directory",
        "path": str(record.original_path),
        "resource_key": resource_key,
        "device": record.original_identity.device,
        "inode": record.original_identity.inode,
    }
    captured_identity = {
        "file_type": "directory",
        "path": str(record.quarantine_path),
        "resource_key": resource_key,
        "device": record.captured_identity.device,
        "inode": record.captured_identity.inode,
    }
    return {
        "path": str(record.original_path),
        "step_resource_key": resource_key,
        "ownership_class": record.ownership_class,
        "captured_directory_cleanup": captured_directory_cleanup,
        "captured_directory": {
            "original_path": str(record.original_path),
            "quarantine_path": str(record.quarantine_path),
            "original_identity": original_identity,
            "captured_identity": captured_identity,
            "ownership_class": record.ownership_class,
        },
    }


def _record_captured_directory_cleanup_plan(
    journal_store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    batch_id: str,
    checkpoint: CheckpointRecord,
    captured: CapturedDirectoryRecord,
    resource_key: str,
    now: datetime,
) -> None:
    artifact_id = _captured_directory_cleanup_artifact_id(captured)
    payload = _captured_directory_cleanup_payload(checkpoint=checkpoint, captured=captured)
    journal_store.record_artifact_cleanup_planned(
        batch_id=batch_id,
        lease=lease,
        artifact_id=artifact_id,
        trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        resource_key=resource_key,
        payload=payload,
        now=now,
    )


def _record_captured_directory_cleanup_retained(
    journal_store: OperationJournalStore,
    *,
    lease: LeaseRecord,
    batch_id: str,
    checkpoint: CheckpointRecord,
    captured: CapturedDirectoryRecord,
    resource_key: str,
    now: datetime,
) -> None:
    artifact_id = _captured_directory_cleanup_artifact_id(captured)
    payload = _captured_directory_cleanup_payload(checkpoint=checkpoint, captured=captured)
    journal_store.record_artifact_cleanup_planned(
        batch_id=batch_id,
        lease=lease,
        artifact_id=artifact_id,
        trigger=ArtifactCleanupTrigger.DEFERRED_CLEANUP,
        resource_key=resource_key,
        payload=payload,
        now=now,
    )
    retained_payload = dict(payload)
    retained_payload["reason_code"] = "captured_directory_retained"
    journal_store.record_artifact_cleanup_skipped(
        batch_id=batch_id,
        lease=lease,
        artifact_id=artifact_id,
        reason="captured_directory_retained",
        payload=retained_payload,
        now=now,
    )


def _captured_directory_cleanup_artifact_id(captured: CapturedDirectoryRecord) -> str:
    return f"captured-directory:{Path(os.path.abspath(captured.quarantine_path))}"


def _captured_directory_cleanup_payload(
    *,
    checkpoint: CheckpointRecord,
    captured: CapturedDirectoryRecord,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "artifact_type": "captured_directory",
        "quarantine_path": str(captured.quarantine_path),
        "original_path": str(captured.original_path),
        "captured_identity": {
            "device": captured.captured_identity.device,
            "inode": captured.captured_identity.inode,
        },
    }
    cleanup_policy = checkpoint.payload.get("captured_directory_cleanup")
    if cleanup_policy in {"automatic", "retain"}:
        payload["captured_directory_cleanup"] = cleanup_policy
    return payload


def _snapshot_or_none(
    snapshot: Callable[[Path | str], ResourceSnapshot],
    path: Path,
) -> ResourceSnapshot | None:
    try:
        return snapshot(path)
    except Exception:
        return None


def _maintain_operation_lease(
    coordinator: JournaledFilesystemMutationMixin,
    lease: LeaseRecord,
    *,
    now: datetime | None,
) -> AbstractContextManager[object]:
    if now is not None or not getattr(coordinator, "_heartbeat_leases", True):
        return nullcontext()
    return maintain_lease(
        coordinator._lease_store,
        lease,
        now=lambda: coordinator._operation_time(None),
    )


def _canonical_rename_path(path: Path | str) -> Path:
    return _canonical_file_path(path)


def _require_matching_rename_resource_key(path: Path, resource_key: str) -> None:
    if resource_key.startswith("file:"):
        _require_matching_file_resource_key(path, resource_key)
        return
    _require_matching_directory_resource_key(path, resource_key)


def _ambiguous_rename_recovery_payload(
    *,
    source_path: Path,
    destination_path: Path,
    resource_key: str,
    before_snapshot: ResourceSnapshot,
    source_after: ResourceSnapshot | None,
    destination_after: ResourceSnapshot | None,
    error: Exception,
) -> dict[str, object] | None:
    if not _looks_physically_moved(source_after, destination_after):
        return None
    rename_payload = _rename_payload_from_snapshots(
        source_path=source_path,
        destination_path=destination_path,
        before_snapshot=before_snapshot,
        destination_after=destination_after,
    )
    if rename_payload is not None:
        return {"rename_record": rename_payload}
    return {
        "manual_intervention_action": _manual_intervention_action_payload(
            operation="rename_no_replace",
            action_id=f"manual-intervention:rename-no-replace:{destination_path}:{source_path}",
            resource_key=resource_key,
            reason="rename_no_replace requires manual intervention after ambiguous post-move verification",
            reason_code="rename_ambiguous_success_unverified",
            detail=str(error),
            source_path=source_path,
            target_path=destination_path,
            before_snapshot=before_snapshot,
            source_after=source_after,
            target_after=destination_after,
        )
    }


def _ambiguous_capture_recovery_payload(
    *,
    source_path: Path,
    quarantine_path: Path,
    resource_key: str,
    before_snapshot: ResourceSnapshot,
    source_after: ResourceSnapshot | None,
    quarantine_after: ResourceSnapshot | None,
    error: Exception,
) -> dict[str, object] | None:
    if not _looks_physically_moved(source_after, quarantine_after):
        return None
    capture_payload = _captured_directory_payload_from_snapshots(
        source_path=source_path,
        quarantine_path=quarantine_path,
        resource_key=resource_key,
        before_snapshot=before_snapshot,
        quarantine_after=quarantine_after,
    )
    if capture_payload is not None:
        return {"captured_directory_restore": capture_payload}
    return {
        "manual_intervention_action": _manual_intervention_action_payload(
            operation="capture_directory",
            action_id=f"manual-intervention:capture-directory:{quarantine_path}:{source_path}",
            resource_key=resource_key,
            reason="capture_directory requires manual intervention after ambiguous post-move verification",
            reason_code="capture_directory_ambiguous_success_unverified",
            detail=str(error),
            source_path=source_path,
            target_path=quarantine_path,
            before_snapshot=before_snapshot,
            source_after=source_after,
            target_after=quarantine_after,
        )
    }


def _looks_physically_moved(
    source_after: ResourceSnapshot | None,
    target_after: ResourceSnapshot | None,
) -> bool:
    return source_after is not None and not source_after.exists and target_after is not None and target_after.exists


def _rename_payload_from_snapshots(
    *,
    source_path: Path,
    destination_path: Path,
    before_snapshot: ResourceSnapshot,
    destination_after: ResourceSnapshot | None,
) -> dict[str, object] | None:
    if destination_after is None:
        return None
    if before_snapshot.file_type != "file":
        return None
    if destination_after.file_type != before_snapshot.file_type:
        return None
    if not _snapshot_identity_matches(destination_after, before_snapshot):
        return None
    if before_snapshot.device is None or before_snapshot.inode is None:
        return None
    if before_snapshot.size is None or before_snapshot.mtime_ns is None or before_snapshot.ctime_ns is None:
        return None
    return rename_record_payload(
        RenameRecord(
            source_path=source_path,
            destination_path=destination_path,
            file_type=cast(RenamedFileType, before_snapshot.file_type),
            device=before_snapshot.device,
            inode=before_snapshot.inode,
            size=before_snapshot.size,
            mtime_ns=before_snapshot.mtime_ns,
            ctime_ns=before_snapshot.ctime_ns,
        )
    )


def _captured_directory_payload_from_snapshots(
    *,
    source_path: Path,
    quarantine_path: Path,
    resource_key: str,
    before_snapshot: ResourceSnapshot,
    quarantine_after: ResourceSnapshot | None,
) -> dict[str, object] | None:
    if quarantine_after is None:
        return None
    if before_snapshot.file_type != "directory" or quarantine_after.file_type != "directory":
        return None
    if not _snapshot_identity_matches(quarantine_after, before_snapshot):
        return None
    if before_snapshot.device is None or before_snapshot.inode is None:
        return None
    return {
        "path": str(source_path),
        "step_resource_key": resource_key,
        "ownership_class": "captured_by_transaction",
        "captured_directory": {
            "original_path": str(source_path),
            "quarantine_path": str(quarantine_path),
            "original_identity": {
                "file_type": "directory",
                "path": str(source_path),
                "resource_key": resource_key,
                "device": before_snapshot.device,
                "inode": before_snapshot.inode,
            },
            "captured_identity": {
                "file_type": "directory",
                "path": str(quarantine_path),
                "resource_key": resource_key,
                "device": before_snapshot.device,
                "inode": before_snapshot.inode,
            },
            "ownership_class": "captured_by_transaction",
        },
    }


def _snapshot_identity_matches(left: ResourceSnapshot, right: ResourceSnapshot) -> bool:
    if left.device is None or left.device != right.device:
        return False
    if left.inode is None or left.inode != right.inode:
        return False
    if left.size is None or left.size != right.size:
        return False
    if left.mtime_ns is None or left.mtime_ns != right.mtime_ns:
        return False
    if left.file_type == right.file_type == "file":
        return left.content_hash is not None and left.content_hash == right.content_hash
    return True


def _default_backup_tree_operation() -> TreeBackupOperation:
    import safe_fs_ops.filesystem_ops as filesystem_ops

    backup_tree = filesystem_ops.__dict__.get("backup_tree")
    if backup_tree is None:
        raise RuntimeError("backup_tree primitive is not available")
    return cast(TreeBackupOperation, backup_tree)


class _TreeBackupPartialArtifactTracker:
    def __init__(
        self,
        journal_store: OperationJournalStore,
        *,
        batch_id: str,
        operation_id: str,
        resource_key: str,
        lease: LeaseRecord,
        root: Path,
        relative_paths: tuple[Path | str, ...],
        store_path: Path,
        operation_time: Callable[[], datetime],
    ) -> None:
        self._journal_store = journal_store
        self._batch_id = batch_id
        self._operation_id = operation_id
        self._resource_key = resource_key
        self._lease = lease
        self._root = root
        self._relative_paths = relative_paths
        self._store_path = store_path
        self._operation_time = operation_time
        self._created_refs: list[ContentRef] = []
        self._checkpointed_count = 0

    @property
    def created_refs(self) -> tuple[ContentRef, ...]:
        return tuple(self._created_refs)

    def record_created_ref(self, ref: ContentRef) -> None:
        self._created_refs.append(ref)
        self.record_checkpoint()

    def record_checkpoint(self) -> CheckpointRecord | None:
        refs = tuple(self._created_refs[self._checkpointed_count :])
        if not refs:
            return None
        checkpoint = self._journal_store.record_checkpoint(
            self._batch_id,
            lease=self._lease,
            operation_id=self._operation_id,
            resource_key=self._resource_key,
            checkpoint_type="tree_backup_partial_artifacts",
            payload=tree_backup_partial_artifacts_checkpoint_payload(
                root=self._root,
                relative_paths=self._relative_paths,
                store_path=self._store_path,
                content_refs=refs,
            ),
            now=self._operation_time(),
        )
        self._checkpointed_count = len(self._created_refs)
        return checkpoint

    def payload(self) -> dict[str, object] | None:
        if not self._created_refs:
            return None
        return tree_backup_partial_artifacts_checkpoint_payload(
            root=self._root,
            relative_paths=self._relative_paths,
            store_path=self._store_path,
            content_refs=tuple(self._created_refs),
        )


class _TrackingContentAddressedStore(ContentAddressedStore):
    def __init__(
        self,
        store: ContentAddressedStore,
        *,
        record_created_ref: Callable[[ContentRef], None],
    ) -> None:
        self._store = store
        self.root = store.root
        self._record_created_ref = record_created_ref

    def put_bytes(self, content: bytes) -> ContentRef:
        return self.put_bytes_with_status(content).ref

    def put_bytes_with_status(self, content: bytes) -> ContentAddressedPutResult:
        result = self._store.put_bytes_with_status(content)
        if result.created:
            self._record_created_ref(result.ref)
        return result

    def put_file(self, path: Path | str) -> ContentRef:
        return self.put_file_with_status(path).ref

    def put_file_with_status(self, path: Path | str) -> ContentAddressedPutResult:
        result = self._store.put_file_with_status(path)
        if result.created:
            self._record_created_ref(result.ref)
        return result

    def verify(self, ref: ContentRef) -> bool:
        return self._store.verify(ref)

    def copy_to(
        self,
        ref: ContentRef,
        destination: Path | str,
        *,
        no_replace: bool = True,
        permissions: int | None = None,
        mtime_ns: int | None = None,
        mtime: float | None = None,
    ) -> None:
        self._store.copy_to(
            ref,
            destination,
            no_replace=no_replace,
            permissions=permissions,
            mtime_ns=mtime_ns,
            mtime=mtime,
        )

    def __getattr__(self, name: str) -> object:
        return getattr(self._store, name)


def _default_restore_tree_backup_operation() -> RestoreTreeBackupOperation:
    import safe_fs_ops.filesystem_ops as filesystem_ops

    restore_tree_backup = filesystem_ops.__dict__.get("restore_tree_backup")
    if restore_tree_backup is None:
        raise RuntimeError("restore_tree_backup primitive is not available")
    return cast(RestoreTreeBackupOperation, restore_tree_backup)


def _tree_artifact_store(artifact_store: object | None, *, default_store_path: Path | None) -> ContentAddressedStore:
    if artifact_store is None:
        if default_store_path is None:
            raise ValueError("artifact_store is required")
        return ContentAddressedStore(default_store_path)
    if not isinstance(artifact_store, ContentAddressedStore):
        raise TypeError("tree backup artifact_store must be a ContentAddressedStore")
    return artifact_store


def _require_artifact_store_path(artifact_store: ContentAddressedStore) -> Path:
    root = getattr(artifact_store, "root", None)
    if root is None:
        raise ValueError("tree backup artifact_store must expose a root path for recovery")
    return Path(str(root))


def _manual_intervention_action_payload(
    *,
    operation: str,
    action_id: str,
    resource_key: str,
    reason: str,
    reason_code: str,
    detail: str,
    source_path: Path,
    target_path: Path,
    before_snapshot: ResourceSnapshot,
    source_after: ResourceSnapshot | None,
    target_after: ResourceSnapshot | None,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action_type": "manual_intervention_required",
        "resource_key": resource_key,
        "payload": {
            "operation": operation,
            "path": str(source_path),
            "target_path": str(target_path),
            "resource_key": resource_key,
            "reason": reason,
            "reason_code": reason_code,
            "detail": detail,
            "before": _snapshot_payload(before_snapshot),
            "source_after": None if source_after is None else _snapshot_payload(source_after),
            "target_after": None if target_after is None else _snapshot_payload(target_after),
        },
    }
