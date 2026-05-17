from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

from safe_fs_ops.filesystem_ops import HashPolicy
from safe_fs_ops.filesystem_ops import snapshot_bundle as filesystem_snapshot_bundle
from safe_fs_ops.operation_journal import JournaledFilesystemResult
from safe_fs_ops.operation_journal.filesystem_support import _canonical_directory_path
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.resources import DirectoryResource, FileResource, ResourceHandle, ResourceSet, TreeResource
from safe_fs_ops.workspace import (
    ResourceNotClaimedError,
    SafeWorkspace,
    SafeWorkspaceBusyError,
    SafeWorkspaceError,
    _ClaimReleaseError,
)
from safe_fs_ops.workspace_rollback import (
    cleanup_committed_transaction_artifacts,
    run_automatic_transaction_rollback,
    run_pending_transaction_rollback,
)
from safe_fs_ops.workspace_state import ClaimConflictError, LeaseLostError
from safe_fs_ops.workspace_state.lease_heartbeat import LeaseHeartbeat, maintain_lease
from safe_fs_ops.workspace_state.models import ClaimRecord, LeaseRecord
from safe_fs_ops.workspace_transaction_bookkeeping import (
    abandon_transaction_for_recovery,
    cleanup_now,
    cleanup_transaction,
    record_cleanup_error,
    resolve_operation_links,
)
from safe_fs_ops.workspace_transaction_operation import (
    claim_details,
    exit_with_implicit_operation,
    finalize_operation_success,
)
from safe_fs_ops.workspace_transaction_state import claim_is_already_released, forget_claim, remember_claim
from safe_fs_ops.workspace_transaction_utils import (
    next_idempotency_key,
    require_claimed_resource,
    require_directory_resource,
    require_file_resource,
)


class _TreeBackupCoordinator(Protocol):
    def backup_tree(
        self,
        path: Path | str,
        relative_paths: Iterable[Path | str],
        *,
        artifact_store: object,
        resource_key: str,
        lease: LeaseRecord,
        owner: str,
        run_id: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        claim_scope: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult: ...

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
        now: datetime | None = None,
    ) -> JournaledFilesystemResult: ...


@dataclass(slots=True)
class SafeTransaction:
    workspace: SafeWorkspace
    name: str
    resources: ResourceSet
    run_id: str
    rollback: Literal["record-only", "automatic"] = "record-only"
    now: datetime | None = None
    cleanup_clock: Callable[[], datetime] | None = None
    _lease: LeaseRecord | None = field(default=None, init=False, repr=False)
    _lease_heartbeat: LeaseHeartbeat | None = field(default=None, init=False, repr=False)
    _cleanup_error: SafeWorkspaceError | None = field(default=None, init=False, repr=False)
    _entered: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _operation_index: int = field(default=0, init=False, repr=False)
    _claims_by_resource_key: dict[str, ClaimRecord] = field(default_factory=dict, init=False, repr=False)
    _claim_release_order: list[str] = field(default_factory=list, init=False, repr=False)
    _operation_run: OperationRunRecord | None = field(default=None, init=False, repr=False)
    _operation_phase: OperationPhaseRecord | None = field(default=None, init=False, repr=False)
    _touched_operation_run_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _touched_operation_phase_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _transaction_instance_id: str = field(default_factory=lambda: uuid.uuid4().hex, init=False, repr=False)
    _state_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    @property
    def r(self) -> ResourceSet:
        return self.resources

    @property
    def lease(self) -> LeaseRecord:
        with self._state_lock:
            if self._lease is None:
                raise SafeWorkspaceError("transaction has not acquired a lease")
            return self._lease

    @property
    def cleanup_error(self) -> SafeWorkspaceError | None:
        with self._state_lock:
            return self._cleanup_error

    @property
    def operation_run(self) -> OperationRunRecord | None:
        with self._state_lock:
            return self._operation_run

    @property
    def operation_phase(self) -> OperationPhaseRecord | None:
        with self._state_lock:
            return self._operation_phase

    @property
    def claim_scope(self) -> str:
        return f"transaction-instance:{self._transaction_instance_id}"

    def __enter__(self) -> SafeTransaction:
        with self._state_lock:
            if self.rollback not in {"record-only", "automatic"}:
                raise ValueError(f"unsupported rollback mode: {self.rollback!r}")
            if not self.name:
                raise ValueError("transaction name must be non-empty")
            if self._closed:
                raise SafeWorkspaceError("transaction is single-use and has already been closed")
            if self._entered:
                raise SafeWorkspaceError("transaction is already active")
            lease = self.workspace.lease_store.acquire(
                self.workspace.lease_name,
                owner=self.workspace.owner,
                ttl=self.workspace.lease_ttl,
                now=self.now,
            )
            if not lease.acquired:
                raise SafeWorkspaceBusyError(
                    f"workspace lease {self.workspace.lease_name!r} is held by {lease.owner!r}"
                )
            self._lease = lease
            self._lease_heartbeat = maintain_lease(
                self.workspace.lease_store,
                lease,
                ttl=self.workspace.lease_ttl,
                now=self._heartbeat_time,
                enabled=self.now is None,
            )
            self._lease_heartbeat.__enter__()
            self._cleanup_error = None
            try:
                for resource in self.resources.sorted:
                    claim = self.workspace.claim_store.upsert(
                        resource.resource_key,
                        lease=lease,
                        owner=self.workspace.owner,
                        scope=self.claim_scope,
                        details=claim_details(
                            resource_kind=resource.kind,
                            run_id=self.run_id,
                            lease=lease,
                        ),
                        now=self.now,
                    )
                    remember_claim(self, claim)
            except ClaimConflictError as exc:
                cleanup_error = cleanup_transaction(self)
                self._cleanup_error = cleanup_error
                self._lease = None
                error = SafeWorkspaceBusyError(
                    f"resource {exc.existing.resource_key!r} is already claimed by {exc.existing.owner!r}"
                )
                record_cleanup_error(self, cleanup_error, error)
                raise error from exc
            except BaseException as exc:
                cleanup_error = cleanup_transaction(self)
                self._cleanup_error = cleanup_error
                self._lease = None
                record_cleanup_error(self, cleanup_error, exc)
                raise
            self._entered = True
            return self

    def _stop_lease_heartbeat(self) -> None:
        heartbeat = self._lease_heartbeat
        self._lease_heartbeat = None
        if heartbeat is not None:
            heartbeat.__exit__(None, None, None)

    def _heartbeat_time(self) -> datetime:
        if self.now is None:
            return datetime.now(UTC)
        if self.now.tzinfo is None:
            return self.now.replace(tzinfo=UTC)
        return self.now.astimezone(UTC)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        with self._state_lock:
            if not self._entered:
                return False
            self._entered = False
            self._closed = True
            lease = self.lease
            if self._operation_run is not None:
                return exit_with_implicit_operation(self, exc_type, exc, traceback)
            body_error = exc if isinstance(exc, BaseException) else None
            unresolved_rollback_error: BaseException | None = None
            if body_error is not None:
                finalization_now = cleanup_now(self)
                try:
                    from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_phases_failed

                    mark_touched_operation_phases_failed(self, now=finalization_now)
                except BaseException as transition_exc:
                    body_error.add_note(f"transaction phase status update also failed: {transition_exc}")
                try:
                    rollback_resolved = run_automatic_transaction_rollback(
                        self,
                        body_error=body_error,
                        now=finalization_now,
                    )
                    if not rollback_resolved:
                        unresolved_rollback_error = SafeWorkspaceError("automatic rollback did not resolve all batches")
                except BaseException as rollback_exc:
                    unresolved_rollback_error = rollback_exc
                    body_error.add_note(f"SafeWorkspace automatic rollback also failed: {rollback_exc}")
                try:
                    from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_runs_failed

                    mark_touched_operation_runs_failed(self, now=finalization_now)
                except BaseException as transition_exc:
                    body_error.add_note(f"transaction operation status update also failed: {transition_exc}")
                if unresolved_rollback_error is not None:
                    abandon_error = abandon_transaction_for_recovery(self)
                    if abandon_error is not None:
                        body_error.add_note(f"SafeWorkspace recovery handoff also failed: {abandon_error}")
                    return False
            elif self._touched_operation_run_ids or self._touched_operation_phase_ids:
                success_failure_stage = "pending_rollback"
                try:
                    run_pending_transaction_rollback(
                        self,
                        now=cleanup_now(self),
                    )
                    success_failure_stage = "artifact_cleanup"
                    artifact_cleanup_error = cleanup_committed_transaction_artifacts(self)
                    self._cleanup_error = artifact_cleanup_error
                    if artifact_cleanup_error is not None:
                        raise artifact_cleanup_error
                    from safe_fs_ops.workspace_transaction_operation import finalize_touched_operation_success

                    success_failure_stage = "finalize_success"
                    finalize_touched_operation_success(self)
                    return False
                except BaseException as success_exc:
                    terminal_status_error: BaseException | None = _mark_touched_finalization_failed(self, lease)
                    if success_failure_stage == "pending_rollback":
                        if terminal_status_error is not None:
                            success_exc.add_note(
                                f"transaction terminal status update also failed: {terminal_status_error}"
                            )
                        abandon_error = abandon_transaction_for_recovery(self)
                        if abandon_error is not None:
                            success_exc.add_note(f"SafeWorkspace recovery handoff also failed: {abandon_error}")
                        raise
                    finalization_cleanup_failure: BaseException | None = None
                    try:
                        cleanup_error = cleanup_transaction(self)
                    except BaseException as exc_cleanup:
                        cleanup_error = None
                        finalization_cleanup_failure = exc_cleanup
                    if cleanup_error is not None:
                        self._cleanup_error = cleanup_error
                    primary_error: BaseException = cleanup_error or finalization_cleanup_failure or success_exc
                    if terminal_status_error is not None:
                        primary_error.add_note(
                            f"transaction terminal status update also failed: {terminal_status_error}"
                        )
                    if primary_error is success_exc:
                        raise
                    raise primary_error from success_exc
            cleanup_failure: BaseException | None = None
            pending_rollback_error: BaseException | None = None
            if body_error is None:
                try:
                    run_pending_transaction_rollback(
                        self,
                        now=cleanup_now(self),
                    )
                except BaseException as rollback_exc:
                    pending_rollback_error = rollback_exc
            committed_artifact_cleanup_error: SafeWorkspaceError | None = None
            if body_error is None and pending_rollback_error is None:
                committed_artifact_cleanup_error = cleanup_committed_transaction_artifacts(self)
                self._cleanup_error = committed_artifact_cleanup_error
            if body_error is None and (
                pending_rollback_error is not None or committed_artifact_cleanup_error is not None
            ):
                cleanup_issue = pending_rollback_error or committed_artifact_cleanup_error
                terminal_status_error = _mark_touched_finalization_failed(self, lease)
                if terminal_status_error is not None:
                    assert cleanup_issue is not None
                    cleanup_issue.add_note(f"transaction terminal status update also failed: {terminal_status_error}")
            if pending_rollback_error is not None:
                abandon_error = abandon_transaction_for_recovery(self)
                if abandon_error is not None:
                    pending_rollback_error.add_note(f"SafeWorkspace recovery handoff also failed: {abandon_error}")
                raise pending_rollback_error
            try:
                cleanup_error = cleanup_transaction(self)
            except BaseException as exc_cleanup:
                cleanup_error = None
                cleanup_failure = exc_cleanup
            self._cleanup_error = cleanup_error
            if committed_artifact_cleanup_error is not None:
                self._cleanup_error = committed_artifact_cleanup_error
            if body_error is None and (cleanup_error is not None or cleanup_failure is not None):
                finalization_now = cleanup_now(self)
                cleanup_issue = cleanup_error or cleanup_failure
                try:
                    from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_phases

                    mark_touched_operation_phases(
                        self,
                        lease=lease,
                        status="finalization_failed",
                        now=finalization_now,
                    )
                except BaseException as transition_exc:
                    assert cleanup_issue is not None
                    cleanup_issue.add_note(f"transaction phase status update also failed: {transition_exc}")
                try:
                    from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_runs

                    mark_touched_operation_runs(
                        self,
                        lease=lease,
                        status="finalization_failed",
                        now=finalization_now,
                    )
                except BaseException as transition_exc:
                    assert cleanup_issue is not None
                    cleanup_issue.add_note(f"transaction operation status update also failed: {transition_exc}")
            if cleanup_error is not None and isinstance(exc, BaseException):
                record_cleanup_error(self, cleanup_error, exc)
            if cleanup_failure is not None and isinstance(exc, BaseException):
                exc.add_note(f"SafeWorkspace cleanup also failed: {cleanup_failure}")
            if cleanup_error is not None and exc_type is None:
                raise cleanup_error
            if cleanup_failure is not None and exc_type is None:
                raise cleanup_failure
            if pending_rollback_error is not None and exc_type is None:
                raise pending_rollback_error
            if committed_artifact_cleanup_error is not None and exc_type is None:
                raise committed_artifact_cleanup_error
            return False

    def write_text(
        self,
        resource: FileResource,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_file_resource(resource)
            assert resource.path is not None
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return self.workspace._coordinator.write_text_file(
                resource.path,
                content,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("write_text", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                encoding=encoding,
                newline=newline,
                now=now or self.now,
            )

    def write_bytes(
        self,
        resource: FileResource,
        content: bytes,
        *,
        permissions: int | None = None,
        max_bytes: int | None = None,
        allow_large: bool | None = None,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_file_resource(resource)
            assert resource.path is not None
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return self.workspace._coordinator.write_bytes_file(
                resource.path,
                content,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("write_bytes", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                permissions=permissions,
                max_bytes=max_bytes,
                allow_large=allow_large,
                now=now or self.now,
            )

    def delete_file(
        self,
        resource: FileResource,
        *,
        missing_ok: bool = False,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_file_resource(resource)
            assert resource.path is not None
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return self.workspace._coordinator.delete_file(
                resource.path,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("delete_file", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                missing_ok=missing_ok,
                now=now or self.now,
            )

    def make_directory(
        self,
        resource: DirectoryResource,
        *,
        parents: bool = False,
        exist_ok: bool = False,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_directory_resource(resource)
            assert resource.path is not None
            if parents:
                from safe_fs_ops.workspace_recursive_mkdir import (
                    RecursiveMkdirContext,
                    execute_transaction_recursive_mkdir,
                )

                operation_run_id, operation_phase_id = resolve_operation_links(
                    self,
                    operation_run_id=operation_run_id,
                    operation_phase_id=operation_phase_id,
                    now=now,
                    create_implicit=True,
                )
                return execute_transaction_recursive_mkdir(
                    RecursiveMkdirContext(
                        transaction=self,
                        operation_run_id=operation_run_id,
                        operation_phase_id=operation_phase_id,
                    ),
                    resource,
                    exist_ok=exist_ok,
                    idempotency_key=idempotency_key or self._next_idempotency_key("make_directory", resource),
                    now=now,
                )
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return self.workspace._coordinator.make_directory(
                resource.path,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("make_directory", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                parents=parents,
                exist_ok=exist_ok,
                now=now or self.now,
            )

    def snapshot_bundle(
        self,
        resource: ResourceHandle,
        paths: Iterable[Path | str] | Path | str | None = None,
        *,
        include_children: bool = False,
        hash_policy: HashPolicy = "metadata-only",
        small_file_max_bytes: int = 1024 * 1024,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_claimed_resource(resource)
            if paths is None:
                if resource.path is None:
                    raise ValueError("paths are required when snapshotting a custom resource")
                paths = resource.path
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            bundle = filesystem_snapshot_bundle(
                paths,
                include_children=include_children,
                hash_policy=hash_policy,
                small_file_max_bytes=small_file_max_bytes,
            )
            return self.workspace._coordinator.snapshot_bundle(
                bundle,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("snapshot_bundle", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                now=now or self.now,
            )

    def rename_no_replace(
        self,
        source: ResourceHandle,
        destination: ResourceHandle,
        *,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_claimed_resource(source)
            self._require_claimed_resource(destination)
            if source.path is None or destination.path is None:
                raise ValueError("rename_no_replace requires path-backed resources")
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return self.workspace._coordinator.rename_no_replace(
                source.path,
                destination.path,
                resource_key=source.resource_key,
                destination_resource_key=destination.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("rename_no_replace", source),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                now=now or self.now,
            )

    def capture_directory(
        self,
        resource: DirectoryResource,
        *,
        quarantine_path: Path | str,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_directory_resource(resource)
            assert resource.path is not None
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            if self.workspace.captured_directory_cleanup not in {"automatic", "retain"}:
                raise ValueError(
                    f"unsupported captured_directory_cleanup: {self.workspace.captured_directory_cleanup!r}"
                )
            self.workspace._coordinator._captured_directory_cleanup = self.workspace.captured_directory_cleanup
            quarantine_resource = self.workspace.directory(quarantine_path)
            self._claim_directory_resource(quarantine_resource.resource_key, now=now)
            return self.workspace._coordinator.capture_directory(
                resource.path,
                quarantine_path=quarantine_path,
                resource_key=resource.resource_key,
                destination_resource_key=quarantine_resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("capture_directory", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                now=now or self.now,
            )

    def backup_tree(
        self,
        resource: TreeResource,
        relative_paths: Iterable[Path | str],
        *,
        artifact_store: object | None = None,
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_tree_resource(resource)
            assert resource.path is not None
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return cast(_TreeBackupCoordinator, self.workspace._coordinator).backup_tree(
                resource.path,
                relative_paths,
                artifact_store=self.workspace.artifact_store if artifact_store is None else artifact_store,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("backup_tree", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                now=now or self.now,
            )

    def restore_tree_backup(
        self,
        resource: TreeResource,
        backup: object,
        *,
        destination_root: Path | str | None = None,
        artifact_store: object | None = None,
        conflict_policy: str = "no_replace",
        idempotency_key: str | None = None,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            self._require_tree_resource(resource)
            assert resource.path is not None
            if self.rollback == "automatic":
                raise SafeWorkspaceError(
                    "restore_tree_backup is not supported in automatic rollback transactions; "
                    "use rollback='record-only' or restore outside the transaction"
                )
            resolved_destination_root = resource.path if destination_root is None else Path(destination_root)
            if _canonical_directory_path(resolved_destination_root) != _canonical_directory_path(resource.path):
                raise ValueError("destination_root must match the claimed tree resource path")
            operation_run_id, operation_phase_id = resolve_operation_links(
                self,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
            )
            return cast(_TreeBackupCoordinator, self.workspace._coordinator).restore_tree_backup(
                backup,
                destination_root=resolved_destination_root,
                artifact_store=self.workspace.artifact_store if artifact_store is None else artifact_store,
                conflict_policy=conflict_policy,
                resource_key=resource.resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key or self._next_idempotency_key("restore_tree_backup", resource),
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                now=now or self.now,
            )

    def _make_directory_path(
        self,
        path: Path | str,
        *,
        resource_key: str,
        idempotency_key: str,
        operation_run_id: str | None = None,
        operation_phase_id: str | None = None,
        parents: bool = False,
        exist_ok: bool = False,
        intent_metadata: dict[str, object] | None = None,
        recovery_metadata: dict[str, object] | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        with self._state_lock:
            self._require_entered()
            return self.workspace._coordinator.make_directory(
                path,
                resource_key=resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                run_id=self.run_id,
                idempotency_key=idempotency_key,
                operation_run_id=operation_run_id,
                operation_phase_id=operation_phase_id,
                claim_scope=self.claim_scope,
                parents=parents,
                exist_ok=exist_ok,
                intent_metadata=intent_metadata,
                recovery_metadata=recovery_metadata,
                now=now or self.now,
            )

    def _claim_directory_resource(self, resource_key: str, *, now: datetime | None = None) -> ClaimRecord:
        with self._state_lock:
            self._require_entered()
            existing = self._claims_by_resource_key.get(resource_key)
            if existing is not None:
                return existing
            claim = self.workspace.claim_store.upsert(
                resource_key,
                lease=self.lease,
                owner=self.workspace.owner,
                scope=self.claim_scope,
                details=claim_details(
                    resource_kind="directory",
                    run_id=self.run_id,
                    lease=self.lease,
                ),
                now=now or self.now,
            )
            remember_claim(self, claim)
            return claim

    def _require_entered(self) -> None:
        with self._state_lock:
            if not self._entered or self._lease is None:
                raise SafeWorkspaceError("transaction is not active")

    def _require_file_resource(self, resource: FileResource) -> None:
        require_file_resource(
            resources=self.resources,
            resource=resource,
            resource_not_claimed_error=ResourceNotClaimedError,
        )

    def _require_directory_resource(self, resource: DirectoryResource) -> None:
        require_directory_resource(
            resources=self.resources,
            resource=resource,
            resource_not_claimed_error=ResourceNotClaimedError,
        )

    def _require_tree_resource(self, resource: TreeResource) -> None:
        if not isinstance(resource, TreeResource):
            raise TypeError("transaction tree operations require a TreeResource")
        self._require_claimed_resource(resource)

    def _require_claimed_resource(self, resource: ResourceHandle) -> None:
        require_claimed_resource(
            resources=self.resources,
            resource=resource,
            resource_not_claimed_error=ResourceNotClaimedError,
        )

    def _next_idempotency_key(self, operation: str, resource: ResourceHandle) -> str:
        with self._state_lock:
            self._operation_index, key = next_idempotency_key(
                name=self.name,
                run_id=self.run_id,
                operation_index=self._operation_index,
                operation=operation,
                resource=resource,
            )
            return key

    def _release_claims_at(self, now: datetime | None, *, tolerate_missing: bool = False) -> None:
        with self._state_lock:
            if self._lease is None:
                return
            for resource_key in reversed(tuple(self._claim_release_order)):
                expected_claim = self._claims_by_resource_key.get(resource_key)
                if expected_claim is None:
                    continue
                try:
                    released = self.workspace.claim_store.release(
                        resource_key,
                        lease=self._lease,
                        owner=self.workspace.owner,
                        scope=self.claim_scope,
                        expected_claim=expected_claim,
                        now=now,
                    )
                except LeaseLostError:
                    released = self.workspace.claim_store.release_if_owner_scope_matches(
                        resource_key,
                        lease=self._lease,
                        owner=self.workspace.owner,
                        scope=self.claim_scope,
                        expected_claim=expected_claim,
                        now=now,
                    )
                if released:
                    forget_claim(self, resource_key)
                    continue
                if tolerate_missing and claim_is_already_released(self, resource_key):
                    forget_claim(self, resource_key)
                    continue
                if not released:
                    raise _ClaimReleaseError(
                        f"claim {resource_key!r} could not be released for owner {self.workspace.owner!r}"
                    )

    def finalize_operation_success(self, operation_run_id: str) -> None:
        with self._state_lock:
            finalize_operation_success(self, operation_run_id)


def _mark_touched_finalization_failed(
    transaction: SafeTransaction,
    lease: LeaseRecord,
) -> BaseException | None:
    terminal_status_error: BaseException | None = None
    try:
        from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_phases

        mark_touched_operation_phases(
            transaction,
            lease=lease,
            status="finalization_failed",
            now=cleanup_now(transaction),
        )
    except BaseException as transition_exc:
        terminal_status_error = transition_exc
    try:
        from safe_fs_ops.workspace_transaction_operation import mark_touched_operation_runs

        mark_touched_operation_runs(
            transaction,
            lease=lease,
            status="finalization_failed",
            now=cleanup_now(transaction),
        )
    except BaseException as transition_exc:
        if terminal_status_error is None:
            terminal_status_error = transition_exc
        else:
            terminal_status_error.add_note(f"transaction operation status update also failed: {transition_exc}")
    return terminal_status_error


SafeTransaction._cleanup_transaction = cleanup_transaction  # type: ignore[attr-defined]
__all__ = ["SafeTransaction"]
