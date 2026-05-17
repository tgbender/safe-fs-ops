from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from safe_fs_ops.filesystem_ops import HashPolicy
from safe_fs_ops.operation_journal import JournaledFilesystemResult
from safe_fs_ops.operation_journal.models import OperationPhaseRecord, OperationRunRecord
from safe_fs_ops.resources import DirectoryResource, FileResource, ResourceHandle, ResourceSet, TreeResource
from safe_fs_ops.workspace import SafeWorkspaceError
from safe_fs_ops.workspace_operation_status import (
    reconcile_active_phase_on_operation_exit,
    record_operation_finalization_failure,
    update_operation_run_terminal_status,
    update_phase_terminal_status,
)
from safe_fs_ops.workspace_rollback import cleanup_committed_transaction_artifacts
from safe_fs_ops.workspace_state.models import LeaseRecord
from safe_fs_ops.workspace_transaction import SafeTransaction
from safe_fs_ops.workspace_transaction_bookkeeping import cleanup_now, cleanup_transaction
from safe_fs_ops.workspace_transaction_state import close_from_operation_exit, remember_touched_operation_links


@dataclass(slots=True)
class SafeOperation:
    _transaction: SafeTransaction
    _active_phase: SafePhase | None = field(default=None, init=False, repr=False)
    _entered: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _operation_run: OperationRunRecord | None = field(default=None, init=False, repr=False)
    _phase_entry_count: int = field(default=0, init=False, repr=False)
    _phase_names: set[str] = field(default_factory=set, init=False, repr=False)
    _state_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def r(self) -> ResourceSet:
        return self._transaction.r

    @property
    def lease(self) -> LeaseRecord:
        return self._transaction.lease

    @property
    def cleanup_error(self) -> SafeWorkspaceError | None:
        return self._transaction.cleanup_error

    @property
    def operation_run(self) -> OperationRunRecord | None:
        with self._state_lock:
            return self._operation_run

    def __enter__(self) -> SafeOperation:
        with self._state_lock:
            if self._transaction.rollback != "record-only":
                raise NotImplementedError(
                    f"rollback={self._transaction.rollback!r} is not implemented for phased operations"
                )
            if self._closed:
                raise SafeWorkspaceError("operation is single-use and has already been closed")
            if self._entered:
                raise SafeWorkspaceError("operation is already active")
            self._transaction.__enter__()
            try:
                self._operation_run = self._transaction.workspace.journal_store.create_operation_run(
                    run_id=self._transaction.run_id,
                    lease=self._transaction.lease,
                    owner=self._transaction.workspace.owner,
                    status="active",
                    payload={
                        "name": self._transaction.name,
                        "rollback": self._transaction.rollback,
                    },
                    now=self._transaction.now,
                )
                remember_touched_operation_links(
                    self._transaction,
                    operation_run_id=self._operation_run.operation_run_id,
                    operation_phase_id=None,
                )
            except BaseException as exc:
                self._transaction.__exit__(type(exc), exc, exc.__traceback__)
                raise
            self._entered = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        with self._state_lock:
            if not self._entered:
                return False
            body_error = exc if isinstance(exc, BaseException) else None
            incoming_body_error = body_error
            finalization_now = cleanup_now(self._transaction)
            lease = self.lease
            operation_exit_error: SafeWorkspaceError | None = None
            status_error: BaseException | None = None
            self._entered = False
            self._closed = True
            if self._active_phase is not None:
                active_phase_error, phase_status_error = reconcile_active_phase_on_operation_exit(
                    self,
                    phase=self._active_phase,
                    now=finalization_now,
                )
                if body_error is None:
                    body_error = active_phase_error
                    operation_exit_error = active_phase_error
                else:
                    body_error.add_note(f"operation exit also reconciled an active phase: {active_phase_error}")
                if status_error is None:
                    status_error = phase_status_error
            self._active_phase = None
            if body_error is None and self._operation_run is not None:
                try:
                    close_from_operation_exit(self._transaction)
                    artifact_cleanup_error = cleanup_committed_transaction_artifacts(self._transaction)
                    self._transaction._cleanup_error = artifact_cleanup_error
                    if artifact_cleanup_error is not None:
                        raise artifact_cleanup_error
                    self._transaction.finalize_operation_success(self._operation_run.operation_run_id)
                    self._operation_run = self._transaction.workspace.journal_store.get_operation_run(
                        self._operation_run.operation_run_id
                    )
                    return False
                except BaseException as success_exc:
                    cleanup_failure = cleanup_transaction(self._transaction)
                    self._transaction._cleanup_error = cleanup_failure
                    status_error = record_operation_finalization_failure(
                        self._transaction,
                        lease=lease,
                        status="finalization_failed",
                        now=finalization_now,
                    )
                    primary_error: BaseException = cleanup_failure or success_exc
                    if status_error is not None:
                        primary_error.add_note(f"operation terminal diagnostic also failed: {status_error}")
                    if primary_error is success_exc:
                        raise
                    raise primary_error from success_exc
            try:
                if self._operation_run is not None:
                    self._operation_run = update_operation_run_terminal_status(
                        self,
                        lease=lease,
                        status="failed",
                        now=finalization_now,
                    )
            except BaseException as transition_exc:
                status_error = transition_exc
                if body_error is not None:
                    body_error.add_note(f"operation status update also failed: {transition_exc}")
            cleanup_error: BaseException | None = None
            transaction_exc = body_error or operation_exit_error
            try:
                if transaction_exc is None:
                    self._transaction.__exit__(exc_type, exc, traceback)
                else:
                    self._transaction.__exit__(type(transaction_exc), transaction_exc, transaction_exc.__traceback__)
            except BaseException as transaction_exc:
                cleanup_error = transaction_exc
            if incoming_body_error is not None:
                return False
            if cleanup_error is not None:
                if operation_exit_error is not None:
                    cleanup_error.add_note(f"operation exit also failed: {operation_exit_error}")
                if status_error is not None:
                    cleanup_error.add_note(f"operation status update also failed: {status_error}")
                raise cleanup_error
            if operation_exit_error is not None:
                if status_error is not None:
                    operation_exit_error.add_note(f"operation status update also failed: {status_error}")
                raise operation_exit_error
            if status_error is not None:
                raise status_error
            return False

    def phase(self, name: str, *, rollback: Literal["record-only"] = "record-only") -> SafePhase:
        if not name:
            raise ValueError("phase name must be non-empty")
        with self._state_lock:
            if name in self._phase_names:
                raise SafeWorkspaceError(f"duplicate phase name {name!r} is not allowed within one operation")
            self._phase_names.add(name)
        return SafePhase(operation=self, name=name, rollback=rollback)

    def _enter_phase(self, phase: SafePhase) -> None:
        with self._state_lock:
            if not self._entered:
                raise SafeWorkspaceError("operation is not active")
            if self._active_phase is not None:
                raise SafeWorkspaceError("another phase is already active")
            if phase.rollback != "record-only":
                raise NotImplementedError(f"rollback={phase.rollback!r} is not implemented for phased operations")
            if self._operation_run is None:
                raise SafeWorkspaceError("operation has no durable operation run record")
            phase_order = self._phase_entry_count + 1
            phase_record = self._transaction.workspace.journal_store.create_operation_phase(
                operation_run_id=self._operation_run.operation_run_id,
                lease=self.lease,
                phase_name=phase.name,
                status="active",
                phase_order=phase_order,
                payload={"rollback": phase.rollback},
                now=self._transaction.now,
            )
            phase._set_phase_record(phase_record)
            remember_touched_operation_links(
                self._transaction,
                operation_run_id=self._operation_run.operation_run_id,
                operation_phase_id=phase_record.operation_phase_id,
            )
            self._phase_entry_count = phase_order
            self._active_phase = phase

    def _exit_phase(self, phase: SafePhase) -> None:
        with self._state_lock:
            if self._active_phase is phase:
                self._active_phase = None

    def _is_active_phase(self, phase: SafePhase) -> bool:
        with self._state_lock:
            return self._active_phase is phase


@dataclass(slots=True)
class SafePhase:
    operation: SafeOperation
    name: str
    rollback: Literal["record-only"] = "record-only"
    _entered: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _phase_record: OperationPhaseRecord | None = field(default=None, init=False, repr=False)
    _operation_index: int = field(default=0, init=False, repr=False)
    _closed_by_operation_exit: bool = field(default=False, init=False, repr=False)
    _state_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @property
    def r(self) -> ResourceSet:
        return self.operation.r

    @property
    def phase_record(self) -> OperationPhaseRecord | None:
        with self._state_lock:
            return self._phase_record

    def __enter__(self) -> SafePhase:
        with self._state_lock:
            if self._closed:
                raise SafeWorkspaceError("phase is single-use and has already been closed")
            if self._entered:
                raise SafeWorkspaceError("phase is already active")
        self.operation._enter_phase(self)
        with self._state_lock:
            self._entered = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> Literal[False]:
        with self.operation._state_lock:
            with self._state_lock:
                if not self._entered:
                    return False
                closed_by_operation_exit = self._closed_by_operation_exit
                if closed_by_operation_exit:
                    self._entered = False
                    self._closed = True
                else:
                    body_error = exc if isinstance(exc, BaseException) else None
                    finalization_now = cleanup_now(self.operation._transaction)
                    status_error: BaseException | None = None
                    try:
                        if self.phase_record is not None:
                            self._set_phase_record(
                                update_phase_terminal_status(
                                    self,
                                    status="failed" if body_error is not None else "succeeded",
                                    now=finalization_now,
                                )
                            )
                    except BaseException as transition_exc:
                        status_error = transition_exc
                        if body_error is not None:
                            body_error.add_note(f"phase status update also failed: {transition_exc}")
                    self._entered = False
                    self._closed = True
            self.operation._exit_phase(self)
        if closed_by_operation_exit:
            return False
        if body_error is None and status_error is not None:
            raise status_error
        return False

    def write_text(
        self,
        resource: FileResource,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="file",
            mutate=lambda: self.operation._transaction.write_text(
                resource,
                content,
                encoding=encoding,
                newline=newline,
                idempotency_key=idempotency_key or self._next_idempotency_key("write_text", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
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
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="file",
            mutate=lambda: self.operation._transaction.write_bytes(
                resource,
                content,
                permissions=permissions,
                max_bytes=max_bytes,
                allow_large=allow_large,
                idempotency_key=idempotency_key or self._next_idempotency_key("write_bytes", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def delete_file(
        self,
        resource: FileResource,
        *,
        missing_ok: bool = False,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="file",
            mutate=lambda: self.operation._transaction.delete_file(
                resource,
                missing_ok=missing_ok,
                idempotency_key=idempotency_key or self._next_idempotency_key("delete_file", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def make_directory(
        self,
        resource: DirectoryResource,
        *,
        parents: bool = False,
        exist_ok: bool = False,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        def mutate() -> JournaledFilesystemResult:
            if parents:
                from safe_fs_ops.workspace_recursive_mkdir import execute_phase_recursive_mkdir

                return execute_phase_recursive_mkdir(
                    self,
                    resource,
                    exist_ok=exist_ok,
                    idempotency_key=idempotency_key or self._next_idempotency_key("make_directory", resource),
                    now=now,
                )
            return self.operation._transaction.make_directory(
                resource,
                parents=parents,
                exist_ok=exist_ok,
                idempotency_key=idempotency_key or self._next_idempotency_key("make_directory", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            )

        return self._run_mutation(resource, resource_kind="directory", mutate=mutate)

    def snapshot_bundle(
        self,
        resource: ResourceHandle,
        paths: Iterable[Path | str] | Path | str | None = None,
        *,
        include_children: bool = False,
        hash_policy: HashPolicy = "metadata-only",
        small_file_max_bytes: int = 1024 * 1024,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="any",
            mutate=lambda: self.operation._transaction.snapshot_bundle(
                resource,
                paths,
                include_children=include_children,
                hash_policy=hash_policy,
                small_file_max_bytes=small_file_max_bytes,
                idempotency_key=idempotency_key or self._next_idempotency_key("snapshot_bundle", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def rename_no_replace(
        self,
        source: ResourceHandle,
        destination: ResourceHandle,
        *,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            source,
            resource_kind="any",
            mutate=lambda: self.operation._transaction.rename_no_replace(
                source,
                destination,
                idempotency_key=idempotency_key or self._next_idempotency_key("rename_no_replace", source),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def capture_directory(
        self,
        resource: DirectoryResource,
        *,
        quarantine_path: Path | str,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="directory",
            mutate=lambda: self.operation._transaction.capture_directory(
                resource,
                quarantine_path=quarantine_path,
                idempotency_key=idempotency_key or self._next_idempotency_key("capture_directory", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def backup_tree(
        self,
        resource: TreeResource,
        relative_paths: Iterable[Path | str],
        *,
        artifact_store: object | None = None,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="tree",
            mutate=lambda: self.operation._transaction.backup_tree(
                resource,
                relative_paths,
                artifact_store=artifact_store,
                idempotency_key=idempotency_key or self._next_idempotency_key("backup_tree", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
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
        now: datetime | None = None,
    ) -> JournaledFilesystemResult:
        return self._run_mutation(
            resource,
            resource_kind="tree",
            mutate=lambda: self.operation._transaction.restore_tree_backup(
                resource,
                backup,
                destination_root=destination_root,
                artifact_store=artifact_store,
                conflict_policy=conflict_policy,
                idempotency_key=idempotency_key or self._next_idempotency_key("restore_tree_backup", resource),
                operation_run_id=self._operation_run_id(),
                operation_phase_id=self._phase_id(),
                now=now,
            ),
        )

    def _run_mutation(
        self,
        resource: ResourceHandle,
        *,
        resource_kind: Literal["any", "directory", "file", "tree"],
        mutate: Callable[[], JournaledFilesystemResult],
    ) -> JournaledFilesystemResult:
        with self.operation._state_lock, self._state_lock:
            self._require_active_locked()
            if resource_kind == "any":
                self.operation._transaction._require_claimed_resource(resource)
            elif resource_kind == "directory":
                self.operation._transaction._require_directory_resource(resource)  # type: ignore[arg-type]
            elif resource_kind == "tree":
                self.operation._transaction._require_tree_resource(resource)  # type: ignore[arg-type]
            else:
                self.operation._transaction._require_file_resource(resource)  # type: ignore[arg-type]
            return mutate()

    def _require_active_locked(self) -> None:
        if not self._entered or self._closed:
            raise SafeWorkspaceError("phase is not active")
        if self.operation._active_phase is not self:
            raise SafeWorkspaceError("phase is not active")

    def _next_idempotency_key(self, operation: str, resource: ResourceHandle) -> str:
        with self._state_lock:
            self._operation_index += 1
            operation_index = self._operation_index
        label = resource.label or resource.resource_key
        transaction = self.operation._transaction
        return f"{transaction.name}:{transaction.run_id}:{self.name}:{operation_index}:{operation}:{label}"

    def _operation_run_id(self) -> str:
        operation_run = self.operation.operation_run
        if operation_run is None:
            raise SafeWorkspaceError("operation has no durable operation run record")
        return operation_run.operation_run_id

    def _phase_id(self) -> str:
        with self._state_lock:
            if self._phase_record is None:
                raise SafeWorkspaceError("phase has no durable phase record")
            return self._phase_record.operation_phase_id

    def _set_phase_record(self, phase_record: OperationPhaseRecord) -> None:
        with self._state_lock:
            self._phase_record = phase_record

    def _close_due_to_operation_exit(self) -> None:
        with self._state_lock:
            self._entered = False
            self._closed = True
            self._closed_by_operation_exit = True


__all__ = ["SafeOperation", "SafePhase"]
