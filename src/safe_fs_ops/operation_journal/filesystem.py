from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    RenameRecord,
    ResourceSnapshot,
    atomic_write_bytes,
    atomic_write_text,
    capture_directory_to_quarantine,
    delete_file,
    make_directory,
    remove_empty_directory,
    remove_existing_empty_directory_by_identity,
    rename_no_replace,
    snapshot_resource,
)
from safe_fs_ops.operation_journal.captured_directory_recovery import (
    RESTORE_CAPTURED_DIRECTORY_ACTION,
    restore_captured_directory_recovery_action,
)
from safe_fs_ops.operation_journal.filesystem_mutations import JournaledFilesystemMutationMixin
from safe_fs_ops.operation_journal.filesystem_recovery import JournaledFilesystemRecoveryMixin
from safe_fs_ops.operation_journal.filesystem_support import (
    DirectoryResourceKeyMismatchError,
    FileResourceKeyMismatchError,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemError,
    JournaledFilesystemMutationError,
    JournaledFilesystemRecoveryError,
    JournaledFilesystemResult,
    MissingResourceClaimError,
    ResourceClaimAuthorityError,
    TreeResourceKeyMismatchError,
    _utcnow,
    directory_resource_key,
    file_resource_key,
    tree_resource_key,
)
from safe_fs_ops.operation_journal.journal import OperationJournalStore
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionHandler,
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
    _bool_from_payload_value,
    _default_recovery_action_handlers,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    REMOVE_CREATED_DIRECTORY_ACTION,
    remove_created_directory_recovery_action,
    remove_empty_directory_recovery_action,
)
from safe_fs_ops.operation_journal.rename_recovery_actions import (
    RESTORE_INVERSE_RENAME_ACTION,
    restore_inverse_rename_recovery_action,
)
from safe_fs_ops.workspace_state.claims import ClaimStore
from safe_fs_ops.workspace_state.leases import LeaseStore

__all__ = [
    "DeleteFileOperation",
    "DirectoryMakeOperation",
    "DirectoryIdentityRemoveOperation",
    "DirectoryCaptureOperation",
    "DirectoryRemoveOperation",
    "DirectoryResourceKeyMismatchError",
    "FileResourceKeyMismatchError",
    "JournaledFilesystemBatchStateError",
    "JournaledFilesystemCoordinator",
    "JournaledFilesystemError",
    "JournaledFilesystemMutationError",
    "JournaledFilesystemRecoveryError",
    "JournaledFilesystemResult",
    "MissingResourceClaimError",
    "RecoveryActionManualInterventionRequired",
    "RecoveryActionSkipped",
    "RenameNoReplaceOperation",
    "ResourceClaimAuthorityError",
    "TreeResourceKeyMismatchError",
    "SnapshotOperation",
    "WriteBytesLargePolicy",
    "WriteBytesOperation",
    "WriteTextOperation",
    "DEFAULT_WRITE_BYTES_MAX_BYTES",
    "_bool_from_payload_value",
    "directory_resource_key",
    "file_resource_key",
    "tree_resource_key",
]

DEFAULT_WRITE_BYTES_MAX_BYTES = 1024 * 1024
WriteBytesLargePolicy = Literal["reject", "allow"]


class SnapshotOperation(Protocol):
    def __call__(self, path: Path | str) -> ResourceSnapshot: ...


class WriteBytesOperation(Protocol):
    def __call__(self, path: Path | str, content: bytes) -> None: ...


class WriteTextOperation(Protocol):
    def __call__(
        self,
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None: ...


class DeleteFileOperation(Protocol):
    def __call__(self, path: Path | str, *, missing_ok: bool = False) -> None: ...


class DirectoryMakeOperation(Protocol):
    def __call__(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None: ...


class DirectoryRemoveOperation(Protocol):
    def __call__(self, path: Path | str, *, missing_ok: bool = False) -> None: ...


class DirectoryIdentityRemoveOperation(Protocol):
    def __call__(self, path: Path | str, *, expected_identity: DirectoryIdentity) -> None: ...


class RenameNoReplaceOperation(Protocol):
    def __call__(self, source: Path | str, destination: Path | str) -> RenameRecord: ...


class DirectoryCaptureOperation(Protocol):
    def __call__(self, source: Path | str, *, quarantine_path: Path | str) -> CapturedDirectoryRecord: ...


class OperationPreflightCheck(Protocol):
    def __call__(self, operation: str) -> None: ...


class JournaledFilesystemCoordinator(
    JournaledFilesystemMutationMixin,
    JournaledFilesystemRecoveryMixin,
):
    """Coordinates lease/claim checks, journal records, and file mutations."""

    def __init__(
        self,
        *,
        lease_store: LeaseStore,
        claim_store: ClaimStore,
        journal_store: OperationJournalStore,
        snapshot: SnapshotOperation = snapshot_resource,
        write_bytes_operation: WriteBytesOperation = atomic_write_bytes,
        write_text_operation: WriteTextOperation = atomic_write_text,
        delete_file_operation: DeleteFileOperation = delete_file,
        make_directory_operation: DirectoryMakeOperation = make_directory,
        remove_directory_operation: DirectoryRemoveOperation = remove_empty_directory,
        identity_remove_directory_operation: DirectoryIdentityRemoveOperation | None = (
            remove_existing_empty_directory_by_identity
        ),
        rename_no_replace_operation: RenameNoReplaceOperation = rename_no_replace,
        capture_directory_operation: DirectoryCaptureOperation = capture_directory_to_quarantine,
        preflight_operation: OperationPreflightCheck | None = None,
        automatic_file_rollback: bool = True,
        write_bytes_max_bytes: int = DEFAULT_WRITE_BYTES_MAX_BYTES,
        write_bytes_large_policy: WriteBytesLargePolicy = "reject",
        captured_directory_cleanup: Literal["automatic", "retain"] = "retain",
        recovery_action_handlers: Mapping[str, RecoveryActionHandler] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if write_bytes_max_bytes < 0:
            raise ValueError("write_bytes_max_bytes must be non-negative")
        if write_bytes_large_policy not in {"reject", "allow"}:
            raise ValueError(f"unsupported write_bytes_large_policy: {write_bytes_large_policy!r}")
        self._lease_store = lease_store
        self._claim_store = claim_store
        self._journal_store = journal_store
        self._snapshot = snapshot
        self._write_bytes = write_bytes_operation
        self._write_text = write_text_operation
        self._delete_file = delete_file_operation
        self._make_directory = make_directory_operation
        self._remove_directory = remove_directory_operation
        self._identity_remove_directory = identity_remove_directory_operation
        self._rename_no_replace = rename_no_replace_operation
        self._capture_directory = capture_directory_operation
        self._preflight_operation = preflight_operation
        self._automatic_file_rollback = automatic_file_rollback
        self._write_bytes_max_bytes = write_bytes_max_bytes
        self._write_bytes_large_policy = write_bytes_large_policy
        if captured_directory_cleanup not in {"automatic", "retain"}:
            raise ValueError(f"unsupported captured_directory_cleanup: {captured_directory_cleanup!r}")
        self._captured_directory_cleanup = captured_directory_cleanup
        self._recovery_action_handlers = dict(_default_recovery_action_handlers())
        self._recovery_action_handlers.setdefault(
            REMOVE_CREATED_DIRECTORY_ACTION,
            lambda context, action: remove_created_directory_recovery_action(
                self._remove_directory,
                self._identity_remove_directory,
                self._snapshot,
                context,
                action,
            ),
        )
        self._recovery_action_handlers.setdefault(
            RESTORE_CAPTURED_DIRECTORY_ACTION,
            lambda context, action: restore_captured_directory_recovery_action(
                context,
                action,
            ),
        )
        self._recovery_action_handlers.setdefault(
            RESTORE_INVERSE_RENAME_ACTION,
            lambda context, action: restore_inverse_rename_recovery_action(context, action),
        )
        self._recovery_action_handlers.setdefault(
            "remove_empty_directory",
            lambda context, action: remove_empty_directory_recovery_action(
                self._remove_directory,
                self._identity_remove_directory,
                self._snapshot,
                context,
                action,
            ),
        )
        if recovery_action_handlers is not None:
            self._recovery_action_handlers.update(recovery_action_handlers)
        self._clock = clock or _utcnow
        self._heartbeat_leases = clock is None

    def _operation_time(self, now: datetime | None) -> datetime:
        return _utcnow(now) if now is not None else _utcnow(self._clock())
