from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from safe_fs_ops.filesystem_ops import (
    ContentAddressedStore,
    DirectoryIdentity,
    DurabilityMode,
    FilesystemBackendCapabilities,
    PathSafety,
    RemoveDirectoryHooks,
    atomic_write_bytes,
    atomic_write_text,
    custom_backend_capabilities,
    detect_default_backend_capabilities,
    inspect_path,
    snapshot_resource,
)
from safe_fs_ops.filesystem_ops import (
    capture_directory_to_quarantine as filesystem_capture_directory,
)
from safe_fs_ops.filesystem_ops import (
    delete_file as filesystem_delete_file,
)
from safe_fs_ops.filesystem_ops import (
    make_directory as filesystem_make_directory,
)
from safe_fs_ops.filesystem_ops import (
    remove_empty_directory as filesystem_remove_empty_directory,
)
from safe_fs_ops.filesystem_ops import (
    remove_existing_empty_directory_by_identity as filesystem_remove_empty_directory_by_identity,
)
from safe_fs_ops.filesystem_ops import (
    rename_no_replace as filesystem_rename_no_replace,
)
from safe_fs_ops.filesystem_ops._windows_operations import (
    atomic_write_bytes_windows,
    atomic_write_text_windows,
    delete_file_windows,
    make_directory_windows,
)
from safe_fs_ops.filesystem_ops._windows_snapshots import snapshot_resource_windows
from safe_fs_ops.operation_journal import JournaledFilesystemCoordinator, OperationJournalStore
from safe_fs_ops.operation_journal.filesystem import (
    DEFAULT_WRITE_BYTES_MAX_BYTES,
    DeleteFileOperation,
    DirectoryCaptureOperation,
    DirectoryIdentityRemoveOperation,
    DirectoryMakeOperation,
    DirectoryRemoveOperation,
    RenameNoReplaceOperation,
    SnapshotOperation,
    WriteBytesLargePolicy,
    WriteBytesOperation,
    WriteTextOperation,
)
from safe_fs_ops.resources import (
    DirectoryResource,
    FileResource,
    ResourceHandle,
    ResourceSet,
    TreeResource,
    custom_resource,
    directory_resource,
    file_resource,
    tree_resource,
)
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore

if TYPE_CHECKING:
    from safe_fs_ops.operation_journal.legacy_captures import LegacyCapture
    from safe_fs_ops.workspace_operation import SafeOperation
    from safe_fs_ops.workspace_transaction import SafeTransaction


class SafeWorkspaceError(RuntimeError):
    pass


class SafeWorkspaceBusyError(SafeWorkspaceError):
    pass


class ResourceNotClaimedError(SafeWorkspaceError):
    pass


class _ClaimReleaseError(RuntimeError):
    pass


@dataclass(slots=True)
class SafeWorkspace:
    state_path: Path
    owner: str
    lease_name: str = "workspace"
    lease_ttl: timedelta = timedelta(seconds=30)
    durability: DurabilityMode = DurabilityMode.FSYNC
    snapshot: SnapshotOperation = snapshot_resource
    write_bytes_operation: WriteBytesOperation = atomic_write_bytes
    write_text_operation: WriteTextOperation = atomic_write_text
    delete_file_operation: DeleteFileOperation = filesystem_delete_file
    make_directory_operation: DirectoryMakeOperation = filesystem_make_directory
    remove_directory_operation: DirectoryRemoveOperation = filesystem_remove_empty_directory
    identity_remove_directory_operation: DirectoryIdentityRemoveOperation | None = (
        filesystem_remove_empty_directory_by_identity
    )
    rename_no_replace_operation: RenameNoReplaceOperation = filesystem_rename_no_replace
    capture_directory_operation: DirectoryCaptureOperation = filesystem_capture_directory
    path_inspector: Callable[[Path], PathSafety] = inspect_path
    enable_automatic_file_rollback: bool | None = None
    backup_artifact_cleanup_operation: Callable[[Path], None] | None = None
    artifact_store: object | None = None
    write_bytes_max_bytes: int = DEFAULT_WRITE_BYTES_MAX_BYTES
    write_bytes_large_policy: WriteBytesLargePolicy = "reject"
    captured_directory_cleanup: Literal["automatic", "retain"] = "retain"
    _write_bytes_accepts_durability: bool = field(init=False, repr=False)
    _write_bytes_accepts_permissions: bool = field(init=False, repr=False)
    _write_text_accepts_durability: bool = field(init=False, repr=False)
    _delete_file_accepts_durability: bool = field(init=False, repr=False)
    _make_directory_accepts_durability: bool = field(init=False, repr=False)
    _remove_directory_accepts_durability: bool = field(init=False, repr=False)
    _remove_directory_accepts_hooks: bool = field(init=False, repr=False)
    _lease_store: LeaseStore = field(init=False, repr=False)
    _claim_store: ClaimStore = field(init=False, repr=False)
    _journal_store: OperationJournalStore = field(init=False, repr=False)
    _coordinator: JournaledFilesystemCoordinator = field(init=False, repr=False)
    _filesystem_backend: FilesystemBackendCapabilities = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("owner must be non-empty")
        if self.write_bytes_max_bytes < 0:
            raise ValueError("write_bytes_max_bytes must be non-negative")
        if self.write_bytes_large_policy not in {"reject", "allow"}:
            raise ValueError(f"unsupported write_bytes_large_policy: {self.write_bytes_large_policy!r}")
        if self.captured_directory_cleanup not in {"automatic", "retain"}:
            raise ValueError(f"unsupported captured_directory_cleanup: {self.captured_directory_cleanup!r}")
        self.state_path = Path(self.state_path).expanduser()
        if self.artifact_store is None:
            self.artifact_store = ContentAddressedStore(self.state_path.parent / f"{self.state_path.name}.objects")
        _apply_platform_default_filesystem_operations(self)
        self._write_bytes_accepts_durability = _supports_keyword_argument(self.write_bytes_operation, "durability")
        self._write_bytes_accepts_permissions = _supports_keyword_argument(self.write_bytes_operation, "permissions")
        self._write_text_accepts_durability = _supports_keyword_argument(self.write_text_operation, "durability")
        self._delete_file_accepts_durability = _supports_keyword_argument(self.delete_file_operation, "durability")
        self._make_directory_accepts_durability = _supports_keyword_argument(
            self.make_directory_operation,
            "durability",
        )
        self._remove_directory_accepts_durability = _supports_keyword_argument(
            self.remove_directory_operation,
            "durability",
        )
        self._remove_directory_accepts_hooks = _supports_keyword_argument(self.remove_directory_operation, "hooks")
        self._filesystem_backend = _detect_workspace_backend(self)
        automatic_file_rollback = (
            self._filesystem_backend.is_default_backend
            if self.enable_automatic_file_rollback is None
            else self.enable_automatic_file_rollback
        )
        self._lease_store = LeaseStore(self.state_path)
        self._claim_store = ClaimStore(self.state_path)
        self._journal_store = OperationJournalStore(self.state_path)
        self._coordinator = JournaledFilesystemCoordinator(
            lease_store=self._lease_store,
            claim_store=self._claim_store,
            journal_store=self._journal_store,
            snapshot=self.snapshot,
            write_bytes_operation=self._write_bytes,
            write_text_operation=self._write_text,
            delete_file_operation=self._delete_file,
            make_directory_operation=self._make_directory,
            remove_directory_operation=self._remove_directory,
            identity_remove_directory_operation=(
                None if self.identity_remove_directory_operation is None else self._identity_remove_directory_recovery
            ),
            rename_no_replace_operation=self.rename_no_replace_operation,
            capture_directory_operation=self.capture_directory_operation,
            preflight_operation=self._require_filesystem_backend_operation,
            automatic_file_rollback=automatic_file_rollback,
            write_bytes_max_bytes=self.write_bytes_max_bytes,
            write_bytes_large_policy=self.write_bytes_large_policy,
            captured_directory_cleanup=self.captured_directory_cleanup,
        )

    @classmethod
    def open(
        cls,
        state_path: Path | str,
        *,
        owner: str,
        lease_name: str = "workspace",
        lease_ttl: timedelta = timedelta(seconds=30),
        durability: DurabilityMode = DurabilityMode.FSYNC,
        snapshot: SnapshotOperation = snapshot_resource,
        write_bytes_operation: WriteBytesOperation = atomic_write_bytes,
        write_text_operation: WriteTextOperation = atomic_write_text,
        delete_file_operation: DeleteFileOperation = filesystem_delete_file,
        make_directory_operation: DirectoryMakeOperation = filesystem_make_directory,
        remove_directory_operation: DirectoryRemoveOperation = filesystem_remove_empty_directory,
        identity_remove_directory_operation: DirectoryIdentityRemoveOperation | None = (
            filesystem_remove_empty_directory_by_identity
        ),
        rename_no_replace_operation: RenameNoReplaceOperation = filesystem_rename_no_replace,
        capture_directory_operation: DirectoryCaptureOperation = filesystem_capture_directory,
        path_inspector: Callable[[Path], PathSafety] = inspect_path,
        enable_automatic_file_rollback: bool | None = None,
        backup_artifact_cleanup_operation: Callable[[Path], None] | None = None,
        artifact_store: object | None = None,
        write_bytes_max_bytes: int = DEFAULT_WRITE_BYTES_MAX_BYTES,
        write_bytes_large_policy: WriteBytesLargePolicy = "reject",
        captured_directory_cleanup: Literal["automatic", "retain"] = "retain",
    ) -> SafeWorkspace:
        return cls(
            state_path=Path(state_path),
            owner=owner,
            lease_name=lease_name,
            lease_ttl=lease_ttl,
            durability=durability,
            snapshot=snapshot,
            write_bytes_operation=write_bytes_operation,
            write_text_operation=write_text_operation,
            delete_file_operation=delete_file_operation,
            make_directory_operation=make_directory_operation,
            remove_directory_operation=remove_directory_operation,
            identity_remove_directory_operation=identity_remove_directory_operation,
            rename_no_replace_operation=rename_no_replace_operation,
            capture_directory_operation=capture_directory_operation,
            path_inspector=path_inspector,
            enable_automatic_file_rollback=enable_automatic_file_rollback,
            backup_artifact_cleanup_operation=backup_artifact_cleanup_operation,
            artifact_store=artifact_store,
            write_bytes_max_bytes=write_bytes_max_bytes,
            write_bytes_large_policy=write_bytes_large_policy,
            captured_directory_cleanup=captured_directory_cleanup,
        )

    @property
    def journal_store(self) -> OperationJournalStore:
        return self._journal_store

    @property
    def claim_store(self) -> ClaimStore:
        return self._claim_store

    @property
    def lease_store(self) -> LeaseStore:
        return self._lease_store

    @property
    def filesystem_backend(self) -> FilesystemBackendCapabilities:
        return self._filesystem_backend

    def cleanup_outstanding_artifacts(self, *, run_id: str | None = None) -> SafeWorkspaceError | None:
        from safe_fs_ops.workspace_rollback import cleanup_outstanding_workspace_artifacts

        return cleanup_outstanding_workspace_artifacts(self, run_id=run_id)

    def recover_pending_batches(self, *, run_id: str | None = None) -> SafeWorkspaceError | None:
        from safe_fs_ops.workspace_rollback import recover_pending_workspace_batches

        return recover_pending_workspace_batches(self, run_id=run_id)

    def list_legacy_captures(self, *, run_id: str | None = None) -> tuple[LegacyCapture, ...]:
        from safe_fs_ops.workspace_legacy_captures import list_legacy_captures

        return list_legacy_captures(self, run_id=run_id)

    def recover_legacy_capture(
        self,
        candidate: LegacyCapture,
        *,
        confirm_ownership: bool,
        reason: str,
        _after_stage: Callable[[str], None] | None = None,
    ) -> None:
        from safe_fs_ops.workspace_legacy_captures import recover_legacy_capture

        recover_legacy_capture(
            self, candidate, confirm_ownership=confirm_ownership, reason=reason, after_stage=_after_stage
        )

    def _require_filesystem_backend_operation(self, operation: str) -> None:
        self._filesystem_backend.require(operation)

    def file(self, path: Path | str, *, scope: str | None = None) -> FileResource:
        return file_resource(path, scope=scope)

    def directory(self, path: Path | str, *, scope: str | None = None) -> DirectoryResource:
        return directory_resource(path, scope=scope)

    def tree(self, path: Path | str, *, scope: str | None = None) -> TreeResource:
        return tree_resource(path, scope=scope)

    def resource(self, resource_key: str, *, scope: str | None = None) -> ResourceHandle:
        return custom_resource(resource_key, scope=scope)

    def resources(self, resources: Mapping[str, ResourceHandle]) -> ResourceSet:
        return ResourceSet(resources)

    def transaction(
        self,
        *,
        name: str,
        resources: ResourceSet | Mapping[str, ResourceHandle],
        run_id: str | None = None,
        rollback: Literal["record-only", "automatic"] = "record-only",
        now: datetime | None = None,
        cleanup_clock: Callable[[], datetime] | None = None,
    ) -> SafeTransaction:
        resource_set = resources if isinstance(resources, ResourceSet) else ResourceSet(resources)
        from safe_fs_ops.workspace_transaction import SafeTransaction

        return SafeTransaction(
            workspace=self,
            name=name,
            resources=resource_set,
            run_id=run_id or uuid.uuid4().hex,
            rollback=rollback,
            now=now,
            cleanup_clock=cleanup_clock,
        )

    def operation(
        self,
        *,
        name: str,
        resources: ResourceSet | Mapping[str, ResourceHandle],
        run_id: str | None = None,
        rollback: Literal["record-only"] = "record-only",
        now: datetime | None = None,
        cleanup_clock: Callable[[], datetime] | None = None,
    ) -> SafeOperation:
        from safe_fs_ops.workspace_operation import SafeOperation

        return SafeOperation(
            _transaction=self.transaction(
                name=name,
                resources=resources,
                run_id=run_id,
                rollback=rollback,
                now=now,
                cleanup_clock=cleanup_clock,
            )
        )

    def _write_text(
        self,
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        self._filesystem_backend.require("write_text")
        if self._write_text_accepts_durability:
            cast(Callable[..., None], self.write_text_operation)(
                path,
                content,
                encoding=encoding,
                newline=newline,
                durability=self.durability,
            )
            return
        self.write_text_operation(path, content, encoding=encoding, newline=newline)

    def _write_bytes(
        self,
        path: Path | str,
        content: bytes,
        *,
        permissions: int | None = None,
    ) -> None:
        self._filesystem_backend.require("write_text")
        kwargs: dict[str, object] = {}
        if permissions is not None:
            if not self._write_bytes_accepts_permissions:
                raise ValueError("configured write_bytes_operation does not accept permissions")
            kwargs["permissions"] = permissions
        if self._write_bytes_accepts_durability:
            kwargs["durability"] = self.durability
        cast(Callable[..., None], self.write_bytes_operation)(path, content, **kwargs)

    def _delete_file(self, path: Path | str, *, missing_ok: bool = False) -> None:
        self._filesystem_backend.require("delete_file")
        if self._delete_file_accepts_durability:
            cast(Callable[..., None], self.delete_file_operation)(
                path,
                missing_ok=missing_ok,
                durability=self.durability,
            )
            return
        self.delete_file_operation(path, missing_ok=missing_ok)

    def _make_directory(self, path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
        self._filesystem_backend.require("make_directory")
        if self._make_directory_accepts_durability:
            cast(Callable[..., None], self.make_directory_operation)(
                path,
                parents=parents,
                exist_ok=exist_ok,
                durability=self.durability,
            )
            return
        self.make_directory_operation(path, parents=parents, exist_ok=exist_ok)

    def _remove_directory(
        self,
        path: Path | str,
        *,
        missing_ok: bool = False,
        hooks: RemoveDirectoryHooks | None = None,
    ) -> None:
        self._filesystem_backend.require("remove_directory")
        kwargs: dict[str, object] = {"missing_ok": missing_ok}
        if self._remove_directory_accepts_durability:
            kwargs["durability"] = self.durability
        if hooks is not None and self._remove_directory_accepts_hooks:
            kwargs["hooks"] = hooks
        if self._remove_directory_accepts_durability:
            cast(Callable[..., None], self.remove_directory_operation)(path, **kwargs)
            return
        cast(Callable[..., None], self.remove_directory_operation)(path, **kwargs)

    def _identity_remove_directory_recovery(
        self,
        path: Path | str,
        *,
        expected_identity: DirectoryIdentity,
    ) -> None:
        if self.identity_remove_directory_operation is None:
            raise RuntimeError("identity_remove_directory_operation is not configured")
        self._filesystem_backend.require("remove_directory")
        self.identity_remove_directory_operation(path, expected_identity=expected_identity)


def _supports_keyword_argument(operation: Callable[..., object], keyword: str) -> bool:
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD or parameter.name == keyword for parameter in parameters)


def _detect_workspace_backend(workspace: SafeWorkspace) -> FilesystemBackendCapabilities:
    if _matches_posix_default_backend(workspace) or _matches_windows_default_backend(workspace):
        return detect_default_backend_capabilities()
    return custom_backend_capabilities()


def _matches_posix_default_backend(workspace: SafeWorkspace) -> bool:
    return (
        workspace.snapshot is snapshot_resource
        and workspace.write_bytes_operation is atomic_write_bytes
        and workspace.write_text_operation is atomic_write_text
        and workspace.delete_file_operation is filesystem_delete_file
        and workspace.make_directory_operation is filesystem_make_directory
        and workspace.remove_directory_operation is filesystem_remove_empty_directory
        and workspace.identity_remove_directory_operation is filesystem_remove_empty_directory_by_identity
        and workspace.path_inspector is inspect_path
    )


def _matches_windows_default_backend(workspace: SafeWorkspace) -> bool:
    return (
        workspace.snapshot is snapshot_resource_windows
        and workspace.write_bytes_operation is atomic_write_bytes_windows
        and workspace.write_text_operation is atomic_write_text_windows
        and workspace.delete_file_operation is delete_file_windows
        and workspace.make_directory_operation is make_directory_windows
        and workspace.remove_directory_operation is filesystem_remove_empty_directory
        and workspace.identity_remove_directory_operation is filesystem_remove_empty_directory_by_identity
        and workspace.path_inspector is inspect_path
    )


def _apply_platform_default_filesystem_operations(workspace: SafeWorkspace) -> None:
    if os.name != "nt":
        return
    if workspace.snapshot is snapshot_resource:
        workspace.snapshot = snapshot_resource_windows
    if workspace.write_bytes_operation is atomic_write_bytes:
        workspace.write_bytes_operation = atomic_write_bytes_windows
    if workspace.write_text_operation is atomic_write_text:
        workspace.write_text_operation = atomic_write_text_windows
    if workspace.delete_file_operation is filesystem_delete_file:
        workspace.delete_file_operation = delete_file_windows
    if workspace.make_directory_operation is filesystem_make_directory:
        workspace.make_directory_operation = make_directory_windows


from safe_fs_ops.workspace_transaction import SafeTransaction  # noqa: E402

__all__ = [
    "ResourceNotClaimedError",
    "SafeTransaction",
    "SafeWorkspace",
    "SafeWorkspaceBusyError",
    "SafeWorkspaceError",
]
