from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from stat import S_IMODE, S_ISREG
from typing import Protocol

from safe_fs_ops.filesystem_ops._windows_directory_operations import delete_file_windows
from safe_fs_ops.filesystem_ops._windows_snapshots import snapshot_resource_windows
from safe_fs_ops.filesystem_ops._windows_write_operations import atomic_write_bytes_windows
from safe_fs_ops.filesystem_ops.models import ContentRef, FileBackup, ResourceSnapshot
from safe_fs_ops.filesystem_ops.mutations import DurabilityMode, atomic_write_bytes, delete_file
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving
from safe_fs_ops.filesystem_ops.snapshots import (
    _ensure_directory_descriptor_matches_path,
    _open_parent_directory_descriptor_relative,
    _open_regular_child,
    _same_snapshot_metadata,
    snapshot_resource,
)

_CHUNK_SIZE = 1024 * 1024


class BackupContentMismatchError(ValueError):
    """Raised when stored backup bytes do not match the recorded digest."""


class RestoreConflictError(UnsafePathError):
    """Raised when restore would overwrite a path with unexpected current state."""


class _SnapshotReader(Protocol):
    def __call__(self, path: Path) -> ResourceSnapshot: ...


class _BackupContentReader(Protocol):
    def __call__(self, path: Path, *, expected_snapshot: ResourceSnapshot) -> tuple[bytes, int | None]: ...


class _BackupContentWriter(Protocol):
    def __call__(
        self,
        path: Path,
        content: bytes,
        *,
        permissions: int | None,
        durability: DurabilityMode,
    ) -> None: ...


class _BackupFileDeleter(Protocol):
    def __call__(self, path: Path, *, durability: DurabilityMode) -> None: ...


class _BackupArtifactStore(Protocol):
    def put_file(self, path: Path | str) -> ContentRef: ...

    def copy_to(
        self,
        ref: ContentRef,
        destination: Path | str,
        *,
        no_replace: bool = True,
        permissions: int | None = None,
        mtime_ns: int | None = None,
        mtime: float | None = None,
    ) -> None: ...


def capture_backup(
    path: Path | str,
    *,
    snapshot: ResourceSnapshot | None = None,
    content_path: Path | str | None = None,
    artifact_store: _BackupArtifactStore | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    _snapshot_reader: _SnapshotReader = snapshot_resource,
    _backup_content_reader: _BackupContentReader | None = None,
    _backup_content_writer: _BackupContentWriter | None = None,
) -> FileBackup:
    if content_path is not None and artifact_store is not None:
        raise ValueError("content_path and artifact_store cannot both be provided")
    if _snapshot_reader is snapshot_resource and os.name == "nt":
        _snapshot_reader = snapshot_resource_windows
    if _backup_content_reader is None:
        _backup_content_reader = _default_backup_content_reader()
    if _backup_content_writer is None:
        _backup_content_writer = _default_backup_content_writer()
    target = absolute_without_resolving(path)
    current_snapshot = snapshot if snapshot is not None else _snapshot_reader(target)
    if current_snapshot.path != target:
        raise ValueError("snapshot.path must match capture target")

    if current_snapshot.file_type == "missing":
        return FileBackup(
            path=target,
            existed=False,
            file_type="missing",
            content_bytes=None,
            content_path=None,
            content_address=None,
            content_hash=None,
            size=None,
            permissions=None,
            snapshot=current_snapshot,
        )
    if current_snapshot.file_type != "file":
        if current_snapshot.file_type == "symlink":
            raise UnsafePathError(f"backup refused for symlink path: {target}")
        raise UnsafePathError(f"backup refused for non-regular file: {target}")

    content_ref: ContentRef | None = None
    if artifact_store is not None:
        permissions = _current_permissions(target)
        if permissions is None:
            raise UnsafePathError(f"backup refused because file permissions could not be captured: {target}")
        content_ref = artifact_store.put_file(target)
        if content_ref.digest != current_snapshot.content_hash or content_ref.size != current_snapshot.size:
            raise UnsafePathError(f"backup refused because file changed before capture completed: {target}")
        content_bytes: bytes | None = None
    else:
        content_bytes, permissions = _backup_content_reader(
            target,
            expected_snapshot=current_snapshot,
        )
        if hashlib.sha256(content_bytes).hexdigest() != current_snapshot.content_hash:
            raise UnsafePathError(f"backup refused because file changed before capture completed: {target}")

    stored_path: Path | None = None
    inline_bytes: bytes | None = content_bytes
    if content_path is not None:
        assert content_bytes is not None
        stored_path = absolute_without_resolving(content_path)
        if stored_path == target:
            raise ValueError("content_path must not alias the backup source path")
        _backup_content_writer(
            stored_path,
            content_bytes,
            permissions=0o600,
            durability=durability,
        )
        inline_bytes = None

    return FileBackup(
        path=target,
        existed=True,
        file_type="file",
        content_bytes=inline_bytes,
        content_path=stored_path,
        content_address=content_ref,
        content_hash=current_snapshot.content_hash,
        size=current_snapshot.size,
        permissions=permissions,
        snapshot=current_snapshot,
    )


def restore_backup(
    backup: FileBackup,
    *,
    expected_current: ResourceSnapshot | None = None,
    allow_overwrite: bool = False,
    artifact_store: _BackupArtifactStore | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    _current_permissions_reader: Callable[[Path], int | None] | None = None,
    _snapshot_reader: _SnapshotReader = snapshot_resource,
    _backup_content_reader: _BackupContentReader | None = None,
    _backup_content_writer: _BackupContentWriter | None = None,
    _backup_file_deleter: _BackupFileDeleter | None = None,
) -> ResourceSnapshot:
    if _snapshot_reader is snapshot_resource and os.name == "nt":
        _snapshot_reader = snapshot_resource_windows
    if _current_permissions_reader is None:
        _current_permissions_reader = _current_permissions
    if _backup_content_reader is None:
        _backup_content_reader = _default_backup_content_reader()
    if _backup_content_writer is None:
        _backup_content_writer = _default_backup_content_writer()
    if _backup_file_deleter is None:
        _backup_file_deleter = _default_backup_file_deleter()
    if backup.file_type == "missing":
        return _restore_missing_backup(
            backup,
            expected_current=expected_current,
            allow_overwrite=allow_overwrite,
            durability=durability,
            snapshot_reader=_snapshot_reader,
            backup_file_deleter=_backup_file_deleter,
        )

    current = _snapshot_reader(backup.path)
    if _snapshot_matches_backup_content(current, backup, current_permissions_reader=_current_permissions_reader):
        if backup.content_address is None:
            _load_and_validate_backup_content(
                backup,
                snapshot_reader=_snapshot_reader,
                backup_content_reader=_backup_content_reader,
            )
        return current
    _require_restore_preconditions(
        backup,
        current=current,
        expected_current=expected_current,
        allow_overwrite=allow_overwrite,
    )

    if backup.content_address is not None:
        if artifact_store is None:
            raise BackupContentMismatchError("backup content address requires an artifact_store for restore")
        try:
            artifact_store.copy_to(
                backup.content_address,
                backup.path,
                no_replace=False,
                permissions=backup.permissions,
            )
        except ValueError as exc:
            raise BackupContentMismatchError(f"backup content address mismatch for {backup.path}") from exc
    else:
        content_bytes = _load_and_validate_backup_content(
            backup,
            snapshot_reader=_snapshot_reader,
            backup_content_reader=_backup_content_reader,
        )
        _backup_content_writer(
            backup.path,
            content_bytes,
            permissions=backup.permissions,
            durability=durability,
        )
    restored = _snapshot_reader(backup.path)
    if restored.file_type != "file" or restored.content_hash != backup.content_hash:
        raise BackupContentMismatchError(f"restore wrote unexpected content for {backup.path}")
    if restored.size != backup.size:
        raise BackupContentMismatchError(f"restore wrote unexpected file size for {backup.path}")
    if _current_permissions_reader(restored.path) != backup.permissions:
        raise BackupContentMismatchError(f"restore wrote unexpected permissions for {backup.path}")
    return restored


def _restore_missing_backup(
    backup: FileBackup,
    *,
    expected_current: ResourceSnapshot | None,
    allow_overwrite: bool,
    durability: DurabilityMode,
    snapshot_reader: _SnapshotReader,
    backup_file_deleter: _BackupFileDeleter,
) -> ResourceSnapshot:
    current = snapshot_reader(backup.path)
    if current.file_type == "missing":
        return current
    if current.file_type != "file":
        if current.file_type == "symlink":
            raise UnsafePathError(f"restore refused for symlink path: {backup.path}")
        raise UnsafePathError(f"restore refused for non-regular file: {backup.path}")
    _require_restore_preconditions(
        backup,
        current=current,
        expected_current=expected_current,
        allow_overwrite=allow_overwrite,
    )
    backup_file_deleter(backup.path, durability=durability)
    restored = snapshot_reader(backup.path)
    if restored.file_type != "missing":
        raise RestoreConflictError(f"restore did not remove file as expected: {backup.path}")
    return restored


def _require_restore_preconditions(
    backup: FileBackup,
    *,
    current: ResourceSnapshot,
    expected_current: ResourceSnapshot | None,
    allow_overwrite: bool,
) -> None:
    if allow_overwrite:
        return
    if expected_current is None:
        raise RestoreConflictError(f"restore refused because current state was not provided for {backup.path}")
    if expected_current.path != backup.path:
        raise ValueError("expected_current.path must match backup path")
    if not _same_resource_snapshot(current, expected_current):
        raise RestoreConflictError(f"restore refused because current state changed unexpectedly: {backup.path}")


def _same_resource_snapshot(left: ResourceSnapshot, right: ResourceSnapshot) -> bool:
    if (
        left.path == right.path
        and left.exists is True
        and right.exists is True
        and left.file_type == right.file_type == "file"
        and left.content_hash is not None
        and right.content_hash is not None
    ):
        return (
            left.content_hash == right.content_hash
            and left.size == right.size
            and left.symlink_target == right.symlink_target
        )
    return (
        left.path == right.path
        and left.exists == right.exists
        and left.file_type == right.file_type
        and left.content_hash == right.content_hash
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
        and left.symlink_target == right.symlink_target
    )


def _snapshot_matches_backup_content(
    snapshot: ResourceSnapshot,
    backup: FileBackup,
    *,
    current_permissions_reader: Callable[[Path], int | None],
) -> bool:
    if backup.file_type != "file":
        return snapshot.file_type == "missing"
    return (
        snapshot.file_type == "file"
        and snapshot.content_hash == backup.content_hash
        and snapshot.size == backup.size
        and current_permissions_reader(snapshot.path) == backup.permissions
    )


def _load_and_validate_backup_content(
    backup: FileBackup,
    *,
    snapshot_reader: _SnapshotReader,
    backup_content_reader: _BackupContentReader,
) -> bytes:
    if backup.file_type != "file":
        raise ValueError("missing backups do not carry file content")
    if backup.content_bytes is not None:
        content_bytes = backup.content_bytes
    else:
        assert backup.content_path is not None
        path_snapshot = snapshot_reader(backup.content_path)
        if path_snapshot.file_type != "file":
            raise BackupContentMismatchError(f"backup content path is not a regular file: {backup.content_path}")
        content_bytes, _permissions = backup_content_reader(
            backup.content_path,
            expected_snapshot=path_snapshot,
        )
    digest = hashlib.sha256(content_bytes).hexdigest()
    if digest != backup.content_hash:
        raise BackupContentMismatchError(f"backup content hash mismatch for {backup.path}")
    if len(content_bytes) != backup.size:
        raise BackupContentMismatchError(f"backup content size mismatch for {backup.path}")
    return content_bytes


def _read_regular_file_backup_bytes(
    path: Path,
    *,
    expected_snapshot: ResourceSnapshot,
) -> tuple[bytes, int | None]:
    if expected_snapshot.file_type != "file":
        raise ValueError("expected_snapshot must describe a regular file")
    parent_descriptor = _open_parent_directory_descriptor_relative(path)
    try:
        stat_result = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not S_ISREG(stat_result.st_mode):
            raise UnsafePathError(f"backup refused for non-regular file: {path}")
        if not _stat_matches_snapshot(stat_result, expected_snapshot):
            raise UnsafePathError(f"backup refused because file changed before capture completed: {path}")
        descriptor = _open_regular_child(
            parent_descriptor,
            path.name,
            path,
            expected_stat=stat_result,
            hooks=None,
        )
        with os.fdopen(descriptor, "rb") as handle:
            content_bytes = handle.read()
            final_stat = os.fstat(handle.fileno())
        if not _same_snapshot_metadata(final_stat, stat_result):
            raise UnsafePathError(f"backup refused because file changed before capture completed: {path}")
        _ensure_directory_descriptor_matches_path(parent_descriptor, path.parent)
        return content_bytes, S_IMODE(stat_result.st_mode)
    finally:
        os.close(parent_descriptor)


def _stat_matches_snapshot(stat_result: os.stat_result, snapshot: ResourceSnapshot) -> bool:
    return (
        snapshot.file_type == "file"
        and stat_result.st_size == snapshot.size
        and stat_result.st_mtime_ns == snapshot.mtime_ns
    )


def _current_permissions(path: Path) -> int | None:
    with contextlib.suppress(OSError):
        return S_IMODE(path.lstat().st_mode)
    return None


def _default_backup_content_reader() -> _BackupContentReader:
    if os.name == "nt":
        return _read_regular_file_backup_bytes_portable
    return _read_regular_file_backup_bytes


def _default_backup_content_writer() -> _BackupContentWriter:
    if os.name == "nt":
        return _write_backup_content_windows
    return _write_backup_content


def _default_backup_file_deleter() -> _BackupFileDeleter:
    if os.name == "nt":
        return _delete_backup_target_windows
    return _delete_backup_target


def _write_backup_content(
    path: Path,
    content: bytes,
    *,
    permissions: int | None,
    durability: DurabilityMode,
) -> None:
    atomic_write_bytes(path, content, permissions=permissions, durability=durability)


def _write_backup_content_windows(
    path: Path,
    content: bytes,
    *,
    permissions: int | None,
    durability: DurabilityMode,
) -> None:
    atomic_write_bytes_windows(path, content, durability=durability)
    if permissions is not None:
        path.chmod(permissions)


def _delete_backup_target(path: Path, *, durability: DurabilityMode) -> None:
    delete_file(path, durability=durability)


def _delete_backup_target_windows(path: Path, *, durability: DurabilityMode) -> None:
    delete_file_windows(path, durability=durability)


def _read_regular_file_backup_bytes_portable(
    path: Path,
    *,
    expected_snapshot: ResourceSnapshot,
) -> tuple[bytes, int | None]:
    if expected_snapshot.file_type != "file":
        raise ValueError("expected_snapshot must describe a regular file")
    try:
        stat_before = path.lstat()
        if not S_ISREG(stat_before.st_mode):
            raise UnsafePathError(f"backup refused for non-regular file: {path}")
        if not _stat_matches_snapshot(stat_before, expected_snapshot):
            raise UnsafePathError(f"backup refused because file changed before capture completed: {path}")
        content = bytearray()
        with path.open("rb") as handle:
            stat_open = os.fstat(handle.fileno())
            if not S_ISREG(stat_open.st_mode) or not _same_portable_file_metadata(stat_open, stat_before):
                raise UnsafePathError(f"backup refused because file changed before capture completed: {path}")
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                content.extend(chunk)
            stat_after = os.fstat(handle.fileno())
        permissions = S_IMODE(stat_before.st_mode)
    except OSError as exc:
        raise UnsafePathError(f"backup refused because file could not be captured safely: {path}") from exc
    if not _same_portable_file_metadata(stat_after, stat_before) or not _stat_matches_snapshot(
        stat_after,
        expected_snapshot,
    ):
        raise UnsafePathError(f"backup refused because file changed before capture completed: {path}")
    return bytes(content), permissions


def _same_portable_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )
