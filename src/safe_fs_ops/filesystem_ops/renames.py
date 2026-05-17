from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import Literal, Protocol

from safe_fs_ops.filesystem_ops.models import FileType
from safe_fs_ops.filesystem_ops.mutations import UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops.no_replace_rename import rename_path_no_replace
from safe_fs_ops.filesystem_ops.paths import (
    UnsafePathError,
    absolute_without_resolving,
    ensure_safe_parent_chain,
    inspect_path,
)

RenamedFileType = Literal["file", "directory"]


class NoReplaceRename(Protocol):
    def __call__(self, source: Path, destination: Path, *, operation: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RenameRecord:
    source_path: Path
    destination_path: Path
    file_type: RenamedFileType
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int | None = None

    def __post_init__(self) -> None:
        if self.file_type not in {"file", "directory"}:
            raise ValueError(f"unsupported file_type: {self.file_type}")


def rename_no_replace(
    source: Path | str,
    destination: Path | str,
    *,
    _rename_no_replace: NoReplaceRename | None = None,
) -> RenameRecord:
    operation = "rename"
    source_path = absolute_without_resolving(source)
    destination_path = absolute_without_resolving(destination)
    source_stat, file_type = _require_safe_existing_source(source_path, operation=operation)
    _require_safe_absent_destination(destination_path, operation=operation)
    _require_same_device(source_stat.st_dev, source_path.parent, destination_path.parent, operation=operation)

    rename_backend = _default_rename_no_replace if _rename_no_replace is None else _rename_no_replace
    rename_backend(source_path, destination_path, operation=operation)

    _require_destination_identity(
        destination_path,
        expected_stat=source_stat,
        expected_file_type=file_type,
        operation=operation,
    )
    _ensure_source_path_vacated(source_path, expected_stat=source_stat, operation=operation)
    return _record_from_stat(
        source_path=source_path,
        destination_path=destination_path,
        file_type=file_type,
        stat_result=source_stat,
    )


def restore_inverse_rename(
    record: RenameRecord,
    *,
    _rename_no_replace: NoReplaceRename | None = None,
) -> RenameRecord:
    operation = "restore inverse rename"
    _require_rename_record(record)
    destination_stat, file_type = _require_safe_existing_source(record.destination_path, operation=operation)
    if file_type != record.file_type or not _same_recorded_identity(destination_stat, record):
        raise UnsafePathError(
            f"{operation} refused because rename destination no longer matches recorded identity: "
            f"{record.destination_path}"
        )
    _require_safe_absent_destination(record.source_path, operation=operation)
    _require_same_device(
        destination_stat.st_dev,
        record.destination_path.parent,
        record.source_path.parent,
        operation=operation,
    )

    rename_backend = _default_rename_no_replace if _rename_no_replace is None else _rename_no_replace
    rename_backend(record.destination_path, record.source_path, operation=operation)

    restored_stat = _require_destination_identity(
        record.source_path,
        expected_stat=destination_stat,
        expected_file_type=file_type,
        operation=operation,
    )
    _ensure_source_path_vacated(record.destination_path, expected_stat=destination_stat, operation=operation)
    return _record_from_stat(
        source_path=record.destination_path,
        destination_path=record.source_path,
        file_type=file_type,
        stat_result=restored_stat,
    )


def _require_rename_record(record: RenameRecord) -> None:
    if not isinstance(record, RenameRecord):
        raise TypeError(f"record must be a RenameRecord, got {type(record).__name__}")


def _require_safe_existing_source(path: Path, *, operation: str) -> tuple[os.stat_result, RenamedFileType]:
    ensure_safe_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    _raise_if_redirecting(safety.file_type, safety.is_windows_reparse_point, safety.is_mount, path, operation=operation)
    if not safety.exists:
        raise FileNotFoundError(path)
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise UnsafePathError(f"{operation} refused because source could not be inspected safely: {path}") from exc
    if S_ISREG(stat_result.st_mode):
        return stat_result, "file"
    if S_ISDIR(stat_result.st_mode):
        return stat_result, "directory"
    raise UnsafePathError(f"{operation} refused because source is not a regular file or directory: {path}")


def _require_safe_absent_destination(path: Path, *, operation: str) -> None:
    ensure_safe_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    _raise_if_redirecting(safety.file_type, safety.is_windows_reparse_point, safety.is_mount, path, operation=operation)
    if safety.exists:
        raise FileExistsError(path)
    _require_existing_directory(path.parent, operation=operation)


def _require_existing_directory(path: Path, *, operation: str) -> os.stat_result:
    ensure_safe_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    _raise_if_redirecting(safety.file_type, safety.is_windows_reparse_point, safety.is_mount, path, operation=operation)
    if not safety.exists:
        raise FileNotFoundError(path)
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because parent path is not a directory: {path}")
    return path.stat()


def _raise_if_redirecting(
    file_type: FileType,
    is_windows_reparse_point: bool,
    is_mount: bool,
    path: Path,
    *,
    operation: str,
) -> None:
    if file_type == "symlink":
        raise UnsafePathError(f"{operation} refused for symlink path: {path}")
    if is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {path}")
    if is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {path}")


def _require_same_device(source_device: int, source_parent: Path, destination_parent: Path, *, operation: str) -> None:
    source_parent_stat = _require_existing_directory(source_parent, operation=operation)
    destination_parent_stat = _require_existing_directory(destination_parent, operation=operation)
    if len({source_device, source_parent_stat.st_dev, destination_parent_stat.st_dev}) != 1:
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because source and destination are not provably on the same filesystem"
        )


def _require_destination_identity(
    path: Path,
    *,
    expected_stat: os.stat_result,
    expected_file_type: RenamedFileType,
    operation: str,
) -> os.stat_result:
    try:
        actual_stat = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(
            f"{operation} refused to report success because destination was not present for verification: {path}"
        ) from exc
    actual_file_type = _renamed_file_type(actual_stat)
    if actual_file_type != expected_file_type or not _same_identity(actual_stat, expected_stat):
        raise UnsafePathError(
            f"{operation} refused to report success because destination identity changed unexpectedly: {path}"
        )
    return actual_stat


def _ensure_source_path_vacated(path: Path, *, expected_stat: os.stat_result, operation: str) -> None:
    try:
        current_stat = path.lstat()
    except FileNotFoundError:
        return
    if _same_identity(current_stat, expected_stat):
        raise UnsafePathError(f"{operation} refused to report success because source path still resolves: {path}")
    raise UnsafePathError(f"{operation} refused to report success because source path was replaced: {path}")


def _renamed_file_type(stat_result: os.stat_result) -> RenamedFileType:
    if S_ISREG(stat_result.st_mode):
        return "file"
    if S_ISDIR(stat_result.st_mode):
        return "directory"
    raise UnsafePathError("renamed path is not a regular file or directory")


def _record_from_stat(
    *,
    source_path: Path,
    destination_path: Path,
    file_type: RenamedFileType,
    stat_result: os.stat_result,
) -> RenameRecord:
    return RenameRecord(
        source_path=source_path,
        destination_path=destination_path,
        file_type=file_type,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _same_recorded_identity(stat_result: os.stat_result, record: RenameRecord) -> bool:
    return (
        stat_result.st_dev == record.device
        and stat_result.st_ino == record.inode
        and stat_result.st_size == record.size
        and stat_result.st_mtime_ns == record.mtime_ns
    )


def _default_rename_no_replace(source: Path, destination: Path, *, operation: str) -> None:
    rename_path_no_replace(source, destination, operation=operation)


__all__ = [
    "NoReplaceRename",
    "RenamedFileType",
    "RenameRecord",
    "rename_no_replace",
    "restore_inverse_rename",
]
