from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR
from typing import Literal

from safe_fs_ops.filesystem_ops.models import PathSafety
from safe_fs_ops.filesystem_ops.mutations import DurabilityMode, UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops.no_replace_rename import DirectoryNoReplaceRename, rename_directory_no_replace
from safe_fs_ops.filesystem_ops.paths import (
    UnsafePathError,
    absolute_without_resolving,
    ensure_safe_parent_chain,
    inspect_path,
)
from safe_fs_ops.filesystem_ops.remove_directories import (
    DirectoryIdentity,
    remove_existing_empty_directory_by_identity,
)

CapturedDirectoryOwnershipClass = Literal["captured_by_transaction"]


@dataclass(frozen=True, slots=True)
class CapturedDirectoryRecord:
    original_path: Path
    quarantine_path: Path
    original_identity: DirectoryIdentity
    captured_identity: DirectoryIdentity
    ownership_class: CapturedDirectoryOwnershipClass = "captured_by_transaction"

    def __post_init__(self) -> None:
        if self.ownership_class != "captured_by_transaction":
            raise ValueError(f"unsupported ownership_class: {self.ownership_class}")


def capture_directory_to_quarantine(
    source: Path | str,
    *,
    quarantine_path: Path | str,
    _rename_no_replace: DirectoryNoReplaceRename = rename_directory_no_replace,
) -> CapturedDirectoryRecord:
    operation = "capture directory"
    source_path = absolute_without_resolving(source)
    quarantine = absolute_without_resolving(quarantine_path)
    if source_path == quarantine:
        raise ValueError("quarantine_path must not alias the source path")

    _require_safe_existing_directory(source_path, operation=operation)
    source_stat = source_path.stat()
    source_parent_stat = _require_existing_directory_stat(source_path.parent, operation=operation)
    quarantine_parent_stat = _require_existing_directory_stat(quarantine.parent, operation=operation)
    _require_same_device(
        source_stat.st_dev,
        source_parent_stat.st_dev,
        quarantine_parent_stat.st_dev,
        operation=operation,
    )
    _require_destination_absent(quarantine, operation=operation)

    _rename_no_replace(source_path, quarantine, operation=operation)

    captured_stat = _require_post_rename_directory(quarantine, operation=operation)
    original_identity = DirectoryIdentity.from_stat(source_stat)
    captured_identity = DirectoryIdentity.from_stat(captured_stat)
    if captured_identity != original_identity:
        raise UnsafePathError(
            f"{operation} refused to report success because captured identity changed unexpectedly: {quarantine}"
        )
    _ensure_original_path_vacated(source_path, original_identity=original_identity, operation=operation)
    _require_same_device(
        captured_stat.st_dev,
        source_parent_stat.st_dev,
        quarantine_parent_stat.st_dev,
        operation=operation,
    )
    return CapturedDirectoryRecord(
        original_path=source_path,
        quarantine_path=quarantine,
        original_identity=original_identity,
        captured_identity=captured_identity,
    )


def restore_captured_directory(
    record: CapturedDirectoryRecord,
    *,
    _rename_no_replace: DirectoryNoReplaceRename = rename_directory_no_replace,
) -> DirectoryIdentity:
    operation = "restore captured directory"
    _require_captured_record(record)
    original_parent_stat = _require_existing_directory_stat(record.original_path.parent, operation=operation)
    quarantine_parent_stat = _require_existing_directory_stat(record.quarantine_path.parent, operation=operation)
    _require_destination_absent(record.original_path, operation=operation)

    quarantine_stat = _require_safe_existing_directory(record.quarantine_path, operation=operation).path.stat()
    quarantine_identity = DirectoryIdentity.from_stat(quarantine_stat)
    if quarantine_identity != record.captured_identity:
        raise UnsafePathError(
            f"{operation} refused because quarantine path no longer matches the captured directory identity: "
            f"{record.quarantine_path}"
        )
    _require_same_device(
        quarantine_stat.st_dev,
        original_parent_stat.st_dev,
        quarantine_parent_stat.st_dev,
        operation=operation,
    )

    _rename_no_replace(record.quarantine_path, record.original_path, operation=operation)

    restored_stat = _require_post_rename_directory(record.original_path, operation=operation)
    restored_identity = DirectoryIdentity.from_stat(restored_stat)
    if restored_identity != record.captured_identity:
        raise UnsafePathError(
            f"{operation} refused to report success because restored identity changed unexpectedly: "
            f"{record.original_path}"
        )
    _ensure_original_path_vacated(
        record.quarantine_path,
        original_identity=record.captured_identity,
        operation=operation,
    )
    return restored_identity


def cleanup_captured_directory(
    record: CapturedDirectoryRecord,
    *,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    missing_ok: bool = False,
    _identity_remove_directory: Callable[..., None] | None = None,
    _platform: str = sys.platform,
) -> None:
    operation = "cleanup captured directory"
    _require_captured_record(record)

    try:
        quarantine_safety = _require_safe_existing_directory(record.quarantine_path, operation=operation)
    except FileNotFoundError:
        if missing_ok:
            return
        raise

    quarantine_stat = quarantine_safety.path.stat()
    quarantine_identity = DirectoryIdentity.from_stat(quarantine_stat)
    if quarantine_identity != record.captured_identity:
        raise UnsafePathError(
            f"{operation} refused because quarantine path no longer matches the captured directory identity: "
            f"{record.quarantine_path}"
        )
    _require_empty_captured_directory(record.quarantine_path, operation=operation)
    if _identity_remove_directory is None:
        remove_existing_empty_directory_by_identity(
            record.quarantine_path,
            expected_identity=record.captured_identity,
            durability=durability,
            _platform=_platform,
        )
        return
    _identity_remove_directory(record.quarantine_path, expected_identity=record.captured_identity)


def _require_empty_captured_directory(path: Path, *, operation: str) -> None:
    try:
        next(path.iterdir())
    except StopIteration:
        return
    except FileNotFoundError:
        raise
    raise UnsupportedFilesystemMutationError(
        f"{operation} refused to recursively delete non-empty captured directory without a safe recursive deleter: "
        f"{path}"
    )


def _require_captured_record(record: CapturedDirectoryRecord) -> None:
    if not isinstance(record, CapturedDirectoryRecord):
        raise TypeError(f"record must be a CapturedDirectoryRecord, got {type(record).__name__}")
    if record.ownership_class != "captured_by_transaction":
        raise ValueError(f"unsupported ownership_class: {record.ownership_class}")


def _require_safe_existing_directory(path: Path, *, operation: str) -> PathSafety:
    ensure_safe_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused for symlink path: {path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {path}")
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {path}")
    if not safety.exists:
        raise FileNotFoundError(path)
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because path is not a directory: {path}")
    return safety


def _require_existing_directory_stat(path: Path, *, operation: str) -> os.stat_result:
    safety = _require_safe_existing_directory(path, operation=operation)
    stat_result = safety.path.stat()
    if not S_ISDIR(stat_result.st_mode):
        raise UnsafePathError(f"{operation} refused because path is not a directory: {path}")
    return stat_result


def _require_destination_absent(path: Path, *, operation: str) -> None:
    ensure_safe_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused for symlink path: {path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {path}")
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {path}")
    if safety.exists:
        raise FileExistsError(path)


def _require_same_device(*devices: int, operation: str) -> None:
    if len(set(devices)) != 1:
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because source and quarantine paths are not provably on the same filesystem"
        )


def _require_post_rename_directory(path: Path, *, operation: str) -> os.stat_result:
    try:
        stat_result = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(
            f"{operation} refused because renamed directory was not present for verification: {path}"
        ) from exc
    if not S_ISDIR(stat_result.st_mode):
        raise UnsafePathError(f"{operation} refused because renamed path is not a directory: {path}")
    return stat_result


def _ensure_original_path_vacated(
    path: Path,
    *,
    original_identity: DirectoryIdentity,
    operation: str,
) -> None:
    safety = inspect_path(path)
    if not safety.exists:
        return
    if safety.is_symlink or safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused to report success because original path was replaced: {path}")
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused to report success because original path was replaced: {path}")
    if DirectoryIdentity.from_stat(path.stat()) != original_identity:
        raise UnsafePathError(f"{operation} refused to report success because original path was replaced: {path}")
    raise UnsafePathError(f"{operation} refused to report success because original path still resolves: {path}")


__all__ = [
    "CapturedDirectoryOwnershipClass",
    "CapturedDirectoryRecord",
    "capture_directory_to_quarantine",
    "cleanup_captured_directory",
    "restore_captured_directory",
]
