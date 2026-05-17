from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from safe_fs_ops.filesystem_ops.models import PathSafety
from safe_fs_ops.filesystem_ops.mutation_support import (
    DurabilityMode,
    ParentChangedAfterMutationError,
    _durability_wrapped_fsync,
)
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, inspect_path

_DirectoryFlush = Callable[[Path], None]
_PathInspector = Callable[[Path], PathSafety]
_TokenHex = Callable[[int], str]
_WindowsReparseChecker = Callable[[Path], bool]


def default_token_hex(nbytes: int) -> str:
    return secrets.token_hex(nbytes)


def create_temp_file_same_directory(parent: Path, target_name: str, *, token_hex: _TokenHex) -> tuple[Path, BinaryIO]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        temp_path = parent / f".{target_name}.{token_hex(8)}.tmp"
        try:
            descriptor = os.open(temp_path, flags, 0o600)
        except FileExistsError:
            continue
        return temp_path, os.fdopen(descriptor, "wb")
    raise FileExistsError(f"could not allocate a unique temp file for {target_name}")


def flush_directory_if_requested(
    durability: DurabilityMode,
    path: Path,
    directory_flush: _DirectoryFlush | None,
) -> None:
    if directory_flush is None:
        return
    _durability_wrapped_fsync(durability, lambda _descriptor: directory_flush(path))(0)


def ensure_safe_write_path(
    path: Path,
    *,
    operation: str,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_parent_chain(
        path,
        operation=operation,
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    safety = safe_inspect(path, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
    raise_if_redirecting(safety, operation=operation)
    if safety.exists and not safety.is_file:
        raise UnsafePathError(f"{operation} refused for non-regular file: {path}")


def ensure_safe_delete_path(
    path: Path,
    *,
    operation: str,
    missing_ok: bool,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_parent_chain(
        path,
        operation=operation,
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    safety = safe_inspect(path, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
    raise_if_redirecting(safety, operation=operation)
    if not safety.exists:
        if missing_ok:
            return
        raise FileNotFoundError(path)
    if not safety.is_file:
        raise UnsafePathError(f"{operation} refused for non-regular file: {path}")


def ensure_safe_mkdir_path(
    path: Path,
    *,
    operation: str,
    exist_ok: bool,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_parent_chain(
        path,
        operation=operation,
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    safety = safe_inspect(path, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
    raise_if_redirecting(safety, operation=operation)
    if not safety.exists:
        return
    if safety.is_dir and exist_ok:
        return
    if safety.is_dir:
        raise FileExistsError(path)
    raise UnsafePathError(f"{operation} refused because path exists and is not a directory: {path}")


def ensure_safe_parent_chain(
    path: Path,
    *,
    operation: str,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    current = path.parent
    anchor = Path(current.anchor)
    while True:
        if current == anchor or current == current.parent:
            return
        safety = safe_inspect(current, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
        if safety.exists:
            raise_if_redirecting(safety, operation=operation)
            if not safety.is_dir:
                raise UnsafePathError(f"{operation} refused because parent path is not a directory: {current}")
        current = current.parent


def ensure_safe_directory(
    path: Path,
    *,
    operation: str,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    safety = safe_inspect(path, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
    raise_if_redirecting(safety, operation=operation)
    if not safety.exists:
        raise UnsafePathError(f"{operation} refused because parent path is no longer present: {path}")
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because parent path is not a directory: {path}")


def same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def path_identity(path: Path) -> os.stat_result:
    return path.stat()


def ensure_same_parent_identity(
    path: Path,
    expected: os.stat_result,
    *,
    operation: str,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_directory(
        path,
        operation=operation,
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    current = path.stat()
    if not same_file_identity(current, expected):
        raise UnsafePathError(f"{operation} refused because parent path redirects elsewhere: {path}")


def ensure_same_parent_identity_after_mutation(
    path: Path,
    expected: os.stat_result,
    *,
    operation: str,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    try:
        ensure_same_parent_identity(
            path,
            expected,
            operation=operation,
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
    except UnsafePathError as exc:
        raise ParentChangedAfterMutationError(
            f"{operation} refused to report success because parent path changed after mutation: {path}"
        ) from exc


def safe_inspect(
    path: Path,
    *,
    path_inspector: _PathInspector = inspect_path,
    windows_reparse_checker: _WindowsReparseChecker,
) -> PathSafety:
    safety = path_inspector(path)
    if not safety.exists:
        return safety
    if safety.is_windows_reparse_point:
        return safety
    reparse = windows_reparse_checker(path)
    if not reparse:
        return safety
    return PathSafety(
        path=safety.path,
        exists=safety.exists,
        file_type=safety.file_type,
        is_mount=safety.is_mount,
        is_windows_reparse_point=True,
        hardlink_count=safety.hardlink_count,
        size=safety.size,
    )


def raise_if_redirecting(safety: PathSafety, *, operation: str) -> None:
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused for symlink path: {safety.path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {safety.path}")
    if safety.is_mount and safety.path != Path(safety.path.anchor):
        raise UnsafePathError(f"{operation} refused for mount point: {safety.path}")


__all__ = [
    "_DirectoryFlush",
    "_PathInspector",
    "_TokenHex",
    "_WindowsReparseChecker",
    "create_temp_file_same_directory",
    "default_token_hex",
    "ensure_safe_delete_path",
    "ensure_safe_directory",
    "ensure_safe_mkdir_path",
    "ensure_safe_parent_chain",
    "ensure_safe_write_path",
    "ensure_same_parent_identity",
    "ensure_same_parent_identity_after_mutation",
    "flush_directory_if_requested",
    "path_identity",
    "safe_inspect",
]
