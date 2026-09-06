from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from safe_fs_ops.filesystem_ops.models import FileType, PathSafety


class UnsafePathError(ValueError):
    """Raised when a primitive refuses to operate on an unsafe path."""


def require_no_nul(path: Path | str) -> None:
    """Reject paths that a NUL-terminated native API would silently shorten."""
    if "\0" in str(path):
        raise ValueError("filesystem path must not contain NUL characters")


def absolute_without_resolving(path: Path | str) -> Path:
    """Return a stable absolute path without following symlinks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def inspect_path(path: Path | str) -> PathSafety:
    checked_path = Path(path)
    file_type = _file_type(checked_path)
    exists = file_type != "missing"
    is_mount = False
    with contextlib.suppress(OSError, NotImplementedError):
        is_mount = checked_path.is_mount()

    hardlink_count = 0
    size: int | None = None
    if exists:
        with contextlib.suppress(OSError):
            stat_result = checked_path.lstat()
            hardlink_count = stat_result.st_nlink
            size = stat_result.st_size

    return PathSafety(
        path=checked_path,
        exists=exists,
        file_type=file_type,
        is_mount=is_mount,
        is_windows_reparse_point=_is_windows_reparse_point(checked_path) if exists else False,
        hardlink_count=hardlink_count,
        size=size,
    )


def ensure_safe_parent_chain(path: Path | str, *, operation: str) -> None:
    """Refuse existing parent directories that redirect writes elsewhere."""
    current = Path(path).expanduser().absolute().parent
    anchor = Path(current.anchor)
    while True:
        if current == anchor or current == current.parent:
            return
        safety = inspect_path(current)
        if safety.exists:
            if safety.is_symlink or safety.is_windows_reparse_point:
                raise UnsafePathError(f"{operation} refused because parent path redirects elsewhere: {current}")
            if not safety.is_dir:
                raise UnsafePathError(f"{operation} refused because parent path is not a directory: {current}")
        current = current.parent


def ensure_safe_write_path(path: Path | str, *, operation: str = "write") -> None:
    checked_path = Path(path)
    ensure_safe_parent_chain(checked_path, operation=operation)
    safety = inspect_path(checked_path)
    _raise_if_redirecting(safety, operation=operation)
    if safety.exists and not safety.is_file:
        raise UnsafePathError(f"{operation} refused for non-regular file: {checked_path}")


def ensure_safe_delete_path(path: Path | str, *, operation: str = "delete", missing_ok: bool = True) -> None:
    checked_path = Path(path)
    ensure_safe_parent_chain(checked_path, operation=operation)
    safety = inspect_path(checked_path)
    _raise_if_redirecting(safety, operation=operation)
    if not safety.exists:
        if missing_ok:
            return
        raise FileNotFoundError(checked_path)
    if not safety.is_file:
        raise UnsafePathError(f"{operation} refused for non-regular file: {checked_path}")


def ensure_safe_mkdir_path(path: Path | str, *, operation: str = "mkdir", exist_ok: bool = False) -> None:
    checked_path = Path(path)
    ensure_safe_parent_chain(checked_path, operation=operation)
    safety = inspect_path(checked_path)
    _raise_if_redirecting(safety, operation=operation)
    if not safety.exists:
        return
    if safety.is_dir and exist_ok:
        return
    if safety.is_dir:
        raise FileExistsError(checked_path)
    raise UnsafePathError(f"{operation} refused because path exists and is not a directory: {checked_path}")


def _raise_if_redirecting(safety: PathSafety, *, operation: str) -> None:
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused for symlink path: {safety.path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {safety.path}")
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {safety.path}")


def _file_type(path: Path) -> FileType:
    try:
        if path.is_symlink():
            return "symlink"
        if not path.exists():
            return "missing"
        if path.is_file():
            return "file"
        if path.is_dir():
            return "directory"
        return "other"
    except OSError:
        return "other"


def _is_windows_reparse_point(path: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        return bool(int(getattr(path.lstat(), "st_file_attributes", 0)) & 0x400)
    except OSError:
        return False
