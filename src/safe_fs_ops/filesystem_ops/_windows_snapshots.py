from __future__ import annotations

import hashlib
import os
from pathlib import Path
from stat import S_ISDIR, S_ISREG

from safe_fs_ops.filesystem_ops._windows_operation_common import _PathInspector, _WindowsReparseChecker, safe_inspect
from safe_fs_ops.filesystem_ops._windows_primitives import is_windows_reparse_point
from safe_fs_ops.filesystem_ops.models import PathSafety, ResourceSnapshot
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path

_CHUNK_SIZE = 1024 * 1024


def snapshot_resource_windows(
    path: Path | str,
    *,
    _path_inspector: _PathInspector = inspect_path,
    _windows_reparse_checker: _WindowsReparseChecker = is_windows_reparse_point,
) -> ResourceSnapshot:
    target = absolute_without_resolving(path)
    _ensure_safe_snapshot_path(
        target,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    try:
        before = target.lstat()
    except FileNotFoundError:
        if target.exists() or target.is_symlink():
            raise UnsafePathError(
                f"snapshot refused because classified missing path appeared before snapshot completed: {target}"
            ) from None
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
            device=None,
            inode=None,
        )

    if S_ISREG(before.st_mode):
        content_hash, after = _hash_regular_file_checked(target, expected_stat=before)
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="file",
            content_hash=content_hash,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            symlink_target=None,
            device=before.st_dev,
            inode=before.st_ino,
        )

    if S_ISDIR(before.st_mode):
        after = target.lstat()
        if not _same_snapshot_metadata(before, after):
            raise UnsafePathError(
                f"snapshot refused because classified path changed before snapshot completed: {target}"
            )
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="directory",
            content_hash=None,
            size=before.st_size,
            mtime_ns=before.st_mtime_ns,
            symlink_target=None,
            device=before.st_dev,
            inode=before.st_ino,
        )

    after = target.lstat()
    if not _same_snapshot_metadata(before, after):
        raise UnsafePathError(f"snapshot refused because classified path changed before snapshot completed: {target}")
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type="other",
        content_hash=None,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        symlink_target=None,
        device=before.st_dev,
        inode=before.st_ino,
    )


def _ensure_safe_snapshot_path(
    path: Path,
    *,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    current = path
    while True:
        safety = safe_inspect(
            current,
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
        if _is_redirecting_snapshot_path(safety):
            raise UnsafePathError(f"snapshot refused because path redirects elsewhere: {current}")
        if current == current.parent:
            break
        current = current.parent


def _is_redirecting_snapshot_path(safety: PathSafety) -> bool:
    return safety.is_symlink or safety.is_windows_reparse_point


def _same_snapshot_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _hash_regular_file_checked(path: Path, *, expected_stat: os.stat_result) -> tuple[str, os.stat_result]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            opened_stat = os.fstat(handle.fileno())
            if not S_ISREG(opened_stat.st_mode) or not _same_snapshot_metadata(opened_stat, expected_stat):
                raise UnsafePathError(f"snapshot refused because classified file changed before open: {path}")
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
            final_stat = os.fstat(handle.fileno())
    except OSError as exc:
        raise UnsafePathError(f"snapshot refused because classified file could not be read safely: {path}") from exc
    if not _same_snapshot_metadata(expected_stat, final_stat):
        raise UnsafePathError(f"snapshot refused because classified file changed before snapshot completed: {path}")
    return digest.hexdigest(), final_stat


__all__ = ["snapshot_resource_windows"]
