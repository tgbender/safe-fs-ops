from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG

from safe_fs_ops.filesystem_ops.models import FileType, ResourceSnapshot
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path

BeforeOpenFileHook = Callable[[Path], None]
AfterLstatHook = Callable[[Path], None]
BeforeCompleteHook = Callable[[Path], None]


class UnsupportedFilesystemSnapshotError(UnsafePathError):
    """Raised when the platform cannot provide conservative snapshot semantics."""


@dataclass(frozen=True, slots=True)
class SnapshotHooks:
    after_lstat: AfterLstatHook | None = None
    before_open_file: BeforeOpenFileHook | None = None
    before_complete: BeforeCompleteHook | None = None


@dataclass(frozen=True, slots=True)
class _SnapshotPlatformSupport:
    open_dir_fd: bool
    stat_dir_fd: bool
    stat_follow_symlinks: bool
    readlink_dir_fd: bool
    nofollow_directory_open: bool
    nofollow_file_open: bool

    @classmethod
    def detect(cls) -> _SnapshotPlatformSupport:
        return cls(
            open_dir_fd=os.open in os.supports_dir_fd,
            stat_dir_fd=os.stat in os.supports_dir_fd,
            stat_follow_symlinks=os.stat in os.supports_follow_symlinks,
            readlink_dir_fd=os.readlink in os.supports_dir_fd,
            nofollow_directory_open=hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"),
            nofollow_file_open=hasattr(os, "O_NOFOLLOW"),
        )


_DEFAULT_PLATFORM_SUPPORT = _SnapshotPlatformSupport.detect()


def snapshot_resource(
    path: Path | str,
    *,
    chunk_size: int = 1024 * 1024,
    hooks: SnapshotHooks | None = None,
    _platform_support: _SnapshotPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
) -> ResourceSnapshot:
    checked_path = absolute_without_resolving(path)
    _require_descriptor_relative_classification(platform_support=_platform_support)
    parent_descriptor = _open_parent_directory_descriptor_relative(checked_path)
    try:
        try:
            stat_result = os.stat(checked_path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            if hooks is not None and hooks.before_complete is not None:
                hooks.before_complete(checked_path)
            _ensure_child_still_missing(checked_path, parent_descriptor=parent_descriptor)
            _ensure_directory_descriptor_matches_path(parent_descriptor, checked_path.parent)
            return ResourceSnapshot(
                path=checked_path,
                exists=False,
                file_type="missing",
                content_hash=None,
                size=None,
                mtime_ns=None,
                symlink_target=None,
                device=None,
                inode=None,
                ctime_ns=None,
            )
        except OSError as exc:
            raise UnsafePathError(
                f"snapshot refused because path could not be classified safely: {checked_path}"
            ) from exc

        file_type = _file_type_from_stat(stat_result)
        if hooks is not None and hooks.after_lstat is not None:
            hooks.after_lstat(checked_path)

        content_hash = (
            _sha256_regular_file(
                checked_path,
                parent_descriptor=parent_descriptor,
                expected_stat=stat_result,
                chunk_size=chunk_size,
                hooks=hooks,
                platform_support=_platform_support,
            )
            if file_type == "file"
            else None
        )
        symlink_target = (
            _readlink_consistent(
                checked_path,
                parent_descriptor=parent_descriptor,
                expected_stat=stat_result,
                platform_support=_platform_support,
            )
            if file_type == "symlink"
            else None
        )
        if file_type not in {"file", "symlink"}:
            _ensure_child_stat_still_matches(
                checked_path,
                parent_descriptor=parent_descriptor,
                expected_stat=stat_result,
                expected_file_type=file_type,
            )
        if hooks is not None and hooks.before_complete is not None:
            hooks.before_complete(checked_path)
        _ensure_child_stat_still_matches(
            checked_path,
            parent_descriptor=parent_descriptor,
            expected_stat=stat_result,
            expected_file_type=file_type,
        )
        _ensure_directory_descriptor_matches_path(parent_descriptor, checked_path.parent)
        return ResourceSnapshot(
            path=checked_path,
            exists=True,
            file_type=file_type,
            content_hash=content_hash,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=symlink_target,
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
            ctime_ns=stat_result.st_ctime_ns,
        )
    finally:
        os.close(parent_descriptor)


def _file_type_from_stat(stat_result: os.stat_result) -> FileType:
    if S_ISLNK(stat_result.st_mode):
        return "symlink"
    if S_ISREG(stat_result.st_mode):
        return "file"
    if S_ISDIR(stat_result.st_mode):
        return "directory"
    return "other"


def _readlink_consistent(
    path: Path,
    *,
    parent_descriptor: int,
    expected_stat: os.stat_result,
    platform_support: _SnapshotPlatformSupport,
) -> str:
    if not platform_support.readlink_dir_fd:
        raise UnsupportedFilesystemSnapshotError(
            "snapshot refused because this platform lacks descriptor-relative symlink reading"
        )
    try:
        target = os.readlink(path.name, dir_fd=parent_descriptor)
    except OSError as exc:
        raise UnsafePathError(f"snapshot refused because classified symlink changed before readlink: {path}") from exc
    _ensure_child_stat_still_matches(
        path,
        parent_descriptor=parent_descriptor,
        expected_stat=expected_stat,
        expected_file_type="symlink",
    )
    return target


def _ensure_child_stat_still_matches(
    path: Path,
    *,
    parent_descriptor: int,
    expected_stat: os.stat_result,
    expected_file_type: FileType,
) -> None:
    try:
        actual_stat = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafePathError(f"snapshot refused because classified path disappeared: {path}") from exc
    except OSError as exc:
        raise UnsafePathError(f"snapshot refused because classified path could not be revalidated: {path}") from exc
    if _file_type_from_stat(actual_stat) != expected_file_type or not _same_snapshot_metadata(
        actual_stat,
        expected_stat,
    ):
        raise UnsafePathError(f"snapshot refused because classified path changed before snapshot completed: {path}")


def _ensure_child_still_missing(path: Path, *, parent_descriptor: int) -> None:
    try:
        os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UnsafePathError(
            f"snapshot refused because classified missing path could not be revalidated: {path}"
        ) from exc
    raise UnsafePathError(
        f"snapshot refused because classified missing path appeared before snapshot completed: {path}"
    )


def _sha256_regular_file(
    path: Path,
    *,
    parent_descriptor: int,
    expected_stat: os.stat_result,
    chunk_size: int,
    hooks: SnapshotHooks | None,
    platform_support: _SnapshotPlatformSupport,
) -> str:
    _require_descriptor_relative_file_snapshot(platform_support=platform_support)
    file_descriptor = _open_regular_child(
        parent_descriptor,
        path.name,
        path,
        expected_stat=expected_stat,
        hooks=hooks,
    )
    digest = hashlib.sha256()
    with os.fdopen(file_descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
        final_stat = os.fstat(handle.fileno())
    if not _same_snapshot_metadata(final_stat, expected_stat):
        raise UnsafePathError(f"snapshot refused because classified file changed before snapshot completed: {path}")
    return digest.hexdigest()


def _require_descriptor_relative_classification(*, platform_support: _SnapshotPlatformSupport) -> None:
    if not all(
        [
            platform_support.open_dir_fd,
            platform_support.stat_dir_fd,
            platform_support.stat_follow_symlinks,
            platform_support.nofollow_directory_open,
        ]
    ):
        raise UnsupportedFilesystemSnapshotError(
            "snapshot refused because this platform lacks descriptor-relative no-follow path classification"
        )


def _require_descriptor_relative_file_snapshot(*, platform_support: _SnapshotPlatformSupport) -> None:
    if not platform_support.nofollow_file_open:
        raise UnsupportedFilesystemSnapshotError(
            "snapshot refused because this platform lacks descriptor-relative no-follow file opening"
        )


def _open_parent_directory_descriptor_relative(path: Path) -> int:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    parent_descriptor = _open_directory_anchor(anchor)
    current_path = anchor
    try:
        for part in absolute.parent.relative_to(anchor).parts:
            current_path = current_path / part
            next_descriptor = _open_child_directory(parent_descriptor, part, current_path)
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor
        _ensure_directory_descriptor_matches_path(parent_descriptor, absolute.parent)
        return parent_descriptor
    except BaseException:
        os.close(parent_descriptor)
        raise


def _open_regular_child(
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    expected_stat: os.stat_result,
    hooks: SnapshotHooks | None,
) -> int:
    if hooks is not None and hooks.before_open_file is not None:
        hooks.before_open_file(path)
    _ensure_directory_descriptor_matches_path(parent_descriptor, path.parent)
    file_descriptor = _open_child_file(parent_descriptor, name, path)
    try:
        actual_stat = os.fstat(file_descriptor)
        if not S_ISREG(actual_stat.st_mode):
            if S_ISLNK(actual_stat.st_mode):
                raise UnsafePathError(f"snapshot refused because file path redirects elsewhere: {path}")
            raise UnsafePathError(f"snapshot refused for non-regular file: {path}")
        if not _same_file_identity(actual_stat, expected_stat):
            raise UnsafePathError(f"snapshot refused because classified file changed before open: {path}")
        return file_descriptor
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(file_descriptor)
        raise


def _open_directory_anchor(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise UnsafePathError(f"snapshot refused because directory could not be opened safely: {path}") from exc


def _open_child_directory(parent_descriptor: int, name: str, path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        safety = inspect_path(path)
        if safety.is_symlink or safety.is_windows_reparse_point:
            raise UnsafePathError(f"snapshot refused because parent path redirects elsewhere: {path}") from exc
        if safety.exists and not safety.is_dir:
            raise UnsafePathError(f"snapshot refused because parent path is not a directory: {path}") from exc
        raise UnsafePathError(f"snapshot refused because directory could not be opened safely: {path}") from exc


def _ensure_directory_descriptor_matches_path(parent_descriptor: int, path: Path) -> None:
    descriptor_stat = os.fstat(parent_descriptor)
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafePathError(f"snapshot refused because parent path is no longer present: {path}") from exc
    if not _same_file_identity(descriptor_stat, path_stat):
        raise UnsafePathError(f"snapshot refused because parent path redirects elsewhere: {path}")


def _open_child_file(
    parent_descriptor: int,
    name: str,
    path: Path,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        safety = inspect_path(path)
        if safety.is_symlink or safety.is_windows_reparse_point:
            raise UnsafePathError(f"snapshot refused because file path redirects elsewhere: {path}") from exc
        if not safety.exists:
            raise UnsafePathError(f"snapshot refused because classified file disappeared before open: {path}") from exc
        raise UnsafePathError(f"snapshot refused because classified file could not be opened safely: {path}") from exc


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_ino == right.st_ino and left.st_dev == right.st_dev


def _same_snapshot_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file_identity(left, right)
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )
