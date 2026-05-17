from __future__ import annotations

import contextlib
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from stat import S_IMODE, S_ISLNK, S_ISREG

from safe_fs_ops.filesystem_ops.paths import (
    UnsafePathError,
    ensure_safe_parent_chain,
    inspect_path,
)

ReplaceFunc = Callable[[Path, Path], None]
BeforeParentMkdirHook = Callable[[Path], None]
BeforeTempFileHook = Callable[[Path], None]
BeforeOpenParentHook = Callable[[Path], None]
BeforeReplaceHook = Callable[[Path, Path], None]
AfterReplaceValidationHook = Callable[[Path, Path], None]
BeforeUnlinkHook = Callable[[Path], None]
AfterUnlinkValidationHook = Callable[[Path], None]
BeforeMkdirHook = Callable[[Path], None]
_Fsync = Callable[[int], None]
_FileFsync = Callable[[int], None]
_DirectoryFsync = Callable[[int], None]


class DurabilityMode(Enum):
    NONE = auto()
    BEST_EFFORT = auto()
    FSYNC = auto()


class UnsupportedFilesystemMutationError(UnsafePathError):
    """Raised when the platform cannot provide conservative mutation semantics."""


class ParentChangedAfterMutationError(UnsafePathError):
    """Raised when a descriptor-relative mutation cannot be confirmed at its lexical parent."""


@dataclass(frozen=True, slots=True)
class AtomicWriteHooks:
    before_parent_mkdir: BeforeParentMkdirHook | None = None
    before_temp_file: BeforeTempFileHook | None = None
    before_replace: BeforeReplaceHook | None = None
    before_open_parent: BeforeOpenParentHook | None = None
    after_replace_validation: AfterReplaceValidationHook | None = None


@dataclass(frozen=True, slots=True)
class DeleteHooks:
    before_unlink: BeforeUnlinkHook | None = None
    before_open_parent: BeforeOpenParentHook | None = None
    after_unlink_validation: AfterUnlinkValidationHook | None = None


@dataclass(frozen=True, slots=True)
class MakeDirectoryHooks:
    before_mkdir: BeforeMkdirHook | None = None
    before_descriptor_mkdir: BeforeMkdirHook | None = None


@dataclass(frozen=True, slots=True)
class _MutationPlatformSupport:
    open_dir_fd: bool
    unlink_dir_fd: bool
    stat_dir_fd: bool
    stat_follow_symlinks: bool
    replace_dir_fd: bool
    rename_dir_fd: bool
    mkdir_dir_fd: bool
    rmdir_dir_fd: bool
    nofollow_directory_open: bool

    @classmethod
    def detect(cls) -> _MutationPlatformSupport:
        return cls(
            open_dir_fd=os.open in os.supports_dir_fd,
            unlink_dir_fd=os.unlink in os.supports_dir_fd,
            stat_dir_fd=os.stat in os.supports_dir_fd,
            stat_follow_symlinks=os.stat in os.supports_follow_symlinks,
            replace_dir_fd=os.replace in os.supports_dir_fd,
            rename_dir_fd=os.rename in os.supports_dir_fd,
            mkdir_dir_fd=os.mkdir in os.supports_dir_fd,
            rmdir_dir_fd=os.rmdir in os.supports_dir_fd,
            nofollow_directory_open=hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW"),
        )


_DEFAULT_PLATFORM_SUPPORT = _MutationPlatformSupport.detect()


def _fsync_file(file_descriptor: int, *, _fsync: _Fsync = os.fsync) -> None:
    _fsync(file_descriptor)


def _fsync_directory_descriptor(descriptor: int, *, _fsync: _Fsync = os.fsync) -> None:
    _fsync(descriptor)


def _validate_durability_mode(durability: DurabilityMode) -> None:
    if not isinstance(durability, DurabilityMode):
        raise TypeError(f"durability must be a DurabilityMode, got {type(durability).__name__}")


def _durability_wrapped_fsync(
    durability: DurabilityMode,
    fsync: _FileFsync | _DirectoryFsync,
) -> _FileFsync | _DirectoryFsync:
    _validate_durability_mode(durability)
    if durability is DurabilityMode.NONE:
        return lambda _descriptor: None
    if durability is DurabilityMode.BEST_EFFORT:

        def best_effort(descriptor: int) -> None:
            with contextlib.suppress(OSError):
                fsync(descriptor)

        return best_effort
    return fsync


def _mkdir_parent_for_write(
    path: Path,
    *,
    hooks: AtomicWriteHooks | None = None,
    platform_support: _MutationPlatformSupport,
    directory_fsync: _DirectoryFsync,
) -> None:
    ensure_safe_parent_chain(path, operation="atomic write")
    if hooks is not None and hooks.before_parent_mkdir is not None:
        hooks.before_parent_mkdir(path.parent)
    ensure_safe_parent_chain(path, operation="atomic write")
    _make_directory_incremental(
        path.parent,
        exist_ok=True,
        operation="atomic write",
        platform_support=platform_support,
        directory_fsync=directory_fsync,
    )
    ensure_safe_parent_chain(path, operation="atomic write")


def _existing_permissions_at(parent_descriptor: int, name: str) -> int | None:
    try:
        stat_result = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not S_ISREG(stat_result.st_mode):
        if S_ISLNK(stat_result.st_mode):
            raise UnsafePathError(f"atomic write refused for symlink path: {name}")
        raise UnsafePathError(f"atomic write refused for non-regular file: {name}")
    return S_IMODE(stat_result.st_mode)


def _supports_same_directory_rename_replace(platform_support: _MutationPlatformSupport) -> bool:
    return os.name != "nt" and platform_support.rename_dir_fd


def _supports_descriptor_relative_replace(platform_support: _MutationPlatformSupport) -> bool:
    return platform_support.replace_dir_fd or _supports_same_directory_rename_replace(platform_support)


def _replace(
    source_name: str,
    target_name: str,
    *,
    parent_descriptor: int,
    platform_support: _MutationPlatformSupport,
) -> None:
    if platform_support.replace_dir_fd:
        os.replace(source_name, target_name, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
        return
    if _supports_same_directory_rename_replace(platform_support):
        os.rename(source_name, target_name, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
        return
    raise UnsupportedFilesystemMutationError(
        "atomic write refused because this platform lacks descriptor-relative replacement"
    )


def _open_directory_for_mutation(
    path: Path,
    *,
    operation: str,
    needs_replace: bool,
    platform_support: _MutationPlatformSupport,
) -> int:
    _require_descriptor_relative_mutation(
        operation=operation,
        needs_replace=needs_replace,
        platform_support=platform_support,
    )
    return _open_existing_directory_descriptor_relative(path, operation=operation)


def _require_descriptor_relative_mutation(
    *,
    operation: str,
    needs_replace: bool,
    platform_support: _MutationPlatformSupport,
) -> None:
    required = [
        platform_support.open_dir_fd,
        platform_support.unlink_dir_fd,
        platform_support.stat_dir_fd,
        platform_support.stat_follow_symlinks,
        platform_support.nofollow_directory_open,
    ]
    if needs_replace:
        required.append(_supports_descriptor_relative_replace(platform_support))
    if not all(required):
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because this platform lacks descriptor-relative filesystem operations"
        )


def _require_descriptor_relative_mkdir(
    *,
    operation: str,
    platform_support: _MutationPlatformSupport,
) -> None:
    required = [
        platform_support.mkdir_dir_fd,
        platform_support.open_dir_fd,
        platform_support.nofollow_directory_open,
    ]
    if not all(required):
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because this platform lacks descriptor-relative directory creation"
        )


def _make_directory_incremental(
    target: Path,
    *,
    exist_ok: bool,
    operation: str,
    hooks: MakeDirectoryHooks | None = None,
    platform_support: _MutationPlatformSupport,
    directory_fsync: _DirectoryFsync,
) -> None:
    _require_descriptor_relative_mkdir(operation=operation, platform_support=platform_support)
    absolute = target.expanduser().absolute()
    anchor = Path(absolute.anchor)
    if absolute == anchor:
        if exist_ok:
            return
        raise FileExistsError(target)

    anchor_descriptor = _open_directory_anchor(anchor, operation=operation)
    current_descriptor = anchor_descriptor
    current_path = anchor
    try:
        parts = absolute.relative_to(anchor).parts
        for index, part in enumerate(parts):
            parent_path = current_path
            current_path = current_path / part
            is_final = index == len(parts) - 1
            try:
                _ensure_directory_descriptor_matches_path(
                    current_descriptor,
                    parent_path,
                    operation=operation,
                )
                if hooks is not None and hooks.before_descriptor_mkdir is not None:
                    hooks.before_descriptor_mkdir(current_path)
                os.mkdir(part, dir_fd=current_descriptor)
            except FileExistsError:
                _ensure_directory_descriptor_matches_path(
                    current_descriptor,
                    parent_path,
                    operation=operation,
                )
                if is_final and not exist_ok:
                    raise
            except OSError as exc:
                message = f"{operation} refused because directory could not be created safely: {current_path}"
                raise UnsafePathError(message) from exc
            else:
                _ensure_directory_descriptor_matches_path_after_mutation(
                    current_descriptor,
                    parent_path,
                    operation=operation,
                )
                directory_fsync(current_descriptor)

            next_descriptor = _open_child_directory_for_mkdir(
                current_descriptor,
                part,
                current_path,
                operation=operation,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        directory_fsync(current_descriptor)
    finally:
        os.close(current_descriptor)


def _make_directory_single(
    target: Path,
    *,
    exist_ok: bool,
    operation: str,
    hooks: MakeDirectoryHooks | None = None,
    platform_support: _MutationPlatformSupport,
    directory_fsync: _DirectoryFsync,
) -> None:
    _require_descriptor_relative_mkdir(operation=operation, platform_support=platform_support)
    absolute = target.expanduser().absolute()
    anchor = Path(absolute.anchor)
    if absolute == anchor:
        if exist_ok:
            return
        raise FileExistsError(target)

    parent_descriptor = _open_existing_directory_descriptor_relative(target.parent, operation=operation)
    try:
        try:
            _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation=operation)
            if hooks is not None and hooks.before_descriptor_mkdir is not None:
                hooks.before_descriptor_mkdir(target)
            os.mkdir(target.name, dir_fd=parent_descriptor)
        except FileExistsError:
            _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation=operation)
            if not exist_ok:
                raise
        except OSError as exc:
            message = f"{operation} refused because directory could not be created safely: {target}"
            raise UnsafePathError(message) from exc
        else:
            _ensure_directory_descriptor_matches_path_after_mutation(
                parent_descriptor,
                target.parent,
                operation=operation,
            )
        directory_fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _open_existing_directory_descriptor_relative(path: Path, *, operation: str) -> int:
    absolute = path.expanduser().absolute()
    anchor = Path(absolute.anchor)
    current_descriptor = _open_directory_anchor(anchor, operation=operation)
    current_path = anchor
    try:
        for part in absolute.relative_to(anchor).parts:
            current_path = current_path / part
            next_descriptor = _open_child_directory_for_mkdir(
                current_descriptor,
                part,
                current_path,
                operation=operation,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return current_descriptor
    except BaseException:
        os.close(current_descriptor)
        raise


def _open_directory_anchor(path: Path, *, operation: str) -> int:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        safety = inspect_path(path)
        if safety.is_symlink or safety.is_windows_reparse_point:
            raise UnsafePathError(f"{operation} refused because parent path redirects elsewhere: {path}") from exc
        message = f"{operation} refused because directory could not be opened safely: {path}"
        raise UnsafePathError(message) from exc


def _open_child_directory_for_mkdir(
    parent_descriptor: int,
    name: str,
    path: Path,
    *,
    operation: str,
) -> int:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        safety = inspect_path(path)
        if safety.is_symlink or safety.is_windows_reparse_point:
            raise UnsafePathError(f"{operation} refused because parent path redirects elsewhere: {path}") from exc
        if safety.exists and not safety.is_dir:
            raise UnsafePathError(f"{operation} refused because path is not a directory: {path}") from exc
        message = f"{operation} refused because directory could not be opened safely: {path}"
        raise UnsafePathError(message) from exc


def _create_temp_file(parent_descriptor: int, target_name: str) -> tuple[int, str]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        temp_name = f".{target_name}.{secrets.token_hex(8)}.tmp"
        try:
            return os.open(temp_name, flags, 0o600, dir_fd=parent_descriptor), temp_name
        except FileExistsError:
            continue
    raise FileExistsError(f"could not allocate a unique temp file for {target_name}")


def _cleanup_temp_file(name: str, *, parent_descriptor: int) -> None:
    with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
        os.unlink(name, dir_fd=parent_descriptor)


def _ensure_open_file_matches_child(
    file_descriptor: int,
    parent_descriptor: int,
    name: str,
    *,
    path: Path,
    operation: str,
) -> None:
    descriptor_stat = os.fstat(file_descriptor)
    try:
        path_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafePathError(f"{operation} refused because temporary file disappeared: {path}") from exc
    if not S_ISREG(path_stat.st_mode):
        raise UnsafePathError(f"{operation} refused because temporary path is not a regular file: {path}")
    if not _same_file_identity(descriptor_stat, path_stat):
        raise UnsafePathError(f"{operation} refused because temporary file identity changed: {path}")


def _ensure_directory_descriptor_matches_path(parent_descriptor: int, path: Path, *, operation: str) -> None:
    descriptor_stat = os.fstat(parent_descriptor)
    try:
        path_stat = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafePathError(f"{operation} refused because parent path is no longer present: {path}") from exc
    if not _same_file_identity(descriptor_stat, path_stat):
        raise UnsafePathError(f"{operation} refused because parent path redirects elsewhere: {path}")


def _ensure_directory_descriptor_matches_path_after_mutation(
    parent_descriptor: int,
    path: Path,
    *,
    operation: str,
) -> None:
    try:
        _ensure_directory_descriptor_matches_path(parent_descriptor, path, operation=operation)
    except UnsafePathError as exc:
        raise ParentChangedAfterMutationError(
            f"{operation} refused to report success because parent path changed after mutation: {path}"
        ) from exc


def _ensure_safe_child_for_replace(parent_descriptor: int, name: str, *, operation: str) -> None:
    try:
        stat_result = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not S_ISREG(stat_result.st_mode):
        if S_ISLNK(stat_result.st_mode):
            raise UnsafePathError(f"{operation} refused for symlink path: {name}")
        raise UnsafePathError(f"{operation} refused for non-regular file: {name}")


def _ensure_safe_child_for_delete(
    parent_descriptor: int,
    name: str,
    *,
    operation: str,
    missing_ok: bool,
) -> None:
    try:
        stat_result = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    if not S_ISREG(stat_result.st_mode):
        if S_ISLNK(stat_result.st_mode):
            raise UnsafePathError(f"{operation} refused for symlink path: {name}")
        raise UnsafePathError(f"{operation} refused for non-regular file: {name}")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_ino == right.st_ino and left.st_dev == right.st_dev


def _chmod_open_file(file_descriptor: int, mode: int, *, operation: str) -> None:
    if not hasattr(os, "fchmod"):
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because this platform cannot chmod an open temp file"
        )
    os.fchmod(file_descriptor, mode)


def _normalize_newlines(content: str, *, newline: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
