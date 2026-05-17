from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR

from safe_fs_ops.filesystem_ops._windows_identity_rmdir import (
    remove_empty_directory_by_identity_windows,
    windows_identity_safe_remove_available,
)
from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    DurabilityMode,
    _DirectoryFsync,
    _durability_wrapped_fsync,
    _ensure_directory_descriptor_matches_path,
    _ensure_directory_descriptor_matches_path_after_mutation,
    _fsync_directory_descriptor,
    _MutationPlatformSupport,
    _open_existing_directory_descriptor_relative,
    _validate_durability_mode,
)
from safe_fs_ops.filesystem_ops.mutations import ParentChangedAfterMutationError, UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops.paths import absolute_without_resolving, ensure_safe_mkdir_path

BeforeOpenParentHook = Callable[[Path], None]
MissingObservedHook = Callable[[Path], None]
BeforeRmdirHook = Callable[[Path], None]
AfterRmdirValidationHook = Callable[[Path], None]


@dataclass(frozen=True, slots=True)
class DirectoryIdentity:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, stat_result: os.stat_result) -> DirectoryIdentity:
        return cls(device=stat_result.st_dev, inode=stat_result.st_ino)


class IdentitySafeRemoveDirectoryUnavailableError(UnsupportedFilesystemMutationError):
    """Raised when identity-safe directory removal is unavailable on this backend."""


@dataclass(frozen=True, slots=True)
class RemoveDirectoryHooks:
    before_open_parent: BeforeOpenParentHook | None = None
    missing_ok_missing_observed: MissingObservedHook | None = None
    before_rmdir: BeforeRmdirHook | None = None
    after_rmdir_validation: AfterRmdirValidationHook | None = None


def remove_empty_directory(
    path: Path | str,
    *,
    missing_ok: bool = False,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: RemoveDirectoryHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _directory_fsync: _DirectoryFsync = _fsync_directory_descriptor,
    _platform: str = sys.platform,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    ensure_safe_mkdir_path(target, operation="rmdir", exist_ok=True)
    if _is_windows_platform(_platform) and windows_identity_safe_remove_available():
        _remove_empty_directory_windows(target, missing_ok=missing_ok, durability=durability, hooks=hooks)
        return
    _require_descriptor_relative_rmdir(operation="rmdir", platform_support=_platform_support)
    if not target.exists():
        if missing_ok:
            return
        raise FileNotFoundError(target)
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    ensure_safe_mkdir_path(target, operation="rmdir", exist_ok=True)
    parent_descriptor = _open_existing_directory_descriptor_relative(target.parent, operation="rmdir")
    try:
        _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="rmdir")
        target_exists = _ensure_safe_child_directory_for_rmdir(
            parent_descriptor,
            target.name,
            operation="rmdir",
            missing_ok=missing_ok,
        )
        if not target_exists:
            if hooks is not None and hooks.missing_ok_missing_observed is not None:
                hooks.missing_ok_missing_observed(target)
            return
        if hooks is not None and hooks.before_rmdir is not None:
            hooks.before_rmdir(target)
        if hooks is not None and hooks.after_rmdir_validation is not None:
            hooks.after_rmdir_validation(target)
        os.rmdir(target.name, dir_fd=parent_descriptor)
        _ensure_directory_descriptor_matches_path_after_mutation(
            parent_descriptor,
            target.parent,
            operation="rmdir",
        )
        _durability_wrapped_fsync(durability, _directory_fsync)(parent_descriptor)
    except FileNotFoundError:
        if not missing_ok:
            raise
        _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="rmdir")
    finally:
        os.close(parent_descriptor)


def _remove_empty_directory_windows(
    target: Path,
    *,
    missing_ok: bool,
    durability: DurabilityMode,
    hooks: RemoveDirectoryHooks | None,
) -> None:
    if not target.exists():
        if missing_ok:
            if hooks is not None and hooks.missing_ok_missing_observed is not None:
                hooks.missing_ok_missing_observed(target)
            return
        raise FileNotFoundError(target)
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    ensure_safe_mkdir_path(target, operation="rmdir", exist_ok=True)
    try:
        expected_identity = DirectoryIdentity.from_stat(target.stat())
    except FileNotFoundError:
        if missing_ok:
            if hooks is not None and hooks.missing_ok_missing_observed is not None:
                hooks.missing_ok_missing_observed(target)
            return
        raise
    try:
        remove_empty_directory_by_identity_windows(
            target,
            expected_identity=(expected_identity.device, expected_identity.inode),
            durability=durability,
            after_identity_validation=_after_identity_validation_hook(hooks),
        )
    except FileNotFoundError:
        if not missing_ok:
            raise
        if hooks is not None and hooks.missing_ok_missing_observed is not None:
            hooks.missing_ok_missing_observed(target)


def remove_existing_empty_directory_by_identity(
    path: Path | str,
    *,
    expected_identity: DirectoryIdentity,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: RemoveDirectoryHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _platform: str = sys.platform,
    _windows_identity_safe_remove_available: Callable[[], bool] = windows_identity_safe_remove_available,
    _windows_remove: Callable[..., None] = remove_empty_directory_by_identity_windows,
) -> None:
    _validate_durability_mode(durability)
    if not isinstance(expected_identity, DirectoryIdentity):
        raise TypeError(f"expected_identity must be a DirectoryIdentity, got {type(expected_identity).__name__}")
    target = absolute_without_resolving(path)
    ensure_safe_mkdir_path(target, operation="identity-safe rmdir", exist_ok=True)
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    ensure_safe_mkdir_path(target, operation="identity-safe rmdir", exist_ok=True)
    if _is_windows_platform(_platform) and _windows_identity_safe_remove_available():
        _windows_remove(
            target,
            expected_identity=(expected_identity.device, expected_identity.inode),
            durability=durability,
            after_identity_validation=_after_identity_validation_hook(hooks),
        )
        return
    raise IdentitySafeRemoveDirectoryUnavailableError(
        identity_safe_remove_directory_unavailable_reason(
            platform_support=_platform_support,
            platform=_platform,
        )
    )


def identity_safe_remove_directory_unavailable_reason(
    *,
    platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    platform: str = sys.platform,
) -> str:
    if _is_windows_platform(platform):
        return "identity-safe rmdir refused because native Windows handle delete disposition API is unavailable"
    missing = [
        name
        for name, supported in {
            "rmdir_dir_fd": platform_support.rmdir_dir_fd,
            "open_dir_fd": platform_support.open_dir_fd,
            "stat_dir_fd": platform_support.stat_dir_fd,
            "stat_follow_symlinks": platform_support.stat_follow_symlinks,
            "nofollow_directory_open": platform_support.nofollow_directory_open,
        }.items()
        if not supported
    ]
    if missing:
        return f"identity-safe rmdir refused because required filesystem features are missing: {', '.join(missing)}"
    return (
        "identity-safe rmdir refused because Python exposes no atomic identity-conditional "
        "directory removal API for this backend"
    )


def _is_windows_platform(platform: str) -> bool:
    return platform in {"win32", "windows"}


def _after_identity_validation_hook(
    hooks: RemoveDirectoryHooks | None,
) -> AfterRmdirValidationHook | None:
    if hooks is None:
        return None
    if hooks.before_rmdir is None:
        return hooks.after_rmdir_validation
    if hooks.after_rmdir_validation is None:
        return hooks.before_rmdir
    before_rmdir = hooks.before_rmdir
    after_rmdir_validation = hooks.after_rmdir_validation

    def run_hooks(path: Path) -> None:
        before_rmdir(path)
        after_rmdir_validation(path)

    return run_hooks


def _require_descriptor_relative_rmdir(
    *,
    operation: str,
    platform_support: _MutationPlatformSupport,
) -> None:
    required = [
        platform_support.rmdir_dir_fd,
        platform_support.open_dir_fd,
        platform_support.stat_dir_fd,
        platform_support.stat_follow_symlinks,
        platform_support.nofollow_directory_open,
    ]
    if not all(required):
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because this platform lacks descriptor-relative directory removal"
        )


def _ensure_safe_child_directory_for_rmdir(
    parent_descriptor: int,
    name: str,
    *,
    operation: str,
    missing_ok: bool,
) -> bool:
    try:
        stat_result = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return False
        raise
    if not S_ISDIR(stat_result.st_mode):
        raise UnsupportedFilesystemMutationError(f"{operation} refused because path is not a directory: {name}")
    return True


__all__ = [
    "DirectoryIdentity",
    "IdentitySafeRemoveDirectoryUnavailableError",
    "ParentChangedAfterMutationError",
    "RemoveDirectoryHooks",
    "UnsupportedFilesystemMutationError",
    "identity_safe_remove_directory_unavailable_reason",
    "remove_existing_empty_directory_by_identity",
    "remove_empty_directory",
]
