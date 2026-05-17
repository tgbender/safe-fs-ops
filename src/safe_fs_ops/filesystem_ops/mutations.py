from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import BinaryIO

from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    AtomicWriteHooks,
    DeleteHooks,
    DurabilityMode,
    MakeDirectoryHooks,
    ParentChangedAfterMutationError,
    ReplaceFunc,
    UnsupportedFilesystemMutationError,
    _chmod_open_file,
    _cleanup_temp_file,
    _create_temp_file,
    _DirectoryFsync,
    _durability_wrapped_fsync,
    _ensure_directory_descriptor_matches_path,
    _ensure_directory_descriptor_matches_path_after_mutation,
    _ensure_open_file_matches_child,
    _ensure_safe_child_for_delete,
    _ensure_safe_child_for_replace,
    _existing_permissions_at,
    _FileFsync,
    _fsync_directory_descriptor,
    _fsync_file,
    _make_directory_incremental,
    _make_directory_single,
    _mkdir_parent_for_write,
    _MutationPlatformSupport,
    _normalize_newlines,
    _open_directory_for_mutation,
    _replace,
    _require_descriptor_relative_mkdir,
    _require_descriptor_relative_mutation,
    _validate_durability_mode,
)
from safe_fs_ops.filesystem_ops.paths import (
    absolute_without_resolving,
    ensure_safe_delete_path,
    ensure_safe_mkdir_path,
    ensure_safe_write_path,
    inspect_path,
)

__all__ = [
    "AtomicWriteHooks",
    "DeleteHooks",
    "DurabilityMode",
    "MakeDirectoryHooks",
    "ParentChangedAfterMutationError",
    "UnsupportedFilesystemMutationError",
    "atomic_write_bytes",
    "atomic_write_text",
    "delete_file",
    "make_directory",
]


def atomic_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    replace: ReplaceFunc | None = None,
    hooks: AtomicWriteHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _file_fsync: _FileFsync = _fsync_file,
    _directory_fsync: _DirectoryFsync = _fsync_directory_descriptor,
) -> None:
    _validate_durability_mode(durability)
    normalized = content if newline is None or newline == "" else _normalize_newlines(content, newline=newline)
    atomic_write_bytes(
        path,
        normalized.encode(encoding),
        durability=durability,
        replace=replace,
        hooks=hooks,
        _platform_support=_platform_support,
        _file_fsync=_file_fsync,
        _directory_fsync=_directory_fsync,
    )


def atomic_write_bytes(
    path: Path | str,
    content: bytes,
    *,
    permissions: int | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    replace: ReplaceFunc | None = None,
    hooks: AtomicWriteHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _file_fsync: _FileFsync = _fsync_file,
    _directory_fsync: _DirectoryFsync = _fsync_directory_descriptor,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    if replace is not None:
        raise UnsupportedFilesystemMutationError(
            "atomic write refused because custom replace callbacks bypass descriptor-relative replace"
        )
    ensure_safe_write_path(target, operation="atomic write")
    _require_descriptor_relative_mutation(
        operation="atomic write",
        needs_replace=True,
        platform_support=_platform_support,
    )
    _require_descriptor_relative_mkdir(operation="atomic write", platform_support=_platform_support)
    _mkdir_parent_for_write(
        target,
        hooks=hooks,
        platform_support=_platform_support,
        directory_fsync=_durability_wrapped_fsync(durability, _directory_fsync),
    )
    ensure_safe_write_path(target, operation="atomic write")

    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)

    temp_path: Path | None = None
    temp_handle: BinaryIO | None = None
    temp_parent_descriptor = _open_directory_for_mutation(
        target.parent,
        operation="atomic write",
        needs_replace=True,
        platform_support=_platform_support,
    )
    temp_name: str | None = None
    try:
        existing_mode = _existing_permissions_at(temp_parent_descriptor, target.name)
        mode = permissions if permissions is not None else existing_mode

        if hooks is not None and hooks.before_temp_file is not None:
            hooks.before_temp_file(target)

        _ensure_directory_descriptor_matches_path(
            temp_parent_descriptor,
            target.parent,
            operation="atomic write",
        )
        temp_descriptor, temp_name = _create_temp_file(temp_parent_descriptor, target.name)
        temp_path = target.parent / temp_name
        _ensure_directory_descriptor_matches_path(
            temp_parent_descriptor,
            target.parent,
            operation="atomic write",
        )
        temp_handle = os.fdopen(temp_descriptor, "wb")
        temp_handle.write(content)
        temp_handle.flush()
        if mode is not None:
            _chmod_open_file(temp_handle.fileno(), mode, operation="atomic write")
        _durability_wrapped_fsync(durability, _file_fsync)(temp_handle.fileno())
        assert temp_name is not None
        _ensure_open_file_matches_child(
            temp_handle.fileno(),
            temp_parent_descriptor,
            temp_name,
            path=temp_path,
            operation="atomic write",
        )

        if hooks is not None and hooks.before_replace is not None:
            hooks.before_replace(temp_path, target)
        _ensure_directory_descriptor_matches_path(
            temp_parent_descriptor,
            target.parent,
            operation="atomic write",
        )
        _ensure_safe_child_for_replace(temp_parent_descriptor, target.name, operation="atomic write")

        temp_handle.close()
        temp_handle = None
        assert temp_path is not None
        if hooks is not None and hooks.after_replace_validation is not None:
            hooks.after_replace_validation(temp_path, target)
        _replace(
            temp_name,
            target.name,
            parent_descriptor=temp_parent_descriptor,
            platform_support=_platform_support,
        )
        temp_path = None
        temp_name = None
        _ensure_directory_descriptor_matches_path_after_mutation(
            temp_parent_descriptor,
            target.parent,
            operation="atomic write",
        )
        _durability_wrapped_fsync(durability, _directory_fsync)(temp_parent_descriptor)
    finally:
        if temp_handle is not None:
            with contextlib.suppress(OSError):
                temp_handle.close()
        if temp_name is not None:
            _cleanup_temp_file(temp_name, parent_descriptor=temp_parent_descriptor)
        os.close(temp_parent_descriptor)


def delete_file(
    path: Path | str,
    *,
    missing_ok: bool = False,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: DeleteHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _directory_fsync: _DirectoryFsync = _fsync_directory_descriptor,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    ensure_safe_delete_path(target, operation="delete", missing_ok=missing_ok)
    if missing_ok and not target.exists():
        return
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    parent_descriptor = _open_directory_for_mutation(
        target.parent,
        operation="delete",
        needs_replace=False,
        platform_support=_platform_support,
    )
    try:
        if hooks is not None and hooks.before_unlink is not None:
            hooks.before_unlink(target)
        _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="delete")
        _ensure_safe_child_for_delete(parent_descriptor, target.name, operation="delete", missing_ok=missing_ok)
        if hooks is not None and hooks.after_unlink_validation is not None:
            hooks.after_unlink_validation(target)
        os.unlink(target.name, dir_fd=parent_descriptor)
        _ensure_directory_descriptor_matches_path_after_mutation(
            parent_descriptor,
            target.parent,
            operation="delete",
        )
        _durability_wrapped_fsync(durability, _directory_fsync)(parent_descriptor)
    except FileNotFoundError:
        if not missing_ok:
            raise
        _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="delete")
    finally:
        os.close(parent_descriptor)


def make_directory(
    path: Path | str,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: MakeDirectoryHooks | None = None,
    _platform_support: _MutationPlatformSupport = _DEFAULT_PLATFORM_SUPPORT,
    _directory_fsync: _DirectoryFsync = _fsync_directory_descriptor,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    ensure_safe_mkdir_path(target, operation="mkdir", exist_ok=exist_ok)
    initial_safety = inspect_path(target)
    if initial_safety.exists and initial_safety.is_dir and exist_ok:
        return
    _require_descriptor_relative_mkdir(operation="mkdir", platform_support=_platform_support)
    if hooks is not None and hooks.before_mkdir is not None:
        hooks.before_mkdir(target)
    ensure_safe_mkdir_path(target, operation="mkdir", exist_ok=exist_ok)
    if parents:
        _make_directory_incremental(
            target,
            exist_ok=exist_ok,
            operation="mkdir",
            hooks=hooks,
            platform_support=_platform_support,
            directory_fsync=_durability_wrapped_fsync(durability, _directory_fsync),
        )
    else:
        _make_directory_single(
            target,
            exist_ok=exist_ok,
            operation="mkdir",
            hooks=hooks,
            platform_support=_platform_support,
            directory_fsync=_durability_wrapped_fsync(durability, _directory_fsync),
        )
    ensure_safe_mkdir_path(target, operation="mkdir", exist_ok=True)
