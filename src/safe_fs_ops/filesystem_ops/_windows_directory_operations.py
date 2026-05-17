from __future__ import annotations

import os
from pathlib import Path

from safe_fs_ops.filesystem_ops._windows_operation_common import (
    _DirectoryFlush,
    _PathInspector,
    _WindowsReparseChecker,
    ensure_safe_delete_path,
    ensure_safe_directory,
    ensure_safe_mkdir_path,
    ensure_same_parent_identity,
    ensure_same_parent_identity_after_mutation,
    flush_directory_if_requested,
    path_identity,
    safe_inspect,
)
from safe_fs_ops.filesystem_ops._windows_primitives import is_windows_reparse_point
from safe_fs_ops.filesystem_ops.mutation_support import (
    DeleteHooks,
    DurabilityMode,
    MakeDirectoryHooks,
    _validate_durability_mode,
)
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path


def delete_file_windows(
    path: Path | str,
    *,
    missing_ok: bool = False,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: DeleteHooks | None = None,
    _directory_flush: _DirectoryFlush | None = None,
    _path_inspector: _PathInspector = inspect_path,
    _windows_reparse_checker: _WindowsReparseChecker = is_windows_reparse_point,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    ensure_safe_delete_path(
        target,
        operation="delete",
        missing_ok=missing_ok,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    ensure_safe_delete_path(
        target,
        operation="delete",
        missing_ok=missing_ok,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    if not target.exists():
        return
    parent_identity = path_identity(target.parent)
    if hooks is not None and hooks.before_unlink is not None:
        hooks.before_unlink(target)
    ensure_safe_delete_path(
        target,
        operation="delete",
        missing_ok=missing_ok,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    ensure_same_parent_identity(
        target.parent,
        parent_identity,
        operation="delete",
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    if hooks is not None and hooks.after_unlink_validation is not None:
        hooks.after_unlink_validation(target)
    try:
        os.unlink(target)
    except FileNotFoundError:
        if not missing_ok:
            raise
        ensure_safe_directory(
            target.parent,
            operation="delete",
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
        return
    ensure_same_parent_identity_after_mutation(
        target.parent,
        parent_identity,
        operation="delete",
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    flush_directory_if_requested(durability, target.parent, _directory_flush)


def make_directory_windows(
    path: Path | str,
    *,
    parents: bool = False,
    exist_ok: bool = False,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    hooks: MakeDirectoryHooks | None = None,
    _directory_flush: _DirectoryFlush | None = None,
    _path_inspector: _PathInspector = inspect_path,
    _windows_reparse_checker: _WindowsReparseChecker = is_windows_reparse_point,
) -> None:
    _validate_durability_mode(durability)
    target = absolute_without_resolving(path)
    ensure_safe_mkdir_path(
        target,
        operation="mkdir",
        exist_ok=exist_ok,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    safety = safe_inspect(target, path_inspector=_path_inspector, windows_reparse_checker=_windows_reparse_checker)
    if safety.exists and safety.is_dir and exist_ok:
        return
    if hooks is not None and hooks.before_mkdir is not None:
        hooks.before_mkdir(target)
    ensure_safe_mkdir_path(
        target,
        operation="mkdir",
        exist_ok=exist_ok,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    if parents:
        _make_directory_incremental_windows(
            target,
            exist_ok=exist_ok,
            durability=durability,
            hooks=hooks,
            directory_flush=_directory_flush,
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
    else:
        _make_directory_single_windows(
            target,
            exist_ok=exist_ok,
            durability=durability,
            hooks=hooks,
            directory_flush=_directory_flush,
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
    ensure_safe_mkdir_path(
        target,
        operation="mkdir",
        exist_ok=True,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )


def _make_directory_incremental_windows(
    target: Path,
    *,
    exist_ok: bool,
    durability: DurabilityMode,
    hooks: MakeDirectoryHooks | None,
    directory_flush: _DirectoryFlush | None,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    anchor = Path(target.anchor)
    current = anchor
    parts = target.relative_to(anchor).parts
    for index, part in enumerate(parts):
        parent = current
        current = current / part
        is_final = index == len(parts) - 1
        ensure_safe_directory(
            parent,
            operation="mkdir",
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
        parent_identity = path_identity(parent)
        safety = safe_inspect(current, path_inspector=path_inspector, windows_reparse_checker=windows_reparse_checker)
        if safety.exists:
            if not safety.is_dir:
                raise UnsafePathError(f"mkdir refused because path exists and is not a directory: {current}")
            if is_final and not exist_ok:
                raise FileExistsError(current)
            continue
        if hooks is not None and hooks.before_descriptor_mkdir is not None:
            hooks.before_descriptor_mkdir(current)
        ensure_same_parent_identity(
            parent,
            parent_identity,
            operation="mkdir",
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
        try:
            os.mkdir(current)
        except FileExistsError:
            updated = safe_inspect(
                current,
                path_inspector=path_inspector,
                windows_reparse_checker=windows_reparse_checker,
            )
            if not updated.is_dir or (is_final and not exist_ok):
                raise
        ensure_same_parent_identity_after_mutation(
            parent,
            parent_identity,
            operation="mkdir",
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
        flush_directory_if_requested(durability, parent, directory_flush)
    flush_directory_if_requested(durability, target, directory_flush)


def _make_directory_single_windows(
    target: Path,
    *,
    exist_ok: bool,
    durability: DurabilityMode,
    hooks: MakeDirectoryHooks | None,
    directory_flush: _DirectoryFlush | None,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_directory(
        target.parent,
        operation="mkdir",
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    parent_identity = path_identity(target.parent)
    if hooks is not None and hooks.before_descriptor_mkdir is not None:
        hooks.before_descriptor_mkdir(target)
    ensure_same_parent_identity(
        target.parent,
        parent_identity,
        operation="mkdir",
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    try:
        os.mkdir(target)
    except FileExistsError:
        if not exist_ok:
            raise
        ensure_safe_mkdir_path(
            target,
            operation="mkdir",
            exist_ok=True,
            path_inspector=path_inspector,
            windows_reparse_checker=windows_reparse_checker,
        )
    ensure_same_parent_identity_after_mutation(
        target.parent,
        parent_identity,
        operation="mkdir",
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    flush_directory_if_requested(durability, target.parent, directory_flush)
    flush_directory_if_requested(durability, target, directory_flush)


__all__ = [
    "delete_file_windows",
    "make_directory_windows",
]
