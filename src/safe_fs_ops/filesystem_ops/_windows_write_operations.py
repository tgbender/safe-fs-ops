from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from safe_fs_ops.filesystem_ops._windows_operation_common import (
    _DirectoryFlush,
    _PathInspector,
    _TokenHex,
    _WindowsReparseChecker,
    create_temp_file_same_directory,
    default_token_hex,
    ensure_safe_directory,
    ensure_safe_parent_chain,
    ensure_safe_write_path,
    ensure_same_parent_identity,
    ensure_same_parent_identity_after_mutation,
    flush_directory_if_requested,
    path_identity,
)
from safe_fs_ops.filesystem_ops._windows_primitives import (
    ReplaceFilePartialFailureError,
    flush_file_descriptor_windows,
    is_windows_reparse_point,
    replace_file_windows,
)
from safe_fs_ops.filesystem_ops.mutation_support import (
    AtomicWriteHooks,
    DurabilityMode,
    UnsupportedFilesystemMutationError,
    _durability_wrapped_fsync,
    _validate_durability_mode,
)
from safe_fs_ops.filesystem_ops.paths import absolute_without_resolving, inspect_path

_FileFlush = Callable[[int], None]


class _ReplaceFile(Protocol):
    def __call__(self, source: Path, destination: Path, *, write_through: bool) -> None: ...


def atomic_write_text_windows(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    replace: object | None = None,
    hooks: AtomicWriteHooks | None = None,
    _file_flush: _FileFlush = flush_file_descriptor_windows,
    _directory_flush: _DirectoryFlush | None = None,
    _path_inspector: _PathInspector = inspect_path,
    _replace_file: _ReplaceFile = replace_file_windows,
    _token_hex: _TokenHex = default_token_hex,
    _windows_reparse_checker: _WindowsReparseChecker = is_windows_reparse_point,
) -> None:
    if newline is None or newline == "":
        normalized = content
    else:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
    atomic_write_bytes_windows(
        path,
        normalized.encode(encoding),
        durability=durability,
        replace=replace,
        hooks=hooks,
        _file_flush=_file_flush,
        _directory_flush=_directory_flush,
        _path_inspector=_path_inspector,
        _replace_file=_replace_file,
        _token_hex=_token_hex,
        _windows_reparse_checker=_windows_reparse_checker,
    )


def atomic_write_bytes_windows(
    path: Path | str,
    content: bytes,
    *,
    permissions: int | None = None,
    durability: DurabilityMode = DurabilityMode.FSYNC,
    replace: object | None = None,
    hooks: AtomicWriteHooks | None = None,
    _file_flush: _FileFlush = flush_file_descriptor_windows,
    _directory_flush: _DirectoryFlush | None = None,
    _path_inspector: _PathInspector = inspect_path,
    _replace_file: _ReplaceFile = replace_file_windows,
    _token_hex: _TokenHex = default_token_hex,
    _windows_reparse_checker: _WindowsReparseChecker = is_windows_reparse_point,
) -> None:
    _validate_durability_mode(durability)
    if replace is not None:
        raise UnsupportedFilesystemMutationError(
            "atomic write refused because custom replace callbacks bypass Windows-native replacement"
        )
    target = absolute_without_resolving(path)
    ensure_safe_write_path(
        target,
        operation="atomic write",
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    _mkdir_parent_for_write_windows(
        target,
        hooks=hooks,
        durability=durability,
        directory_flush=_directory_flush,
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    ensure_safe_write_path(
        target,
        operation="atomic write",
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    if hooks is not None and hooks.before_open_parent is not None:
        hooks.before_open_parent(target.parent)
    ensure_safe_directory(
        target.parent,
        operation="atomic write",
        path_inspector=_path_inspector,
        windows_reparse_checker=_windows_reparse_checker,
    )
    parent_identity = path_identity(target.parent)

    temp_path: Path | None = None
    temp_handle = None
    try:
        if hooks is not None and hooks.before_temp_file is not None:
            hooks.before_temp_file(target)
        temp_path, temp_handle = create_temp_file_same_directory(target.parent, target.name, token_hex=_token_hex)
        temp_handle.write(content)
        temp_handle.flush()
        _durability_wrapped_fsync(durability, _file_flush)(temp_handle.fileno())
        temp_handle.close()
        temp_handle = None
        if permissions is not None:
            temp_path.chmod(permissions)
        if hooks is not None and hooks.before_replace is not None:
            hooks.before_replace(temp_path, target)
        ensure_safe_write_path(
            target,
            operation="atomic write",
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
        ensure_same_parent_identity(
            target.parent,
            parent_identity,
            operation="atomic write",
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
        if hooks is not None and hooks.after_replace_validation is not None:
            hooks.after_replace_validation(temp_path, target)
        try:
            _replace_file(temp_path, target, write_through=durability is DurabilityMode.FSYNC)
        except ReplaceFilePartialFailureError:
            temp_path = None
            raise
        temp_path = None
        ensure_same_parent_identity_after_mutation(
            target.parent,
            parent_identity,
            operation="atomic write",
            path_inspector=_path_inspector,
            windows_reparse_checker=_windows_reparse_checker,
        )
        flush_directory_if_requested(durability, target.parent, _directory_flush)
    finally:
        if temp_handle is not None:
            with contextlib.suppress(OSError):
                temp_handle.close()
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(temp_path)


def _mkdir_parent_for_write_windows(
    path: Path,
    *,
    hooks: AtomicWriteHooks | None,
    durability: DurabilityMode,
    directory_flush: _DirectoryFlush | None,
    path_inspector: _PathInspector,
    windows_reparse_checker: _WindowsReparseChecker,
) -> None:
    ensure_safe_parent_chain(
        path,
        operation="atomic write",
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    if hooks is not None and hooks.before_parent_mkdir is not None:
        hooks.before_parent_mkdir(path.parent)
    ensure_safe_parent_chain(
        path,
        operation="atomic write",
        path_inspector=path_inspector,
        windows_reparse_checker=windows_reparse_checker,
    )
    from safe_fs_ops.filesystem_ops._windows_directory_operations import make_directory_windows

    make_directory_windows(
        path.parent,
        parents=True,
        exist_ok=True,
        durability=durability,
        _directory_flush=directory_flush,
        _path_inspector=path_inspector,
        _windows_reparse_checker=windows_reparse_checker,
    )


__all__ = [
    "atomic_write_bytes_windows",
    "atomic_write_text_windows",
]
