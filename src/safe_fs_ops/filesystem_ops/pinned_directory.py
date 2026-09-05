from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from safe_fs_ops.filesystem_ops.directory_capture_token import _require_identity
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, ensure_safe_parent_chain, inspect_path


@contextmanager
def pin_directory(path: Path, *, device: int, inode: int) -> Iterator[None]:
    """Keep the inspected directory allocated while recording its adoption."""
    ensure_safe_parent_chain(path, operation="adopt legacy capture")
    safety = inspect_path(path)
    if not safety.is_dir or safety.is_symlink or safety.is_windows_reparse_point or safety.is_mount:
        raise UnsafePathError(f"legacy capture must be an ordinary directory: {path}")
    if os.name == "nt":
        from safe_fs_ops.filesystem_ops._windows_identity_rmdir import (
            _close_handle,
            _file_information,
            _kernel32,
            _open_parent_handle,
        )

        api = _kernel32()
        handle = _open_parent_handle(api, path, for_flush=False)
        try:
            attributes, identity = _file_information(api, handle, path=path)
            if attributes & 0x400 or identity != (device, inode):
                raise UnsafePathError(f"legacy capture identity changed: {path}")
            _require_identity(path, path.lstat(), device=device, inode=inode)
            yield
        finally:
            _close_handle(api, handle)
    else:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            _require_identity(path, os.fstat(descriptor), device=device, inode=inode)
            _require_identity(path, path.lstat(), device=device, inode=inode)
            yield
        finally:
            os.close(descriptor)
