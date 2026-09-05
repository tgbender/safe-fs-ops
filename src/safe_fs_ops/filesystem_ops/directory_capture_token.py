"""Persist a capture token in directory metadata, independent of inode reuse."""

from __future__ import annotations

import errno
import os
import re
import sys
from pathlib import Path
from stat import S_ISDIR
from uuid import uuid4

from safe_fs_ops.filesystem_ops.paths import UnsafePathError

_ATTRIBUTE = "user.safe_fs_ops.capture"
_STREAM = ":safe_fs_ops.capture"


def directory_capture_token(
    path: Path,
    *,
    device: int,
    inode: int,
    create: bool = False,
    _token: str | None = None,
) -> str | None:
    """Read a tag; initialize before capture or during explicit legacy adoption.

    POSIX uses an extended attribute; Windows uses a named data stream. Neither
    creates a visible child. Existing malformed tags and unsupported metadata
    backends fail closed. Automatic recovery must never initialize a tag.
    """
    if _token is not None and (not create or re.fullmatch(r"[0-9a-f]{32}", _token) is None):
        raise ValueError("an explicit token must be 32 lowercase hex characters and requires create=True")
    _require_identity(path, path.lstat(), device=device, inode=inode)
    try:
        if os.name == "nt":
            value = _windows_token(path, create=create, token=_token)
        else:
            value = _posix_token(path, device=device, inode=inode, create=create, token=_token)
    except OSError:
        return None
    _require_identity(path, path.lstat(), device=device, inode=inode)
    if value is None or re.fullmatch(rb"[0-9a-f]{32}", value) is None:
        return None
    return value.decode("ascii")


def _require_identity(path: Path, observed: os.stat_result, *, device: int, inode: int) -> None:
    if not S_ISDIR(observed.st_mode) or (observed.st_dev, observed.st_ino) != (device, inode):
        raise UnsafePathError(f"directory identity changed while reading capture token: {path}")


def _windows_token(path: Path, *, create: bool, token: str | None) -> bytes | None:
    stream = Path(str(path) + _STREAM)
    if create:
        try:
            with stream.open("xb") as handle:
                handle.write((token or uuid4().hex).encode("ascii"))
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            pass
    with stream.open("rb") as handle:
        return handle.read(33)


def _posix_token(path: Path, *, device: int, inode: int, create: bool, token: str | None) -> bytes | None:
    getxattr = getattr(os, "getxattr", None)
    setxattr = getattr(os, "setxattr", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nofollow_flag = getattr(os, "O_NOFOLLOW", None)
    create_flag = getattr(os, "XATTR_CREATE", None)
    if sys.platform == "darwin":
        from safe_fs_ops.filesystem_ops.darwin_xattrs import DarwinXattrs

        native = DarwinXattrs()
        getxattr, setxattr = native.get, native.set
        create_flag = 2  # Darwin XATTR_CREATE; Linux uses a different value.
    if getxattr is None or setxattr is None or directory_flag is None or nofollow_flag is None or create_flag is None:
        return None
    descriptor = os.open(path, os.O_RDONLY | directory_flag | nofollow_flag)
    try:
        _require_identity(path, os.fstat(descriptor), device=device, inode=inode)
        if create:
            try:
                setxattr(descriptor, _ATTRIBUTE, (token or uuid4().hex).encode("ascii"), create_flag)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
            # Persist the tag before its journal checkpoint can become durable.
            os.fsync(descriptor)
        value: bytes = getxattr(descriptor, _ATTRIBUTE)
        return value
    finally:
        os.close(descriptor)
