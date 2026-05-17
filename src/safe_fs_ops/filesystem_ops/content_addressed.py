from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISREG
from typing import BinaryIO

from safe_fs_ops.filesystem_ops._windows_directory_operations import make_directory_windows
from safe_fs_ops.filesystem_ops._windows_primitives import replace_file_windows
from safe_fs_ops.filesystem_ops.models import ContentRef
from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    UnsupportedFilesystemMutationError,
    _chmod_open_file,
    _cleanup_temp_file,
    _ensure_directory_descriptor_matches_path,
    _ensure_directory_descriptor_matches_path_after_mutation,
    _ensure_safe_child_for_replace,
    _fsync_directory_descriptor,
    _open_directory_for_mutation,
    _replace,
)
from safe_fs_ops.filesystem_ops.mutation_support import (
    _create_temp_file as _create_descriptor_temp_file,
)
from safe_fs_ops.filesystem_ops.mutations import make_directory
from safe_fs_ops.filesystem_ops.no_replace_rename import rename_path_no_replace
from safe_fs_ops.filesystem_ops.paths import (
    UnsafePathError,
    absolute_without_resolving,
    ensure_safe_parent_chain,
    inspect_path,
)

_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ContentAddressedPutResult:
    ref: ContentRef
    created: bool


class ContentAddressedStore:
    """Filesystem-backed SHA-256 content-addressed object store."""

    def __init__(self, root: Path | str) -> None:
        self.root = absolute_without_resolving(root)

    def put_bytes(self, content: bytes) -> ContentRef:
        return self.put_bytes_with_status(content).ref

    def put_bytes_with_status(self, content: bytes) -> ContentAddressedPutResult:
        digest = hashlib.sha256(content).hexdigest()
        return self._write_verified_object(digest=digest, content=content)

    def put_file(self, path: Path | str) -> ContentRef:
        return self.put_file_with_status(path).ref

    def put_file_with_status(self, path: Path | str) -> ContentAddressedPutResult:
        source = absolute_without_resolving(path)
        digest, size, source_stat = self._digest_regular_file(source)
        object_path = self._object_path(digest)
        _require_safe_store_root_if_exists(self.root, operation="content store")
        if _path_exists_no_follow(object_path):
            ref = ContentRef(digest=digest, size=size, path=object_path)
            if not self.verify(ref):
                raise ValueError(f"existing content object failed verification: {object_path}")
            return ContentAddressedPutResult(ref=ref, created=False)

        self._ensure_safe_object_parent(object_path)
        temp_handle: BinaryIO | None = None
        temp_path: Path | None = None
        temp_name: str | None = None
        parent_descriptor: int | None = None
        created = False
        try:
            temp_handle, temp_path, temp_name, parent_descriptor = self._create_temp_file(
                object_path.parent,
                object_path.name,
                operation="content store",
                needs_replace=False,
            )
            self._copy_regular_file_to_temp(source, temp_handle, expected_stat=source_stat)
            temp_handle.seek(0)
            temp_digest, temp_size = self._hash_stream(temp_handle)
            if temp_digest != digest or temp_size != size:
                raise ValueError(f"source file changed while storing content: {source}")
            if parent_descriptor is not None:
                _require_temp_name_matches_open_file(
                    temp_handle.fileno(),
                    temp_name,
                    parent_descriptor=parent_descriptor,
                    path=temp_path,
                    operation="content store",
                )
            else:
                temp_handle.close()
                temp_handle = None
            created = self._install_new_object(
                temp_path,
                object_path,
                digest=digest,
                size=size,
                temp_name=temp_name,
                parent_descriptor=parent_descriptor,
            )
            if temp_handle is not None:
                temp_handle.close()
                temp_handle = None
            temp_name = None
            temp_path = None
        finally:
            if temp_handle is not None:
                with contextlib.suppress(OSError):
                    temp_handle.close()
            if temp_name is not None and parent_descriptor is not None:
                _cleanup_temp_file(temp_name, parent_descriptor=parent_descriptor)
            elif temp_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()
            if parent_descriptor is not None:
                os.close(parent_descriptor)
        return ContentAddressedPutResult(
            ref=ContentRef(digest=digest, size=size, path=object_path),
            created=created,
        )

    def verify(self, ref: ContentRef) -> bool:
        self._require_sha256(ref)
        object_path = self._object_path(ref.digest)
        if ref.path != object_path:
            return False
        self._require_safe_object_parent(object_path)
        try:
            digest, size = self._hash_regular_file(object_path)
        except (OSError, ValueError):
            return False
        return digest == ref.digest and size == ref.size

    def copy_to(
        self,
        ref: ContentRef,
        destination: Path | str,
        *,
        no_replace: bool = True,
        permissions: int | None = None,
        mtime_ns: int | None = None,
        mtime: float | None = None,
    ) -> None:
        self._require_sha256(ref)
        if mtime_ns is not None and mtime is not None:
            raise ValueError("mtime_ns and mtime cannot both be provided")
        if not self.verify(ref):
            raise ValueError(f"content object failed verification: {ref.path}")

        target = absolute_without_resolving(destination)
        ensure_safe_parent_chain(target, operation="content copy")
        if no_replace and _path_exists_no_follow(target):
            raise FileExistsError(f"destination already exists: {target}")
        _safe_make_directory(target.parent, parents=True, exist_ok=True)
        parent_stat = _snapshot_destination_parent(target)
        temp_handle: BinaryIO | None = None
        temp_path: Path | None = None
        temp_name: str | None = None
        parent_descriptor: int | None = None
        object_path = self._object_path(ref.digest)
        self._require_safe_object_parent(object_path)
        try:
            temp_handle, temp_path, temp_name, parent_descriptor = self._create_temp_file(
                target.parent,
                target.name,
                operation="content copy",
                needs_replace=not no_replace,
            )
            with self._open_verified_object(object_path) as object_handle:
                digest, size = self._copy_hashing(object_handle, temp_handle)
                temp_handle.flush()
                os.fsync(temp_handle.fileno())
            if digest != ref.digest or size != ref.size:
                raise ValueError(f"content object changed while copying: {object_path}")
            if permissions is not None:
                if parent_descriptor is None:
                    temp_path.chmod(permissions)
                else:
                    _chmod_open_file(temp_handle.fileno(), permissions, operation="content copy")
            if mtime_ns is not None:
                if parent_descriptor is None:
                    os.utime(temp_path, ns=(mtime_ns, mtime_ns))
                else:
                    os.utime(temp_handle.fileno(), ns=(mtime_ns, mtime_ns))
            elif mtime is not None:
                if parent_descriptor is None:
                    os.utime(temp_path, times=(mtime, mtime))
                else:
                    os.utime(temp_handle.fileno(), times=(mtime, mtime))
            if parent_descriptor is not None:
                _require_temp_name_matches_open_file(
                    temp_handle.fileno(),
                    temp_name,
                    parent_descriptor=parent_descriptor,
                    path=temp_path,
                    operation="content copy",
                )
            else:
                temp_handle.close()
                temp_handle = None

            _ensure_destination_parent_matches(target, parent_stat)
            if parent_descriptor is not None:
                _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="content copy")
            if no_replace:
                if parent_descriptor is None:
                    rename_path_no_replace(temp_path, target, operation="content copy")
                else:
                    _link_temp_file_no_replace(
                        temp_name,
                        target.name,
                        parent_descriptor=parent_descriptor,
                        operation="content copy",
                    )
                    _cleanup_temp_file(temp_name, parent_descriptor=parent_descriptor)
                    _ensure_directory_descriptor_matches_path_after_mutation(
                        parent_descriptor,
                        target.parent,
                        operation="content copy",
                    )
                    _fsync_directory_descriptor(parent_descriptor)
            else:
                _replace_file(temp_path, target, temp_name=temp_name, parent_descriptor=parent_descriptor)
                if parent_descriptor is not None:
                    _fsync_directory_descriptor(parent_descriptor)
            if temp_handle is not None:
                temp_handle.close()
                temp_handle = None
            temp_name = None
            temp_path = None
            _ensure_destination_parent_matches(target, parent_stat)
            copied_digest, copied_size = self._hash_regular_file(target)
            if copied_digest != ref.digest or copied_size != ref.size:
                raise ValueError(f"destination content changed while copying: {target}")
            copied_stat = _require_regular_file_stat(target, operation="content copy")
            if mtime_ns is not None and copied_stat.st_mtime_ns != mtime_ns:
                raise ValueError(f"destination mtime changed while copying: {target}")
        finally:
            if temp_handle is not None:
                with contextlib.suppress(OSError):
                    temp_handle.close()
            if temp_name is not None and parent_descriptor is not None:
                _cleanup_temp_file(temp_name, parent_descriptor=parent_descriptor)
            elif temp_path is not None:
                with contextlib.suppress(FileNotFoundError, OSError):
                    temp_path.unlink()
            if parent_descriptor is not None:
                os.close(parent_descriptor)

    def _object_path(self, digest: str) -> Path:
        return self.root / "sha256" / digest[:2] / digest

    def _write_verified_object(self, *, digest: str, content: bytes) -> ContentAddressedPutResult:
        object_path = self._object_path(digest)
        _require_safe_store_root_if_exists(self.root, operation="content store")
        if _path_exists_no_follow(object_path):
            ref = ContentRef(digest=digest, size=len(content), path=object_path)
            if not self.verify(ref):
                raise ValueError(f"existing content object failed verification: {object_path}")
            return ContentAddressedPutResult(ref=ref, created=False)
        self._ensure_safe_object_parent(object_path)
        temp_handle: BinaryIO | None = None
        temp_path: Path | None = None
        temp_name: str | None = None
        parent_descriptor: int | None = None
        created = False
        try:
            temp_handle, temp_path, temp_name, parent_descriptor = self._create_temp_file(
                object_path.parent,
                object_path.name,
                operation="content store",
                needs_replace=False,
            )
            temp_handle.write(content)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
            temp_handle.seek(0)
            temp_digest, temp_size = self._hash_stream(temp_handle)
            if temp_digest != digest or temp_size != len(content):
                raise ValueError("temporary content object failed verification")
            if parent_descriptor is not None:
                _require_temp_name_matches_open_file(
                    temp_handle.fileno(),
                    temp_name,
                    parent_descriptor=parent_descriptor,
                    path=temp_path,
                    operation="content store",
                )
            else:
                temp_handle.close()
                temp_handle = None
            created = self._install_new_object(
                temp_path,
                object_path,
                digest=digest,
                size=len(content),
                temp_name=temp_name,
                parent_descriptor=parent_descriptor,
            )
            if temp_handle is not None:
                temp_handle.close()
                temp_handle = None
            temp_name = None
            temp_path = None
        finally:
            if temp_handle is not None:
                with contextlib.suppress(OSError):
                    temp_handle.close()
            if temp_name is not None and parent_descriptor is not None:
                _cleanup_temp_file(temp_name, parent_descriptor=parent_descriptor)
            elif temp_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    temp_path.unlink()
            if parent_descriptor is not None:
                os.close(parent_descriptor)
        return ContentAddressedPutResult(
            ref=ContentRef(digest=digest, size=len(content), path=object_path),
            created=created,
        )

    def _digest_regular_file(self, path: Path) -> tuple[str, int, os.stat_result]:
        stat_before = path.lstat()
        if not S_ISREG(stat_before.st_mode):
            raise ValueError(f"content store only accepts regular files: {path}")
        with path.open("rb") as handle:
            stat_open = os.fstat(handle.fileno())
            if not S_ISREG(stat_open.st_mode) or not _same_file_metadata(stat_before, stat_open):
                raise ValueError(f"content store only accepts regular files: {path}")
            digest, size = self._hash_stream(handle)
        stat_after = path.lstat()
        if not _same_file_metadata(stat_before, stat_after):
            raise ValueError(f"source file changed while storing content: {path}")
        return digest, size, stat_before

    def _copy_regular_file_to_temp(
        self,
        source: Path,
        temp_handle: BinaryIO,
        *,
        expected_stat: os.stat_result,
    ) -> None:
        if not _same_file_metadata(source.lstat(), expected_stat):
            raise ValueError(f"source file changed while storing content: {source}")
        with source.open("rb") as source_handle:
            stat_open = os.fstat(source_handle.fileno())
            if not S_ISREG(stat_open.st_mode) or not _same_file_metadata(stat_open, expected_stat):
                raise ValueError(f"source file changed while storing content: {source}")
            self._copy_stream(source_handle, temp_handle)
            temp_handle.flush()
            os.fsync(temp_handle.fileno())
        if not _same_file_metadata(source.lstat(), expected_stat):
            raise ValueError(f"source file changed while storing content: {source}")

    def _hash_regular_file(self, path: Path) -> tuple[str, int]:
        stat_result = path.lstat()
        if not S_ISREG(stat_result.st_mode):
            raise ValueError(f"content object is not a regular file: {path}")
        with path.open("rb") as handle:
            stat_open = os.fstat(handle.fileno())
            if not S_ISREG(stat_open.st_mode) or not _same_file_identity(stat_result, stat_open):
                raise ValueError(f"content object is not a regular file: {path}")
            return self._hash_stream(handle)

    def _open_verified_object(self, path: Path) -> BinaryIO:
        stat_result = path.lstat()
        if not S_ISREG(stat_result.st_mode):
            raise ValueError(f"content object is not a regular file: {path}")
        handle = path.open("rb")
        try:
            stat_open = os.fstat(handle.fileno())
            if not S_ISREG(stat_open.st_mode) or not _same_file_identity(stat_result, stat_open):
                raise ValueError(f"content object changed before open completed: {path}")
            return handle
        except BaseException:
            handle.close()
            raise

    def _install_new_object(
        self,
        temp_path: Path,
        object_path: Path,
        *,
        digest: str,
        size: int,
        temp_name: str,
        parent_descriptor: int | None,
    ) -> bool:
        self._require_safe_object_parent(object_path)
        try:
            if parent_descriptor is None:
                rename_path_no_replace(temp_path, object_path, operation="content store")
            else:
                _ensure_directory_descriptor_matches_path(
                    parent_descriptor,
                    object_path.parent,
                    operation="content store",
                )
                _link_temp_file_no_replace(
                    temp_name,
                    object_path.name,
                    parent_descriptor=parent_descriptor,
                    operation="content store",
                )
        except FileExistsError:
            ref = ContentRef(digest=digest, size=size, path=object_path)
            if not self.verify(ref):
                raise ValueError(f"existing content object failed verification: {object_path}") from None
            return False
        if parent_descriptor is not None:
            _cleanup_temp_file(temp_name, parent_descriptor=parent_descriptor)
            _ensure_directory_descriptor_matches_path_after_mutation(
                parent_descriptor,
                object_path.parent,
                operation="content store",
            )
            _fsync_directory_descriptor(parent_descriptor)
        ref = ContentRef(digest=digest, size=size, path=object_path)
        if not self.verify(ref):
            raise ValueError(f"installed content object failed verification: {object_path}")
        return True

    def _hash_stream(self, handle: BinaryIO) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size

    def _copy_hashing(self, source: BinaryIO, destination: BinaryIO) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        while chunk := source.read(_CHUNK_SIZE):
            destination.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        return digest.hexdigest(), size

    def _copy_stream(self, source: BinaryIO, destination: BinaryIO) -> None:
        while chunk := source.read(_CHUNK_SIZE):
            destination.write(chunk)

    def _ensure_safe_object_parent(self, object_path: Path) -> None:
        _ensure_safe_store_root(self.root, operation="content store")
        _ensure_safe_directory_chain(self.root, object_path.parent, operation="content store")
        self._require_safe_object_parent(object_path)

    def _require_safe_object_parent(self, object_path: Path) -> None:
        _require_safe_directory_chain(self.root, object_path.parent, operation="content store")

    def _create_temp_file(
        self,
        parent: Path,
        target_name: str,
        *,
        operation: str,
        needs_replace: bool,
    ) -> tuple[BinaryIO, Path, str, int | None]:
        if os.name != "nt":
            parent_descriptor = _open_directory_for_mutation(
                parent,
                operation=operation,
                needs_replace=needs_replace,
                platform_support=_DEFAULT_PLATFORM_SUPPORT,
            )
            try:
                _ensure_directory_descriptor_matches_path(parent_descriptor, parent, operation=operation)
                temp_descriptor, temp_name = _create_descriptor_temp_file(parent_descriptor, target_name)
                return os.fdopen(temp_descriptor, "w+b"), parent / temp_name, temp_name, parent_descriptor
            except BaseException:
                os.close(parent_descriptor)
                raise
        handle, path = _create_temp_file_windows(parent)
        return handle, path, path.name, None

    def _require_sha256(self, ref: ContentRef) -> None:
        if ref.algo != "sha256":
            raise ValueError(f"unsupported content address algorithm: {ref.algo}")


def _create_temp_file_windows(parent: Path) -> tuple[BinaryIO, Path]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        path = parent / f".safe-fs-ops-{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        try:
            return os.fdopen(descriptor, "w+b"), path
        except BaseException:
            os.close(descriptor)
            with contextlib.suppress(FileNotFoundError, OSError):
                path.unlink()
            raise
    raise FileExistsError(f"could not allocate a unique temp file in {parent}")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino and left.st_size == right.st_size


def _same_file_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_file_identity(left, right) and left.st_mtime_ns == right.st_mtime_ns


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_safe_store_parent_chain(path: Path, *, operation: str) -> None:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.relative_to(anchor).parts[:-1]:
        current = current / part
        safety = inspect_path(current)
        if safety.is_symlink or safety.is_windows_reparse_point:
            raise UnsafePathError(f"{operation} refused because store path redirects elsewhere: {current}")
        if safety.exists and not safety.is_dir:
            raise UnsafePathError(f"{operation} refused because store parent is not a directory: {current}")


def _require_safe_store_root_if_exists(path: Path, *, operation: str) -> None:
    _require_safe_store_parent_chain(path, operation=operation)
    safety = inspect_path(path)
    if not safety.exists:
        return
    if safety.is_symlink or safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused because store path redirects elsewhere: {path}")
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because store path is not a directory: {path}")


def _ensure_safe_store_root(path: Path, *, operation: str) -> None:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.relative_to(anchor).parts:
        current = current / part
        safety = inspect_path(current)
        if safety.exists:
            _require_safe_existing_directory(current, operation=operation)
            continue
        _safe_make_directory(current, exist_ok=True)
        _require_safe_existing_directory(current, operation=operation)


def _ensure_safe_directory_chain(root: Path, directory: Path, *, operation: str) -> None:
    absolute_root = absolute_without_resolving(root)
    absolute_directory = absolute_without_resolving(directory)
    try:
        parts = absolute_directory.relative_to(absolute_root).parts
    except ValueError as exc:
        raise UnsafePathError(
            f"{operation} refused because object path escapes store root: {absolute_directory}"
        ) from exc
    _require_safe_existing_directory(absolute_root, operation=operation)
    current = absolute_root
    for part in parts:
        current = current / part
        safety = inspect_path(current)
        if safety.exists:
            _require_safe_existing_directory(current, operation=operation)
            continue
        _safe_make_directory(current, exist_ok=True)
        _require_safe_existing_directory(current, operation=operation)


def _require_safe_directory_chain(root: Path, directory: Path, *, operation: str) -> None:
    absolute_root = absolute_without_resolving(root)
    absolute_directory = absolute_without_resolving(directory)
    try:
        parts = absolute_directory.relative_to(absolute_root).parts
    except ValueError as exc:
        raise UnsafePathError(
            f"{operation} refused because object path escapes store root: {absolute_directory}"
        ) from exc
    _require_safe_existing_directory(absolute_root, operation=operation)
    current = absolute_root
    for part in parts:
        current = current / part
        _require_safe_existing_directory(current, operation=operation)


def _require_safe_existing_directory(path: Path, *, operation: str) -> None:
    safety = inspect_path(path)
    if not safety.exists:
        raise UnsafePathError(f"{operation} refused because store directory is missing: {path}")
    if safety.is_symlink or safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused because store path redirects elsewhere: {path}")
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because store path is not a directory: {path}")


def _snapshot_destination_parent(path: Path) -> os.stat_result:
    ensure_safe_parent_chain(path, operation="content copy")
    safety = inspect_path(path.parent)
    if not safety.exists or safety.is_symlink or safety.is_windows_reparse_point or not safety.is_dir:
        raise UnsafePathError(f"content copy refused because parent path is not a safe directory: {path.parent}")
    stat_result = path.parent.lstat()
    if not S_ISDIR(stat_result.st_mode):
        raise UnsafePathError(f"content copy refused because parent path is not a directory: {path.parent}")
    return stat_result


def _ensure_destination_parent_matches(path: Path, expected_stat: os.stat_result) -> None:
    ensure_safe_parent_chain(path, operation="content copy")
    try:
        actual_stat = path.parent.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"content copy refused because parent path disappeared: {path.parent}") from exc
    if not S_ISDIR(actual_stat.st_mode) or not _same_path_identity(actual_stat, expected_stat):
        raise UnsafePathError(f"content copy refused because parent path changed: {path.parent}")


def _same_path_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _replace_file(
    source: Path,
    destination: Path,
    *,
    temp_name: str,
    parent_descriptor: int | None,
) -> None:
    if parent_descriptor is None:
        replace_file_windows(source, destination, write_through=True)
        return

    _ensure_directory_descriptor_matches_path(parent_descriptor, destination.parent, operation="content copy")
    _ensure_safe_child_for_replace(parent_descriptor, destination.name, operation="content copy")
    _replace(
        temp_name,
        destination.name,
        parent_descriptor=parent_descriptor,
        platform_support=_DEFAULT_PLATFORM_SUPPORT,
    )
    _ensure_directory_descriptor_matches_path_after_mutation(
        parent_descriptor,
        destination.parent,
        operation="content copy",
    )


def _link_temp_file_no_replace(source_name: str, target_name: str, *, parent_descriptor: int, operation: str) -> None:
    if os.link not in os.supports_dir_fd:
        raise UnsupportedFilesystemMutationError(
            f"{operation} refused because this platform lacks descriptor-relative hard links"
        )
    try:
        os.link(source_name, target_name, src_dir_fd=parent_descriptor, dst_dir_fd=parent_descriptor)
    except FileExistsError:
        raise
    except OSError as exc:
        raise UnsafePathError(f"{operation} refused because temporary file could not be linked safely") from exc


def _require_regular_file_stat(path: Path, *, operation: str) -> os.stat_result:
    try:
        stat_result = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"{operation} refused because destination disappeared: {path}") from exc
    if not S_ISREG(stat_result.st_mode):
        raise UnsafePathError(f"{operation} refused because destination is not a regular file: {path}")
    return stat_result


def _require_temp_name_matches_open_file(
    file_descriptor: int,
    name: str,
    *,
    parent_descriptor: int,
    path: Path,
    operation: str,
) -> None:
    descriptor_stat = os.fstat(file_descriptor)
    try:
        name_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise UnsafePathError(f"{operation} refused because temporary file disappeared: {path}") from exc
    if not S_ISREG(name_stat.st_mode):
        raise UnsafePathError(f"{operation} refused because temporary path is not a regular file: {path}")
    if not _same_path_identity(descriptor_stat, name_stat):
        raise UnsafePathError(f"{operation} refused because temporary file identity changed: {path}")


def _safe_make_directory(path: Path, *, parents: bool = False, exist_ok: bool = False) -> None:
    if os.name == "nt":
        make_directory_windows(path, parents=parents, exist_ok=exist_ok)
        return
    make_directory(path, parents=parents, exist_ok=exist_ok)
