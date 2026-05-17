from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FileType = Literal["missing", "file", "directory", "symlink", "other"]


@dataclass(frozen=True, slots=True)
class PathSafety:
    path: Path
    exists: bool
    file_type: FileType
    is_mount: bool
    is_windows_reparse_point: bool
    hardlink_count: int
    size: int | None

    @property
    def is_file(self) -> bool:
        return self.file_type == "file"

    @property
    def is_dir(self) -> bool:
        return self.file_type == "directory"

    @property
    def is_symlink(self) -> bool:
        return self.file_type == "symlink"


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    path: Path
    exists: bool
    file_type: FileType
    content_hash: str | None
    size: int | None
    mtime_ns: int | None
    symlink_target: str | None
    device: int | None = None
    inode: int | None = None
    ctime_ns: int | None = None


BackupFileType = Literal["missing", "file"]
TreeBackupEntryKind = Literal["file", "symlink"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentRef:
    algo: Literal["sha256"] = "sha256"
    digest: str
    size: int
    path: Path

    def __post_init__(self) -> None:
        if self.algo != "sha256":
            raise ValueError(f"unsupported content address algorithm: {self.algo}")
        if len(self.digest) != 64 or any(character not in string.hexdigits for character in self.digest):
            raise ValueError("content digest must be a 64-character SHA-256 hex digest")
        if self.digest.lower() != self.digest:
            raise ValueError("content digest must be lowercase")
        if self.size < 0:
            raise ValueError("content size must be non-negative")


@dataclass(frozen=True, slots=True)
class FileBackup:
    path: Path
    existed: bool
    file_type: BackupFileType
    content_bytes: bytes | None
    content_path: Path | None
    content_hash: str | None
    size: int | None
    permissions: int | None
    snapshot: ResourceSnapshot
    content_address: ContentRef | None = None

    def __post_init__(self) -> None:
        if self.file_type not in {"missing", "file"}:
            raise ValueError(f"unsupported backup file_type: {self.file_type}")
        if self.existed != (self.file_type == "file"):
            raise ValueError("existed must agree with file_type")
        if self.snapshot.path != self.path:
            raise ValueError("snapshot.path must match backup path")
        if self.snapshot.file_type != self.file_type:
            raise ValueError("snapshot.file_type must match backup file_type")
        if self.snapshot.exists != self.existed:
            raise ValueError("snapshot.exists must match backup existed flag")
        if self.file_type == "missing":
            if any(
                value is not None
                for value in (
                    self.content_bytes,
                    self.content_path,
                    self.content_address,
                    self.content_hash,
                    self.size,
                    self.permissions,
                )
            ):
                raise ValueError("missing backups cannot store file content or file metadata")
            return
        if self.permissions is None:
            raise ValueError("file backups must store permissions")
        content_source_count = sum(
            value is not None for value in (self.content_bytes, self.content_path, self.content_address)
        )
        if content_source_count != 1:
            raise ValueError("file backups must store exactly one content source")
        if self.content_hash is None or self.size is None:
            raise ValueError("file backups must store content_hash and size")
        if self.content_address is not None:
            if self.content_address.algo != "sha256":
                raise ValueError("file backup content address must use sha256")
            if self.content_address.digest != self.content_hash:
                raise ValueError("file backup content address digest must match content_hash")
            if self.content_address.size != self.size:
                raise ValueError("file backup content address size must match backup size")


@dataclass(frozen=True, slots=True, kw_only=True)
class TreeBackupEntry:
    relative_path: Path
    kind: TreeBackupEntryKind
    snapshot: ResourceSnapshot
    content_address: ContentRef | None = None
    content_hash: str | None = None
    size: int | None = None
    permissions: int | None = None
    symlink_target: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"file", "symlink"}:
            raise ValueError(f"unsupported tree backup entry kind: {self.kind}")
        if self.relative_path.is_absolute() or self.relative_path.drive or self.relative_path.root:
            raise ValueError("tree backup entry paths must be relative")
        if self.relative_path == Path(".") or any(part == ".." for part in self.relative_path.parts):
            raise ValueError("tree backup entry paths must name an explicit child without '..'")

        if self.kind == "file":
            if self.snapshot.file_type != "file":
                raise ValueError("file tree backup entries require a file snapshot")
            if self.content_address is None:
                raise ValueError("file tree backup entries require a content address")
            if self.content_hash is None or self.size is None:
                raise ValueError("file tree backup entries require content_hash and size")
            if self.permissions is None:
                raise ValueError("file tree backup entries require permissions")
            if self.symlink_target is not None:
                raise ValueError("file tree backup entries cannot store a symlink target")
            if self.snapshot.content_hash != self.content_hash:
                raise ValueError("file tree backup content_hash must match snapshot")
            if self.snapshot.size != self.size:
                raise ValueError("file tree backup size must match snapshot")
            if self.content_address.digest != self.content_hash:
                raise ValueError("file tree backup content address digest must match content_hash")
            if self.content_address.size != self.size:
                raise ValueError("file tree backup content address size must match entry size")
            return

        if self.snapshot.file_type != "symlink":
            raise ValueError("symlink tree backup entries require a symlink snapshot")
        if self.symlink_target is None:
            raise ValueError("symlink tree backup entries require a symlink target")
        if self.snapshot.symlink_target != self.symlink_target:
            raise ValueError("symlink tree backup target must match snapshot")
        if any(
            value is not None
            for value in (
                self.content_address,
                self.content_hash,
                self.size,
                self.permissions,
            )
        ):
            raise ValueError("symlink tree backup entries cannot store file content metadata")


@dataclass(frozen=True, slots=True)
class TreeBackup:
    root: Path
    entries: tuple[TreeBackupEntry, ...]

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("tree backup root must be absolute")
        seen: set[Path] = set()
        for entry in self.entries:
            if entry.relative_path in seen:
                raise ValueError(f"duplicate tree backup entry path: {entry.relative_path}")
            for existing in seen:
                if _is_path_prefix(existing, entry.relative_path) or _is_path_prefix(entry.relative_path, existing):
                    raise ValueError(
                        f"tree backup entry path {entry.relative_path} conflicts with prefix path {existing}"
                    )
            seen.add(entry.relative_path)

    def by_relative_path(self) -> dict[Path, TreeBackupEntry]:
        return {entry.relative_path: entry for entry in self.entries}


def _is_path_prefix(prefix: Path, path: Path) -> bool:
    prefix_parts = prefix.parts
    path_parts = path.parts
    return len(prefix_parts) < len(path_parts) and path_parts[: len(prefix_parts)] == prefix_parts
