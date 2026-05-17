from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from safe_fs_ops.filesystem_ops.models import ResourceSnapshot
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path
from safe_fs_ops.filesystem_ops.snapshots import snapshot_resource

HashPolicy = Literal["metadata-only", "small-files"]
_DEFAULT_SMALL_FILE_MAX_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SnapshotBundleEntry:
    path: Path
    snapshot: ResourceSnapshot
    requested: bool

    def __post_init__(self) -> None:
        if self.snapshot.path != self.path:
            raise ValueError("snapshot.path must match entry path")


@dataclass(frozen=True, slots=True)
class SnapshotBundle:
    requested_paths: tuple[Path, ...]
    entries: tuple[SnapshotBundleEntry, ...]
    include_children: bool
    hash_policy: HashPolicy
    small_file_max_bytes: int

    def __post_init__(self) -> None:
        if self.hash_policy not in {"metadata-only", "small-files"}:
            raise ValueError(f"unsupported hash_policy: {self.hash_policy}")
        if self.small_file_max_bytes < 0:
            raise ValueError("small_file_max_bytes must be non-negative")

    def by_path(self) -> dict[Path, SnapshotBundleEntry]:
        return {entry.path: entry for entry in self.entries}


def snapshot_bundle(
    paths: Iterable[Path | str] | Path | str,
    *,
    include_children: bool = False,
    hash_policy: HashPolicy = "metadata-only",
    small_file_max_bytes: int = _DEFAULT_SMALL_FILE_MAX_BYTES,
) -> SnapshotBundle:
    if hash_policy not in {"metadata-only", "small-files"}:
        raise ValueError(f"unsupported hash_policy: {hash_policy}")
    if small_file_max_bytes < 0:
        raise ValueError("small_file_max_bytes must be non-negative")

    requested_paths = _normalize_paths(paths)
    requested_set = set(requested_paths)
    entries: list[SnapshotBundleEntry] = []
    entries_by_path: dict[Path, SnapshotBundleEntry] = {}

    for requested_path in requested_paths:
        entry = _append_snapshot_entry(
            entries,
            entries_by_path=entries_by_path,
            path=requested_path,
            requested=requested_path in requested_set,
            hash_policy=hash_policy,
            small_file_max_bytes=small_file_max_bytes,
        )
        if not include_children:
            continue
        if entry.snapshot.file_type != "directory":
            continue
        for child in _iter_child_paths(requested_path, expected_root=entry.snapshot):
            _append_snapshot_entry(
                entries,
                entries_by_path=entries_by_path,
                path=child,
                requested=child in requested_set,
                hash_policy=hash_policy,
                small_file_max_bytes=small_file_max_bytes,
            )

    return SnapshotBundle(
        requested_paths=requested_paths,
        entries=tuple(entries),
        include_children=include_children,
        hash_policy=hash_policy,
        small_file_max_bytes=small_file_max_bytes,
    )


def _normalize_paths(paths: Iterable[Path | str] | Path | str) -> tuple[Path, ...]:
    if isinstance(paths, (Path, str)):
        return (absolute_without_resolving(paths),)
    return tuple(absolute_without_resolving(path) for path in paths)


def _append_snapshot_entry(
    entries: list[SnapshotBundleEntry],
    *,
    entries_by_path: dict[Path, SnapshotBundleEntry],
    path: Path,
    requested: bool,
    hash_policy: HashPolicy,
    small_file_max_bytes: int,
) -> SnapshotBundleEntry:
    existing = entries_by_path.get(path)
    if existing is not None:
        return existing
    snapshot = _snapshot_for_policy(
        path,
        hash_policy=hash_policy,
        small_file_max_bytes=small_file_max_bytes,
    )
    entry = SnapshotBundleEntry(path=path, snapshot=snapshot, requested=requested)
    entries_by_path[path] = entry
    entries.append(entry)
    return entry


def _snapshot_for_policy(
    path: Path,
    *,
    hash_policy: HashPolicy,
    small_file_max_bytes: int,
) -> ResourceSnapshot:
    snapshot = _platform_snapshot_resource(path)
    if snapshot.file_type != "file":
        return snapshot
    if hash_policy == "metadata-only":
        return replace(snapshot, content_hash=None)
    if snapshot.size is None or snapshot.size > small_file_max_bytes:
        return replace(snapshot, content_hash=None)
    if snapshot.content_hash is not None:
        return snapshot
    return replace(snapshot, content_hash=_sha256(path))


def _platform_snapshot_resource(path: Path) -> ResourceSnapshot:
    if os.name == "nt":
        from safe_fs_ops.filesystem_ops._windows_snapshots import snapshot_resource_windows

        return snapshot_resource_windows(path)
    return snapshot_resource(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_child_paths(root: Path, *, expected_root: ResourceSnapshot) -> tuple[Path, ...]:
    children: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        if current == root:
            _ensure_directory_matches_snapshot(root, expected_root)
        child_safety = [
            (child, inspect_path(child)) for child in sorted(current.iterdir(), key=lambda child: str(child))
        ]
        for child, safety in child_safety:
            if not safety.is_windows_reparse_point:
                children.append(child)
        for child, safety in reversed(child_safety):
            if (
                safety.exists
                and safety.file_type == "directory"
                and not safety.is_symlink
                and not safety.is_mount
                and not safety.is_windows_reparse_point
            ):
                stack.append(child)
        if current == root:
            _ensure_directory_matches_snapshot(root, expected_root)
    return tuple(children)


def _ensure_directory_matches_snapshot(path: Path, snapshot: ResourceSnapshot) -> None:
    try:
        stat_result = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"snapshot bundle refused because root directory disappeared: {path}") from exc
    if (
        snapshot.file_type != "directory"
        or snapshot.device != stat_result.st_dev
        or snapshot.inode != stat_result.st_ino
        or snapshot.size != stat_result.st_size
        or snapshot.mtime_ns != stat_result.st_mtime_ns
    ):
        raise UnsafePathError(f"snapshot bundle refused because root directory changed during traversal: {path}")


__all__ = [
    "HashPolicy",
    "SnapshotBundle",
    "SnapshotBundleEntry",
    "snapshot_bundle",
]
