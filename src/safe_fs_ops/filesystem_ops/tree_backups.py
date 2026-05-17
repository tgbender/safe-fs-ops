from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from stat import S_IMODE, S_ISDIR, S_ISLNK
from typing import Literal, Protocol, cast

from safe_fs_ops.filesystem_ops._windows_directory_operations import make_directory_windows
from safe_fs_ops.filesystem_ops.backups import BackupContentMismatchError, RestoreConflictError
from safe_fs_ops.filesystem_ops.models import (
    ContentRef,
    FileType,
    PathSafety,
    ResourceSnapshot,
    TreeBackup,
    TreeBackupEntry,
    TreeBackupEntryKind,
)
from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    UnsupportedFilesystemMutationError,
    _ensure_directory_descriptor_matches_path,
    _ensure_directory_descriptor_matches_path_after_mutation,
    _open_directory_for_mutation,
)
from safe_fs_ops.filesystem_ops.mutations import make_directory
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path
from safe_fs_ops.filesystem_ops.snapshots import snapshot_resource

TreeRestoreConflictPolicy = Literal["no_replace", "replace"]


class _TreeBackupArtifactStore(Protocol):
    def put_file(self, path: Path | str) -> ContentRef: ...

    def verify(self, ref: ContentRef) -> bool: ...

    def copy_to(
        self,
        ref: ContentRef,
        destination: Path | str,
        *,
        no_replace: bool = True,
        permissions: int | None = None,
        mtime_ns: int | None = None,
        mtime: float | None = None,
    ) -> None: ...


def backup_tree(
    root: Path | str,
    relative_paths: Iterable[Path | str],
    *,
    artifact_store: _TreeBackupArtifactStore,
) -> TreeBackup:
    source_root = absolute_without_resolving(root)
    _require_existing_safe_directory(source_root, operation="tree backup root")

    entries: list[TreeBackupEntry] = []
    seen: set[Path] = set()
    for relative_path in relative_paths:
        normalized = _normalize_relative_path(relative_path)
        if normalized in seen:
            raise ValueError(f"duplicate tree backup path: {normalized}")
        seen.add(normalized)
        target = source_root / normalized
        _require_existing_safe_parent_chain(source_root, target, operation="tree backup")
        entries.append(_backup_entry(target, normalized, artifact_store=artifact_store))

    return TreeBackup(root=source_root, entries=tuple(entries))


def restore_tree_backup(
    backup: TreeBackup,
    destination_root: Path | str | None = None,
    *,
    artifact_store: _TreeBackupArtifactStore,
    conflict_policy: TreeRestoreConflictPolicy = "no_replace",
    check_authority: Callable[[], None] | None = None,
) -> tuple[ResourceSnapshot, ...]:
    if conflict_policy not in {"no_replace", "replace"}:
        raise ValueError(f"unsupported conflict_policy: {conflict_policy}")

    root = backup.root if destination_root is None else absolute_without_resolving(destination_root)
    _preflight_restore_tree_backup(
        backup,
        root,
        artifact_store=artifact_store,
        conflict_policy=conflict_policy,
    )
    if check_authority is not None:
        check_authority()
    _ensure_safe_directory(root, operation="tree restore root")

    restored: list[ResourceSnapshot] = []
    for entry in backup.entries:
        if check_authority is not None:
            check_authority()
        target = root / entry.relative_path
        _ensure_safe_directory(target.parent, operation="tree restore parent")
        restored.append(
            _restore_entry(
                entry,
                target,
                artifact_store=artifact_store,
                conflict_policy=conflict_policy,
            )
        )
        if check_authority is not None:
            check_authority()
    return tuple(restored)


def tree_backup_from_journal_payload(payload: Mapping[str, object]) -> TreeBackup:
    source = _tree_backup_payload_source(payload)
    root = _tree_backup_root_from_payload(source)
    entries_payload = _required_sequence(source, "entries", "tree backup payload")
    return TreeBackup(
        root=root,
        entries=tuple(_tree_backup_entry_from_payload(entry) for entry in entries_payload),
    )


def _preflight_restore_tree_backup(
    backup: TreeBackup,
    root: Path,
    *,
    artifact_store: _TreeBackupArtifactStore,
    conflict_policy: TreeRestoreConflictPolicy,
) -> None:
    _require_safe_directory_chain_if_exists(root, operation="tree restore root")
    seen: set[Path] = set()
    for entry in backup.entries:
        normalized = _normalize_relative_path(entry.relative_path)
        if normalized in seen:
            raise RestoreConflictError(f"tree restore refused duplicate destination path: {normalized}")
        seen.add(normalized)
        if normalized != entry.relative_path:
            raise UnsafePathError(f"tree backup paths must be normalized: {entry.relative_path}")
        _preflight_restore_entry(
            entry,
            root / normalized,
            artifact_store=artifact_store,
            conflict_policy=conflict_policy,
        )


def _tree_backup_payload_source(payload: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("tree_backup", "backup"):
        nested = _optional_mapping(payload.get(key), f"tree backup {key!r} payload")
        if nested is not None:
            return nested
    return payload


def _tree_backup_root_from_payload(payload: Mapping[str, object]) -> Path:
    for key in ("root", "root_path", "path", "source_path"):
        value = payload.get(key)
        if value is not None:
            return Path(str(value))
    raise ValueError("tree backup payload is missing a root path")


def _tree_backup_entry_from_payload(payload: object) -> TreeBackupEntry:
    entry_payload = _required_mapping(payload, "tree backup entry payload")
    snapshot_payload = _required_mapping(entry_payload.get("snapshot"), "tree backup entry snapshot")
    content_address = _optional_mapping(entry_payload.get("content_address"), "tree backup content address")
    return TreeBackupEntry(
        relative_path=Path(str(_required_value(entry_payload, "relative_path", "tree backup entry payload"))),
        kind=cast(TreeBackupEntryKind, str(_required_value(entry_payload, "kind", "tree backup entry payload"))),
        snapshot=_resource_snapshot_from_payload(snapshot_payload),
        content_address=None if content_address is None else _content_ref_from_payload(content_address),
        content_hash=_optional_str(entry_payload.get("content_hash")),
        size=_optional_int(entry_payload.get("size")),
        permissions=_optional_int(entry_payload.get("permissions")),
        symlink_target=_optional_str(entry_payload.get("symlink_target")),
    )


def _resource_snapshot_from_payload(payload: Mapping[str, object]) -> ResourceSnapshot:
    return ResourceSnapshot(
        path=Path(str(_required_value(payload, "path", "resource snapshot payload"))),
        exists=_required_bool(payload.get("exists"), "resource snapshot exists"),
        file_type=cast(FileType, str(_required_value(payload, "file_type", "resource snapshot payload"))),
        content_hash=_optional_str(payload.get("content_hash")),
        size=_optional_int(payload.get("size")),
        mtime_ns=_optional_int(payload.get("mtime_ns")),
        symlink_target=_optional_str(payload.get("symlink_target")),
        device=_optional_int(payload.get("device")),
        inode=_optional_int(payload.get("inode")),
        ctime_ns=_optional_int(payload.get("ctime_ns")),
    )


def _content_ref_from_payload(payload: Mapping[str, object]) -> ContentRef:
    return ContentRef(
        algo="sha256",
        digest=str(_required_value(payload, "digest", "content address payload")),
        size=_required_int(payload.get("size"), "content address size"),
        path=Path(str(_required_value(payload, "path", "content address payload"))),
    )


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    mapping = _optional_mapping(value, label)
    if mapping is None:
        raise ValueError(f"{label} must be a mapping")
    return mapping


def _optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _required_sequence(
    payload: Mapping[str, object],
    key: str,
    label: str,
) -> Sequence[object]:
    value = _required_value(payload, key, label)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} {key!r} must be a sequence")
    return value


def _required_value(payload: Mapping[str, object], key: str, label: str) -> object:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{label} is missing {key!r}")
    return value


def _optional_str(value: object | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return _required_int(value, "integer payload value")


def _required_int(value: object | None, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must not be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"{label} must be an int or string")


def _required_bool(value: object | None, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a bool")
    return value


def _preflight_restore_entry(
    entry: TreeBackupEntry,
    target: Path,
    *,
    artifact_store: _TreeBackupArtifactStore,
    conflict_policy: TreeRestoreConflictPolicy,
) -> None:
    _preflight_entry_artifact(entry, artifact_store=artifact_store)
    if not _existing_safe_parent_chain_allows_target(target, operation="tree restore"):
        return

    current = inspect_path(target)
    if not current.exists:
        return
    _raise_if_unsafe_leaf(current, operation="tree restore")
    if _matching_restored_snapshot(entry, target) is not None:
        return
    if conflict_policy == "no_replace":
        raise RestoreConflictError(f"tree restore refused because destination already exists: {target}")
    if not (current.is_file or current.is_symlink):
        raise UnsafePathError(f"tree restore refused to replace non-file path: {target}")
    if current.is_symlink and entry.kind == "file":
        raise UnsafePathError(f"tree restore refused to replace symlink with file: {target}")
    if entry.kind == "symlink":
        _require_symlink_restore_supported(target, replace=True)


def _preflight_entry_artifact(entry: TreeBackupEntry, *, artifact_store: _TreeBackupArtifactStore) -> None:
    if entry.kind == "file":
        if entry.content_address is None or entry.content_hash is None or entry.size is None:
            raise BackupContentMismatchError(f"tree backup content address mismatch for {entry.relative_path}")
        if entry.content_address.digest != entry.content_hash or entry.content_address.size != entry.size:
            raise BackupContentMismatchError(f"tree backup content address mismatch for {entry.relative_path}")
        if entry.snapshot.content_hash != entry.content_hash or entry.snapshot.size != entry.size:
            raise BackupContentMismatchError(f"tree backup snapshot mismatch for {entry.relative_path}")
        if not artifact_store.verify(entry.content_address):
            raise BackupContentMismatchError(f"tree backup content address mismatch for {entry.relative_path}")
        return

    if entry.symlink_target is None or entry.snapshot.symlink_target != entry.symlink_target:
        raise BackupContentMismatchError(f"tree backup symlink snapshot mismatch for {entry.relative_path}")
    _require_symlink_restore_supported(Path(entry.relative_path), replace=False)


def _backup_entry(
    target: Path,
    relative_path: Path,
    *,
    artifact_store: _TreeBackupArtifactStore,
) -> TreeBackupEntry:
    safety = inspect_path(target)
    if not safety.exists:
        raise FileNotFoundError(target)
    _raise_if_unsafe_leaf(safety, operation="tree backup")

    if safety.is_file:
        snapshot = _snapshot_regular_file(target)
        permissions = _current_permissions(target)
        if permissions is None:
            raise UnsafePathError(f"tree backup refused because file permissions could not be captured: {target}")
        content_ref = artifact_store.put_file(target)
        if content_ref.digest != snapshot.content_hash or content_ref.size != snapshot.size:
            raise UnsafePathError(f"tree backup refused because file changed before capture completed: {target}")
        return TreeBackupEntry(
            relative_path=relative_path,
            kind="file",
            snapshot=snapshot,
            content_address=content_ref,
            content_hash=snapshot.content_hash,
            size=snapshot.size,
            permissions=permissions,
        )

    if safety.is_symlink:
        snapshot = _snapshot_symlink(target)
        assert snapshot.symlink_target is not None
        return TreeBackupEntry(
            relative_path=relative_path,
            kind="symlink",
            snapshot=snapshot,
            symlink_target=snapshot.symlink_target,
        )

    if safety.is_dir:
        raise UnsafePathError(f"tree backup refused for directory path: {target}")
    raise UnsafePathError(f"tree backup refused for non-regular file: {target}")


def _restore_entry(
    entry: TreeBackupEntry,
    target: Path,
    *,
    artifact_store: _TreeBackupArtifactStore,
    conflict_policy: TreeRestoreConflictPolicy,
) -> ResourceSnapshot:
    current = inspect_path(target)
    if current.exists:
        _raise_if_unsafe_leaf(current, operation="tree restore")
        matched = _matching_restored_snapshot(entry, target)
        if matched is not None:
            return matched
        if conflict_policy == "no_replace":
            raise RestoreConflictError(f"tree restore refused because destination already exists: {target}")
        if not (current.is_file or current.is_symlink):
            raise UnsafePathError(f"tree restore refused to replace non-file path: {target}")

    if entry.kind == "file":
        return _restore_file_entry(
            entry,
            target,
            artifact_store=artifact_store,
            no_replace=not current.exists,
        )
    return _restore_symlink_entry(entry, target, replace=current.exists)


def _restore_file_entry(
    entry: TreeBackupEntry,
    target: Path,
    *,
    artifact_store: _TreeBackupArtifactStore,
    no_replace: bool,
) -> ResourceSnapshot:
    assert entry.content_address is not None
    if not artifact_store.verify(entry.content_address):
        raise BackupContentMismatchError(f"tree backup content address mismatch for {entry.relative_path}")
    try:
        artifact_store.copy_to(
            entry.content_address,
            target,
            no_replace=no_replace,
            permissions=entry.permissions,
            mtime_ns=entry.snapshot.mtime_ns,
        )
    except FileExistsError as exc:
        raise RestoreConflictError(f"tree restore refused because destination already exists: {target}") from exc
    except ValueError as exc:
        raise BackupContentMismatchError(f"tree backup content address mismatch for {entry.relative_path}") from exc

    restored = _snapshot_regular_file(target)
    _require_file_matches_entry(restored, entry)
    return restored


def _restore_symlink_entry(entry: TreeBackupEntry, target: Path, *, replace: bool) -> ResourceSnapshot:
    assert entry.symlink_target is not None
    parent_identity = _snapshot_restore_parent(target)
    if replace:
        _replace_symlink_entry(entry.symlink_target, target, parent_identity=parent_identity)
        restored = _snapshot_symlink(target)
        if restored.symlink_target != entry.symlink_target:
            raise BackupContentMismatchError(f"tree restore wrote unexpected symlink target for {target}")
        return restored

    try:
        _create_symlink_entry(entry.symlink_target, target, parent_identity=parent_identity)
    except FileExistsError as exc:
        raise RestoreConflictError(f"tree restore refused because destination already exists: {target}") from exc
    except OSError as exc:
        raise UnsafePathError(f"tree restore refused because symlink could not be created: {target}") from exc

    restored = _snapshot_symlink(target)
    if restored.symlink_target != entry.symlink_target:
        raise BackupContentMismatchError(f"tree restore wrote unexpected symlink target for {target}")
    return restored


def _require_symlink_restore_supported(target: Path, *, replace: bool) -> None:
    if replace:
        raise UnsupportedFilesystemMutationError(
            f"tree restore refused because safe symlink replacement is unavailable: {target}"
        )
    if os.name == "nt":
        raise UnsupportedFilesystemMutationError(
            f"tree restore refused because safe symlink creation is unavailable on Windows: {target}"
        )
    if os.symlink not in os.supports_dir_fd:
        raise UnsupportedFilesystemMutationError(
            "tree restore refused because this platform lacks descriptor-relative symlink creation"
        )


def _create_symlink_entry(symlink_target: str, target: Path, *, parent_identity: os.stat_result) -> None:
    _require_symlink_restore_supported(target, replace=False)

    parent_descriptor = _open_directory_for_mutation(
        target.parent,
        operation="tree restore",
        needs_replace=False,
        platform_support=_DEFAULT_PLATFORM_SUPPORT,
    )
    try:
        _ensure_directory_descriptor_matches_path(parent_descriptor, target.parent, operation="tree restore")
        _ensure_restore_parent_matches(target, parent_identity)
        os.symlink(symlink_target, target.name, dir_fd=parent_descriptor)
        _ensure_directory_descriptor_matches_path_after_mutation(
            parent_descriptor,
            target.parent,
            operation="tree restore",
        )
        _ensure_restore_parent_matches(target, parent_identity)
    finally:
        os.close(parent_descriptor)


def _replace_symlink_entry(symlink_target: str, target: Path, *, parent_identity: os.stat_result) -> None:
    del symlink_target, parent_identity
    raise UnsupportedFilesystemMutationError(
        f"tree restore refused because safe symlink replacement is unavailable: {target}"
    )


def _snapshot_restore_parent(target: Path) -> os.stat_result:
    parent = target.parent
    safety = inspect_path(parent)
    _raise_if_unsafe_directory(safety, operation="tree restore parent")
    if not safety.exists or not safety.is_dir:
        raise UnsafePathError(f"tree restore refused because parent is not a directory: {parent}")
    try:
        parent_identity = parent.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"tree restore refused because parent disappeared: {parent}") from exc
    if not S_ISDIR(parent_identity.st_mode):
        raise UnsafePathError(f"tree restore refused because parent is not a directory: {parent}")
    return parent_identity


def _ensure_restore_parent_matches(target: Path, expected: os.stat_result) -> None:
    parent = target.parent
    safety = inspect_path(parent)
    _raise_if_unsafe_directory(safety, operation="tree restore parent")
    if not safety.exists or not safety.is_dir:
        raise UnsafePathError(f"tree restore refused because parent is not a directory: {parent}")
    try:
        actual = parent.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"tree restore refused because parent disappeared: {parent}") from exc
    if not S_ISDIR(actual.st_mode) or not _same_path_identity(expected, actual):
        raise UnsafePathError(f"tree restore refused because parent changed during restore: {parent}")


def _matching_restored_snapshot(entry: TreeBackupEntry, target: Path) -> ResourceSnapshot | None:
    if entry.kind == "file":
        try:
            snapshot = _snapshot_regular_file(target)
        except UnsafePathError:
            return None
        if (
            _file_snapshot_matches_entry(snapshot, entry)
            and _current_permissions(target) == entry.permissions
            and _file_mtime_matches_entry(snapshot, entry)
        ):
            return snapshot
        return None

    try:
        snapshot = _snapshot_symlink(target)
    except UnsafePathError:
        return None
    if snapshot.symlink_target == entry.symlink_target:
        return snapshot
    return None


def _require_file_matches_entry(snapshot: ResourceSnapshot, entry: TreeBackupEntry) -> None:
    if not _file_snapshot_matches_entry(snapshot, entry):
        raise BackupContentMismatchError(f"tree restore wrote unexpected file content for {snapshot.path}")
    if _current_permissions(snapshot.path) != entry.permissions:
        raise BackupContentMismatchError(f"tree restore wrote unexpected permissions for {snapshot.path}")
    if not _file_mtime_matches_entry(snapshot, entry):
        raise BackupContentMismatchError(f"tree restore wrote unexpected mtime for {snapshot.path}")


def _file_snapshot_matches_entry(snapshot: ResourceSnapshot, entry: TreeBackupEntry) -> bool:
    return snapshot.file_type == "file" and snapshot.content_hash == entry.content_hash and snapshot.size == entry.size


def _file_mtime_matches_entry(snapshot: ResourceSnapshot, entry: TreeBackupEntry) -> bool:
    expected_mtime_ns = entry.snapshot.mtime_ns
    return expected_mtime_ns is None or snapshot.mtime_ns is None or snapshot.mtime_ns == expected_mtime_ns


def _remove_replaceable_leaf(path: Path) -> None:
    safety = inspect_path(path)
    if not safety.exists:
        return
    _raise_if_unsafe_leaf(safety, operation="tree restore")
    if not (safety.is_file or safety.is_symlink):
        raise UnsafePathError(f"tree restore refused to replace non-file path: {path}")
    path.unlink()


def _snapshot_regular_file(path: Path) -> ResourceSnapshot:
    snapshot = _platform_snapshot_resource(path)
    if snapshot.file_type != "file":
        raise UnsafePathError(f"tree backup refused for non-regular file: {path}")
    return snapshot


def _snapshot_symlink(path: Path) -> ResourceSnapshot:
    if os.name != "nt":
        snapshot = snapshot_resource(path)
        if snapshot.file_type != "symlink":
            raise UnsafePathError(f"tree backup refused because path is not a symlink: {path}")
        return snapshot

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"tree backup refused because symlink disappeared: {path}") from exc
    if not S_ISLNK(before.st_mode):
        raise UnsafePathError(f"tree backup refused because path is not a symlink: {path}")
    try:
        target = os.readlink(path)
    except OSError as exc:
        raise UnsafePathError(f"tree backup refused because symlink target could not be read: {path}") from exc
    try:
        after = path.lstat()
    except FileNotFoundError as exc:
        raise UnsafePathError(f"tree backup refused because symlink disappeared: {path}") from exc
    if not S_ISLNK(after.st_mode) or not _same_snapshot_metadata(before, after):
        raise UnsafePathError(f"tree backup refused because symlink changed before capture completed: {path}")
    return ResourceSnapshot(
        path=path,
        exists=True,
        file_type="symlink",
        content_hash=None,
        size=before.st_size,
        mtime_ns=before.st_mtime_ns,
        symlink_target=target,
        device=before.st_dev,
        inode=before.st_ino,
        ctime_ns=before.st_ctime_ns,
    )


def _platform_snapshot_resource(path: Path) -> ResourceSnapshot:
    if os.name == "nt":
        from safe_fs_ops.filesystem_ops._windows_snapshots import snapshot_resource_windows

        return snapshot_resource_windows(path)
    return snapshot_resource(path)


def _normalize_relative_path(path: Path | str) -> Path:
    relative_path = Path(path)
    if relative_path.is_absolute() or relative_path.drive or relative_path.root:
        raise UnsafePathError(f"tree backup paths must be relative: {relative_path}")
    if relative_path == Path(".") or any(part == ".." for part in relative_path.parts):
        raise UnsafePathError(f"tree backup paths must not escape the root: {relative_path}")
    return relative_path


def _require_existing_safe_directory(path: Path, *, operation: str) -> None:
    safety = inspect_path(path)
    if not safety.exists:
        raise FileNotFoundError(path)
    _raise_if_unsafe_directory(safety, operation=operation)
    if not safety.is_dir:
        raise UnsafePathError(f"{operation} refused because path is not a directory: {path}")


def _ensure_safe_directory(path: Path, *, operation: str) -> None:
    target = absolute_without_resolving(path)
    _safe_make_directory(target, parents=True, exist_ok=True)
    _require_existing_safe_directory(target, operation=operation)


def _require_safe_directory_chain_if_exists(path: Path, *, operation: str) -> None:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.relative_to(anchor).parts:
        current = current / part
        safety = inspect_path(current)
        if not safety.exists:
            return
        _raise_if_unsafe_directory(safety, operation=operation)
        if not safety.is_dir:
            raise UnsafePathError(f"{operation} refused because path is not a directory: {current}")


def _existing_safe_parent_chain_allows_target(target: Path, *, operation: str) -> bool:
    absolute_parent = absolute_without_resolving(target.parent)
    anchor = Path(absolute_parent.anchor)
    current = anchor
    for part in absolute_parent.relative_to(anchor).parts:
        current = current / part
        safety = inspect_path(current)
        if not safety.exists:
            return False
        _raise_if_unsafe_directory(safety, operation=f"{operation} parent")
        if not safety.is_dir:
            raise UnsafePathError(f"{operation} refused because parent path is not a directory: {current}")
    return True


def _safe_make_directory(path: Path, *, parents: bool = False, exist_ok: bool = False) -> None:
    if os.name == "nt":
        make_directory_windows(path, parents=parents, exist_ok=exist_ok)
        return
    make_directory(path, parents=parents, exist_ok=exist_ok)


def _require_existing_safe_parent_chain(root: Path, target: Path, *, operation: str) -> None:
    try:
        relative_parent = target.parent.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError(f"{operation} refused because path escapes the root: {target}") from exc
    current = root
    _require_existing_safe_directory(root, operation=operation)
    for part in relative_parent.parts:
        current = current / part
        _require_existing_safe_directory(current, operation=operation)


def _raise_if_unsafe_directory(safety: PathSafety, *, operation: str) -> None:
    if safety.is_symlink:
        raise UnsafePathError(f"{operation} refused because path redirects elsewhere: {safety.path}")
    if safety.is_windows_reparse_point:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {safety.path}")
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {safety.path}")


def _raise_if_unsafe_leaf(safety: PathSafety, *, operation: str) -> None:
    if safety.is_mount:
        raise UnsafePathError(f"{operation} refused for mount point: {safety.path}")
    if safety.is_windows_reparse_point and not safety.is_symlink:
        raise UnsafePathError(f"{operation} refused for Windows reparse point: {safety.path}")


def _current_permissions(path: Path) -> int | None:
    with contextlib.suppress(OSError):
        return S_IMODE(path.lstat().st_mode)
    return None


def _same_snapshot_metadata(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
    )


def _same_path_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


__all__ = [
    "TreeRestoreConflictPolicy",
    "backup_tree",
    "restore_tree_backup",
    "tree_backup_from_journal_payload",
]
