from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal.models import CheckpointRecord, OperationBatchRecord, OperationRecord, RecoveryRecord
from safe_fs_ops.workspace_state.models import ClaimRecord


@dataclass(frozen=True, slots=True)
class JournaledFilesystemResult:
    batch: OperationBatchRecord
    operation: OperationRecord | None
    before_checkpoint: CheckpointRecord | None
    after_checkpoint: CheckpointRecord | None
    recovery_record: RecoveryRecord | None
    skipped: bool = False


class JournaledFilesystemError(RuntimeError):
    pass


class MissingResourceClaimError(JournaledFilesystemError):
    pass


class ResourceClaimAuthorityError(JournaledFilesystemError):
    def __init__(self, message: str, *, existing: ClaimRecord) -> None:
        super().__init__(message)
        self.existing = existing


class FileResourceKeyMismatchError(JournaledFilesystemError):
    pass


class DirectoryResourceKeyMismatchError(JournaledFilesystemError):
    pass


class TreeResourceKeyMismatchError(JournaledFilesystemError):
    pass


class JournaledFilesystemBatchStateError(JournaledFilesystemError):
    pass


class JournaledFilesystemMutationError(JournaledFilesystemError):
    def __init__(self, message: str, *, batch_id: str) -> None:
        super().__init__(message)
        self.batch_id = batch_id


class JournaledFilesystemRecoveryError(JournaledFilesystemError):
    def __init__(
        self,
        message: str,
        *,
        batch_id: str,
        recovery_record: RecoveryRecord | None = None,
        recording_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.batch_id = batch_id
        self.recovery_record = recovery_record
        self.recording_error = recording_error


def _snapshot_payload(snapshot: ResourceSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "exists": snapshot.exists,
        "file_type": snapshot.file_type,
        "content_hash": snapshot.content_hash,
        "size": snapshot.size,
        "mtime_ns": snapshot.mtime_ns,
        "symlink_target": snapshot.symlink_target,
        "device": snapshot.device,
        "inode": snapshot.inode,
        "ctime_ns": snapshot.ctime_ns,
    }


def _normalized_text(content: str, *, newline: str | None) -> str:
    if newline is None or newline == "":
        return content
    return content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)


def file_resource_key(path: Path | str) -> str:
    return f"file:{_resource_key_path(path)}"


def directory_resource_key(path: Path | str) -> str:
    return f"directory:{_resource_key_path(path)}"


def tree_resource_key(path: Path | str) -> str:
    return f"tree:{_resource_key_path(path)}"


def _canonical_file_path(path: Path | str) -> Path:
    return _absolute_file_path(path)


def _canonical_directory_path(path: Path | str) -> Path:
    return _absolute_file_path(path)


def _absolute_file_path(path: Path | str) -> Path:
    return Path(os.path.abspath(Path(path).expanduser()))


def _resource_key_path(path: Path | str) -> Path:
    return Path(os.path.normcase(os.path.abspath(Path(path).expanduser())))


def _require_matching_file_resource_key(path: Path, resource_key: str) -> None:
    expected = file_resource_key(path)
    if resource_key != expected:
        raise FileResourceKeyMismatchError(
            f"resource_key {resource_key!r} does not match canonical file resource {expected!r}"
        )


def _require_matching_directory_resource_key(path: Path, resource_key: str) -> None:
    expected = directory_resource_key(path)
    if resource_key != expected:
        raise DirectoryResourceKeyMismatchError(
            f"resource_key {resource_key!r} does not match canonical directory resource {expected!r}"
        )


def _require_matching_tree_resource_key(path: Path, resource_key: str) -> None:
    expected = tree_resource_key(path)
    if resource_key != expected:
        raise TreeResourceKeyMismatchError(
            f"resource_key {resource_key!r} does not match canonical tree resource {expected!r}"
        )


def _utcnow(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "DirectoryResourceKeyMismatchError",
    "FileResourceKeyMismatchError",
    "JournaledFilesystemBatchStateError",
    "JournaledFilesystemError",
    "JournaledFilesystemMutationError",
    "JournaledFilesystemRecoveryError",
    "JournaledFilesystemResult",
    "MissingResourceClaimError",
    "ResourceClaimAuthorityError",
    "TreeResourceKeyMismatchError",
    "directory_resource_key",
    "file_resource_key",
    "tree_resource_key",
]
