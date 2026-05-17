from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    RenameRecord,
    ResourceSnapshot,
    capture_directory_to_quarantine,
    rename_no_replace,
)
from safe_fs_ops.workspace_state import ClaimConflictError
from safe_fs_ops.workspace_state.models import ClaimRecord


def workspace_with_portable_ops(
    state_path: Path,
    *,
    artifact_store: object | None = None,
    captured_directory_cleanup: Literal["automatic", "retain"] = "automatic",
) -> SafeWorkspace:
    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        snapshot=portable_snapshot,
        write_bytes_operation=portable_write_bytes,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=portable_remove_empty_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        rename_no_replace_operation=portable_rename_no_replace,
        capture_directory_operation=portable_capture_directory,
        artifact_store=artifact_store,
        captured_directory_cleanup=captured_directory_cleanup,
    )


def portable_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding, newline=newline)


def portable_write_bytes(
    path: Path | str,
    content: bytes,
    *,
    permissions: int | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    if permissions is not None:
        target.chmod(permissions)


def portable_delete_file(path: Path | str, *, missing_ok: bool = False) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def portable_make_directory(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def portable_remove_empty_directory(path: Path | str, *, missing_ok: bool = False) -> None:
    try:
        Path(path).rmdir()
    except FileNotFoundError:
        if not missing_ok:
            raise


def portable_remove_empty_directory_by_identity(
    path: Path | str,
    *,
    expected_identity: DirectoryIdentity,
) -> None:
    target = Path(path)
    current_identity = DirectoryIdentity.from_stat(target.stat())
    if current_identity != expected_identity:
        raise RuntimeError(f"identity mismatch for {target}")
    target.rmdir()


def portable_rename_no_replace(path: Path | str, destination: Path | str) -> RenameRecord:
    return rename_no_replace(
        path,
        destination,
        _rename_no_replace=lambda source, target, *, operation: source.rename(target),
    )


def portable_capture_directory(path: Path | str, *, quarantine_path: Path | str) -> CapturedDirectoryRecord:
    return capture_directory_to_quarantine(
        path,
        quarantine_path=quarantine_path,
        _rename_no_replace=lambda source, target, *, operation: source.rename(target),
    )


def portable_snapshot(path: Path | str) -> ResourceSnapshot:
    target = Path(path)
    if target.is_symlink():
        stat_result = target.lstat()
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="symlink",
            content_hash=None,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=os.readlink(target),
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
        )
    if not target.exists():
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
        )
    stat_result = target.lstat()
    content_hash = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None
    file_type = "file" if target.is_file() else "directory" if target.is_dir() else "other"
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type=file_type,
        content_hash=content_hash,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )


class EnterConflictClaimStore:
    def __init__(self, *, conflict_key: str) -> None:
        self._conflict_key = conflict_key
        self._claimed: list[str] = []

    def upsert(
        self,
        resource_key: str,
        *,
        lease: object,
        owner: str,
        scope: str | None = None,
        details: str | None = None,
        now: datetime | None = None,
    ) -> None:
        del lease, now
        if resource_key == self._conflict_key:
            raise ClaimConflictError(
                f"resource {resource_key!r} is already claimed",
                existing=ClaimRecord(
                    resource_key=resource_key,
                    owner="owner-b",
                    scope="transaction:other-run",
                    details="custom",
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                    updated_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            )
        self._claimed.append(resource_key)
        return ClaimRecord(
            resource_key=resource_key,
            owner=owner,
            scope=scope,
            details=details,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def release(
        self,
        resource_key: str,
        *,
        lease: object,
        owner: str,
        scope: str | None = None,
        expected_claim: ClaimRecord | None = None,
        now: datetime | None = None,
    ) -> bool:
        del lease, owner, scope, expected_claim, now
        return not self._claimed or resource_key != self._claimed[0]
