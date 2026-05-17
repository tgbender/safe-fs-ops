from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal import JournaledFilesystemCoordinator, OperationJournalStore
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def _coordinator(
    state_path: Path,
    *,
    journal: OperationJournalStore | None = None,
    snapshot=None,
    write_text_operation=None,
    delete_file_operation=None,
    make_directory_operation=None,
    clock=None,
) -> JournaledFilesystemCoordinator:
    kwargs = {
        "snapshot": _portable_snapshot,
        "write_text_operation": _portable_write_text,
        "delete_file_operation": _portable_delete_file,
        "make_directory_operation": _portable_make_directory,
    }
    if snapshot is not None:
        kwargs["snapshot"] = snapshot
    if write_text_operation is not None:
        kwargs["write_text_operation"] = write_text_operation
    if delete_file_operation is not None:
        kwargs["delete_file_operation"] = delete_file_operation
    if make_directory_operation is not None:
        kwargs["make_directory_operation"] = make_directory_operation
    return JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal or OperationJournalStore(state_path),
        clock=clock,
        **kwargs,
    )


def _portable_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    Path(path).write_text(content, encoding=encoding, newline=newline)


def _portable_delete_file(path: Path | str, *, missing_ok: bool = False) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def _portable_make_directory(path: Path | str, *, parents: bool = False, exist_ok: bool = False) -> None:
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def _portable_snapshot(path: Path | str) -> ResourceSnapshot:
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
    if target.is_file():
        content_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        file_type = "file"
    elif target.is_dir():
        content_hash = None
        file_type = "directory"
    else:
        content_hash = None
        file_type = "other"
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


def _acquire_and_claim(state_path: Path, resource_key: str, *, now: datetime) -> LeaseRecord:
    lease = LeaseStore(state_path).acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    ClaimStore(state_path).upsert(
        resource_key,
        lease=lease,
        owner="owner-a",
        details=_claim_details(lease),
        now=now,
    )
    return lease


def _claim_details(lease: LeaseRecord) -> str:
    return json.dumps(
        lease_claim_details_payload(lease, claim_id=f"claim:{lease.name}:{lease.fencing_token}"),
        separators=(",", ":"),
        sort_keys=True,
    )


def _sequence_clock(values: list[datetime]):
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        if calls >= len(values):
            return values[-1]
        value = values[calls]
        calls += 1
        return value

    return clock


__all__ = [name for name in globals() if not name.startswith("__")]
