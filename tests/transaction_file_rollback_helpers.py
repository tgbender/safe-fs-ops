from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import pytest
from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_bytes,
    portable_write_text,
)

from safe_fs_ops import SafeWorkspace

requires_backup_restore_support = pytest.mark.skipif(
    not (
        os.name == "nt"
        or (
            os.open in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.stat in os.supports_follow_symlinks
            and os.replace in os.supports_dir_fd
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
        )
    ),
    reason="descriptor-relative backup restore support is unavailable on this platform",
)


def workspace_with_portable_file_rollback(
    state_path: Path,
    *,
    write_bytes_max_bytes: int | None = None,
    write_bytes_large_policy: Literal["reject", "allow"] = "reject",
) -> SafeWorkspace:
    write_bytes_kwargs = {} if write_bytes_max_bytes is None else {"write_bytes_max_bytes": write_bytes_max_bytes}
    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        snapshot=portable_snapshot,
        write_bytes_operation=portable_write_bytes,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
        write_bytes_large_policy=write_bytes_large_policy,
        **write_bytes_kwargs,
    )


def hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()


def latest_batch_for_run(workspace: SafeWorkspace, run_id: str):
    matches = [batch for batch in workspace.journal_store.list_batches() if batch.run_id == run_id]
    assert matches, f"expected batch for run {run_id!r}"
    return matches[-1]


def backup_content_path_for_batch(workspace: SafeWorkspace, batch_id: str) -> Path:
    for checkpoint in workspace.journal_store.list_checkpoints(batch_id):
        if checkpoint.checkpoint_type != "backup":
            continue
        content_path = checkpoint.payload.get("content_path")
        if content_path is not None:
            return Path(str(content_path))
    raise AssertionError(f"expected backup content path for batch {batch_id!r}")


def rewrite_backup_checkpoint_payload(
    workspace: SafeWorkspace,
    *,
    batch_id: str,
    content_path: str | None,
    existed: bool,
    file_type: str,
) -> None:
    for checkpoint in workspace.journal_store.list_checkpoints(batch_id):
        if checkpoint.checkpoint_type != "backup":
            continue
        payload = jsonable(checkpoint.payload)
        assert isinstance(payload, dict)
        payload["content_path"] = content_path
        payload["existed"] = existed
        payload["file_type"] = file_type
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            snapshot_payload = dict(snapshot)
            snapshot_payload["exists"] = existed
            snapshot_payload["file_type"] = file_type
            payload["snapshot"] = snapshot_payload
        with workspace.journal_store.sqlite_store.transaction() as connection:
            connection.execute(
                """
                UPDATE operation_journal_checkpoints
                SET payload = ?
                WHERE checkpoint_id = ?
                """,
                (json.dumps(payload, separators=(",", ":"), sort_keys=True), checkpoint.checkpoint_id),
            )
        return
    raise AssertionError(f"expected backup checkpoint for batch {batch_id!r}")


def jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [jsonable(item) for item in value]
    return value
