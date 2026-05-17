from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    DirectoryIdentity,
    FileBackup,
    ResourceSnapshot,
    remove_empty_directory,
    snapshot_resource,
)
from safe_fs_ops.operation_journal import (
    JournaledFilesystemCoordinator,
    OperationBatchRecord,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.filesystem import (
    DirectoryIdentityRemoveOperation,
    DirectoryRemoveOperation,
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload
from safe_fs_ops.workspace_state.models import LeaseRecord


def _descriptor_relative_snapshot_supported() -> bool:
    import os

    return (
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def _descriptor_relative_mutations_supported() -> bool:
    import os

    return (
        os.open in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.replace in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


requires_backup_restore_support = pytest.mark.skipif(
    not (_descriptor_relative_snapshot_supported() and _descriptor_relative_mutations_supported()),
    reason="descriptor-relative snapshot/mutation support is unavailable on this platform",
)


def _coordinator(
    state_path: Path,
    *,
    journal: OperationJournalStore | None = None,
    recovery_action_handlers: Mapping[str, Callable[..., object | None]] | None = None,
    remove_directory_operation: DirectoryRemoveOperation | None = None,
    identity_remove_directory_operation: DirectoryIdentityRemoveOperation | None = None,
    snapshot: Callable[[Path | str], ResourceSnapshot] | None = None,
) -> JournaledFilesystemCoordinator:
    return JournaledFilesystemCoordinator(
        lease_store=LeaseStore(state_path),
        claim_store=ClaimStore(state_path),
        journal_store=journal or OperationJournalStore(state_path),
        recovery_action_handlers=recovery_action_handlers,
        remove_directory_operation=remove_directory_operation or remove_empty_directory,
        identity_remove_directory_operation=identity_remove_directory_operation,
        snapshot=snapshot if snapshot is not None else snapshot_resource,
    )


def fake_identity_remove_directory(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
    target = Path(path)
    current_identity = DirectoryIdentity.from_stat(target.stat())
    if current_identity != expected_identity:
        raise RecoveryActionManualInterventionRequired(f"identity mismatch for {target}")
    target.rmdir()


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


def _start_recovering_batch(
    journal: OperationJournalStore,
    *,
    lease: LeaseRecord,
    resource_key: str,
    now: datetime,
) -> OperationBatchRecord:
    batch = journal.create_batch(
        idempotency_key=f"recover:{resource_key}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload={"operation": "write_text"},
        batch_id="batch-1",
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload={"resource_key": resource_key},
        now=now,
    )
    journal.mark_failed(
        batch.batch_id, lease=lease, error="write failed", observed_state={"resource_key": resource_key}, now=now
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="restore after failure",
        payload={"resource_key": resource_key},
        recovery_id="recovery-desired",
        now=now,
    )
    journal.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-attempt",
        now=now + timedelta(microseconds=1),
    )
    return batch


def _current_recovery_attempt_id(journal: OperationJournalStore, batch_id: str) -> str:
    context = journal.read_recovery_context(batch_id)
    assert context.recovery_attempt is not None
    return context.recovery_attempt.recovery_id


def _restore_backup_payload(
    backup: FileBackup,
    *,
    expected_current: ResourceSnapshot | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "backup": {
            "path": str(backup.path),
            "existed": backup.existed,
            "file_type": backup.file_type,
            "content_path": str(backup.content_path) if backup.content_path is not None else None,
            "content_hash": backup.content_hash,
            "size": backup.size,
            "permissions": backup.permissions,
            "snapshot": _snapshot_payload(backup.snapshot),
        }
    }
    if expected_current is not None:
        payload["expected_current"] = _snapshot_payload(expected_current)
    return payload


def _snapshot_payload(snapshot: ResourceSnapshot) -> dict[str, object]:
    return {
        "path": str(snapshot.path),
        "exists": snapshot.exists,
        "file_type": snapshot.file_type,
        "content_hash": snapshot.content_hash,
        "size": snapshot.size,
        "mtime_ns": snapshot.mtime_ns,
        "symlink_target": snapshot.symlink_target,
    }


def _skip_handler(*_args: object, **_kwargs: object) -> None:
    raise RecoveryActionSkipped("already handled")


def _fail_handler(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("injected recovery action failure")


def _manual_handler(*_args: object, **_kwargs: object) -> None:
    raise RecoveryActionManualInterventionRequired("chmod unsupported")


def _success_handler(*_args: object, **_kwargs: object) -> str:
    return "ok"
