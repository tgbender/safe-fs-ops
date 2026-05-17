from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import ResourceSnapshot
from safe_fs_ops.operation_journal import (
    BatchIdempotencyMismatchError,
    BatchPhase,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemCoordinator,
    OperationJournalStore,
    ResourceClaimAuthorityError,
    directory_resource_key,
    file_resource_key,
)
from safe_fs_ops.workspace_state import ClaimStore, LeaseStore
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


def test_empty_planned_retry_with_newer_lease_restarts_without_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    old_lease = _acquire_and_claim(state_path, resource_key, now=first_now)
    journal = OperationJournalStore(state_path)
    desired = b"new\n"
    payload = {
        "operation": "write_text",
        "path": str(target),
        "resource_key": resource_key,
        "encoding": "utf-8",
        "newline": None,
        "desired_size": len(desired),
        "desired_sha256": hashlib.sha256(desired).hexdigest(),
    }
    batch = journal.create_batch(
        idempotency_key="write:config",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload=payload,
        batch_id="batch-1",
        now=first_now,
    )
    new_now = first_now + timedelta(seconds=31)
    new_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=new_now,
    )
    ClaimStore(state_path).upsert(
        resource_key,
        lease=new_lease,
        owner="owner-a",
        details=_claim_details(new_lease),
        now=new_now,
    )

    result = _coordinator(state_path, journal=journal).write_text_file(
        target,
        "new\n",
        resource_key=resource_key,
        lease=new_lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="write:config",
        now=new_now,
    )

    assert result.batch.batch_id == batch.batch_id
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert result.batch.lease_fencing_token == new_lease.fencing_token
    assert journal.list_recovery_records(batch.batch_id) == []
    assert target.read_text(encoding="utf-8") == "new\n"


def test_successful_idempotent_retry_reuses_batch_without_mutating_again(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    first = coordinator.write_text_file(
        target,
        "new\n",
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="write:config",
        now=now,
    )

    def fail_if_called(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        raise AssertionError("retry should not mutate")

    second = _coordinator(state_path, journal=journal, write_text_operation=fail_if_called).write_text_file(
        target,
        "new\n",
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="write:config",
        now=now,
    )

    assert second.skipped is True
    assert second.batch.batch_id == first.batch.batch_id
    assert target.read_text(encoding="utf-8") == "new\n"
    assert len(journal.list_batches()) == 1
    assert len(journal.list_operations(first.batch.batch_id)) == 1
    assert len(journal.list_checkpoints(first.batch.batch_id)) == 5


def test_make_directory_records_journal_batch_and_checkpoints(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "nested" / "state"
    target.parent.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = directory_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)

    result = _coordinator(state_path, journal=journal).make_directory(
        target,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="mkdir:state",
        parents=False,
        exist_ok=False,
        now=now,
    )

    assert target.is_dir()
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert {key: result.batch.payload[key] for key in ("operation", "path", "resource_key", "parents", "exist_ok")} == {
        "operation": "make_directory",
        "path": str(target),
        "resource_key": resource_key,
        "parents": False,
        "exist_ok": False,
    }
    assert result.batch.payload["rollback_diagnostic"] == {
        "automatic_recursive_delete": False,
        "manual_intervention_possible": True,
        "remove_only_if": {
            "created_by_exact_step": True,
            "directory_is_empty": True,
            "path_is_still_safe_directory": True,
            "resource_key_still_matches": True,
        },
    }
    assert journal.list_operations(result.batch.batch_id)[0].operation_type == "make_directory"
    before, after = journal.list_checkpoints(result.batch.batch_id)
    assert before.checkpoint_type == "before"
    assert before.payload["file_type"] == "missing"
    assert after.checkpoint_type == "after"
    assert after.payload["file_type"] == "directory"


def test_make_directory_idempotent_retry_with_different_payload_is_rejected(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "nested" / "state"
    target.parent.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = directory_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    coordinator.make_directory(
        target,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="mkdir:state",
        parents=False,
        exist_ok=False,
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="payload"):
        coordinator.make_directory(
            target,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="mkdir:state",
            parents=False,
            exist_ok=True,
            now=now,
        )


def test_make_directory_rejects_parents_true_before_batch_or_side_effects(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "nested" / "state"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = directory_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)

    with pytest.raises(NotImplementedError, match="parents=True"):
        _coordinator(state_path, journal=journal).make_directory(
            target,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="mkdir:state",
            parents=True,
            exist_ok=False,
            now=now,
        )

    assert not target.parent.exists()
    assert not target.exists()
    assert journal.list_batches() == []


def test_idempotent_retry_with_different_requested_mutation_is_rejected(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    coordinator.write_text_file(
        target,
        "new\n",
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="write:config",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="payload"):
        coordinator.write_text_file(
            target,
            "different\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=now,
        )

    assert target.read_text(encoding="utf-8") == "new\n"
    assert len(journal.list_batches()) == 1


def test_same_key_retry_with_newer_lease_marks_abandoned_attempt_for_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    old_lease = _acquire_and_claim(state_path, resource_key, now=first_now)
    journal = OperationJournalStore(state_path)
    desired = b"new\n"
    payload = {
        "operation": "write_text",
        "path": str(target),
        "resource_key": resource_key,
        "encoding": "utf-8",
        "newline": None,
        "desired_size": len(desired),
        "desired_sha256": hashlib.sha256(desired).hexdigest(),
    }
    batch = journal.create_batch(
        idempotency_key="write:config",
        lease=old_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload=payload,
        batch_id="batch-1",
        now=first_now,
    )
    journal.append_operation(
        batch.batch_id,
        lease=old_lease,
        operation_type="write_text",
        resource_key=resource_key,
        payload=payload,
        now=first_now,
    )
    journal.mark_attempting(batch.batch_id, lease=old_lease, now=first_now)
    new_now = first_now + timedelta(seconds=31)
    new_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=new_now,
    )
    ClaimStore(state_path).upsert(
        resource_key,
        lease=new_lease,
        owner="owner-a",
        details=_claim_details(new_lease),
        now=new_now,
    )

    with pytest.raises(JournaledFilesystemBatchStateError, match="cannot start"):
        _coordinator(state_path, journal=journal).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=new_lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=new_now,
        )

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert (
        journal.list_recovery_records(batch.batch_id)[0].reason == "retry found abandoned batch outside planned phase"
    )
    assert target.read_text(encoding="utf-8") == "old\n"


def test_current_unrelated_lease_cannot_use_claim_bound_to_workspace_lease(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    workspace_lease = _acquire_and_claim(state_path, resource_key, now=now)
    unrelated_lease = LeaseStore(state_path).acquire(
        "unrelated",
        owner=workspace_lease.owner,
        ttl=timedelta(seconds=30),
        now=now,
    )

    with pytest.raises(ResourceClaimAuthorityError, match="different lease identity"):
        _coordinator(state_path).write_text_file(
            target,
            "hijacked\n",
            resource_key=resource_key,
            lease=unrelated_lease,
            owner="owner-a",
            run_id="run-unrelated-lease",
            idempotency_key="write:unrelated-lease",
            now=now,
        )

    assert target.read_text(encoding="utf-8") == "old\n"


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
