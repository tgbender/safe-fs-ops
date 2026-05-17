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


def test_write_text_rejects_current_lease_owner_that_does_not_match_operation_owner(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="lease-owner",
        ttl=timedelta(seconds=30),
        now=now,
    )
    ClaimStore(state_path).upsert(resource_key, lease=lease, owner="owner-a", now=now)

    with pytest.raises(ResourceClaimAuthorityError, match="lease owner"):
        _coordinator(state_path).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=now,
        )

    assert target.read_text(encoding="utf-8") == "old\n"


def test_write_text_rejects_stale_lease_bound_claim_owner_scope_forge(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    old_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=first_now,
    )
    claim_scope = "transaction-instance:stale"
    ClaimStore(state_path).upsert(
        resource_key,
        lease=old_lease,
        owner="owner-a",
        scope=claim_scope,
        details=json.dumps(
            lease_claim_details_payload(
                old_lease,
                claim_id="old-claim",
                extra={"resource_kind": "file", "run_id": "run-1"},
            ),
            separators=(",", ":"),
            sort_keys=True,
        ),
        now=first_now,
    )
    new_now = first_now + timedelta(seconds=31)
    new_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=new_now,
    )
    write_calls: list[Path | str] = []

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        write_calls.append(path)
        Path(path).write_text(content, encoding=encoding, newline=newline)

    with pytest.raises(ResourceClaimAuthorityError, match="fencing token") as raised:
        _coordinator(state_path, write_text_operation=record_write).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=new_lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            claim_scope=claim_scope,
            now=new_now,
        )

    assert raised.value.existing.details is not None
    assert write_calls == []
    assert target.read_text(encoding="utf-8") == "old\n"
    assert OperationJournalStore(state_path).list_batches() == []


def test_write_text_rejects_stale_unbound_claim_owner_scope_forge(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    old_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=first_now,
    )
    claim_scope = "transaction-instance:stale-unbound"
    ClaimStore(state_path).upsert(
        resource_key,
        lease=old_lease,
        owner="owner-a",
        scope=claim_scope,
        now=first_now,
    )
    new_now = first_now + timedelta(seconds=31)
    new_lease = LeaseStore(state_path).acquire(
        "workspace",
        owner="owner-a",
        ttl=timedelta(seconds=30),
        now=new_now,
    )
    write_calls: list[Path | str] = []

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        write_calls.append(path)
        Path(path).write_text(content, encoding=encoding, newline=newline)

    with pytest.raises(ResourceClaimAuthorityError, match="lease-bound"):
        _coordinator(state_path, write_text_operation=record_write).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=new_lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:stale-unbound-claim",
            claim_scope=claim_scope,
            now=new_now,
        )

    assert write_calls == []
    assert target.read_text(encoding="utf-8") == "old\n"
    assert OperationJournalStore(state_path).list_batches() == []


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


def test_claim_change_after_journal_records_is_detected_before_side_effect(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    snapshot_calls = 0
    write_calls: list[Path | str] = []

    def change_claim_after_before_snapshot(path: Path | str):
        nonlocal snapshot_calls
        snapshot_calls += 1
        observed = _portable_snapshot(path)
        if snapshot_calls == 1:
            claims = ClaimStore(state_path)
            assert claims.release(resource_key, lease=lease, owner="owner-a", now=now) is True
            claims.upsert(resource_key, lease=lease, owner="owner-b", details=_claim_details(lease), now=now)
        return observed

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        write_calls.append(path)

    with pytest.raises(ResourceClaimAuthorityError) as raised:
        _coordinator(
            state_path,
            journal=journal,
            snapshot=change_claim_after_before_snapshot,
            write_text_operation=record_write,
        ).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=now,
        )

    assert raised.value.existing.owner == "owner-b"
    assert write_calls == []
    assert target.read_text(encoding="utf-8") == "old\n"
    batch = journal.list_batches()[0]
    assert batch.resource_key == resource_key
    assert batch.claim_owner == "owner-a"
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem mutation failed"
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
        "failure",
    ]
    recovery = journal.list_recovery_records(batch.batch_id)[0]
    assert recovery.reason == "filesystem mutation failed"
    assert "claimed by" in recovery.payload["failure"]["error"]


def _coordinator(
    state_path: Path,
    *,
    journal: OperationJournalStore | None = None,
    snapshot=None,
    write_text_operation=None,
    delete_file_operation=None,
    clock=None,
) -> JournaledFilesystemCoordinator:
    kwargs = {
        "snapshot": _portable_snapshot,
        "write_text_operation": _portable_write_text,
        "delete_file_operation": _portable_delete_file,
    }
    if snapshot is not None:
        kwargs["snapshot"] = snapshot
    if write_text_operation is not None:
        kwargs["write_text_operation"] = write_text_operation
    if delete_file_operation is not None:
        kwargs["delete_file_operation"] = delete_file_operation
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
