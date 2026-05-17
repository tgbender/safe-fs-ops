from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from journaled_filesystem_helpers import (
    _acquire_and_claim,
    _claim_details,
    _coordinator,
    _portable_delete_file,
    _portable_make_directory,
    _portable_snapshot,
    _portable_write_text,
    _sequence_clock,
)

from safe_fs_ops.filesystem_ops import ResourceSnapshot, UnsafePathError
from safe_fs_ops.operation_journal import (
    BatchPhase,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemCoordinator,
    JournaledFilesystemMutationError,
    OperationJournalStore,
    file_resource_key,
)
from safe_fs_ops.workspace_state import ClaimStore, LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = pytest.mark.safe_fs_ops


class _RecordingLeaseStore:
    def __init__(self, delegate: LeaseStore) -> None:
        self._delegate = delegate
        self.heartbeat_times: list[datetime | None] = []

    def heartbeat(self, lease: LeaseRecord, *, ttl: timedelta, now: datetime | None = None) -> LeaseRecord | None:
        self.heartbeat_times.append(now)
        return self._delegate.heartbeat(lease, ttl=ttl, now=now)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


def test_mutation_failure_records_failed_state_and_recovery_intent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)

    def fail_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        raise OSError("injected write failure")

    with pytest.raises(JournaledFilesystemMutationError, match="injected write failure") as raised:
        _coordinator(state_path, journal=journal, write_text_operation=fail_write).write_text_file(
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
    batch = journal.get_batch(raised.value.batch_id)
    assert batch is not None
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem mutation failed"
    checkpoints = journal.list_checkpoints(batch.batch_id)
    assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
        "failure",
    ]
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert len(recovery_records) == 1
    assert recovery_records[0].phase == BatchPhase.RECOVERY_DESIRED
    assert recovery_records[0].payload["failure"]["error"] == "injected write failure"


def test_mutation_with_explicit_logical_now_does_not_heartbeat_lease(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease_store = _RecordingLeaseStore(LeaseStore(state_path))
    lease = lease_store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    ClaimStore(state_path).upsert(resource_key, lease=lease, owner="owner-a", details=_claim_details(lease), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = JournaledFilesystemCoordinator(
        lease_store=lease_store,  # type: ignore[arg-type]
        claim_store=ClaimStore(state_path),
        journal_store=journal,
        snapshot=_portable_snapshot,
        write_text_operation=_portable_write_text,
        delete_file_operation=_portable_delete_file,
        make_directory_operation=_portable_make_directory,
    )

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

    assert lease_store.heartbeat_times == []


def test_mutation_side_effect_failure_with_lease_loss_records_takeover_needed_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    expired_now = first_now + timedelta(seconds=31)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=first_now)
    journal = OperationJournalStore(state_path)
    clock = _sequence_clock([first_now] * 8 + [expired_now])

    def write_then_fail(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise OSError("injected write failure after side effect")

    with pytest.raises(JournaledFilesystemMutationError, match="injected write failure after side effect") as raised:
        _coordinator(
            state_path,
            journal=journal,
            write_text_operation=write_then_fail,
            clock=clock,
        ).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
        )

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = journal.get_batch(raised.value.batch_id)
    assert batch is not None
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem mutation failed"
    assert batch.status_payload["failure"]["error"] == "injected write failure after side effect"
    assert (
        batch.status_payload["failure"]["snapshot"]["content_hash"]
        == hashlib.sha256(
            target.read_bytes(),
        ).hexdigest()
    )
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
    ]
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert len(recovery_records) == 1
    assert recovery_records[0].phase == BatchPhase.RECOVERY_DESIRED
    assert recovery_records[0].reason == "filesystem mutation failed"


def test_before_snapshot_failure_records_recovery_without_mutating(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    write_calls: list[Path | str] = []

    def fail_snapshot(path: Path | str) -> ResourceSnapshot:
        raise UnsafePathError("injected before snapshot failure")

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        write_calls.append(path)

    with pytest.raises(UnsafePathError, match="injected before snapshot failure"):
        _coordinator(
            state_path,
            journal=journal,
            snapshot=fail_snapshot,
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

    assert write_calls == []
    assert target.read_text(encoding="utf-8") == "old\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem before-snapshot failed"
    assert batch.status_payload["failure"]["stage"] == "before_snapshot"
    assert journal.list_operations(batch.batch_id)[0].operation_type == "write_text"
    assert journal.list_checkpoints(batch.batch_id) == []
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert len(recovery_records) == 1
    assert recovery_records[0].reason == "filesystem before-snapshot failed"
    assert recovery_records[0].payload["failure"]["stage"] == "before_snapshot"


def test_before_snapshot_failure_falls_back_to_lease_lost_recovery_desired(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=first_now)
    expired_now = first_now + timedelta(seconds=31)

    class LeaseLosingJournal(OperationJournalStore):
        def mark_failed(self, *args, **kwargs):
            LeaseStore(state_path).acquire("workspace", owner="runner-b", ttl=timedelta(seconds=30), now=expired_now)
            return super().mark_failed(*args, **kwargs)

    journal = LeaseLosingJournal(state_path)

    def fail_snapshot(path: Path | str) -> ResourceSnapshot:
        raise UnsafePathError("injected before snapshot failure")

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        raise AssertionError("write should not be attempted after snapshot failure")

    with pytest.raises(UnsafePathError, match="injected before snapshot failure"):
        _coordinator(
            state_path,
            journal=journal,
            snapshot=fail_snapshot,
            write_text_operation=record_write,
        ).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=first_now,
        )

    assert target.read_text(encoding="utf-8") == "old\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem before-snapshot failed"
    assert batch.status_payload["failure"]["stage"] == "before_snapshot"
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert len(recovery_records) == 1
    assert recovery_records[0].reason == "filesystem before-snapshot failed"
    assert recovery_records[0].phase == BatchPhase.RECOVERY_DESIRED
    assert recovery_records[0].payload["failure"]["stage"] == "before_snapshot"


def test_after_snapshot_failure_after_mutation_records_recovery_intent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    snapshot_calls = 0

    def fail_after_mutation_snapshot(path: Path | str) -> ResourceSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls > 1:
            raise UnsafePathError("injected after snapshot failure")
        return _portable_snapshot(path)

    def write_and_change_file(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)

    with pytest.raises(UnsafePathError, match="injected after snapshot failure"):
        _coordinator(
            state_path,
            journal=journal,
            snapshot=fail_after_mutation_snapshot,
            write_text_operation=write_and_change_file,
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

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem post-mutation journal failed"
    assert batch.status_payload["failure"]["stage"] == "after_snapshot"
    assert batch.status_payload["failure"]["snapshot_error"]["error"] == "injected after snapshot failure"
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
    ]
    recovery_records = journal.list_recovery_records(batch.batch_id)
    assert len(recovery_records) == 1
    assert recovery_records[0].reason == "filesystem post-mutation journal failed"
    assert recovery_records[0].payload["before"]["content_hash"] != recovery_records[0].payload["desired"]["sha256"]


def test_after_checkpoint_failure_after_mutation_records_recovery_intent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)

    class FailingAfterCheckpointJournal(OperationJournalStore):
        def record_checkpoint(self, *args, **kwargs):
            if kwargs.get("checkpoint_type") == "after":
                raise RuntimeError("injected after checkpoint failure")
            return super().record_checkpoint(*args, **kwargs)

    journal = FailingAfterCheckpointJournal(state_path)

    with pytest.raises(RuntimeError, match="injected after checkpoint failure"):
        _coordinator(state_path, journal=journal).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=now,
        )

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_payload["failure"]["stage"] == "after_checkpoint"
    assert batch.status_payload["failure"]["after"]["content_hash"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
    ]


def test_mark_succeeded_failure_after_mutation_records_recovery_intent(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)

    class FailingSuccessJournal(OperationJournalStore):
        def mark_succeeded(self, *args, **kwargs):
            raise RuntimeError("injected mark_succeeded failure")

    journal = FailingSuccessJournal(state_path)

    with pytest.raises(RuntimeError, match="injected mark_succeeded failure"):
        _coordinator(state_path, journal=journal).write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="write:config",
            now=now,
        )

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_payload["failure"]["stage"] == "mark_succeeded"
    assert [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)] == [
        "before",
        "backup_artifact_intent",
        "backup",
        "expected_after",
        "after",
    ]


@pytest.mark.parametrize(
    ("stage", "first_times"),
    [
        ("after_snapshot", 8),
        ("after_checkpoint", 8),
        ("mark_succeeded", 9),
    ],
)
def test_post_mutation_lease_loss_records_takeover_needed_state(
    tmp_path: Path,
    stage: str,
    first_times: int,
) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    expired_now = first_now + timedelta(seconds=31)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=first_now)
    journal = OperationJournalStore(state_path)
    clock = _sequence_clock([first_now] * first_times + [expired_now])
    snapshot_calls = 0

    def snapshot_with_optional_after_failure(path: Path | str) -> ResourceSnapshot:
        nonlocal snapshot_calls
        snapshot_calls += 1
        if stage == "after_snapshot" and snapshot_calls == 2:
            raise UnsafePathError("injected after snapshot failure")
        return _portable_snapshot(path)

    coordinator = _coordinator(state_path, journal=journal, snapshot=snapshot_with_optional_after_failure, clock=clock)

    expected_error = UnsafePathError if stage == "after_snapshot" else LeaseLostError
    with pytest.raises(expected_error):
        coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key=f"write:config:{stage}",
        )

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = journal.list_batches()[0]
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    assert batch.status_message == "filesystem post-mutation journal failed"
    assert batch.status_payload["failure"]["stage"] == stage
    assert [record.phase for record in journal.list_recovery_records(batch.batch_id)] == [BatchPhase.RECOVERY_DESIRED]
    checkpoint_types = [checkpoint.checkpoint_type for checkpoint in journal.list_checkpoints(batch.batch_id)]
    if stage == "mark_succeeded":
        assert checkpoint_types == ["before", "backup_artifact_intent", "backup", "expected_after", "after"]
    else:
        assert checkpoint_types == ["before", "backup_artifact_intent", "backup", "expected_after"]


def test_same_lease_retry_after_before_snapshot_failure_preserves_recovery_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    calls: list[str] = []

    def fail_snapshot(path: Path | str) -> ResourceSnapshot:
        calls.append("snapshot")
        raise UnsafePathError("injected before snapshot failure")

    def record_write(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        calls.append("write")

    coordinator = _coordinator(
        state_path,
        journal=journal,
        snapshot=fail_snapshot,
        write_text_operation=record_write,
    )
    with pytest.raises(UnsafePathError, match="injected before snapshot failure"):
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
    batch = journal.list_batches()[0]
    recovery_before = journal.list_recovery_records(batch.batch_id)

    with pytest.raises(JournaledFilesystemBatchStateError, match="cannot start"):
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

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert journal.list_recovery_records(batch.batch_id) == recovery_before
    assert calls == ["snapshot"]
    assert target.read_text(encoding="utf-8") == "old\n"
