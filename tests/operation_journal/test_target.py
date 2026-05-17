from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from safe_fs_ops.operation_journal import FileStateCommit, JournalTarget, TargetJournal

pytestmark = pytest.mark.safe_fs_ops


@dataclass(slots=True)
class RunRecord:
    id: int
    command: str


@dataclass(slots=True)
class TargetAttemptRecord:
    id: int
    run_id: int
    target_type: str
    target_id: str | None
    subject: str
    address: str
    owner_id: str
    phase: str
    error: str | None = None


@dataclass(slots=True)
class BaselineRecord:
    run_id: int
    path: Path
    content_text: str
    format: str | None
    original_exists: bool


@dataclass(slots=True)
class CheckpointRecord:
    run_id: int
    path: Path
    content_text: str
    format: str | None
    original_exists: bool


@dataclass(slots=True)
class SnapshotRecord:
    run_id: int
    path: Path
    content_hash: bytes
    size: int
    mtime_ns: int | None
    format: str | None
    spec_hash: bytes | None


@dataclass(slots=True)
class ChangeBatchRecord:
    run_id: int
    path: Path
    operations: list[dict[str, Any]]
    original_exists: bool
    format: str | None


@dataclass(slots=True)
class EventRecord:
    run_id: int
    event_type: str
    path: Path | None
    changed: bool
    summary: str | None


class InMemoryTargetStateStore:
    def __init__(self) -> None:
        self._next_run_id = 1
        self._next_attempt_id = 1
        self._runs: list[RunRecord] = []
        self._attempts: list[TargetAttemptRecord] = []
        self._baselines: list[BaselineRecord] = []
        self._checkpoints: list[CheckpointRecord] = []
        self._snapshots: list[SnapshotRecord] = []
        self._change_batches: list[ChangeBatchRecord] = []
        self._events: list[EventRecord] = []

    def initialize(self) -> None:
        return None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        snapshot = deepcopy(self.__dict__)
        try:
            yield None
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    def start_run(self, *, command: str) -> RunRecord:
        run = RunRecord(id=self._next_run_id, command=command)
        self._next_run_id += 1
        self._runs.append(run)
        return run

    def record_target_attempt(
        self,
        *,
        run_id: int,
        target_type: str,
        target_id: str | None,
        subject: str,
        address: str,
        owner_id: str,
        phase: str,
        connection: object | None = None,
    ) -> TargetAttemptRecord:
        attempt = TargetAttemptRecord(
            id=self._next_attempt_id,
            run_id=run_id,
            target_type=target_type,
            target_id=target_id,
            subject=subject,
            address=address,
            owner_id=owner_id,
            phase=phase,
        )
        self._next_attempt_id += 1
        self._attempts.append(attempt)
        return attempt

    def target_attempts(self, run_id: int) -> list[TargetAttemptRecord]:
        return [attempt for attempt in self._attempts if attempt.run_id == run_id]

    def update_target_attempt_ids(
        self,
        attempt_ids: set[int],
        *,
        phase: str,
        error: str | None = None,
        connection: object | None = None,
    ) -> None:
        for attempt in self._attempts:
            if attempt.id in attempt_ids:
                attempt.phase = phase
                attempt.error = error

    def record_baseline(
        self,
        *,
        run_id: int,
        path: Path,
        content_text: str,
        format: str | None,
        original_exists: bool,
        connection: object | None = None,
    ) -> BaselineRecord:
        baseline = BaselineRecord(run_id, path, content_text, format, original_exists)
        self._baselines.append(baseline)
        return baseline

    def original_baseline(self, path: Path, *, connection: object | None = None) -> BaselineRecord | None:
        return next((baseline for baseline in self._baselines if baseline.path == path), None)

    def record_checkpoint(
        self,
        *,
        run_id: int,
        path: Path,
        content_text: str,
        format: str | None,
        original_exists: bool,
        connection: object | None = None,
    ) -> CheckpointRecord:
        checkpoint = CheckpointRecord(run_id, path, content_text, format, original_exists)
        self._checkpoints.append(checkpoint)
        return checkpoint

    def latest_checkpoint(self, path: Path) -> CheckpointRecord | None:
        return next((checkpoint for checkpoint in reversed(self._checkpoints) if checkpoint.path == path), None)

    def record_snapshot(
        self,
        *,
        run_id: int,
        path: Path,
        content_hash: bytes,
        size: int,
        mtime_ns: int | None,
        format: str | None,
        spec_hash: bytes | None,
        connection: object | None = None,
    ) -> SnapshotRecord:
        snapshot = SnapshotRecord(run_id, path, content_hash, size, mtime_ns, format, spec_hash)
        self._snapshots.append(snapshot)
        return snapshot

    def latest_snapshot(self, path: Path) -> SnapshotRecord | None:
        return next((snapshot for snapshot in reversed(self._snapshots) if snapshot.path == path), None)

    def record_change_batch(
        self,
        *,
        run_id: int,
        path: Path,
        operations: list[dict[str, Any]],
        original_exists: bool,
        format: str | None,
        connection: object | None = None,
    ) -> ChangeBatchRecord:
        batch = ChangeBatchRecord(run_id, path, deepcopy(operations), original_exists, format)
        self._change_batches.append(batch)
        return batch

    def change_batches(self, path: Path) -> list[ChangeBatchRecord]:
        return [batch for batch in self._change_batches if batch.path == path]

    def record_event(
        self,
        *,
        run_id: int,
        event_type: str,
        path: Path | None,
        changed: bool,
        summary: str | None,
        connection: object | None = None,
    ) -> EventRecord:
        event = EventRecord(run_id, event_type, path, changed, summary)
        self._events.append(event)
        return event

    def latest_event(self, run_id: int) -> EventRecord | None:
        return next((event for event in reversed(self._events) if event.run_id == run_id), None)


class SnapshotFailingStore(InMemoryTargetStateStore):
    def record_snapshot(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("snapshot failed")


def _target(path: Path) -> JournalTarget:
    return JournalTarget(
        target_type="file",
        target_id="spec.toml#files[0]",
        subject=str(path),
        address="count",
        owner_id="spec.toml#files[0]",
    )


def _commit(path: Path) -> FileStateCommit:
    return FileStateCommit(
        path=path,
        content_text="count = 2\n",
        content_hash=b"new-hash",
        size=10,
        mtime_ns=123,
        format="toml",
        original_exists=True,
        original_text="count = 1\n",
        operations=[
            {
                "kind": "update",
                "key": "count",
                "value": 2,
                "before_value": 1,
            }
        ],
        event_type="applied",
        summary="applied with 1 ops",
        spec_hash=b"spec-hash",
    )


def _state_store(tmp_path: Path) -> InMemoryTargetStateStore:
    store = InMemoryTargetStateStore()
    store.initialize()
    return store


def test_target_journal_records_attempting_target(tmp_path: Path) -> None:
    state_store = _state_store(tmp_path)
    run = state_store.start_run(command="apply")
    path = tmp_path / "config.toml"

    attempt = TargetJournal(state_store).begin_attempt(run_id=run.id, target=_target(path))

    attempts = state_store.target_attempts(run.id)
    assert len(attempts) == 1
    assert attempts[0].id == attempt.id
    assert attempts[0].phase == "attempting"
    assert attempts[0].subject == str(path)


def test_target_journal_commits_file_success_in_one_transaction(tmp_path: Path) -> None:
    state_store = _state_store(tmp_path)
    run = state_store.start_run(command="apply")
    path = tmp_path / "config.toml"
    journal = TargetJournal(state_store)
    attempt = journal.begin_attempt(run_id=run.id, target=_target(path))

    journal.commit_file_success(attempt, _commit(path))

    assert state_store.target_attempts(run.id)[0].phase == "succeeded"
    assert state_store.original_baseline(path).content_text == "count = 1\n"
    assert state_store.latest_checkpoint(path).content_text == "count = 2\n"
    assert state_store.latest_snapshot(path).content_hash == b"new-hash"
    assert state_store.change_batches(path)[0].operations[0]["before_value"] == 1
    assert state_store.latest_event(run.id).event_type == "applied"


def test_file_state_commit_copies_operations_at_construction(tmp_path: Path) -> None:
    state_store = _state_store(tmp_path)
    run = state_store.start_run(command="apply")
    path = tmp_path / "config.toml"
    journal = TargetJournal(state_store)
    attempt = journal.begin_attempt(run_id=run.id, target=_target(path))
    operations = [{"kind": "update", "key": "count", "value": 2, "nested": {"before": 1}}]
    commit = FileStateCommit(
        path=path,
        content_text="count = 2\n",
        content_hash=b"new-hash",
        size=10,
        mtime_ns=123,
        format="toml",
        original_exists=True,
        original_text="count = 1\n",
        operations=operations,
        event_type="applied",
        summary="applied with 1 ops",
    )
    operations[0]["value"] = 3
    operations[0]["nested"]["before"] = 0  # type: ignore[index]

    journal.commit_file_success(attempt, commit)

    recorded = state_store.change_batches(path)[0].operations[0]
    assert recorded["value"] == 2
    assert recorded["nested"]["before"] == 1
    with pytest.raises(TypeError):
        commit.operations[0]["value"] = 4  # type: ignore[index]


def test_target_journal_file_success_rolls_back_partial_state_on_failure(tmp_path: Path) -> None:
    store = SnapshotFailingStore()
    store.initialize()
    run = store.start_run(command="apply")
    path = tmp_path / "config.toml"
    journal = TargetJournal(store)
    attempt = journal.begin_attempt(run_id=run.id, target=_target(path))

    with pytest.raises(RuntimeError, match="snapshot failed"):
        journal.commit_file_success(attempt, _commit(path))

    assert store.target_attempts(run.id)[0].phase == "attempting"
    assert store.original_baseline(path) is None
    assert store.latest_checkpoint(path) is None
    assert store.latest_snapshot(path) is None
    assert store.change_batches(path) == []
    assert store.latest_event(run.id) is None


def test_target_journal_commits_failure_in_one_transaction(tmp_path: Path) -> None:
    state_store = _state_store(tmp_path)
    run = state_store.start_run(command="rollback")
    path = tmp_path / "config.toml"
    journal = TargetJournal(state_store)
    attempt = journal.begin_attempt(run_id=run.id, target=_target(path))

    journal.commit_failure(attempt, error="write failed", path=path, changed=True)

    attempt_after = state_store.target_attempts(run.id)[0]
    assert attempt_after.phase == "failed"
    assert attempt_after.error == "write failed"
    event = state_store.latest_event(run.id)
    assert event.event_type == "error"
    assert event.changed is True
    assert event.summary == "write failed"
