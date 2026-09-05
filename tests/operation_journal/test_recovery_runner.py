from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from recovery_runner_helpers import (
    _acquire_and_claim,
    _coordinator,
    _current_recovery_attempt_id,
    _fail_handler,
    _manual_handler,
    _skip_handler,
    _start_recovering_batch,
    _success_handler,
)

from safe_fs_ops.operation_journal import (
    JournaledFilesystemCoordinator,
    JournaledFilesystemRecoveryError,
    OperationJournalStore,
)
from safe_fs_ops.operation_journal.filesystem import _bool_from_payload_value
from safe_fs_ops.operation_journal.filesystem_support import file_resource_key
from safe_fs_ops.operation_journal.models import require_recovery_action_authority
from safe_fs_ops.workspace_state import ClaimStore, LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


class _RecordingLeaseStore:
    def __init__(self, delegate: LeaseStore) -> None:
        self._delegate = delegate
        self.heartbeat_times: list[datetime | None] = []

    def heartbeat(self, lease: LeaseRecord, *, ttl: timedelta, now: datetime | None = None) -> LeaseRecord | None:
        self.heartbeat_times.append(now)
        return self._delegate.heartbeat(lease, ttl=ttl, now=now)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


@pytest.mark.parametrize("string_boolean", ["false", "0"])
def test_recovery_runner_rejects_string_boolean_payload_fields(string_boolean: str) -> None:
    with pytest.raises(ValueError, match="boolean"):
        _bool_from_payload_value(string_boolean, default=False)


def test_recovery_runner_records_success_and_skipped_actions_portably(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={
            "custom_success": _success_handler,
            "custom_skip": _skip_handler,
        },
    )

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="custom_success",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-success",
        now=now + timedelta(seconds=1),
    )
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="custom_skip",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-skip",
        now=now + timedelta(seconds=2),
    )

    context = journal.read_recovery_context(batch.batch_id)
    results = coordinator.run_recovery_actions(context, lease=lease, now=now + timedelta(seconds=3))

    assert [record.status for record in results] == ["succeeded", "skipped"]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "skipped",
    ]


def test_recovery_runner_rerun_uses_latest_terminal_action_state(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal, recovery_action_handlers={"noop": _success_handler})

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="noop",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-noop",
        now=now + timedelta(seconds=1),
    )

    stale_context = journal.read_recovery_context(batch.batch_id)
    first_results = coordinator.run_recovery_actions(stale_context, lease=lease, now=now + timedelta(seconds=2))
    second_results = coordinator.run_recovery_actions(stale_context, lease=lease, now=now + timedelta(seconds=3))

    assert [record.status for record in first_results] == ["succeeded"]
    assert [record.status for record in second_results] == ["succeeded"]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


def test_recovery_runner_completes_attempting_action_on_retry(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    recovery_attempts: list[str] = []

    def record_success(context, action) -> None:
        assert context.recovery_attempt_id is not None
        recovery_attempts.append(action.action_id)

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="noop",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-noop",
        now=now + timedelta(seconds=1),
    )
    journal.mark_recovery_action_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_id="action-noop",
        payload={"event_type": "attempting"},
        now=now + timedelta(seconds=2),
    )

    coordinator = _coordinator(state_path, journal=journal, recovery_action_handlers={"noop": record_success})
    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=3),
    )

    assert recovery_attempts == ["action-noop"]
    assert [record.status for record in results] == ["succeeded"]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


def test_recovery_runner_runs_handler_after_attempting_commit_and_outside_transaction(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    observed_action_statuses: list[list[str]] = []
    observed_immediate_transactions: list[bool] = []

    def record_handler_state(_context, _action) -> None:
        observed_action_statuses.append(
            [record.status for record in OperationJournalStore(state_path).list_recovery_actions("batch-1")]
        )
        observed_immediate_transactions.append(_can_begin_immediate_transaction(state_path))
        target.write_text("restored\n", encoding="utf-8")

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="record_state",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-record-state",
        now=now + timedelta(seconds=1),
    )

    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"record_state": record_handler_state},
    )
    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=2),
    )

    assert observed_action_statuses == [["planned", "attempting"]]
    assert observed_immediate_transactions == [True]
    assert target.read_text(encoding="utf-8") == "restored\n"
    assert [record.status for record in results] == ["succeeded"]
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


def test_recovery_runner_records_failure_after_handler_runs_outside_transaction(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    observed_action_statuses: list[list[str]] = []
    observed_immediate_transactions: list[bool] = []

    def fail_after_recording_state(_context, _action) -> None:
        observed_action_statuses.append(
            [record.status for record in OperationJournalStore(state_path).list_recovery_actions("batch-1")]
        )
        observed_immediate_transactions.append(_can_begin_immediate_transaction(state_path))
        target.write_text("partially-restored\n", encoding="utf-8")
        raise RuntimeError("handler failed after side effect")

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="fail_after_recording_state",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-fail-after-recording-state",
        now=now + timedelta(seconds=1),
    )

    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"fail_after_recording_state": fail_after_recording_state},
    )
    with pytest.raises(JournaledFilesystemRecoveryError, match="action-fail-after-recording-state.*failed") as raised:
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "handler failed after side effect"
    assert observed_action_statuses == [["planned", "attempting"]]
    assert observed_immediate_transactions == [True]
    assert target.read_text(encoding="utf-8") == "partially-restored\n"
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "failed",
    ]


def test_recovery_handler_authority_check_stops_after_lease_takeover(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    first_side_effect = tmp_path / "first.txt"
    second_side_effect = tmp_path / "second.txt"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    takeover_now = now + timedelta(seconds=31)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)

    def stop_after_takeover(context, _action) -> None:
        first_side_effect.write_text("first\n", encoding="utf-8")
        LeaseStore(state_path).acquire("workspace", owner="runner-b", ttl=timedelta(seconds=30), now=takeover_now)
        context.require_recovery_authority()
        second_side_effect.write_text("second\n", encoding="utf-8")

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="stop_after_takeover",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-stop-after-takeover",
        now=now + timedelta(seconds=1),
    )
    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"stop_after_takeover": stop_after_takeover},
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="no longer active") as raised:
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    assert isinstance(raised.value.__cause__, LeaseLostError)
    assert first_side_effect.read_text(encoding="utf-8") == "first\n"
    assert second_side_effect.exists() is False
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
    ]


def test_recovery_action_authority_live_check_stops_after_action_terminalized(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    side_effect = tmp_path / "side-effect.txt"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)

    def terminalize_then_try_to_mutate(context, action) -> None:
        journal.record_recovery_action_skipped(
            batch_id=batch.batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            action_id=action.action_id,
            reason="terminalized elsewhere",
            payload={"event_type": "skipped"},
            now=now + timedelta(seconds=3),
        )
        require_recovery_action_authority(context, action)
        side_effect.write_text("mutated\n", encoding="utf-8")

    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="terminalize_then_try_to_mutate",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-terminalized-before-mutation",
        now=now + timedelta(seconds=1),
    )
    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"terminalize_then_try_to_mutate": terminalize_then_try_to_mutate},
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="no longer active"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    assert side_effect.exists() is False
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "skipped",
    ]


def test_recovery_runner_heartbeats_lease_during_long_handler(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime.now(UTC)
    resource_key = file_resource_key(target)
    lease_store = LeaseStore(state_path)
    claim_store = ClaimStore(state_path)
    lease = lease_store.acquire(
        "workspace",
        owner="runner-a",
        ttl=timedelta(seconds=30),
        now=now,
    )
    assert lease.acquired
    claim_store.upsert(resource_key, lease=lease, owner="owner-a", now=now)
    journal = OperationJournalStore(state_path)

    def slow_handler(_context, _action) -> None:
        time.sleep(6)
        target.write_text("restored\n", encoding="utf-8")

    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="slow_handler",
        resource_key=resource_key,
        payload={"path": str(target)},
        action_id="action-slow-handler",
        now=now + timedelta(milliseconds=1),
    )

    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"slow_handler": slow_handler},
    )
    context = journal.read_recovery_context(batch.batch_id)
    # Setup does not exercise the heartbeat and can be slow on native Windows.
    # Start the short deadline only now; the handler still outlasts its TTL.
    lease = lease_store.heartbeat(lease, ttl=timedelta(seconds=5))
    assert lease is not None
    results = coordinator.run_recovery_actions(
        context,
        lease=lease,
    )

    assert [record.status for record in results] == ["succeeded"]
    assert lease_store.is_current(lease, now=datetime.now(UTC))
    assert target.read_text(encoding="utf-8") == "restored\n"


def test_recovery_runner_with_explicit_logical_now_does_not_heartbeat_lease(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = file_resource_key(target)
    lease_store = _RecordingLeaseStore(LeaseStore(state_path))
    lease = lease_store.acquire("workspace", owner="runner-a", ttl=timedelta(seconds=30), now=now)
    ClaimStore(state_path).upsert(resource_key, lease=lease, owner="owner-a", now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="restore",
        resource_key=resource_key,
        payload={"path": str(target)},
        action_id="action-restore",
        now=now + timedelta(milliseconds=1),
    )

    coordinator = JournaledFilesystemCoordinator(
        lease_store=lease_store,  # type: ignore[arg-type]
        claim_store=ClaimStore(state_path),
        journal_store=journal,
        recovery_action_handlers={"restore": lambda _context, _action: target.write_text("restored\n")},
    )

    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=1),
    )

    assert [record.status for record in results] == ["succeeded"]
    assert lease_store.heartbeat_times == []


def test_recovery_runner_retry_executes_with_original_planned_payload(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    seen_payloads: list[object] = []

    def record_payload(_context, action) -> None:
        seen_payloads.append(action.payload.get("path"))

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="noop",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-noop",
        now=now + timedelta(seconds=1),
    )
    journal.mark_recovery_action_attempting(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_id="action-noop",
        payload={"event_type": "attempting"},
        now=now + timedelta(seconds=2),
    )

    coordinator = _coordinator(state_path, journal=journal, recovery_action_handlers={"noop": record_payload})
    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch.batch_id),
        lease=lease,
        now=now + timedelta(seconds=3),
    )

    assert seen_payloads == [str(target)]
    assert [record.status for record in results] == ["succeeded"]


def _can_begin_immediate_transaction(state_path: Path) -> bool:
    connection = sqlite3.connect(state_path, timeout=0)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.commit()
    except sqlite3.OperationalError:
        connection.rollback()
        return False
    finally:
        connection.close()
    return True


def test_recovery_runner_does_not_invoke_handler_after_attempt_is_completed_mid_action(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)

    class CompletingAttemptJournal(OperationJournalStore):
        def mark_recovery_action_attempting(self, batch_id: str, **kwargs):
            record = super().mark_recovery_action_attempting(batch_id=batch_id, **kwargs)
            if kwargs.get("action_id") == "action-restore":
                self.record_recovery_succeeded(
                    batch_id,
                    lease=kwargs["lease"],
                    recovery_attempt_id=kwargs["recovery_attempt_id"],
                    reason="completed by another same-lease actor",
                    payload={"event_type": "completed_elsewhere"},
                    now=kwargs.get("now"),
                )
            return record

    journal = CompletingAttemptJournal(state_path)
    handler_calls: list[str] = []

    def record_handler(_context, action) -> None:
        handler_calls.append(action.action_id)

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="record",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-restore",
        now=now + timedelta(seconds=1),
    )

    coordinator = _coordinator(state_path, journal=journal, recovery_action_handlers={"record": record_handler})
    context = journal.read_recovery_context(batch.batch_id)

    with pytest.raises(JournaledFilesystemRecoveryError, match="no longer active"):
        coordinator.run_recovery_actions(context, lease=lease, now=now + timedelta(seconds=2))

    assert handler_calls == []
    assert journal.get_batch(batch.batch_id).phase == "recovery_succeeded"  # type: ignore[union-attr]


def test_recovery_runner_records_failed_action_rows(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"explode": _fail_handler},
    )

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="explode",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-fail",
        now=now + timedelta(seconds=1),
    )

    context = journal.read_recovery_context(batch.batch_id)
    with pytest.raises(JournaledFilesystemRecoveryError, match="action-fail.*failed"):
        coordinator.run_recovery_actions(context, lease=lease, now=now + timedelta(seconds=2))

    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "failed",
    ]


def test_recovery_runner_records_manual_intervention_required_action_rows(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, file_resource_key(target), now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={"needs_manual": _manual_handler},
    )

    batch = _start_recovering_batch(journal, lease=lease, resource_key=file_resource_key(target), now=now)
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type="needs_manual",
        resource_key=file_resource_key(target),
        payload={"path": str(target)},
        action_id="action-manual",
        now=now + timedelta(seconds=1),
    )

    context = journal.read_recovery_context(batch.batch_id)
    with pytest.raises(JournaledFilesystemRecoveryError, match="manual intervention"):
        coordinator.run_recovery_actions(context, lease=lease, now=now + timedelta(seconds=2))

    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
