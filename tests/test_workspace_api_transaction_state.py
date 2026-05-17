from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from workspace_api_helpers import workspace_with_portable_ops

from safe_fs_ops import SafeWorkspaceError
from safe_fs_ops.workspace_state.models import ClaimRecord, LeaseRecord

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def test_transaction_recursive_mkdir_concurrency_reuses_one_implicit_operation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    delayed_store = _DelayedCreateJournalStore(workspace.journal_store)
    workspace._journal_store = delayed_store
    workspace._coordinator._journal_store = delayed_store
    resources = workspace.resources(
        {
            "cache_a": workspace.directory(tmp_path / "config" / "a" / "cache"),
            "cache_b": workspace.directory(tmp_path / "data" / "b" / "cache"),
        }
    )
    failures: list[BaseException] = []

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        threads = [
            threading.Thread(
                target=_run_recursive_mkdir,
                args=(tx, tx.r.cache_a, "mkdir:cache-a", failures),
            ),
            threading.Thread(
                target=_run_recursive_mkdir,
                args=(tx, tx.r.cache_b, "mkdir:cache-b", failures),
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert tx.operation_run is not None
        assert tx.operation_phase is not None

    assert delayed_store.create_operation_run_calls == 1
    assert delayed_store.create_operation_phase_calls == 1
    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner=workspace.owner)
    assert len(operation_runs) == 1
    assert workspace.journal_store.list_operation_phases(operation_runs[0].operation_run_id) == [tx.operation_phase]
    assert {(batch.operation_run_id, batch.operation_phase_id) for batch in workspace.journal_store.list_batches()} == {
        (tx.operation_run.operation_run_id, tx.operation_phase.operation_phase_id)
    }
    assert all(workspace.claim_store.get(resource.resource_key) is None for resource in resources.sorted)
    lease = workspace.lease_store.acquire(workspace.lease_name, owner=workspace.owner, ttl=timedelta(seconds=30))
    assert lease.acquired
    workspace.lease_store.release(lease)


def test_transaction_is_single_use_after_exit_and_ignores_duplicate_exit(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})
    tx = workspace.transaction(name="apply", resources=resources, run_id="run-1")

    tx.__enter__()
    tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert tx.__exit__(None, None, None) is False
    assert tx.__exit__(None, None, None) is False

    with pytest.raises(SafeWorkspaceError, match="transaction is single-use"), tx:
        pass

    with pytest.raises(SafeWorkspaceError, match="transaction is not active"):
        tx.make_directory(tx.r.state_dir, idempotency_key="mkdir:after-exit")


def test_transaction_concurrent_enter_serializes_lifecycle(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    workspace._lease_store = _PermissiveLeaseStore()
    workspace._claim_store = _TrackingClaimStore()
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})
    tx = workspace.transaction(name="apply", resources=resources, run_id="run-1")
    entered_count = 0
    errors: list[BaseException] = []
    result_lock = threading.Lock()

    def enter_transaction() -> None:
        nonlocal entered_count
        try:
            tx.__enter__()
        except BaseException as exc:
            with result_lock:
                errors.append(exc)
            return
        with result_lock:
            entered_count += 1

    threads = [threading.Thread(target=enter_transaction), threading.Thread(target=enter_transaction)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert entered_count == 1
    assert len(errors) == 1
    assert isinstance(errors[0], SafeWorkspaceError)
    assert "transaction is already active" in str(errors[0])
    assert workspace.lease_store.acquire_calls == 1

    assert tx.__exit__(None, None, None) is False
    assert tx.__exit__(None, None, None) is False


def test_transaction_claim_directory_resource_deduplicates_concurrent_state_updates(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    claim_store = _TrackingClaimStore()
    workspace._claim_store = claim_store
    resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "state")})
    tx = workspace.transaction(name="apply", resources=resources, run_id="run-1")

    tx.__enter__()
    try:
        dynamic_resource_key = "directory:dynamic/cache"
        claims: list[ClaimRecord] = []
        failures: list[BaseException] = []
        result_lock = threading.Lock()

        def claim_directory() -> None:
            try:
                claim = tx._claim_directory_resource(dynamic_resource_key)
            except BaseException as exc:
                with result_lock:
                    failures.append(exc)
                return
            with result_lock:
                claims.append(claim)

        threads = [threading.Thread(target=claim_directory), threading.Thread(target=claim_directory)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert failures == []
        assert len(claims) == 2
        assert claims[0] == claims[1]
        assert claim_store.upsert_calls_by_resource_key[dynamic_resource_key] == 1
        assert tx._claim_release_order.count(dynamic_resource_key) == 1
        assert tx._claims_by_resource_key[dynamic_resource_key] == claims[0]
    finally:
        assert tx.__exit__(None, None, None) is False


def test_transaction_default_operations_link_to_existing_implicit_operation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    ready_parent = tmp_path / "ready"
    ready_parent.mkdir()
    resources = workspace.resources(
        {
            "state_dir": workspace.directory(tmp_path / "config" / "state" / "cache"),
            "config": workspace.file(tmp_path / "config.txt"),
            "ready_dir": workspace.directory(ready_parent / "child"),
        }
    )

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        tx.write_text(tx.r.config, "value = 1\n", idempotency_key="write:config")
        tx.delete_file(tx.r.config, idempotency_key="delete:config")
        tx.make_directory(tx.r.ready_dir, idempotency_key="mkdir:ready")
        operation_run = tx.operation_run
        operation_phase = tx.operation_phase

    assert operation_run is not None
    assert operation_phase is not None
    batches_by_key = {batch.idempotency_key: batch for batch in workspace.journal_store.list_batches()}
    assert (batches_by_key["write:config"].operation_run_id, batches_by_key["write:config"].operation_phase_id) == (
        operation_run.operation_run_id,
        operation_phase.operation_phase_id,
    )
    assert (batches_by_key["delete:config"].operation_run_id, batches_by_key["delete:config"].operation_phase_id) == (
        operation_run.operation_run_id,
        operation_phase.operation_phase_id,
    )
    assert (batches_by_key["mkdir:ready"].operation_run_id, batches_by_key["mkdir:ready"].operation_phase_id) == (
        operation_run.operation_run_id,
        operation_phase.operation_phase_id,
    )


def test_transactions_with_same_public_run_id_create_distinct_implicit_operation_runs(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    first_resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "first")})
    second_resources = workspace.resources({"state_dir": workspace.directory(tmp_path / "config" / "second")})

    with workspace.transaction(name="apply", resources=first_resources, run_id="run-1") as first_tx:
        first_tx.make_directory(first_tx.r.state_dir, parents=True, idempotency_key="mkdir:first")
        first_run = first_tx.operation_run

    with workspace.transaction(name="apply", resources=second_resources, run_id="run-1") as second_tx:
        second_tx.make_directory(second_tx.r.state_dir, parents=True, idempotency_key="mkdir:second")
        second_run = second_tx.operation_run

    assert first_run is not None
    assert second_run is not None
    operation_runs = workspace.journal_store.list_operation_runs(run_id="run-1", owner=workspace.owner)
    assert [record.operation_run_id for record in operation_runs] == [
        first_run.operation_run_id,
        second_run.operation_run_id,
    ]


def test_transaction_explicit_operation_links_override_existing_implicit_operation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources(
        {
            "state_dir": workspace.directory(tmp_path / "config" / "state" / "cache"),
            "config": workspace.file(tmp_path / "config.txt"),
        }
    )

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        implicit_run = tx.operation_run
        implicit_phase = tx.operation_phase
        assert implicit_run is not None
        assert implicit_phase is not None

        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=implicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=2,
            payload={"explicit": True},
            operation_phase_id="operation-phase-explicit",
            now=tx.now,
        )

        tx.write_text(
            tx.r.config,
            "value = 1\n",
            idempotency_key="write:explicit",
            operation_run_id=implicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    batches_by_key = {batch.idempotency_key: batch for batch in workspace.journal_store.list_batches()}
    assert (batches_by_key["write:explicit"].operation_run_id, batches_by_key["write:explicit"].operation_phase_id) == (
        implicit_run.operation_run_id,
        explicit_phase.operation_phase_id,
    )
    recursive_batch_links = {
        (batch.operation_run_id, batch.operation_phase_id)
        for key, batch in batches_by_key.items()
        if key.startswith("mkdir:state:recursive:") or key == "mkdir:state"
    }
    assert recursive_batch_links == {(implicit_run.operation_run_id, implicit_phase.operation_phase_id)}


def _run_recursive_mkdir(
    tx: object,
    resource: object,
    idempotency_key: str,
    failures: list[BaseException],
) -> None:
    try:
        tx.make_directory(resource, parents=True, idempotency_key=idempotency_key)  # type: ignore[call-arg]
    except BaseException as exc:
        failures.append(exc)


class _DelayedCreateJournalStore:
    def __init__(self, delegate: object, *, delay_seconds: float = 0.05) -> None:
        self._delegate = delegate
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.create_operation_run_calls = 0
        self.create_operation_phase_calls = 0

    def create_operation_run(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            self.create_operation_run_calls += 1
        time.sleep(self._delay_seconds)
        return self._delegate.create_operation_run(*args, **kwargs)

    def create_operation_phase(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            self.create_operation_phase_calls += 1
        time.sleep(self._delay_seconds)
        return self._delegate.create_operation_phase(*args, **kwargs)

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)


class _PermissiveLeaseStore:
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(
        self,
        name: str,
        *,
        owner: str,
        ttl: timedelta,
        token: str | None = None,
        now: datetime | None = None,
    ) -> LeaseRecord:
        del ttl, token
        with self._lock:
            self.acquire_calls += 1
            acquire_calls = self.acquire_calls
        time.sleep(self._delay_seconds)
        acquired_at = _normalized_now(now)
        return LeaseRecord(
            name=name,
            owner=owner,
            token=f"token-{acquire_calls}",
            fencing_token=acquire_calls,
            acquired_at=acquired_at,
            heartbeat_at=acquired_at,
            expires_at=acquired_at + timedelta(minutes=5),
            acquired=True,
        )

    def release(self, lease: LeaseRecord, *, now: datetime | None = None) -> bool:
        del lease, now
        with self._lock:
            self.release_calls += 1
        return True


class _TrackingClaimStore:
    def __init__(self, *, delay_seconds: float = 0.05) -> None:
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._claims_by_resource_key: dict[str, ClaimRecord] = {}
        self.upsert_calls_by_resource_key: dict[str, int] = {}

    def upsert(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        details: str | None = None,
        now: datetime | None = None,
    ) -> ClaimRecord:
        del lease
        time.sleep(self._delay_seconds)
        with self._lock:
            self.upsert_calls_by_resource_key[resource_key] = self.upsert_calls_by_resource_key.get(resource_key, 0) + 1
            claim = self._claims_by_resource_key.get(resource_key)
            if claim is None:
                timestamp = _normalized_now(now)
                claim = ClaimRecord(
                    resource_key=resource_key,
                    owner=owner,
                    scope=scope,
                    details=details,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                self._claims_by_resource_key[resource_key] = claim
            return claim

    def release(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        expected_claim: ClaimRecord | None = None,
        now: datetime | None = None,
    ) -> bool:
        del lease, owner, scope, expected_claim, now
        with self._lock:
            return self._claims_by_resource_key.pop(resource_key, None) is not None

    def release_if_owner_scope_matches(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        expected_claim: ClaimRecord | None = None,
        now: datetime | None = None,
    ) -> bool:
        return self.release(
            resource_key,
            lease=lease,
            owner=owner,
            scope=scope,
            expected_claim=expected_claim,
            now=now,
        )

    def get(self, resource_key: str) -> ClaimRecord | None:
        with self._lock:
            return self._claims_by_resource_key.get(resource_key)


def _normalized_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)
