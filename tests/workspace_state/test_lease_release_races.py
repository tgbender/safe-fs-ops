import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.leases import _release_lease_in_connection


@pytest.fixture(params=["sqlalchemy", "sqlite3"])
def store(tmp_path: Path, request: pytest.FixtureRequest) -> LeaseStore:
    path = tmp_path / "state.db"
    factory = sqlite3.connect if request.param == "sqlite3" else None
    return LeaseStore(path, sqlite_store=SqliteStore(path, connection_factory=factory))


def test_delayed_heartbeat_cannot_revive_released_lease(store: LeaseStore) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="a", ttl=timedelta(seconds=60), now=now)
    entered, resume = Event(), Event()

    class PausedStore(LeaseStore):
        def initialize(self) -> None:
            entered.set()
            if not resume.wait(10):
                raise TimeoutError("heartbeat was not resumed")
            super().initialize()

    worker = PausedStore(
        store.path,
        sqlite_store=SqliteStore(store.path, connection_factory=store.sqlite_store.connection_factory),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(worker.heartbeat, lease, ttl=timedelta(seconds=60), now=now + timedelta(seconds=1))
        try:
            assert entered.wait(10)
            assert store.release(lease, now=now + timedelta(seconds=2))
            assert store.active("workspace", now=now + timedelta(seconds=2)) is None
        finally:
            resume.set()
        assert pending.result(timeout=10) is None

    assert store.active("workspace", now=now + timedelta(seconds=3)) is None
    replacement = store.acquire("workspace", owner="b", ttl=timedelta(seconds=60), now=now + timedelta(seconds=3))
    assert replacement.acquired
    assert replacement.fencing_token == lease.fencing_token + 1


@pytest.mark.parametrize("release_in_transaction", [False, True])
def test_release_revokes_old_authority_even_with_earlier_timestamp(
    store: LeaseStore, release_in_transaction: bool
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="a", ttl=timedelta(seconds=60), now=now)
    released_at = now + timedelta(seconds=2)
    if release_in_transaction:
        with store.sqlite_store.transaction() as connection:
            assert _release_lease_in_connection(connection, lease, now=released_at)
    else:
        assert store.release(lease, now=released_at)

    earlier = now + timedelta(seconds=1)
    with pytest.raises(LeaseLostError):
        store.require_current(lease, now=earlier)
    assert store.release(lease, now=earlier) is False
    assert store.heartbeat(lease, ttl=timedelta(seconds=60), now=earlier) is None
    assert store.active("workspace", now=released_at) is None
    replacement = store.acquire("workspace", owner="b", ttl=timedelta(seconds=60), now=released_at)
    assert replacement.acquired
    assert replacement.fencing_token == lease.fencing_token + 1
