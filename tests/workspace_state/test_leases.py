from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.workspace_state import LeaseLostError, LeaseRecord, LeaseStore, maintain_lease

pytestmark = pytest.mark.safe_fs_ops


def test_lease_store_acquires_fresh_lease(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)

    lease = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)

    assert lease.acquired is True
    assert lease.owner == "owner-a"
    assert lease.token
    assert lease.fencing_token == 1
    assert lease.heartbeat_at == now
    assert store.is_current(lease, now=now + timedelta(seconds=1)) is True


def test_lease_store_reports_active_competing_lease(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)

    second = store.acquire(
        "workspace",
        owner="owner-b",
        ttl=timedelta(seconds=30),
        now=now + timedelta(seconds=1),
    )

    assert second.acquired is False
    assert second.owner == first.owner
    assert second.token == ""
    assert second.fencing_token == 0
    assert store.heartbeat(second, ttl=timedelta(seconds=30), now=now + timedelta(seconds=2)) is None
    assert store.release(second, now=now + timedelta(seconds=2)) is False
    assert store.is_current(first, now=now + timedelta(seconds=2)) is True


def test_lease_store_active_report_does_not_expose_authority(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)

    active = store.active("workspace", now=now + timedelta(seconds=1))

    assert active is not None
    assert active.acquired is False
    assert active.owner == first.owner
    assert active.token == ""
    assert active.fencing_token == 0
    assert store.heartbeat(active, ttl=timedelta(seconds=30), now=now + timedelta(seconds=2)) is None


def test_lease_store_stale_takeover_advances_fencing_token(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=31)
    first = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=first_now)

    second = store.acquire("workspace", owner="owner-b", ttl=timedelta(seconds=30), now=second_now)

    assert second.acquired is True
    assert second.owner == "owner-b"
    assert second.fencing_token == first.fencing_token + 1
    assert store.is_current(first, now=second_now) is False
    assert store.is_current(second, now=second_now) is True


def test_lease_store_rejects_stale_holder_after_takeover(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=31)
    first = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=first_now)
    store.acquire("workspace", owner="owner-b", ttl=timedelta(seconds=30), now=second_now)

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.require_current(first, now=second_now)


def test_lease_store_heartbeat_extends_current_lease(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)

    refreshed = store.heartbeat(lease, ttl=timedelta(seconds=30), now=now + timedelta(seconds=10))

    assert refreshed is not None
    assert refreshed.fencing_token == lease.fencing_token
    assert refreshed.heartbeat_at == now + timedelta(seconds=10)
    assert refreshed.expires_at == now + timedelta(seconds=40)


def test_lease_heartbeat_can_be_disabled_for_logical_time(tmp_path: Path) -> None:
    class RecordingLeaseStore:
        def __init__(self, delegate: LeaseStore) -> None:
            self._delegate = delegate
            self.heartbeat_times: list[datetime | None] = []

        def heartbeat(self, lease: LeaseRecord, *, ttl: timedelta, now: datetime | None = None) -> LeaseRecord | None:
            self.heartbeat_times.append(now)
            return self._delegate.heartbeat(lease, ttl=ttl, now=now)

        def __getattr__(self, name: str) -> object:
            return getattr(self._delegate, name)

    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    recording_store = RecordingLeaseStore(store)

    with maintain_lease(
        recording_store,  # type: ignore[arg-type]
        lease,
        ttl=timedelta(seconds=30),
        now=lambda: now,
        enabled=False,
    ):
        pass

    assert recording_store.heartbeat_times == []


def test_lease_store_heartbeat_requires_matching_token_and_fencing_token(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    stale = LeaseRecord(
        name=lease.name,
        owner=lease.owner,
        token=lease.token,
        fencing_token=lease.fencing_token + 1,
        acquired_at=lease.acquired_at,
        heartbeat_at=lease.heartbeat_at,
        expires_at=lease.expires_at,
        acquired=True,
    )

    assert store.heartbeat(stale, ttl=timedelta(seconds=30), now=now + timedelta(seconds=1)) is None


def test_lease_store_release_requires_matching_token_and_fencing_token(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)
    stale = LeaseRecord(
        name=lease.name,
        owner=lease.owner,
        token=lease.token,
        fencing_token=lease.fencing_token + 1,
        acquired_at=lease.acquired_at,
        heartbeat_at=lease.heartbeat_at,
        expires_at=lease.expires_at,
        acquired=True,
    )

    assert store.release(stale, now=now + timedelta(seconds=1)) is False
    assert store.is_current(lease, now=now) is True
    assert store.release(lease, now=now + timedelta(seconds=1)) is True
    assert store.is_current(lease, now=now + timedelta(seconds=1)) is False


def test_lease_store_reacquire_after_release_advances_fencing_token(tmp_path: Path) -> None:
    store = LeaseStore(tmp_path / "state.db")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = store.acquire("workspace", owner="owner-a", ttl=timedelta(seconds=30), now=now)

    assert store.release(first, now=now + timedelta(seconds=1)) is True
    second = store.acquire(
        "workspace",
        owner="owner-b",
        ttl=timedelta(seconds=30),
        now=now + timedelta(seconds=1),
    )

    assert second.acquired is True
    assert second.fencing_token == first.fencing_token + 1
    assert store.is_current(first, now=now + timedelta(seconds=1)) is False
    assert store.is_current(second, now=now + timedelta(seconds=1)) is True
