from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import TracebackType

from safe_fs_ops.workspace_state.leases import LeaseLostError, LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord


class LeaseHeartbeat:
    """Keep a lease current while a long filesystem side effect runs."""

    def __init__(
        self,
        lease_store: LeaseStore,
        lease: LeaseRecord,
        *,
        ttl: timedelta | None = None,
        now: Callable[[], datetime] | None = None,
        enabled: bool = True,
    ) -> None:
        self._lease_store = lease_store
        self._lease = lease
        self._ttl = ttl or (lease.expires_at - lease.heartbeat_at)
        self._now = now or (lambda: datetime.now(UTC))
        self._enabled = enabled
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> LeaseHeartbeat:
        if not self._enabled:
            return self
        if self._ttl.total_seconds() <= 0:
            return self
        if not hasattr(self._lease_store, "heartbeat"):
            return self
        self._heartbeat_once()
        self._thread = threading.Thread(
            target=self._run,
            name=f"safe-fs-ops-lease-heartbeat:{self._lease.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_seconds() * 2))
        if exc_type is None:
            self.raise_if_failed()

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds()):
            try:
                self._heartbeat_once()
            except BaseException as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return

    def _heartbeat_once(self) -> None:
        refreshed = self._lease_store.heartbeat(
            self._lease,
            ttl=self._ttl,
            now=self._now(),
        )
        if refreshed is None:
            raise LeaseLostError(f"lease {self._lease.name!r} is no longer current")

    def _interval_seconds(self) -> float:
        return max(0.01, min(5.0, self._ttl.total_seconds() / 3.0))


def maintain_lease(
    lease_store: LeaseStore,
    lease: LeaseRecord,
    *,
    ttl: timedelta | None = None,
    now: Callable[[], datetime] | None = None,
    enabled: bool = True,
) -> LeaseHeartbeat:
    return LeaseHeartbeat(lease_store, lease, ttl=ttl, now=now, enabled=enabled)


__all__ = ["LeaseHeartbeat", "maintain_lease"]
