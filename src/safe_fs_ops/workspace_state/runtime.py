from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path

from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving
from safe_fs_ops.workspace_state.claims import ClaimStore
from safe_fs_ops.workspace_state.leases import LeaseStore
from safe_fs_ops.workspace_state.models import ClaimRecord, LeaseRecord


class WorkspaceRuntime:
    """Thread-safe workspace coordinator facade.

    The runtime may be cached in-process, but all cross-process authority lives
    in SQLite lease rows.
    """

    _registry_lock = threading.RLock()
    _registry: dict[Path, WorkspaceRuntime] = {}

    def __init__(self, root: Path, *, state_dir: str = ".safe-fs-ops", state_filename: str = "state.db") -> None:
        self.root = root.expanduser().resolve()
        self.state_path = _workspace_state_path(self.root, state_dir=state_dir, state_filename=state_filename)
        self._lock = threading.RLock()
        self._claims = ClaimStore(self.state_path)
        self._leases = LeaseStore(self.state_path)

    @classmethod
    def open(
        cls,
        root: Path | str,
        *,
        state_dir: str = ".safe-fs-ops",
        state_filename: str = "state.db",
    ) -> WorkspaceRuntime:
        resolved = Path(root).expanduser().resolve()
        state_path = _workspace_state_path(resolved, state_dir=state_dir, state_filename=state_filename)
        with cls._registry_lock:
            runtime = cls._registry.get(state_path)
            if runtime is None:
                runtime = cls(resolved, state_dir=state_dir, state_filename=state_filename)
                cls._registry[state_path] = runtime
            return runtime

    @property
    def claim_store(self) -> ClaimStore:
        return self._claims

    @property
    def lease_store(self) -> LeaseStore:
        return self._leases

    def claim_resource(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        details: str | None = None,
        now: datetime | None = None,
    ) -> ClaimRecord:
        with self._lock:
            return self._claims.upsert(
                resource_key,
                lease=lease,
                owner=owner,
                scope=scope,
                details=details,
                now=now,
            )

    def claim_conflict_for(
        self,
        resource_key: str,
        *,
        owner: str,
        scope: str | None = None,
    ) -> ClaimRecord | None:
        with self._lock:
            return self._claims.conflict_for(resource_key, owner=owner, scope=scope)

    def list_claims(self, *, owner: str | None = None) -> list[ClaimRecord]:
        with self._lock:
            return self._claims.list_claims(owner=owner)

    def release_claim(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        with self._lock:
            return self._claims.release(resource_key, lease=lease, owner=owner, scope=scope, now=now)

    def acquire_lease(
        self,
        name: str,
        *,
        owner: str,
        ttl: timedelta,
        token: str | None = None,
        now: datetime | None = None,
    ) -> LeaseRecord:
        with self._lock:
            return self._leases.acquire(name, owner=owner, ttl=ttl, token=token, now=now)

    def heartbeat_lease(
        self,
        lease: LeaseRecord,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> LeaseRecord | None:
        with self._lock:
            return self._leases.heartbeat(lease, ttl=ttl, now=now)

    def release_lease(self, lease: LeaseRecord, *, now: datetime | None = None) -> bool:
        with self._lock:
            return self._leases.release(lease, now=now)

    def require_current_lease(self, lease: LeaseRecord, *, now: datetime | None = None) -> None:
        with self._lock:
            self._leases.require_current(lease, now=now)


def _workspace_state_path(root: Path, *, state_dir: str, state_filename: str) -> Path:
    state_path = absolute_without_resolving(root / state_dir / state_filename)
    try:
        state_path.relative_to(root)
    except ValueError as exc:
        raise UnsafePathError(f"workspace state path must stay inside workspace root: {state_path}") from exc
    return state_path
