from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from safe_fs_ops.workspace_state import ClaimConflictError, ClaimStore, LeaseLostError, LeaseStore, WorkspaceRuntime
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload

pytestmark = pytest.mark.safe_fs_ops


def test_claim_store_creates_and_reads_claim(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)

    claim = store.create(
        "file:C:/Users/me/.config/tool.toml",
        lease=lease,
        owner="owner-a",
        scope="install",
        details="settings.theme",
        now=now,
    )

    assert claim.resource_key == "file:C:/Users/me/.config/tool.toml"
    assert claim.owner == "owner-a"
    assert claim.scope == "install"
    assert claim.details == "settings.theme"
    assert claim.created_at == now
    assert claim.updated_at == now
    assert store.get(claim.resource_key) == claim


def test_claim_store_upserts_same_owner_and_scope(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=10)
    lease = _acquire_lease(state_path, now=first_now)
    first = store.upsert(
        "asset:tool.exe",
        lease=lease,
        owner="owner-a",
        scope="install",
        details="v1",
        now=first_now,
    )

    second = store.upsert(
        "asset:tool.exe",
        lease=lease,
        owner="owner-a",
        scope="install",
        details="v2",
        now=second_now,
    )

    assert second.resource_key == first.resource_key
    assert second.owner == first.owner
    assert second.scope == first.scope
    assert second.details == "v2"
    assert second.created_at == first_now
    assert second.updated_at == second_now


def test_claim_store_reports_conflicts_across_separate_store_instances(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    first_store = ClaimStore(state_path)
    second_store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    first_store.upsert("json-key:settings.toml::theme", lease=lease, owner="owner-a", scope="profile", now=now)

    conflict = second_store.conflict_for("json-key:settings.toml::theme", owner="owner-b", scope="profile")

    assert conflict is not None
    assert conflict.owner == "owner-a"
    with pytest.raises(ClaimConflictError) as raised:
        second_store.upsert(
            "json-key:settings.toml::theme",
            lease=lease,
            owner="owner-b",
            scope="profile",
            now=now,
        )
    assert raised.value.existing == conflict


def test_claim_store_create_reports_existing_claim_conflict(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    first = store.create("json-key:settings.toml::theme", lease=lease, owner="owner-a", scope="profile", now=now)

    with pytest.raises(ClaimConflictError) as raised:
        store.create(
            "json-key:settings.toml::theme",
            lease=lease,
            owner="owner-b",
            scope="profile",
            now=now + timedelta(seconds=1),
        )

    assert raised.value.existing == first


def test_claim_store_treats_scope_mismatch_as_conflict(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    store.upsert("file:.env", lease=lease, owner="owner-a", scope="install", now=now)

    conflict = store.conflict_for("file:.env", owner="owner-a", scope="upgrade")

    assert conflict is not None
    assert conflict.owner == "owner-a"
    assert conflict.scope == "install"


def test_claim_store_rejects_same_owner_transaction_scope_reassignment(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    store.upsert("file:.env", lease=lease, owner="owner-a", scope="transaction:run-1", now=now)

    with pytest.raises(ClaimConflictError) as raised:
        store.upsert(
            "file:.env",
            lease=lease,
            owner="owner-a",
            scope="transaction:run-2",
            now=now + timedelta(seconds=1),
        )

    assert raised.value.existing.owner == "owner-a"
    assert raised.value.existing.scope == "transaction:run-1"
    claim = store.get("file:.env")
    assert claim is not None
    assert claim.scope == "transaction:run-1"


def test_claim_store_lists_claims_deterministically(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    store.upsert("file:z", lease=lease, owner="owner-b", now=now)
    store.upsert("file:a", lease=lease, owner="owner-a", now=now)
    store.upsert("file:m", lease=lease, owner="owner-a", now=now)

    all_claims = store.list_claims()
    owner_claims = store.list_claims(owner="owner-a")

    assert [claim.resource_key for claim in all_claims] == ["file:a", "file:m", "file:z"]
    assert [claim.resource_key for claim in owner_claims] == ["file:a", "file:m"]


def test_claim_store_release_requires_matching_owner_and_scope(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    claim = store.upsert("file:.env", lease=lease, owner="owner-a", scope="install", now=now)

    assert store.release(claim.resource_key, lease=lease, owner="owner-b", scope="install", now=now) is False
    assert store.release(claim.resource_key, lease=lease, owner="owner-a", scope="upgrade", now=now) is False
    assert store.get(claim.resource_key) == claim
    assert store.release(claim.resource_key, lease=lease, owner="owner-a", scope="install", now=now) is True
    assert store.get(claim.resource_key) is None


def test_claim_store_stale_fallback_does_not_release_replacement_claim_with_reused_scope(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=31)
    first_lease = _acquire_lease(state_path, now=first_now)
    original = store.upsert(
        "file:.env",
        lease=first_lease,
        owner="owner-a",
        scope="transaction:run-1",
        details=_lease_bound_claim_details(first_lease),
        now=first_now,
    )
    assert store.release(
        original.resource_key,
        lease=first_lease,
        owner="owner-a",
        scope="transaction:run-1",
        now=first_now,
    )

    second_lease = _acquire_lease(state_path, now=second_now)
    replacement = store.upsert(
        "file:.env",
        lease=second_lease,
        owner="owner-a",
        scope="transaction:run-1",
        details=_lease_bound_claim_details(second_lease),
        now=second_now,
    )

    assert (
        store.release_if_owner_scope_matches(
            original.resource_key,
            lease=first_lease,
            owner="owner-a",
            scope="transaction:run-1",
            expected_claim=original,
            now=second_now,
        )
        is False
    )
    assert store.get(original.resource_key) == replacement


def test_claim_store_stale_fallback_requires_exact_claim_identity(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    claim = store.upsert(
        "file:.env",
        lease=lease,
        owner="owner-a",
        scope="transaction:run-1",
        details="file",
        now=now,
    )

    with pytest.raises(ValueError, match="expected_claim is required"):
        store.release_if_owner_scope_matches(
            claim.resource_key,
            lease=lease,
            owner="owner-a",
            scope="transaction:run-1",
        )

    assert store.get(claim.resource_key) == claim


def test_claim_store_stale_fallback_requires_lease_bound_claim_authority(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    later = now + timedelta(seconds=31)
    lease = _acquire_lease(state_path, now=now)
    claim = store.upsert(
        "file:.env",
        lease=lease,
        owner="owner-a",
        scope="transaction:run-1",
        details="legacy-unbound-claim",
        now=now,
    )
    _acquire_lease(state_path, now=later, owner="runner-b")

    with pytest.raises(ValueError, match="current lease or lease-bound"):
        store.release_if_owner_scope_matches(
            claim.resource_key,
            lease=lease,
            owner="owner-a",
            scope="transaction:run-1",
            expected_claim=claim,
        )

    assert store.get(claim.resource_key) == claim


def test_claim_store_stale_fallback_rejects_forged_same_fencing_lease_identity(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_lease(state_path, now=now)
    claim = store.upsert(
        "file:.env",
        lease=lease,
        owner="owner-a",
        scope="transaction:run-1",
        details=_lease_bound_claim_details(lease),
        now=now,
    )
    forged = replace(lease, name="other")

    with pytest.raises(ValueError, match="current lease or lease-bound"):
        store.release_if_owner_scope_matches(
            claim.resource_key,
            lease=forged,
            owner="owner-a",
            scope="transaction:run-1",
            expected_claim=claim,
        )

    assert store.get(claim.resource_key) == claim


def test_claim_store_rejects_stale_lease_without_writing_claim(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = ClaimStore(state_path)
    first_now = datetime(2026, 1, 1, tzinfo=UTC)
    second_now = first_now + timedelta(seconds=31)
    first_lease = _acquire_lease(state_path, now=first_now)
    _acquire_lease(state_path, now=second_now, owner="runner-b")

    with pytest.raises(LeaseLostError, match="no longer current"):
        store.upsert("file:.env", lease=first_lease, owner="owner-a", now=second_now)

    assert store.get("file:.env") is None


def test_workspace_runtime_reports_claim_conflicts_across_separate_instances(tmp_path: Path) -> None:
    first_runtime = WorkspaceRuntime(tmp_path)
    second_runtime = WorkspaceRuntime(tmp_path)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = first_runtime.acquire_lease("workspace", owner="runner-a", ttl=timedelta(seconds=30), now=now)
    first_runtime.claim_resource("asset:tool.exe", lease=lease, owner="owner-a", scope="install", now=now)

    conflict = second_runtime.claim_conflict_for("asset:tool.exe", owner="owner-b", scope="install")

    assert conflict is not None
    assert conflict.owner == "owner-a"
    with pytest.raises(ClaimConflictError):
        second_runtime.claim_resource("asset:tool.exe", lease=lease, owner="owner-b", scope="install", now=now)


def _acquire_lease(state_path: Path, *, now: datetime, owner: str = "runner-a"):
    return LeaseStore(state_path).acquire("workspace", owner=owner, ttl=timedelta(seconds=30), now=now)


def _lease_bound_claim_details(lease) -> str:
    return json.dumps(
        lease_claim_details_payload(lease, claim_id=f"claim:{lease.name}:{lease.fencing_token}"),
        separators=(",", ":"),
        sort_keys=True,
    )
