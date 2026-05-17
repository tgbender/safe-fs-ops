from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from workspace_api_helpers import EnterConflictClaimStore, workspace_with_portable_ops

from safe_fs_ops import ResourceNotClaimedError, SafeWorkspace, SafeWorkspaceBusyError, SafeWorkspaceError
from safe_fs_ops.operation_journal import BatchPhase, file_resource_key
from safe_fs_ops.resources import FileResource
from safe_fs_ops.workspace_state import LeaseStore
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


def test_workspace_resources_are_named_durable_handles(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    config_dir = workspace.directory(tmp_path / "config", scope="settings")
    resources = workspace.resources(
        {
            "config_dir": config_dir,
            "settings": config_dir.file("settings.toml"),
            "cache": workspace.tree(tmp_path / "cache"),
        }
    )

    assert resources.config_dir.label == "config_dir"
    assert resources["settings"].label == "settings"
    assert resources.settings.path == tmp_path / "config" / "settings.toml"
    assert resources.settings.resource_key == file_resource_key(tmp_path / "config" / "settings.toml")
    assert resources.settings.scope == "settings"
    assert [resource.resource_key for resource in resources.sorted] == sorted(
        resource.resource_key for resource in resources.values()
    )


def test_workspace_resources_are_immutable_with_attr_and_mapping_access(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    assert resources.config is resources["config"]
    with pytest.raises(TypeError):
        resources["other"] = workspace.file(tmp_path / "other.txt")  # type: ignore[index]


def test_workspace_resources_reject_duplicate_resource_keys(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = workspace.file(tmp_path / "config.txt")

    with pytest.raises(ValueError, match="both refer"):
        workspace.resources({"first": target, "second": workspace.file(tmp_path / "config.txt")})


@pytest.mark.skipif(os.name != "nt", reason="Windows path aliases are case-insensitive")
def test_workspace_resources_reject_windows_case_aliases(tmp_path: Path) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a")
    target = tmp_path / "Config.txt"
    target.write_text("old\n", encoding="utf-8")
    lower_path = str(target).lower()
    upper_path = str(target).upper()

    assert file_resource_key(lower_path) == file_resource_key(upper_path)
    assert workspace.directory(lower_path).resource_key == workspace.directory(upper_path).resource_key
    assert workspace.tree(lower_path).resource_key == workspace.tree(upper_path).resource_key
    with pytest.raises(ValueError, match="both refer"):
        workspace.resources({"lower": workspace.file(lower_path), "upper": workspace.file(upper_path)})


def test_transaction_writes_claimed_file_resource(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target_path = tmp_path / "config.txt"
    workspace = _workspace(state_path)
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        result = tx.write_text(tx.r.config, "value = 1\n", idempotency_key="apply:config")

    assert target_path.read_text(encoding="utf-8") == "value = 1\n"
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert workspace.claim_store.get(resources.config.resource_key) is None
    batches = workspace.journal_store.list_batches()
    assert len(batches) == 1
    assert batches[0].run_id == "run-1"
    assert batches[0].resource_key == resources.config.resource_key


def test_transaction_with_explicit_logical_now_does_not_heartbeat_lease(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    target_path = tmp_path / "config.txt"
    workspace = _workspace(tmp_path / "state.db")
    lease_store = _RecordingLeaseStore(workspace.lease_store)
    workspace._lease_store = lease_store  # type: ignore[assignment]
    workspace._coordinator._lease_store = lease_store  # type: ignore[assignment]
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
        now=now,
        cleanup_clock=lambda: now,
    ) as tx:
        tx.write_text(tx.r.config, "value = 1\n", idempotency_key="apply:config")

    assert target_path.read_text(encoding="utf-8") == "value = 1\n"
    assert lease_store.heartbeat_times == []


def test_transaction_deletes_claimed_file_resource(tmp_path: Path) -> None:
    target_path = tmp_path / "config.txt"
    target_path.write_text("old\n", encoding="utf-8")
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        result = tx.delete_file(tx.r.config, idempotency_key="delete:config")

    assert target_path.exists() is False
    assert result.batch.phase == BatchPhase.SUCCEEDED
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_transaction_rejects_unclaimed_file_resource(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    other = workspace.file(tmp_path / "other.txt")

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(
        ResourceNotClaimedError,
        match="was not claimed",
    ):
        tx.write_text(other, "value\n")


def test_transaction_rejects_alias_handle_with_same_resource_key(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    alias = workspace.file(tmp_path / "config.txt")

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(
        ResourceNotClaimedError,
        match="claimed ResourceSet member",
    ):
        tx.write_text(alias, "value\n")


def test_transaction_rejects_raw_paths_for_file_operations(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.transaction(name="apply", resources=resources) as tx, pytest.raises(TypeError, match="FileResource"):
        tx.write_text(tmp_path / "config.txt", "value\n")  # type: ignore[arg-type]


def test_transaction_releases_claim_and_lease_when_body_fails(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with pytest.raises(RuntimeError, match="boom"), workspace.transaction(name="apply", resources=resources) as tx:
        assert isinstance(tx.r.config, FileResource)
        raise RuntimeError("boom")

    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active("workspace") is None


def test_transaction_body_exception_is_not_masked_when_cleanup_loses_lease(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        now=now,
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=2),
    )

    with pytest.raises(RuntimeError, match="boom") as excinfo, tx:
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-b",
            ttl=timedelta(seconds=30),
            now=now + workspace.lease_ttl + timedelta(seconds=1),
        )
        raise RuntimeError("boom")

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction lease"
    assert excinfo.value.__notes__ is not None
    assert any("cleanup also failed" in note for note in excinfo.value.__notes__)
    assert workspace.claim_store.get(resources.config.resource_key) is None
    active_lease = workspace.lease_store.active(workspace.lease_name, now=now)
    assert active_lease is not None
    assert active_lease.owner == "owner-b"


def test_transaction_claim_conflict_surfaces_as_workspace_busy_error(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace_a = _workspace(tmp_path / "state.db")
    workspace_b = _workspace(tmp_path / "state.db")
    resources = workspace_a.resources({"config": workspace_a.file(tmp_path / "config.txt")})
    lease = workspace_b.lease_store.acquire(
        workspace_b.lease_name,
        owner=workspace_b.owner,
        ttl=workspace_b.lease_ttl,
        now=now,
    )
    workspace_b.claim_store.upsert(
        resources.config.resource_key,
        lease=lease,
        owner="owner-b",
        scope="transaction:other-run",
        details="file",
        now=now,
    )
    assert workspace_b.lease_store.release(lease, now=now) is True

    with (
        pytest.raises(SafeWorkspaceBusyError, match="already claimed"),
        workspace_a.transaction(
            name="apply",
            resources=resources,
            now=now + timedelta(seconds=1),
            cleanup_clock=lambda: now + timedelta(seconds=2),
        ),
    ):
        pass


def test_transaction_enter_cleanup_failure_is_recorded_without_masking_claim_conflict(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources(
        {
            "claimed": workspace.resource("custom:a"),
            "conflict": workspace.resource("custom:b"),
        }
    )
    workspace._claim_store = EnterConflictClaimStore(conflict_key=resources.conflict.resource_key)
    tx = workspace.transaction(name="apply", resources=resources, run_id="run-1")

    with pytest.raises(SafeWorkspaceBusyError, match="already claimed") as excinfo, tx:
        pass

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction claims"
    assert excinfo.value.__notes__ is not None
    assert any("cleanup also failed" in note for note in excinfo.value.__notes__)


def test_transaction_enter_cleanup_does_not_release_unacquired_same_owner_claims(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace_a = _workspace(tmp_path / "state.db")
    workspace_b = _workspace(tmp_path / "state.db")
    resources = workspace_a.resources(
        {
            "conflict": workspace_a.resource("custom:a"),
            "preexisting": workspace_a.resource("custom:z"),
        }
    )
    lease = workspace_b.lease_store.acquire(
        workspace_b.lease_name,
        owner=workspace_b.owner,
        ttl=workspace_b.lease_ttl,
        now=now,
    )
    preexisting_claim = workspace_b.claim_store.upsert(
        resources.preexisting.resource_key,
        lease=lease,
        owner=workspace_a.owner,
        scope="transaction:run-1",
        details="preexisting",
        now=now,
    )
    workspace_b.claim_store.upsert(
        resources.conflict.resource_key,
        lease=lease,
        owner="owner-b",
        scope="transaction:other-run",
        details="conflict",
        now=now,
    )
    assert workspace_b.lease_store.release(lease, now=now) is True

    with (
        pytest.raises(SafeWorkspaceBusyError, match="already claimed"),
        workspace_a.transaction(
            name="apply",
            resources=resources,
            run_id="run-1",
            now=now + timedelta(seconds=1),
        ),
    ):
        pass

    assert workspace_a.claim_store.get(resources.preexisting.resource_key) == preexisting_claim


def test_transaction_cleanup_failures_surface_as_workspace_error_on_normal_exit(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with (
        pytest.raises(SafeWorkspaceError, match="failed to release transaction claims"),
        workspace.transaction(
            name="apply",
            resources=resources,
            now=now,
            cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=33),
        ),
    ):
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-b",
            ttl=timedelta(seconds=30),
            now=now + workspace.lease_ttl + timedelta(seconds=1),
        )
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-c",
            ttl=timedelta(seconds=30),
            now=now + workspace.lease_ttl + timedelta(seconds=2),
        )
        with sqlite3.connect(tmp_path / "state.db") as connection:
            connection.execute(
                """
                    UPDATE workspace_resource_claims
                    SET owner = ?
                    WHERE resource_key = ?
                    """,
                ("owner-z", resources.config.resource_key),
            )
            connection.commit()

    claim = workspace.claim_store.get(resources.config.resource_key)
    assert claim is not None
    assert claim.owner == "owner-z"


def test_transaction_exit_cleanup_failure_is_recorded_on_body_exception(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        now=now,
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=33),
    )

    with pytest.raises(RuntimeError, match="boom") as excinfo, tx:
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-b",
            ttl=timedelta(seconds=30),
            now=now + workspace.lease_ttl + timedelta(seconds=1),
        )
        workspace.lease_store.acquire(
            workspace.lease_name,
            owner="owner-c",
            ttl=timedelta(seconds=30),
            now=now + workspace.lease_ttl + timedelta(seconds=2),
        )
        with sqlite3.connect(tmp_path / "state.db") as connection:
            connection.execute(
                """
                    UPDATE workspace_resource_claims
                    SET owner = ?
                    WHERE resource_key = ?
                    """,
                ("owner-z", resources.config.resource_key),
            )
            connection.commit()
        raise RuntimeError("boom")

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction claims"
    assert excinfo.value.__notes__ is not None
    assert any("cleanup also failed" in note for note in excinfo.value.__notes__)
    claim = workspace.claim_store.get(resources.config.resource_key)
    assert claim is not None
    assert claim.owner == "owner-z"


def test_transaction_cleanup_failure_keeps_remaining_claims_when_lease_has_expired(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"a": workspace.resource("custom:a"), "b": workspace.resource("custom:b")})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
        now=now,
        cleanup_clock=lambda: now + timedelta(seconds=2),
    )

    with pytest.raises(SafeWorkspaceError, match="failed to release transaction lease"), tx:
        assert workspace.claim_store.get(resources.a.resource_key) is not None
        assert workspace.claim_store.get(resources.b.resource_key) is not None
        with sqlite3.connect(tmp_path / "state.db") as connection:
            connection.execute(
                "DELETE FROM workspace_resource_claims WHERE resource_key = ?",
                (resources.b.resource_key,),
            )
            connection.execute(
                """
                    UPDATE workspace_leases
                    SET owner = ?, token = ?, fencing_token = fencing_token + 1, expires_at = ?, heartbeat_at = ?
                    WHERE name = ?
                    """,
                (
                    "owner-b",
                    "replacement-token",
                    (now - timedelta(seconds=1)).isoformat(),
                    (now - timedelta(seconds=1)).isoformat(),
                    workspace.lease_name,
                ),
            )
            connection.commit()

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction lease"
    assert workspace.claim_store.get(resources.a.resource_key) is None
    assert workspace.claim_store.get(resources.b.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name, now=now + timedelta(seconds=2)) is None


def test_transaction_exposes_claimed_resources_via_r(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        assert tx.r is resources
        assert tx.r.config is resources.config


def test_directory_resource_rejects_child_paths_that_escape_parent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    directory = workspace.directory(tmp_path / "root")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        directory.file("../escape.txt")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        directory.directory(Path("nested") / ".." / ".." / "escape")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        directory.tree("child/../../escape")


def test_tree_resource_rejects_child_paths_that_escape_parent(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "state.db")
    tree = workspace.tree(tmp_path / "root")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        tree.file("../escape.txt")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        tree.directory("../escape")

    with pytest.raises(ValueError, match="must stay within the parent resource"):
        tree.tree(Path("child") / ".." / ".." / "escape")


def _workspace(state_path: Path) -> SafeWorkspace:
    return workspace_with_portable_ops(state_path)
