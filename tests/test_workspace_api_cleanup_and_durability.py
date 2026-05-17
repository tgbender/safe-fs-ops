from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from workspace_api_helpers import workspace_with_portable_ops

from safe_fs_ops import SafeWorkspace, SafeWorkspaceBusyError, SafeWorkspaceError
from safe_fs_ops.filesystem_ops import DurabilityMode, ResourceSnapshot
from safe_fs_ops.operation_journal import OperationJournalStore
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state import LeaseStore

pytestmark = pytest.mark.safe_fs_ops


def test_transaction_normal_exit_after_real_lease_expiry_surfaces_cleanup_error(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        now=now,
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=1),
    )

    with pytest.raises(SafeWorkspaceError, match="failed to release transaction lease"), tx:
        pass

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction lease"
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_transaction_body_exception_keeps_primary_error_after_real_lease_expiry(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    tx = workspace.transaction(
        name="apply",
        resources=resources,
        now=now,
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=1),
    )

    with pytest.raises(RuntimeError, match="boom") as excinfo, tx:
        raise RuntimeError("boom")

    assert tx.cleanup_error is not None
    assert str(tx.cleanup_error) == "failed to release transaction lease"
    assert excinfo.value.__notes__ is not None
    assert any("cleanup also failed" in note for note in excinfo.value.__notes__)
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_new_transaction_reclaims_same_owner_stale_transaction_claim_after_lease_loss(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    expired = workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
        now=now,
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=1),
    )

    with pytest.raises(SafeWorkspaceError, match="failed to release transaction lease"), expired:
        pass

    assert expired.cleanup_error is not None
    assert str(expired.cleanup_error) == "failed to release transaction lease"
    assert workspace.claim_store.get(resources.config.resource_key) is None

    with workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-2",
        now=now + workspace.lease_ttl + timedelta(seconds=2),
        cleanup_clock=lambda: now + workspace.lease_ttl + timedelta(seconds=2),
    ) as tx:
        active_claim = workspace.claim_store.get(resources.config.resource_key)
        assert active_claim is not None
        assert active_claim.owner == workspace.owner
        assert active_claim.scope == tx.claim_scope
        assert active_claim.scope.startswith("transaction-instance:")

    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_same_owner_different_lease_names_cannot_rewrite_active_transaction_claim(tmp_path: Path) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    workspace_a = _workspace(tmp_path / "state.db", lease_name="workspace-a")
    workspace_b = _workspace(tmp_path / "state.db", lease_name="workspace-b")
    resources = workspace_a.resources({"config": workspace_a.file(tmp_path / "config.txt")})

    with workspace_a.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
        now=now,
        cleanup_clock=lambda: now,
    ) as tx:
        claim = workspace_a.claim_store.get(resources.config.resource_key)
        assert claim is not None
        assert claim.owner == workspace_a.owner
        assert claim.scope == tx.claim_scope
        assert claim.scope.startswith("transaction-instance:")

        with (
            pytest.raises(SafeWorkspaceBusyError, match="already claimed"),
            workspace_b.transaction(
                name="apply",
                resources=resources,
                run_id="run-2",
                now=now,
                cleanup_clock=lambda: now,
            ),
        ):
            pass

    assert workspace_a.claim_store.get(resources.config.resource_key) is None


def test_finalize_operation_success_keeps_claim_ledger_until_commit_and_cleanup_can_release_after_rollback(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "state.db")
    resources = workspace.resources({"config": workspace.file(tmp_path / "config.txt")})
    tx = workspace.transaction(name="apply", resources=resources, run_id="run-1")
    tx.__enter__()

    operation_run = workspace.journal_store.create_operation_run(
        run_id="run-1",
        lease=tx.lease,
        owner=workspace.owner,
        status="active",
        payload={"name": "apply", "rollback": "record-only"},
        now=tx.now,
    )
    workspace._journal_store = OperationJournalStore(
        workspace.state_path,
        sqlite_store=_FailingSqliteStore(workspace.state_path, fail_on_sql="UPDATE workspace_leases"),
    )

    with pytest.raises(sqlite3.OperationalError, match="injected sqlite failure"):
        tx.finalize_operation_success(operation_run.operation_run_id)

    assert workspace.claim_store.get(resources.config.resource_key) is not None

    cleanup_error = tx._cleanup_transaction()

    assert cleanup_error is None
    assert workspace.claim_store.get(resources.config.resource_key) is None
    assert workspace.lease_store.active(workspace.lease_name) is None


def test_transaction_implicit_phase_records_finalization_failed_when_commit_time_cleanup_fails(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    workspace._journal_store = OperationJournalStore(
        workspace.state_path,
        sqlite_store=_FailingSqliteStore(workspace.state_path, fail_on_sql="UPDATE workspace_leases"),
    )
    workspace._coordinator._journal_store = workspace.journal_store

    with pytest.raises(sqlite3.OperationalError, match="injected sqlite failure"), workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
    ) as tx:
        tx_ref = tx
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")

    assert tx_ref is not None
    assert tx_ref.operation_run is not None
    assert tx_ref.operation_phase is not None
    assert (
        workspace.journal_store.get_operation_run(tx_ref.operation_run.operation_run_id).status == "finalization_failed"
    )
    assert workspace.journal_store.get_operation_phase(tx_ref.operation_phase.operation_phase_id).status == (
        "finalization_failed"
    )
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


def test_transaction_explicit_operation_links_record_finalization_failed_when_commit_time_cleanup_fails(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config.txt"
    resources = workspace.resources({"config": workspace.file(target_path)})
    tx_ref = None
    explicit_run_id = "operation-run-explicit"
    explicit_phase_id = "operation-phase-explicit"

    workspace._lease_store = LeaseStore(
        workspace.state_path,
        sqlite_store=_FailingSqliteStore(workspace.state_path, fail_on_sql="UPDATE workspace_leases"),
    )

    with pytest.raises(sqlite3.OperationalError, match="injected sqlite failure"), workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
    ) as tx:
        tx_ref = tx
        explicit_run = workspace.journal_store.create_operation_run(
            run_id=tx.run_id,
            lease=tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
            operation_run_id=explicit_run_id,
            now=tx.now,
        )
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=explicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=1,
            payload={"explicit": True},
            operation_phase_id=explicit_phase_id,
            now=tx.now,
        )
        tx.write_text(
            tx.r.config,
            "value = 1\n",
            idempotency_key="write:explicit",
            operation_run_id=explicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert workspace.journal_store.get_operation_run(explicit_run_id).status == "finalization_failed"
    assert workspace.journal_store.get_operation_phase(explicit_phase_id).status == "finalization_failed"
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_transaction_explicit_operation_links_reconcile_to_succeeded_on_clean_success(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config.txt"
    resources = workspace.resources({"config": workspace.file(target_path)})
    explicit_run_id = "operation-run-explicit"
    explicit_phase_id = "operation-phase-explicit"

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        explicit_run = workspace.journal_store.create_operation_run(
            run_id=tx.run_id,
            lease=tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
            operation_run_id=explicit_run_id,
            now=tx.now,
        )
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=explicit_run.operation_run_id,
            lease=tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=1,
            payload={"explicit": True},
            operation_phase_id=explicit_phase_id,
            now=tx.now,
        )
        tx.write_text(
            tx.r.config,
            "value = 1\n",
            idempotency_key="write:explicit",
            operation_run_id=explicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    assert workspace.journal_store.get_operation_run(explicit_run_id).status == "succeeded"
    assert workspace.journal_store.get_operation_phase(explicit_phase_id).status == "succeeded"
    assert target_path.read_text(encoding="utf-8") == "value = 1\n"
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_transaction_implicit_and_explicit_links_all_record_finalization_failed_when_commit_time_cleanup_fails(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    resources = workspace.resources(
        {
            "state_dir": workspace.directory(tmp_path / "config" / "state"),
            "config": workspace.file(tmp_path / "config.txt"),
        }
    )
    tx_ref = None
    explicit_phase_id = "operation-phase-explicit"

    workspace._journal_store = OperationJournalStore(
        workspace.state_path,
        sqlite_store=_FailingSqliteStore(workspace.state_path, fail_on_sql="UPDATE workspace_leases"),
    )
    workspace._coordinator._journal_store = workspace.journal_store

    with pytest.raises(sqlite3.OperationalError, match="injected sqlite failure"), workspace.transaction(
        name="apply",
        resources=resources,
        run_id="run-1",
    ) as tx:
        tx_ref = tx
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
            operation_phase_id=explicit_phase_id,
            now=tx.now,
        )
        tx.write_text(
            tx.r.config,
            "value = 1\n",
            idempotency_key="write:explicit",
            operation_run_id=implicit_run.operation_run_id,
            operation_phase_id=explicit_phase.operation_phase_id,
        )

    assert tx_ref is not None
    assert tx_ref.operation_run is not None
    assert tx_ref.operation_phase is not None
    assert (
        workspace.journal_store.get_operation_run(tx_ref.operation_run.operation_run_id).status == "finalization_failed"
    )
    assert workspace.journal_store.get_operation_phase(tx_ref.operation_phase.operation_phase_id).status == (
        "finalization_failed"
    )
    assert workspace.journal_store.get_operation_phase(explicit_phase_id).status == "finalization_failed"
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None
    assert workspace.claim_store.get(resources.config.resource_key) is None


def test_workspace_threads_durability_only_to_operations_that_accept_it(tmp_path: Path) -> None:
    calls: list[tuple[str, DurabilityMode | None]] = []

    def write_text_with_durability(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
        durability: DurabilityMode = DurabilityMode.FSYNC,
    ) -> None:
        calls.append(("write", durability))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding=encoding, newline=newline)

    def delete_file_without_durability(path: Path | str, *, missing_ok: bool = False) -> None:
        calls.append(("delete", None))
        try:
            Path(path).unlink()
        except FileNotFoundError:
            if not missing_ok:
                raise

    target_path = tmp_path / "config.txt"
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        durability=DurabilityMode.NONE,
        snapshot=_portable_snapshot,
        write_text_operation=write_text_with_durability,
        delete_file_operation=delete_file_without_durability,
    )
    resources = workspace.resources({"config": workspace.file(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
        tx.write_text(tx.r.config, "value = 1\n", idempotency_key="apply:config")
        tx.delete_file(tx.r.config, idempotency_key="delete:config")

    assert calls == [("write", DurabilityMode.NONE), ("delete", None)]
    batches = workspace.journal_store.list_batches()
    assert [batch.payload["operation"] for batch in batches] == ["write_text", "delete_file"]
    assert [
        checkpoint.checkpoint_type for checkpoint in workspace.journal_store.list_checkpoints(batches[0].batch_id)
    ] == [
        "before",
        "after",
    ]
    assert [
        checkpoint.checkpoint_type for checkpoint in workspace.journal_store.list_checkpoints(batches[1].batch_id)
    ] == [
        "before",
        "after",
    ]


def test_custom_workspace_file_batches_report_unsupported_automatic_rollback_without_backup_proof(
    tmp_path: Path,
) -> None:
    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=_portable_snapshot,
        write_text_operation=_portable_write_text,
        delete_file_operation=_portable_delete_file,
    )
    target_path = tmp_path / "config.txt"

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"config": workspace.file(target_path)},
            run_id="run-1",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.config, "value = 1\n", idempotency_key="apply:config")
        raise RuntimeError("boom")

    assert target_path.read_text(encoding="utf-8") == "value = 1\n"
    batch = workspace.journal_store.list_batches()[-1]
    assert batch.phase == "recovery_desired"
    notes = getattr(excinfo.value, "__notes__", ())
    assert any("write_text" in note and "does not support automatic rollback" in note for note in notes)


def _workspace(state_path: Path, *, lease_name: str = "workspace") -> SafeWorkspace:
    return SafeWorkspace.open(
        state_path,
        owner="owner-a",
        lease_name=lease_name,
        snapshot=_portable_snapshot,
        write_text_operation=_portable_write_text,
        delete_file_operation=_portable_delete_file,
    )


def _portable_write_text(
    path: Path | str,
    content: str,
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding=encoding, newline=newline)


def _portable_delete_file(path: Path | str, *, missing_ok: bool = False) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def _portable_snapshot(path: Path | str) -> ResourceSnapshot:
    target = Path(path)
    if not target.exists():
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
        )
    stat_result = target.lstat()
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type="file" if target.is_file() else "directory",
        content_hash=None,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
    )


class _FailingSqliteStore(SqliteStore):
    def __init__(self, path: Path | str, *, fail_on_sql: str) -> None:
        super().__init__(path)
        self._fail_on_sql = fail_on_sql

    @contextmanager
    def transaction(self) -> sqlite3.Connection:
        with super().transaction() as connection:
            yield _FailingConnection(connection, fail_on_sql=self._fail_on_sql)


class _FailingConnection:
    def __init__(self, connection: sqlite3.Connection, *, fail_on_sql: str) -> None:
        self._connection = connection
        self._fail_on_sql = fail_on_sql

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if self._fail_on_sql in " ".join(sql.split()):
            raise sqlite3.OperationalError("injected sqlite failure")
        return self._connection.execute(sql, parameters)

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)
