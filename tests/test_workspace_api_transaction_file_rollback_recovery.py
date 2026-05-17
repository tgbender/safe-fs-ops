from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from transaction_file_rollback_helpers import (
    backup_content_path_for_batch,
    hook_capable_remove_directory,
    latest_batch_for_run,
    requires_backup_restore_support,
    workspace_with_portable_file_rollback,
)
from workspace_api_helpers import (
    portable_delete_file,
    portable_make_directory,
    portable_remove_empty_directory_by_identity,
    portable_snapshot,
    portable_write_text,
    workspace_with_portable_ops,
)

from safe_fs_ops import SafeWorkspace
from safe_fs_ops.operation_journal import BatchPhase, OperationJournalStore
from safe_fs_ops.workspace_state.claims import lease_claim_details_payload

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.slow_recovery]


def _claim_details(lease) -> str:
    return json.dumps(
        lease_claim_details_payload(lease, claim_id=f"claim:{lease.name}:{lease.fencing_token}"),
        separators=(",", ":"),
        sort_keys=True,
    )


@requires_backup_restore_support
def test_transaction_automatic_file_rollback_removes_created_file_after_body_exception(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-created-file",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:create")
        raise RuntimeError("boom")

    batch = latest_batch_for_run(workspace, "run-created-file")
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.exists() is False
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


@requires_backup_restore_support
def test_workspace_recover_pending_batches_takes_over_abandoned_attempting_file_mutation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )

    def write_then_crash(
        path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise KeyboardInterrupt("simulated process death")

    workspace._coordinator._write_text = write_then_crash
    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-abandoned-attempting",
            idempotency_key="write:abandoned",
            newline="\n",
            now=crash_time,
        )

    batch = latest_batch_for_run(workspace, "run-abandoned-attempting")
    assert batch.phase == BatchPhase.ATTEMPTING
    assert target.read_text(encoding="utf-8") == "new\n"

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-attempting")

    batch = latest_batch_for_run(workspace, "run-abandoned-attempting")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert workspace.claim_store.get(resource.resource_key) is None

    with workspace.transaction(
        name="after-recovery",
        resources={"file": workspace.file(target)},
        run_id="run-after-public-recovery",
    ) as tx:
        tx.write_text(tx.r.file, "next\n", idempotency_key="write:after-recovery", newline="\n")

    assert target.read_text(encoding="utf-8") == "next\n"


@requires_backup_restore_support
def test_workspace_recover_pending_batches_takes_over_abandoned_recovering_file_mutation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )

    def write_then_crash(
        path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise KeyboardInterrupt("simulated process death")

    workspace._coordinator._write_text = write_then_crash
    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-abandoned-recovering",
            idempotency_key="write:abandoned-recovering",
            newline="\n",
            now=crash_time,
        )

    batch = latest_batch_for_run(workspace, "run-abandoned-recovering")
    takeover_time = crash_time + timedelta(seconds=2)
    takeover_lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=takeover_time,
    )
    assert takeover_lease.acquired
    workspace.journal_store.record_interrupted_recovery_desired(
        batch.batch_id,
        lease=takeover_lease,
        resource_key=resource.resource_key,
        reason="simulated recovery attempt",
        payload={"batch_id": batch.batch_id},
        now=takeover_time,
    )
    workspace.journal_store.start_recovery(
        batch.batch_id,
        lease=takeover_lease,
        reason="simulated abandoned recovery",
        payload={"batch_id": batch.batch_id},
        now=takeover_time,
    )
    batch = latest_batch_for_run(workspace, "run-abandoned-recovering")
    assert batch.phase == BatchPhase.RECOVERING

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-recovering")

    batch = latest_batch_for_run(workspace, "run-abandoned-recovering")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert workspace.claim_store.get(resource.resource_key) is None


def test_workspace_recover_pending_batches_releases_orphan_transaction_claim_without_batch(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    tx = workspace.transaction(
        name="apply",
        resources={"file": resource},
        run_id="run-orphan-claim",
        now=crash_time,
    )
    tx.__enter__()
    try:
        tx._stop_lease_heartbeat()
        assert workspace.journal_store.list_batches(run_id="run-orphan-claim") == []
        assert workspace.claim_store.get(resource.resource_key) is not None

        recovery_error = workspace.recover_pending_batches(run_id="run-orphan-claim")
    finally:
        tx._stop_lease_heartbeat()

    assert recovery_error is None
    assert workspace.journal_store.list_batches(run_id="run-orphan-claim") == []
    assert workspace.claim_store.get(resource.resource_key) is None

    with workspace.transaction(
        name="after-orphan-recovery",
        resources={"file": workspace.file(target)},
        run_id="run-after-orphan-recovery",
    ) as next_tx:
        next_tx.write_text(next_tx.r.file, "next\n", idempotency_key="write:after-orphan-recovery", newline="\n")

    assert target.read_text(encoding="utf-8") == "next\n"


def test_workspace_recover_pending_batches_handles_abandoned_read_only_succeeded_batch_with_live_claim(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "config.toml"
    target.write_text("enabled = true\n", encoding="utf-8")
    resource = workspace.tree(root)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    tx = workspace.transaction(
        name="sync",
        resources={"tree": resource},
        run_id="run-abandoned-read-only",
        rollback="automatic",
        now=crash_time,
    )
    tx.__enter__()
    try:
        tx.backup_tree(tx.r.tree, ["config.toml"], idempotency_key="backup:abandoned-read-only")
        tx._stop_lease_heartbeat()

        batch = latest_batch_for_run(workspace, "run-abandoned-read-only")
        assert batch.phase == BatchPhase.SUCCEEDED
        assert workspace.claim_store.get(resource.resource_key) is not None

        recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-read-only")
    finally:
        tx._stop_lease_heartbeat()

    batch = latest_batch_for_run(workspace, "run-abandoned-read-only")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "enabled = true\n"
    assert workspace.claim_store.get(resource.resource_key) is None
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []


@requires_backup_restore_support
def test_workspace_recover_pending_batches_rolls_back_abandoned_succeeded_file_mutation(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    tx = workspace.transaction(
        name="apply",
        resources={"file": resource},
        run_id="run-abandoned-succeeded",
        rollback="automatic",
        now=crash_time,
    )
    tx.__enter__()
    try:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:abandoned-succeeded", newline="\n")
        tx._stop_lease_heartbeat()

        batch = latest_batch_for_run(workspace, "run-abandoned-succeeded")
        assert batch.phase == BatchPhase.SUCCEEDED
        assert target.read_text(encoding="utf-8") == "new\n"
        assert workspace.claim_store.get(resource.resource_key) is not None

        recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-succeeded")
    finally:
        tx._stop_lease_heartbeat()

    batch = latest_batch_for_run(workspace, "run-abandoned-succeeded")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert workspace.claim_store.get(resource.resource_key) is None


@requires_backup_restore_support
def test_workspace_recover_pending_batches_terminalizes_abandoned_operation_links(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    operation = workspace.operation(
        name="sync",
        resources={"file": resource},
        run_id="run-abandoned-operation-links",
        now=crash_time,
    )
    operation.__enter__()
    phase_context = operation.phase("apply")
    phase = phase_context.__enter__()
    try:
        phase.write_text(phase.r.file, "new\n", idempotency_key="write:abandoned-operation-links", newline="\n")
        operation_run = operation.operation_run
        operation_phase = phase.phase_record
        assert operation_run is not None
        assert operation_phase is not None
        operation._transaction._stop_lease_heartbeat()

        batch = latest_batch_for_run(workspace, "run-abandoned-operation-links")
        assert batch.phase == BatchPhase.SUCCEEDED
        assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "active"
        assert workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "active"

        recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-operation-links")
    finally:
        operation._transaction._stop_lease_heartbeat()

    batch = latest_batch_for_run(workspace, "run-abandoned-operation-links")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert workspace.claim_store.get(resource.resource_key) is None
    assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "finalization_failed"
    assert (
        workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "finalization_failed"
    )


@requires_backup_restore_support
def test_workspace_recover_pending_batches_retries_terminalization_before_claim_release(tmp_path: Path) -> None:
    class FailFirstPhaseTerminalDiagnosticJournal(OperationJournalStore):
        attempts = 0

        def record_operation_phase_terminal_diagnostic(self, *args: object, **kwargs: object) -> object:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("injected terminal diagnostic failure")
            return super().record_operation_phase_terminal_diagnostic(*args, **kwargs)

    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    journal = FailFirstPhaseTerminalDiagnosticJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    operation = workspace.operation(
        name="sync",
        resources={"file": resource},
        run_id="run-terminalization-retry",
        now=crash_time,
    )
    operation.__enter__()
    phase_context = operation.phase("apply")
    phase = phase_context.__enter__()
    try:
        phase.write_text(phase.r.file, "new\n", idempotency_key="write:terminalization-retry", newline="\n")
        operation_run = operation.operation_run
        operation_phase = phase.phase_record
        assert operation_run is not None
        assert operation_phase is not None
        operation._transaction._stop_lease_heartbeat()

        first_error = workspace.recover_pending_batches(run_id="run-terminalization-retry")
        batch = latest_batch_for_run(workspace, "run-terminalization-retry")

        assert first_error is not None
        assert "injected terminal diagnostic failure" in str(first_error.__cause__)
        assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
        assert target.read_text(encoding="utf-8") == "old\n"
        assert workspace.claim_store.get(resource.resource_key) is not None
        assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "active"
        assert workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "active"

        retry_error = workspace.recover_pending_batches(run_id="run-terminalization-retry")
    finally:
        operation._transaction._stop_lease_heartbeat()

    assert retry_error is None
    assert workspace.claim_store.get(resource.resource_key) is None
    assert workspace.journal_store.get_operation_run(operation_run.operation_run_id).status == "finalization_failed"
    assert (
        workspace.journal_store.get_operation_phase(operation_phase.operation_phase_id).status == "finalization_failed"
    )


@requires_backup_restore_support
def test_transaction_heartbeats_lease_during_long_file_mutation(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"

    def slow_write_text(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        time.sleep(0.35)
        portable_write_text(path, content, encoding=encoding, newline=newline)

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        lease_ttl=timedelta(milliseconds=180),
        snapshot=portable_snapshot,
        write_text_operation=slow_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
    )

    with workspace.transaction(
        name="apply",
        resources={"file": workspace.file(target)},
        run_id="run-slow-write",
        rollback="automatic",
    ) as tx:
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:slow")

    assert target.read_text(encoding="utf-8") == "new\n"
    batch = latest_batch_for_run(workspace, "run-slow-write")
    assert batch.phase == BatchPhase.SUCCEEDED


@requires_backup_restore_support
def test_workspace_recover_pending_batches_releases_claim_after_recovery_cleanup_debt(tmp_path: Path) -> None:
    cleanup_paths: list[Path] = []

    def fail_backup_cleanup(path: Path) -> None:
        cleanup_paths.append(path)
        raise PermissionError("injected backup cleanup failure")

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
        backup_artifact_cleanup_operation=fail_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )

    def write_then_crash(
        path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise KeyboardInterrupt("simulated process death")

    workspace._coordinator._write_text = write_then_crash
    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-public-recovery-cleanup-debt",
            idempotency_key="write:public-recovery-cleanup-debt",
            newline="\n",
            now=crash_time,
        )

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id="run-public-recovery-cleanup-debt")

    batch = latest_batch_for_run(workspace, "run-public-recovery-cleanup-debt")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert recovery_error is not None
    assert "artifact cleanup debt" in str(recovery_error.__cause__ or recovery_error)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is True
    assert cleanup_paths == [backup_path]
    assert workspace.claim_store.get(resource.resource_key) is None

    workspace.backup_artifact_cleanup_operation = None
    with workspace.transaction(
        name="after-recovery-cleanup-debt",
        resources={"file": workspace.file(target)},
        run_id="run-after-public-recovery-cleanup-debt",
    ) as tx:
        tx.write_text(tx.r.file, "next\n", idempotency_key="write:after-public-recovery-cleanup-debt", newline="\n")

    assert target.read_text(encoding="utf-8") == "next\n"


@requires_backup_restore_support
def test_workspace_recover_pending_batches_keeps_unrecovered_sibling_transaction_claim(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    manual_target = tmp_path / "manual.txt"
    recovered_target = tmp_path / "recovered.txt"
    manual_target.write_text("manual-old\n", encoding="utf-8")
    recovered_target.write_text("recovered-old\n", encoding="utf-8")
    manual_resource = workspace.file(manual_target)
    recovered_resource = workspace.file(recovered_target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    run_id = "run-public-recovery-sibling-claims"
    claim_scope = "transaction-instance:public-recovery-sibling-claims"
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        manual_resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        scope=claim_scope,
        details=_claim_details(lease),
        now=crash_time,
    )
    workspace.claim_store.upsert(
        recovered_resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        scope=claim_scope,
        details=_claim_details(lease),
        now=crash_time,
    )

    def write_then_crash(
        path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise KeyboardInterrupt("simulated process death")

    workspace._coordinator._write_text = write_then_crash
    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            manual_target,
            "manual-new\n",
            resource_key=manual_resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id=run_id,
            idempotency_key="write:manual",
            claim_scope=claim_scope,
            newline="\n",
            now=crash_time,
        )
    manual_batch = next(
        batch
        for batch in workspace.journal_store.list_batches()
        if batch.run_id == run_id and batch.resource_key == manual_resource.resource_key
    )
    backup_content_path_for_batch(workspace, manual_batch.batch_id).unlink()

    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            recovered_target,
            "recovered-new\n",
            resource_key=recovered_resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id=run_id,
            idempotency_key="write:recovered",
            claim_scope=claim_scope,
            newline="\n",
            now=crash_time,
        )
    recovered_batch = next(
        batch
        for batch in workspace.journal_store.list_batches()
        if batch.run_id == run_id and batch.resource_key == recovered_resource.resource_key
    )
    assert recovered_batch.batch_id != manual_batch.batch_id

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id=run_id)

    manual_batch = workspace.journal_store.get_batch(manual_batch.batch_id)
    recovered_batch = workspace.journal_store.get_batch(recovered_batch.batch_id)
    assert recovery_error is not None
    assert manual_batch is not None
    assert recovered_batch is not None
    assert manual_batch.phase == BatchPhase.RECOVERY_FAILED
    assert recovered_batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert manual_target.read_text(encoding="utf-8") == "manual-new\n"
    assert recovered_target.read_text(encoding="utf-8") == "recovered-old\n"
    assert workspace.claim_store.get(manual_resource.resource_key) is not None
    assert workspace.claim_store.get(recovered_resource.resource_key) is None


@requires_backup_restore_support
def test_workspace_recover_pending_batches_keeps_same_resource_claim_until_sibling_recovery_resolves(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    run_id = "run-public-recovery-same-resource-sibling-claims"
    claim_scope = "transaction-instance:public-recovery-same-resource-sibling-claims"
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        scope=claim_scope,
        details=_claim_details(lease),
        now=crash_time,
    )

    def write_then_crash(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise KeyboardInterrupt("simulated process death")

    workspace._coordinator._write_text = write_then_crash
    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            target,
            "manual-new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id=run_id,
            idempotency_key="write:manual-same-resource",
            claim_scope=claim_scope,
            newline="\n",
            now=crash_time,
        )
    manual_batch = next(
        batch
        for batch in workspace.journal_store.list_batches()
        if batch.run_id == run_id and batch.idempotency_key == "write:manual-same-resource"
    )
    backup_content_path_for_batch(workspace, manual_batch.batch_id).unlink()

    with pytest.raises(KeyboardInterrupt, match="simulated process death"):
        workspace._coordinator.write_text_file(
            target,
            "recovered-new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id=run_id,
            idempotency_key="write:recovered-same-resource",
            claim_scope=claim_scope,
            newline="\n",
            now=crash_time,
        )
    recovered_batch = next(
        batch
        for batch in workspace.journal_store.list_batches()
        if batch.run_id == run_id and batch.idempotency_key == "write:recovered-same-resource"
    )
    assert recovered_batch.batch_id != manual_batch.batch_id

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id=run_id)

    manual_batch = workspace.journal_store.get_batch(manual_batch.batch_id)
    recovered_batch = workspace.journal_store.get_batch(recovered_batch.batch_id)
    claim = workspace.claim_store.get(resource.resource_key)
    assert recovery_error is not None
    assert manual_batch is not None
    assert recovered_batch is not None
    assert manual_batch.phase == BatchPhase.RECOVERY_FAILED
    assert recovered_batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "manual-new\n"
    assert claim is not None
    assert claim.owner == workspace.owner
    assert claim.scope == claim_scope


@requires_backup_restore_support
def test_workspace_recover_pending_batches_noops_abandoned_file_mutation_without_side_effect(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)
    crash_time = datetime(2026, 1, 1, tzinfo=UTC)
    lease = workspace.lease_store.acquire(
        workspace.lease_name,
        owner=workspace.owner,
        ttl=timedelta(seconds=1),
        now=crash_time,
    )
    assert lease.acquired
    workspace.claim_store.upsert(
        resource.resource_key,
        lease=lease,
        owner=workspace.owner,
        details=_claim_details(lease),
        now=crash_time,
    )

    def crash_before_write(
        path: Path | str,
        content: str,
        *,
        encoding: str = "utf-8",
        newline: str | None = None,
    ) -> None:
        del path, content, encoding, newline
        raise KeyboardInterrupt("simulated process death before mutation")

    workspace._coordinator._write_text = crash_before_write
    with pytest.raises(KeyboardInterrupt, match="before mutation"):
        workspace._coordinator.write_text_file(
            target,
            "new\n",
            resource_key=resource.resource_key,
            lease=lease,
            owner=workspace.owner,
            run_id="run-abandoned-before-side-effect",
            idempotency_key="write:abandoned-before-side-effect",
            newline="\n",
            now=crash_time,
        )

    batch = latest_batch_for_run(workspace, "run-abandoned-before-side-effect")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert batch.phase == BatchPhase.ATTEMPTING
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is True

    workspace._coordinator._write_text = portable_write_text
    recovery_error = workspace.recover_pending_batches(run_id="run-abandoned-before-side-effect")

    batch = latest_batch_for_run(workspace, "run-abandoned-before-side-effect")
    assert recovery_error is None
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is False
    assert workspace.journal_store.list_recovery_actions(batch.batch_id) == []
    assert workspace.claim_store.get(resource.resource_key) is None


@requires_backup_restore_support
def test_transaction_clean_exit_rolls_back_failed_batch_without_recovery_intent(tmp_path: Path) -> None:
    class FailFirstRecoveryDesiredJournal(OperationJournalStore):
        attempts = 0

        def record_recovery_desired(self, *args: object, **kwargs: object) -> object:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("simulated recovery intent write loss")
            return super().record_recovery_desired(*args, **kwargs)

    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    journal = FailFirstRecoveryDesiredJournal(tmp_path / "state.db")
    workspace._journal_store = journal
    workspace._coordinator._journal_store = journal
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    def write_then_fail(path: Path | str, content: str, *, encoding: str = "utf-8", newline: str | None = None) -> None:
        Path(path).write_text(content, encoding=encoding, newline=newline)
        raise OSError("write failed after side effect")

    workspace._coordinator._write_text = write_then_fail
    with (
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-failed-gap",
            rollback="automatic",
        ) as tx,
        pytest.raises(RuntimeError, match="simulated recovery intent write loss"),
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:failed-gap", newline="\n")

    batch = latest_batch_for_run(workspace, "run-failed-gap")

    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"


@requires_backup_restore_support
def test_transaction_automatic_file_rollback_restores_overwritten_file_after_body_exception(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-overwrite-file",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:overwrite")
        raise RuntimeError("boom")

    batch = latest_batch_for_run(workspace, "run-overwrite-file")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is False
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]
    assert [record.status for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


@requires_backup_restore_support
def test_transaction_automatic_file_rollback_restores_deleted_file_after_body_exception(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-delete-file",
            rollback="automatic",
        ) as tx,
    ):
        tx.delete_file(tx.r.file, idempotency_key="delete:file")
        raise RuntimeError("boom")

    batch = latest_batch_for_run(workspace, "run-delete-file")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is False
    assert [record.status for record in workspace.journal_store.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "succeeded",
    ]


@requires_backup_restore_support
def test_transaction_automatic_rollback_rolls_back_file_and_recursive_mkdir_operations(tmp_path: Path) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target_dir = tmp_path / "config" / "state" / "cache"
    target_file = target_dir / "settings.json"

    with (
        pytest.raises(RuntimeError, match="boom"),
        workspace.transaction(
            name="apply",
            resources={
                "state_dir": workspace.directory(target_dir),
                "settings_file": workspace.file(target_file),
            },
            run_id="run-mixed",
            rollback="automatic",
        ) as tx,
    ):
        tx.make_directory(tx.r.state_dir, parents=True, idempotency_key="mkdir:state")
        tx.write_text(tx.r.settings_file, '{"mode":"test"}\n', idempotency_key="write:settings")
        raise RuntimeError("boom")

    batches = [batch for batch in workspace.journal_store.list_batches() if batch.run_id == "run-mixed"]
    assert batches
    assert all(batch.phase == BatchPhase.RECOVERY_SUCCEEDED for batch in batches)
    assert target_file.exists() is False
    assert target_dir.exists() is False
    assert target_dir.parent.exists() is False
    assert target_dir.parent.parent.exists() is False


@requires_backup_restore_support
@pytest.mark.parametrize("tamper_mode", ["missing", "tampered"])
def test_transaction_automatic_file_rollback_records_manual_intervention_for_missing_or_tampered_backup_artifact(
    tmp_path: Path,
    tamper_mode: str,
) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": resource},
            run_id=f"run-backup-{tamper_mode}",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key=f"write:{tamper_mode}")
        batch = latest_batch_for_run(workspace, f"run-backup-{tamper_mode}")
        backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
        if tamper_mode == "missing":
            backup_path.unlink()
        else:
            backup_path.write_text("tampered\n", encoding="utf-8")
        raise RuntimeError("boom")

    notes = getattr(excinfo.value, "__notes__", None)
    assert notes is not None
    assert any("automatic rollback also failed" in note for note in notes)
    batch = latest_batch_for_run(workspace, f"run-backup-{tamper_mode}")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    recovery_actions = workspace.journal_store.list_recovery_actions(batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_FAILED
    assert target.read_text(encoding="utf-8") == "new\n"
    if tamper_mode == "tampered":
        assert backup_path.exists() is True
    assert recovery_actions[-1].status == "manual_intervention_required"
    assert workspace.lease_store.active(workspace.lease_name) is not None
    assert workspace.claim_store.get(resource.resource_key) is not None


@requires_backup_restore_support
def test_transaction_automatic_file_rollback_surfaces_backup_artifact_cleanup_debt_without_failing_recovery(
    tmp_path: Path,
) -> None:
    backup_paths: list[Path] = []

    def fail_backup_cleanup(path: Path) -> None:
        backup_paths.append(path)
        raise PermissionError("injected backup cleanup failure")

    workspace = SafeWorkspace.open(
        tmp_path / "state.db",
        owner="owner-a",
        snapshot=portable_snapshot,
        write_text_operation=portable_write_text,
        delete_file_operation=portable_delete_file,
        make_directory_operation=portable_make_directory,
        remove_directory_operation=hook_capable_remove_directory,
        identity_remove_directory_operation=portable_remove_empty_directory_by_identity,
        enable_automatic_file_rollback=True,
        backup_artifact_cleanup_operation=fail_backup_cleanup,
    )
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": workspace.file(target)},
            run_id="run-rollback-cleanup-debt",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:rollback-cleanup-debt")
        raise RuntimeError("boom")

    batch = latest_batch_for_run(workspace, "run-rollback-cleanup-debt")
    backup_path = backup_content_path_for_batch(workspace, batch.batch_id)
    notes = getattr(excinfo.value, "__notes__", None)
    assert notes is not None
    assert any("artifact cleanup also failed" in note for note in notes)
    assert batch.phase == BatchPhase.RECOVERY_SUCCEEDED
    assert target.read_text(encoding="utf-8") == "old\n"
    assert backup_path.exists() is True
    assert backup_paths == [backup_path]
    assert [record.status for record in workspace.journal_store.list_artifact_cleanup_records(batch.batch_id)] == [
        "planned",
        "attempting",
        "failed",
    ]


@requires_backup_restore_support
def test_transaction_automatic_file_rollback_records_manual_intervention_for_tampered_current_target(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_file_rollback(tmp_path / "state.db")
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    resource = workspace.file(target)

    with (
        pytest.raises(RuntimeError, match="boom") as excinfo,
        workspace.transaction(
            name="apply",
            resources={"file": resource},
            run_id="run-current-tamper",
            rollback="automatic",
        ) as tx,
    ):
        tx.write_text(tx.r.file, "new\n", idempotency_key="write:tamper-current")
        target.write_text("tampered-current\n", encoding="utf-8")
        raise RuntimeError("boom")

    notes = getattr(excinfo.value, "__notes__", None)
    assert notes is not None
    assert any("automatic rollback also failed" in note for note in notes)
    batch = latest_batch_for_run(workspace, "run-current-tamper")
    recovery_actions = workspace.journal_store.list_recovery_actions(batch.batch_id)
    assert batch.phase == BatchPhase.RECOVERY_FAILED
    assert target.read_text(encoding="utf-8") == "tampered-current\n"
    assert recovery_actions[-1].status == "manual_intervention_required"
    assert workspace.lease_store.active(workspace.lease_name) is not None
    assert workspace.claim_store.get(resource.resource_key) is not None
