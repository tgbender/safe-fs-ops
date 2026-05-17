from __future__ import annotations

from pathlib import Path

import pytest
from workspace_api_helpers import workspace_with_portable_ops

pytestmark = pytest.mark.safe_fs_ops


def test_transaction_recursive_mkdir_rejects_phase_without_run_id(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    with pytest.raises(ValueError, match="operation_phase_id requires operation_run_id"):
        with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
            tx_ref = tx
            operation_run = workspace.journal_store.create_operation_run(
                run_id=tx.run_id,
                lease=tx.lease,
                owner=workspace.owner,
                status="active",
                payload={"name": tx.name, "rollback": tx.rollback, "api": "transaction"},
                now=tx.now,
            )
            operation_phase = workspace.journal_store.create_operation_phase(
                operation_run_id=operation_run.operation_run_id,
                lease=tx.lease,
                phase_name="transaction",
                status="active",
                phase_order=1,
                payload={"implicit": False, "rollback": tx.rollback},
                now=tx.now,
            )
            tx.make_directory(
                tx.r.state_dir,
                parents=True,
                idempotency_key="mkdir:state",
                operation_phase_id=operation_phase.operation_phase_id,
            )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


def test_transaction_recursive_mkdir_rejects_mismatched_explicit_phase_and_run_ids(tmp_path: Path) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    with pytest.raises(ValueError, match="belongs to operation_run_id"):
        with workspace.transaction(name="apply", resources=resources, run_id="run-2") as tx:
            tx_ref = tx
            first_run = workspace.journal_store.create_operation_run(
                run_id="run-1",
                lease=tx.lease,
                owner=workspace.owner,
                status="active",
                payload={"name": "first"},
                operation_run_id="operation-run-1",
                now=tx.now,
            )
            second_run = workspace.journal_store.create_operation_run(
                run_id=tx.run_id,
                lease=tx.lease,
                owner=workspace.owner,
                status="active",
                payload={"name": "second"},
                operation_run_id="operation-run-2",
                now=tx.now,
            )
            phase = workspace.journal_store.create_operation_phase(
                operation_run_id=first_run.operation_run_id,
                lease=tx.lease,
                phase_name="transaction",
                status="active",
                phase_order=1,
                payload={"source": "first"},
                operation_phase_id="phase-1",
                now=tx.now,
            )
            tx.make_directory(
                tx.r.state_dir,
                parents=True,
                idempotency_key="mkdir:state",
                operation_run_id=second_run.operation_run_id,
                operation_phase_id=phase.operation_phase_id,
            )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert workspace.journal_store.get_operation_run("operation-run-1").status == "active"
    assert workspace.journal_store.get_operation_run("operation-run-2").status == "active"
    assert workspace.journal_store.get_operation_phase("phase-1").status == "active"
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


@pytest.mark.parametrize(
    ("run_id", "owner", "expected_message"),
    [
        ("run-2", "owner-a", "belongs to run_id 'run-2'"),
        ("run-1", "owner-b", "belongs to owner 'owner-b'"),
    ],
)
def test_transaction_recursive_mkdir_rejects_explicit_operation_run_with_wrong_identity(
    tmp_path: Path,
    run_id: str,
    owner: str,
    expected_message: str,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    with pytest.raises(ValueError, match=expected_message):
        with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
            tx_ref = tx
            operation_run = workspace.journal_store.create_operation_run(
                run_id=run_id,
                lease=tx.lease,
                owner=owner,
                status="active",
                payload={"name": "foreign"},
                operation_run_id="operation-run-foreign",
                now=tx.now,
            )
            tx.make_directory(
                tx.r.state_dir,
                parents=True,
                idempotency_key="mkdir:state",
                operation_run_id=operation_run.operation_run_id,
            )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert workspace.journal_store.get_operation_run("operation-run-foreign").status == "active"
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


def test_transaction_recursive_mkdir_rejects_foreign_explicit_links_from_prior_transaction_instance(
    tmp_path: Path,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})

    with workspace.transaction(name="apply", resources=resources, run_id="run-1") as first_tx:
        explicit_run = workspace.journal_store.create_operation_run(
            run_id=first_tx.run_id,
            lease=first_tx.lease,
            owner=workspace.owner,
            status="active",
            payload={"name": first_tx.name, "instance": "first"},
            operation_run_id="operation-run-foreign",
            now=first_tx.now,
        )
        explicit_phase = workspace.journal_store.create_operation_phase(
            operation_run_id=explicit_run.operation_run_id,
            lease=first_tx.lease,
            phase_name="explicit",
            status="active",
            phase_order=1,
            payload={"instance": "first"},
            operation_phase_id="operation-phase-foreign",
            now=first_tx.now,
        )

    tx_ref = None
    with pytest.raises(ValueError, match="belongs to fencing token"):
        with workspace.transaction(name="apply", resources=resources, run_id="run-1") as second_tx:
            tx_ref = second_tx
            second_tx.make_directory(
                second_tx.r.state_dir,
                parents=True,
                idempotency_key="mkdir:state",
                operation_run_id=explicit_run.operation_run_id,
                operation_phase_id=explicit_phase.operation_phase_id,
            )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert workspace.journal_store.get_operation_run(explicit_run.operation_run_id).status == "active"
    assert workspace.journal_store.get_operation_phase(explicit_phase.operation_phase_id).status == "active"
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None


@pytest.mark.parametrize("status", ["succeeded", "failed", "finalization_failed"])
def test_transaction_recursive_mkdir_rejects_terminal_explicit_operation_links(
    tmp_path: Path,
    status: str,
) -> None:
    workspace = workspace_with_portable_ops(tmp_path / "state.db")
    target_path = tmp_path / "config" / "state" / "cache"
    resources = workspace.resources({"state_dir": workspace.directory(target_path)})
    tx_ref = None

    with pytest.raises(ValueError, match=f"status {status!r} is terminal"):
        with workspace.transaction(name="apply", resources=resources, run_id="run-1") as tx:
            tx_ref = tx
            explicit_run = workspace.journal_store.create_operation_run(
                run_id=tx.run_id,
                lease=tx.lease,
                owner=workspace.owner,
                status=status,
                payload={"name": "terminal"},
                operation_run_id="operation-run-terminal",
                now=tx.now,
            )
            explicit_phase = workspace.journal_store.create_operation_phase(
                operation_run_id=explicit_run.operation_run_id,
                lease=tx.lease,
                phase_name="explicit",
                status=status,
                phase_order=1,
                payload={"terminal": True},
                operation_phase_id="operation-phase-terminal",
                now=tx.now,
            )
            tx.make_directory(
                tx.r.state_dir,
                parents=True,
                idempotency_key="mkdir:state",
                operation_run_id=explicit_run.operation_run_id,
                operation_phase_id=explicit_phase.operation_phase_id,
            )

    assert tx_ref is not None
    assert tx_ref.operation_run is None
    assert tx_ref.operation_phase is None
    assert not target_path.exists()
    assert workspace.journal_store.list_batches() == []
    assert workspace.claim_store.get(resources.state_dir.resource_key) is None
