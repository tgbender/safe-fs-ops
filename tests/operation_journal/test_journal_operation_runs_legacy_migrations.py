from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _journal_test_helpers import (
    acquire_lease,
    create_legacy_journal_db,
    create_legacy_operation_run_identity_db,
)

from safe_fs_ops.operation_journal import OperationJournalStore

pytestmark = pytest.mark.safe_fs_ops


def test_initialize_creates_operation_run_tables_and_batch_link_columns(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    store = OperationJournalStore(state_path)

    store.initialize()

    with sqlite3.connect(state_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'",
            ).fetchall()
        }
        batch_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(operation_batches)",
            ).fetchall()
        }

    assert "operation_runs" in tables
    assert "operation_phase_records" in tables
    assert "operation_run_id" in batch_columns
    assert "operation_phase_id" in batch_columns


def test_initialize_legacy_db_preserves_batch_data_with_null_operation_links(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    create_legacy_journal_db(state_path, now=now)
    store = OperationJournalStore(state_path)

    store.initialize()

    batch = store.get_batch("batch-1")
    operations = store.list_operations("batch-1")

    assert batch is not None
    assert batch.operation_run_id is None
    assert batch.operation_phase_id is None
    assert batch.batch_id == "batch-1"
    assert [record.operation_id for record in operations] == ["operation-1"]


def test_list_batches_and_recovery_reads_tolerate_null_operation_links(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    create_legacy_journal_db(state_path, now=now)
    store = OperationJournalStore(state_path)

    store.initialize()

    batches = store.list_batches(run_id="run-1")
    context = store.read_recovery_context("batch-1")

    assert [batch.batch_id for batch in batches] == ["batch-1"]
    assert batches[0].operation_run_id is None
    assert batches[0].operation_phase_id is None
    assert context.batch.operation_run_id is None
    assert context.batch.operation_phase_id is None
    assert [record.operation_id for record in context.operations] == ["operation-1"]
    assert [record.checkpoint_id for record in context.checkpoints] == ["checkpoint-1"]
    assert [record.recovery_id for record in context.recovery_records] == ["recovery-1"]


def test_initialize_migrates_operation_run_identity_to_allow_shared_public_run_ids(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    create_legacy_operation_run_identity_db(state_path, now=now)
    store = OperationJournalStore(state_path)

    store.initialize()

    lease = acquire_lease(state_path, now=now + timedelta(seconds=1))
    second_run = store.create_operation_run(
        run_id="run-1",
        lease=lease,
        owner="owner-a",
        status="active",
        operation_run_id="operation-run-2",
        now=now + timedelta(seconds=2),
    )

    operation_runs = store.list_operation_runs(run_id="run-1", owner="owner-a")
    assert [record.operation_run_id for record in operation_runs] == ["operation-run-1", second_run.operation_run_id]


def test_initialize_rebuilds_operation_run_identity_without_breaking_phase_foreign_keys(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    create_legacy_operation_run_identity_db(state_path, now=now)
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            INSERT INTO operation_phase_records (
                operation_phase_id,
                operation_run_id,
                phase_name,
                status,
                phase_order,
                payload,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "phase-1",
                "operation-run-1",
                "prepare",
                "active",
                1,
                "{}",
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.commit()
    store = OperationJournalStore(state_path)

    store.initialize()

    with sqlite3.connect(state_path) as connection:
        connection.row_factory = sqlite3.Row
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()

    lease = acquire_lease(state_path, now=now + timedelta(seconds=1))
    second_phase = store.create_operation_phase(
        operation_run_id="operation-run-1",
        lease=lease,
        phase_name="apply",
        status="active",
        phase_order=2,
        operation_phase_id="phase-2",
        now=now + timedelta(seconds=2),
    )

    assert foreign_key_errors == []
    assert second_phase.operation_phase_id == "phase-2"
    assert [record.phase_name for record in store.list_operation_phases("operation-run-1")] == ["prepare", "apply"]
