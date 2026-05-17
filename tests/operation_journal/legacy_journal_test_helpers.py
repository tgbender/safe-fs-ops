from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from safe_fs_ops.operation_journal import BatchPhase
from safe_fs_ops.workspace_state import LeaseStore
from safe_fs_ops.workspace_state.models import LeaseRecord


def create_legacy_journal_db(state_path: Path, *, now: datetime) -> None:
    payload_text = json.dumps({"kind": "legacy"})
    status_payload_text = json.dumps({"legacy": True})
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            CREATE TABLE operation_batches (
                batch_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                lease_name TEXT NOT NULL,
                lease_fencing_token INTEGER NOT NULL,
                owner TEXT NOT NULL,
                run_id TEXT NOT NULL,
                resource_key TEXT,
                claim_owner TEXT,
                claim_scope TEXT,
                phase TEXT NOT NULL,
                payload TEXT NOT NULL,
                status_message TEXT,
                status_payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_journal_operations (
                operation_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES operation_batches (batch_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                operation_type TEXT NOT NULL,
                resource_key TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (batch_id, sequence)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_journal_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES operation_batches (batch_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                operation_id TEXT REFERENCES operation_journal_operations (operation_id) ON DELETE SET NULL,
                resource_key TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (batch_id, sequence)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_recovery_records (
                recovery_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES operation_batches (batch_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                phase TEXT NOT NULL,
                reason TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (batch_id, sequence)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_recovery_action_records (
                action_record_id TEXT PRIMARY KEY,
                action_id TEXT NOT NULL,
                recovery_attempt_id TEXT NOT NULL REFERENCES operation_recovery_records (recovery_id) ON DELETE CASCADE,
                batch_id TEXT NOT NULL REFERENCES operation_batches (batch_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                status TEXT NOT NULL,
                resource_key TEXT,
                reason TEXT,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (batch_id, sequence)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO operation_batches (
                batch_id,
                idempotency_key,
                lease_name,
                lease_fencing_token,
                owner,
                run_id,
                resource_key,
                claim_owner,
                claim_scope,
                phase,
                payload,
                status_message,
                status_payload,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "batch-1",
                "legacy:batch-1",
                "workspace",
                7,
                "owner-a",
                "run-1",
                "file:a",
                "owner-a",
                "install",
                BatchPhase.PLANNED,
                payload_text,
                None,
                status_payload_text,
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO operation_journal_operations (
                operation_id,
                batch_id,
                sequence,
                operation_type,
                resource_key,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "operation-1",
                "batch-1",
                1,
                "write_text",
                "file:a",
                payload_text,
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO operation_journal_checkpoints (
                checkpoint_id,
                batch_id,
                sequence,
                operation_id,
                resource_key,
                checkpoint_type,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "checkpoint-1",
                "batch-1",
                2,
                "operation-1",
                "file:a",
                "before",
                payload_text,
                now.isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO operation_recovery_records (
                recovery_id,
                batch_id,
                sequence,
                phase,
                reason,
                payload,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "recovery-1",
                "batch-1",
                3,
                BatchPhase.RECOVERY_DESIRED,
                "legacy recovery",
                payload_text,
                now.isoformat(),
            ),
        )
        connection.commit()


def create_legacy_operation_run_identity_db(state_path: Path, *, now: datetime) -> None:
    with sqlite3.connect(state_path) as connection:
        connection.execute(
            """
            CREATE TABLE workspace_leases (
                name TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                token TEXT NOT NULL,
                fencing_token INTEGER NOT NULL,
                acquired_at TEXT NOT NULL,
                heartbeat_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_runs (
                operation_run_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                lease_name TEXT NOT NULL,
                lease_fencing_token INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (run_id, owner)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE operation_phase_records (
                operation_phase_id TEXT PRIMARY KEY,
                operation_run_id TEXT NOT NULL
                    REFERENCES operation_runs (operation_run_id) ON DELETE CASCADE,
                phase_name TEXT NOT NULL,
                status TEXT NOT NULL,
                phase_order INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (operation_run_id, phase_name)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_operation_runs_run_owner
            ON operation_runs (run_id, owner, created_at, operation_run_id)
            """
        )
        connection.execute(
            """
            CREATE INDEX idx_operation_phase_records_run_order
            ON operation_phase_records (operation_run_id, phase_order, operation_phase_id)
            """
        )
        connection.execute(
            """
            INSERT INTO operation_runs (
                operation_run_id,
                run_id,
                owner,
                lease_name,
                lease_fencing_token,
                status,
                payload,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "operation-run-1",
                "run-1",
                "owner-a",
                "workspace",
                1,
                "active",
                json.dumps({"legacy": True}),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        connection.commit()


def acquire_lease(
    state_path: Path,
    *,
    now: datetime,
    owner: str = "runner-a",
    name: str = "workspace",
) -> LeaseRecord:
    return LeaseStore(state_path).acquire(name, owner=owner, ttl=timedelta(seconds=30), now=now)
