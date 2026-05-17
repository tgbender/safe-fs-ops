from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypeVar

from safe_fs_ops.operation_journal.journal_artifact_cleanup import JournalArtifactCleanupMixin
from safe_fs_ops.operation_journal.journal_batch import JournalBatchMixin
from safe_fs_ops.operation_journal.journal_errors import (
    BatchIdempotencyMismatchError,
    BatchNotFoundError,
    BatchStartConflictError,
    InvalidBatchPhaseTransitionError,
    JournalError,
    JournalLeaseMismatchError,
    RecoveryAttemptMismatchError,
    RecoveryContextError,
)
from safe_fs_ops.operation_journal.journal_operation_runs import JournalOperationRunMixin
from safe_fs_ops.operation_journal.journal_recovery import JournalRecoveryMixin
from safe_fs_ops.operation_journal.journal_recovery_actions import JournalRecoveryActionMixin
from safe_fs_ops.operation_journal.journal_recovery_reads import JournalRecoveryReadMixin
from safe_fs_ops.operation_journal.journal_rows import (
    _ensure_batch_authority_columns,
    _ensure_batch_operation_link_columns,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _JOURNAL_SCHEMA,
    _OPERATION_PHASE_SELECT_LIST,
    _OPERATION_PHASE_TABLE,
    _OPERATION_RUN_SELECT_LIST,
    _OPERATION_RUN_TABLE,
    BatchPhase,
    _OperationPhaseColumn,
    _OperationRunColumn,
)
from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import lease_schema

__all__ = [
    "BatchIdempotencyMismatchError",
    "BatchNotFoundError",
    "BatchPhase",
    "BatchStartConflictError",
    "JournalError",
    "JournalLeaseMismatchError",
    "InvalidBatchPhaseTransitionError",
    "OperationJournalStore",
    "RecoveryAttemptMismatchError",
    "RecoveryContextError",
]


_T = TypeVar("_T")


class OperationJournalStore(
    JournalOperationRunMixin,
    JournalBatchMixin,
    JournalArtifactCleanupMixin,
    JournalRecoveryReadMixin,
    JournalRecoveryMixin,
    JournalRecoveryActionMixin,
):
    """SQLite-backed operation batches, journal records, checkpoints, and recovery state."""

    def __init__(self, path: Path | str, *, sqlite_store: SqliteStore | None = None) -> None:
        self.sqlite_store = sqlite_store or SqliteStore(path)

    @property
    def path(self) -> Path | str:
        return self.sqlite_store.path

    def initialize(self) -> None:
        self.sqlite_store.initialize((*lease_schema(), *_JOURNAL_SCHEMA))
        with self.sqlite_store.transaction() as connection:
            _migrate_operation_run_identity(connection)
            _ensure_batch_authority_columns(connection)
            _ensure_batch_operation_link_columns(connection)


def _migrate_operation_run_identity(connection: sqlite3.Connection) -> None:
    if not _operation_run_identity_uses_public_columns(connection):
        return
    legacy_run_table = f"{_OPERATION_RUN_TABLE}_legacy_identity"
    legacy_phase_table = f"{_OPERATION_PHASE_TABLE}_legacy_identity"
    has_phase_table = _table_exists(connection, _OPERATION_PHASE_TABLE)
    if has_phase_table:
        connection.execute(f"ALTER TABLE {_OPERATION_PHASE_TABLE} RENAME TO {legacy_phase_table}")
    connection.execute(f"ALTER TABLE {_OPERATION_RUN_TABLE} RENAME TO {legacy_run_table}")
    _create_operation_run_table(connection)
    connection.execute(
        f"""
        INSERT INTO {_OPERATION_RUN_TABLE} (
            {_OperationRunColumn.OPERATION_RUN_ID},
            {_OperationRunColumn.RUN_ID},
            {_OperationRunColumn.OWNER},
            {_OperationRunColumn.LEASE_NAME},
            {_OperationRunColumn.LEASE_FENCING_TOKEN},
            {_OperationRunColumn.STATUS},
            {_OperationRunColumn.PAYLOAD},
            {_OperationRunColumn.CREATED_AT},
            {_OperationRunColumn.UPDATED_AT}
        )
        SELECT {_OPERATION_RUN_SELECT_LIST}
        FROM {legacy_run_table}
        """
    )
    if has_phase_table:
        _create_operation_phase_table(connection)
        connection.execute(
            f"""
            INSERT INTO {_OPERATION_PHASE_TABLE} (
                {_OperationPhaseColumn.OPERATION_PHASE_ID},
                {_OperationPhaseColumn.OPERATION_RUN_ID},
                {_OperationPhaseColumn.PHASE_NAME},
                {_OperationPhaseColumn.STATUS},
                {_OperationPhaseColumn.PHASE_ORDER},
                {_OperationPhaseColumn.PAYLOAD},
                {_OperationPhaseColumn.CREATED_AT},
                {_OperationPhaseColumn.UPDATED_AT}
            )
            SELECT {_OPERATION_PHASE_SELECT_LIST}
            FROM {legacy_phase_table}
            """
        )
        connection.execute(f"DROP TABLE {legacy_phase_table}")
    connection.execute(f"DROP TABLE {legacy_run_table}")
    _create_operation_run_indexes(connection)
    _create_operation_phase_indexes(connection)


def _operation_run_identity_uses_public_columns(connection: sqlite3.Connection) -> bool:
    table_names = {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'",
        ).fetchall()
    }
    if _OPERATION_RUN_TABLE not in table_names:
        return False
    for row in connection.execute(f"PRAGMA index_list({_OPERATION_RUN_TABLE})").fetchall():
        if int(row["unique"]) != 1:
            continue
        index_name = str(row["name"])
        columns = [
            str(index_row["name"]) for index_row in connection.execute(f"PRAGMA index_info({index_name})").fetchall()
        ]
        if columns == [_OperationRunColumn.RUN_ID, _OperationRunColumn.OWNER]:
            return True
    return False


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _create_operation_run_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE {_OPERATION_RUN_TABLE} (
            {_OperationRunColumn.OPERATION_RUN_ID} TEXT PRIMARY KEY,
            {_OperationRunColumn.RUN_ID} TEXT NOT NULL,
            {_OperationRunColumn.OWNER} TEXT NOT NULL,
            {_OperationRunColumn.LEASE_NAME} TEXT NOT NULL,
            {_OperationRunColumn.LEASE_FENCING_TOKEN} INTEGER NOT NULL,
            {_OperationRunColumn.STATUS} TEXT NOT NULL,
            {_OperationRunColumn.PAYLOAD} TEXT NOT NULL,
            {_OperationRunColumn.CREATED_AT} TEXT NOT NULL,
            {_OperationRunColumn.UPDATED_AT} TEXT NOT NULL
        )
        """
    )


def _create_operation_phase_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE {_OPERATION_PHASE_TABLE} (
            {_OperationPhaseColumn.OPERATION_PHASE_ID} TEXT PRIMARY KEY,
            {_OperationPhaseColumn.OPERATION_RUN_ID} TEXT NOT NULL
                REFERENCES {_OPERATION_RUN_TABLE} ({_OperationRunColumn.OPERATION_RUN_ID}) ON DELETE CASCADE,
            {_OperationPhaseColumn.PHASE_NAME} TEXT NOT NULL,
            {_OperationPhaseColumn.STATUS} TEXT NOT NULL,
            {_OperationPhaseColumn.PHASE_ORDER} INTEGER NOT NULL,
            {_OperationPhaseColumn.PAYLOAD} TEXT NOT NULL,
            {_OperationPhaseColumn.CREATED_AT} TEXT NOT NULL,
            {_OperationPhaseColumn.UPDATED_AT} TEXT NOT NULL,
            UNIQUE ({_OperationPhaseColumn.OPERATION_RUN_ID}, {_OperationPhaseColumn.PHASE_NAME})
        )
        """
    )


def _create_operation_run_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{_OPERATION_RUN_TABLE}_run_owner
        ON {_OPERATION_RUN_TABLE} (
            {_OperationRunColumn.RUN_ID},
            {_OperationRunColumn.OWNER},
            {_OperationRunColumn.CREATED_AT},
            {_OperationRunColumn.OPERATION_RUN_ID}
        )
        """
    )


def _create_operation_phase_indexes(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{_OPERATION_PHASE_TABLE}_run_order
        ON {_OPERATION_PHASE_TABLE} (
            {_OperationPhaseColumn.OPERATION_RUN_ID},
            {_OperationPhaseColumn.PHASE_ORDER},
            {_OperationPhaseColumn.OPERATION_PHASE_ID}
        )
        """
    )
