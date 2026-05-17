from __future__ import annotations

import sqlite3
from typing import cast

from safe_fs_ops.operation_journal.journal_payloads import _frozen_payload, _optional_str, _parse_datetime
from safe_fs_ops.operation_journal.journal_schema import (
    _ARTIFACT_CLEANUP_SELECT_LIST,
    _ARTIFACT_CLEANUP_TABLE,
    _BATCH_SELECT_LIST,
    _BATCH_TABLE,
    _CHECKPOINT_SELECT_LIST,
    _CHECKPOINT_TABLE,
    _OPERATION_SELECT_LIST,
    _OPERATION_TABLE,
    _RECOVERY_ACTION_SELECT_LIST,
    _RECOVERY_ACTION_TABLE,
    _RECOVERY_SELECT_LIST,
    _RECOVERY_TABLE,
    BatchPhase,
    _ArtifactCleanupColumn,
    _BatchColumn,
    _CheckpointColumn,
    _OperationColumn,
    _RecoveryActionColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import (
    ARTIFACT_CLEANUP_DEBT_STATUSES,
    ARTIFACT_CLEANUP_OUTSTANDING_STATUSES,
    ArtifactCleanupRecord,
    CheckpointRecord,
    OperationBatchRecord,
    OperationRecord,
    RecoveryActionRecord,
    RecoveryRecord,
)


def _next_sequence(connection: sqlite3.Connection, batch_id: str) -> int:
    operation_sequence = _max_sequence(connection, _OPERATION_TABLE, _OperationColumn.BATCH_ID, batch_id)
    checkpoint_sequence = _max_sequence(connection, _CHECKPOINT_TABLE, _CheckpointColumn.BATCH_ID, batch_id)
    recovery_sequence = _max_sequence(connection, _RECOVERY_TABLE, _RecoveryColumn.BATCH_ID, batch_id)
    recovery_action_sequence = _max_sequence(
        connection,
        _RECOVERY_ACTION_TABLE,
        _RecoveryActionColumn.BATCH_ID,
        batch_id,
    )
    artifact_cleanup_sequence = _max_sequence(
        connection,
        _ARTIFACT_CLEANUP_TABLE,
        _ArtifactCleanupColumn.BATCH_ID,
        batch_id,
    )
    return (
        max(
            operation_sequence,
            checkpoint_sequence,
            recovery_sequence,
            recovery_action_sequence,
            artifact_cleanup_sequence,
        )
        + 1
    )


def _max_sequence(connection: sqlite3.Connection, table: str, batch_column: str, batch_id: str) -> int:
    row = connection.execute(
        f"""
        SELECT MAX(sequence) AS sequence
        FROM {table}
        WHERE {batch_column} = ?
        """,
        (batch_id,),
    ).fetchone()
    value = row["sequence"]
    return 0 if value is None else int(value)


def _batch_has_journal_records(connection: sqlite3.Connection, batch_id: str) -> bool:
    for table, batch_column in (
        (_OPERATION_TABLE, _OperationColumn.BATCH_ID),
        (_CHECKPOINT_TABLE, _CheckpointColumn.BATCH_ID),
        (_RECOVERY_TABLE, _RecoveryColumn.BATCH_ID),
        (_RECOVERY_ACTION_TABLE, _RecoveryActionColumn.BATCH_ID),
        (_ARTIFACT_CLEANUP_TABLE, _ArtifactCleanupColumn.BATCH_ID),
    ):
        row = connection.execute(
            f"""
            SELECT 1
            FROM {table}
            WHERE {batch_column} = ?
            LIMIT 1
            """,
            (batch_id,),
        ).fetchone()
        if row is not None:
            return True
    return False


def _batch_row(connection: sqlite3.Connection, batch_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_BATCH_SELECT_LIST}
            FROM {_BATCH_TABLE}
            WHERE {_BatchColumn.BATCH_ID} = ?
            """,
            (batch_id,),
        ).fetchone(),
    )


def _batch_row_by_idempotency_key(connection: sqlite3.Connection, idempotency_key: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_BATCH_SELECT_LIST}
            FROM {_BATCH_TABLE}
            WHERE {_BatchColumn.IDEMPOTENCY_KEY} = ?
            """,
            (idempotency_key,),
        ).fetchone(),
    )


def _list_operations_in_connection(connection: sqlite3.Connection, batch_id: str) -> list[OperationRecord]:
    rows = connection.execute(
        f"""
        SELECT {_OPERATION_SELECT_LIST}
        FROM {_OPERATION_TABLE}
        WHERE {_OperationColumn.BATCH_ID} = ?
        ORDER BY {_OperationColumn.SEQUENCE}, {_OperationColumn.OPERATION_ID}
        """,
        (batch_id,),
    ).fetchall()
    return [_operation_from_row(row) for row in rows]


def _list_checkpoints_in_connection(connection: sqlite3.Connection, batch_id: str) -> list[CheckpointRecord]:
    rows = connection.execute(
        f"""
        SELECT {_CHECKPOINT_SELECT_LIST}
        FROM {_CHECKPOINT_TABLE}
        WHERE {_CheckpointColumn.BATCH_ID} = ?
        ORDER BY {_CheckpointColumn.SEQUENCE}, {_CheckpointColumn.CHECKPOINT_ID}
        """,
        (batch_id,),
    ).fetchall()
    return [_checkpoint_from_row(row) for row in rows]


def _list_recovery_records_in_connection(connection: sqlite3.Connection, batch_id: str) -> list[RecoveryRecord]:
    rows = connection.execute(
        f"""
        SELECT {_RECOVERY_SELECT_LIST}
        FROM {_RECOVERY_TABLE}
        WHERE {_RecoveryColumn.BATCH_ID} = ?
        ORDER BY {_RecoveryColumn.SEQUENCE}, {_RecoveryColumn.RECOVERY_ID}
        """,
        (batch_id,),
    ).fetchall()
    return [_recovery_from_row(row) for row in rows]


def _list_recovery_actions_in_connection(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    recovery_attempt_id: str | None = None,
) -> list[RecoveryActionRecord]:
    where_parts = [f"{_RecoveryActionColumn.BATCH_ID} = ?"]
    parameters: list[str] = [batch_id]
    if recovery_attempt_id is not None:
        where_parts.append(f"{_RecoveryActionColumn.RECOVERY_ATTEMPT_ID} = ?")
        parameters.append(recovery_attempt_id)
    rows = connection.execute(
        f"""
        SELECT {_RECOVERY_ACTION_SELECT_LIST}
        FROM {_RECOVERY_ACTION_TABLE}
        WHERE {" AND ".join(where_parts)}
        ORDER BY {_RecoveryActionColumn.SEQUENCE}, {_RecoveryActionColumn.ACTION_RECORD_ID}
        """,
        tuple(parameters),
    ).fetchall()
    return [_recovery_action_from_row(row) for row in rows]


def _list_artifact_cleanup_records_in_connection(
    connection: sqlite3.Connection,
    batch_id: str,
    *,
    artifact_id: str | None = None,
) -> list[ArtifactCleanupRecord]:
    where_parts = [f"{_ArtifactCleanupColumn.BATCH_ID} = ?"]
    parameters: list[str] = [batch_id]
    if artifact_id is not None:
        where_parts.append(f"{_ArtifactCleanupColumn.ARTIFACT_ID} = ?")
        parameters.append(artifact_id)
    rows = connection.execute(
        f"""
        SELECT {_ARTIFACT_CLEANUP_SELECT_LIST}
        FROM {_ARTIFACT_CLEANUP_TABLE}
        WHERE {" AND ".join(where_parts)}
        ORDER BY {_ArtifactCleanupColumn.SEQUENCE}, {_ArtifactCleanupColumn.CLEANUP_RECORD_ID}
        """,
        tuple(parameters),
    ).fetchall()
    return [_artifact_cleanup_from_row(row) for row in rows]


def _list_unresolved_artifact_cleanup_debt_in_connection(connection: sqlite3.Connection) -> list[ArtifactCleanupRecord]:
    return _list_latest_artifact_cleanup_rows_with_statuses_in_connection(
        connection,
        statuses=ARTIFACT_CLEANUP_DEBT_STATUSES,
    )


def _list_outstanding_artifact_cleanup_records_in_connection(
    connection: sqlite3.Connection,
) -> list[ArtifactCleanupRecord]:
    return _list_latest_artifact_cleanup_rows_with_statuses_in_connection(
        connection,
        statuses=ARTIFACT_CLEANUP_OUTSTANDING_STATUSES,
    )


def _list_latest_artifact_cleanup_rows_with_statuses_in_connection(
    connection: sqlite3.Connection,
    *,
    statuses: frozenset[str],
) -> list[ArtifactCleanupRecord]:
    placeholders = ", ".join("?" for _ in statuses)
    rows = connection.execute(
        f"""
        SELECT {_ARTIFACT_CLEANUP_SELECT_LIST}
        FROM {_ARTIFACT_CLEANUP_TABLE}
        WHERE {_ArtifactCleanupColumn.CLEANUP_RECORD_ID} IN (
            SELECT latest.{_ArtifactCleanupColumn.CLEANUP_RECORD_ID}
            FROM {_ARTIFACT_CLEANUP_TABLE} AS latest
            INNER JOIN (
                SELECT
                    {_ArtifactCleanupColumn.BATCH_ID} AS batch_id,
                    {_ArtifactCleanupColumn.ARTIFACT_ID} AS artifact_id,
                    MAX({_ArtifactCleanupColumn.SEQUENCE}) AS max_sequence
                FROM {_ARTIFACT_CLEANUP_TABLE}
                GROUP BY
                    {_ArtifactCleanupColumn.BATCH_ID},
                    {_ArtifactCleanupColumn.ARTIFACT_ID}
            ) AS grouped
                ON latest.{_ArtifactCleanupColumn.BATCH_ID} = grouped.batch_id
               AND latest.{_ArtifactCleanupColumn.ARTIFACT_ID} = grouped.artifact_id
               AND latest.{_ArtifactCleanupColumn.SEQUENCE} = grouped.max_sequence
        )
          AND {_ArtifactCleanupColumn.STATUS} IN ({placeholders})
        ORDER BY {_ArtifactCleanupColumn.CREATED_AT}, {_ArtifactCleanupColumn.CLEANUP_RECORD_ID}
        """,
        tuple(sorted(statuses)),
    ).fetchall()
    return [_artifact_cleanup_from_row(row) for row in rows]


def _ensure_batch_authority_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(
            f"PRAGMA table_info({_BATCH_TABLE})",
        ).fetchall()
    }
    for column in (
        _BatchColumn.RESOURCE_KEY,
        _BatchColumn.CLAIM_OWNER,
        _BatchColumn.CLAIM_SCOPE,
        _BatchColumn.LEASE_OWNER,
        _BatchColumn.LEASE_TOKEN_DIGEST,
    ):
        if column not in columns:
            connection.execute(f"ALTER TABLE {_BATCH_TABLE} ADD COLUMN {column} TEXT")


def _ensure_batch_operation_link_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"])
        for row in connection.execute(
            f"PRAGMA table_info({_BATCH_TABLE})",
        ).fetchall()
    }
    for column in (_BatchColumn.OPERATION_RUN_ID, _BatchColumn.OPERATION_PHASE_ID):
        if column not in columns:
            connection.execute(f"ALTER TABLE {_BATCH_TABLE} ADD COLUMN {column} TEXT")


def _operation_row(connection: sqlite3.Connection, operation_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_OPERATION_SELECT_LIST}
            FROM {_OPERATION_TABLE}
            WHERE {_OperationColumn.OPERATION_ID} = ?
            """,
            (operation_id,),
        ).fetchone(),
    )


def _checkpoint_row(connection: sqlite3.Connection, checkpoint_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_CHECKPOINT_SELECT_LIST}
            FROM {_CHECKPOINT_TABLE}
            WHERE {_CheckpointColumn.CHECKPOINT_ID} = ?
            """,
            (checkpoint_id,),
        ).fetchone(),
    )


def _recovery_row(connection: sqlite3.Connection, recovery_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_RECOVERY_SELECT_LIST}
            FROM {_RECOVERY_TABLE}
            WHERE {_RecoveryColumn.RECOVERY_ID} = ?
            """,
            (recovery_id,),
        ).fetchone(),
    )


def _recovery_action_row(connection: sqlite3.Connection, action_record_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_RECOVERY_ACTION_SELECT_LIST}
            FROM {_RECOVERY_ACTION_TABLE}
            WHERE {_RecoveryActionColumn.ACTION_RECORD_ID} = ?
            """,
            (action_record_id,),
        ).fetchone(),
    )


def _artifact_cleanup_row(
    connection: sqlite3.Connection,
    cleanup_record_id: str,
) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_ARTIFACT_CLEANUP_SELECT_LIST}
            FROM {_ARTIFACT_CLEANUP_TABLE}
            WHERE {_ArtifactCleanupColumn.CLEANUP_RECORD_ID} = ?
            """,
            (cleanup_record_id,),
        ).fetchone(),
    )


def _latest_recovery_row(connection: sqlite3.Connection, batch_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_RECOVERY_SELECT_LIST}
            FROM {_RECOVERY_TABLE}
            WHERE {_RecoveryColumn.BATCH_ID} = ?
            ORDER BY {_RecoveryColumn.SEQUENCE} DESC, {_RecoveryColumn.RECOVERY_ID} DESC
            LIMIT 1
            """,
            (batch_id,),
        ).fetchone(),
    )


def _latest_recovering_recovery_row(connection: sqlite3.Connection, batch_id: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_RECOVERY_SELECT_LIST}
            FROM {_RECOVERY_TABLE}
            WHERE {_RecoveryColumn.BATCH_ID} = ?
              AND {_RecoveryColumn.PHASE} = ?
            ORDER BY {_RecoveryColumn.SEQUENCE} DESC, {_RecoveryColumn.RECOVERY_ID} DESC
            LIMIT 1
            """,
            (batch_id, BatchPhase.RECOVERING),
        ).fetchone(),
    )


def _latest_artifact_cleanup_row(
    connection: sqlite3.Connection,
    batch_id: str,
    artifact_id: str,
) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_ARTIFACT_CLEANUP_SELECT_LIST}
            FROM {_ARTIFACT_CLEANUP_TABLE}
            WHERE {_ArtifactCleanupColumn.BATCH_ID} = ?
              AND {_ArtifactCleanupColumn.ARTIFACT_ID} = ?
            ORDER BY {_ArtifactCleanupColumn.SEQUENCE} DESC, {_ArtifactCleanupColumn.CLEANUP_RECORD_ID} DESC
            LIMIT 1
            """,
            (batch_id, artifact_id),
        ).fetchone(),
    )


def _latest_recovery_action_for_attempt(
    connection: sqlite3.Connection,
    recovery_attempt_id: str,
    action_id: str,
) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_RECOVERY_ACTION_SELECT_LIST}
            FROM {_RECOVERY_ACTION_TABLE}
            WHERE {_RecoveryActionColumn.RECOVERY_ATTEMPT_ID} = ?
              AND {_RecoveryActionColumn.ACTION_ID} = ?
            ORDER BY {_RecoveryActionColumn.SEQUENCE} DESC, {_RecoveryActionColumn.ACTION_RECORD_ID} DESC
            LIMIT 1
            """,
            (recovery_attempt_id, action_id),
        ).fetchone(),
    )


def _batch_from_row(row: sqlite3.Row) -> OperationBatchRecord:
    return OperationBatchRecord(
        batch_id=str(row[_BatchColumn.BATCH_ID]),
        idempotency_key=str(row[_BatchColumn.IDEMPOTENCY_KEY]),
        lease_name=str(row[_BatchColumn.LEASE_NAME]),
        lease_fencing_token=int(row[_BatchColumn.LEASE_FENCING_TOKEN]),
        owner=str(row[_BatchColumn.OWNER]),
        run_id=str(row[_BatchColumn.RUN_ID]),
        operation_run_id=_optional_str(row[_BatchColumn.OPERATION_RUN_ID]),
        operation_phase_id=_optional_str(row[_BatchColumn.OPERATION_PHASE_ID]),
        resource_key=_optional_str(row[_BatchColumn.RESOURCE_KEY]),
        claim_owner=_optional_str(row[_BatchColumn.CLAIM_OWNER]),
        claim_scope=_optional_str(row[_BatchColumn.CLAIM_SCOPE]),
        phase=str(row[_BatchColumn.PHASE]),
        payload=_frozen_payload(str(row[_BatchColumn.PAYLOAD])),
        status_message=_optional_str(row[_BatchColumn.STATUS_MESSAGE]),
        status_payload=_frozen_payload(str(row[_BatchColumn.STATUS_PAYLOAD])),
        created_at=_parse_datetime(str(row[_BatchColumn.CREATED_AT])),
        updated_at=_parse_datetime(str(row[_BatchColumn.UPDATED_AT])),
        storage_order=int(row[_BatchColumn.STORAGE_ORDER]),
    )


def _operation_from_row(row: sqlite3.Row) -> OperationRecord:
    return OperationRecord(
        operation_id=str(row[_OperationColumn.OPERATION_ID]),
        batch_id=str(row[_OperationColumn.BATCH_ID]),
        sequence=int(row[_OperationColumn.SEQUENCE]),
        operation_type=str(row[_OperationColumn.OPERATION_TYPE]),
        resource_key=_optional_str(row[_OperationColumn.RESOURCE_KEY]),
        payload=_frozen_payload(str(row[_OperationColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_OperationColumn.CREATED_AT])),
    )


def _checkpoint_from_row(row: sqlite3.Row) -> CheckpointRecord:
    return CheckpointRecord(
        checkpoint_id=str(row[_CheckpointColumn.CHECKPOINT_ID]),
        batch_id=str(row[_CheckpointColumn.BATCH_ID]),
        sequence=int(row[_CheckpointColumn.SEQUENCE]),
        operation_id=_optional_str(row[_CheckpointColumn.OPERATION_ID]),
        resource_key=str(row[_CheckpointColumn.RESOURCE_KEY]),
        checkpoint_type=str(row[_CheckpointColumn.CHECKPOINT_TYPE]),
        payload=_frozen_payload(str(row[_CheckpointColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_CheckpointColumn.CREATED_AT])),
    )


def _recovery_from_row(row: sqlite3.Row) -> RecoveryRecord:
    return RecoveryRecord(
        recovery_id=str(row[_RecoveryColumn.RECOVERY_ID]),
        batch_id=str(row[_RecoveryColumn.BATCH_ID]),
        sequence=int(row[_RecoveryColumn.SEQUENCE]),
        phase=str(row[_RecoveryColumn.PHASE]),
        reason=_optional_str(row[_RecoveryColumn.REASON]),
        payload=_frozen_payload(str(row[_RecoveryColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_RecoveryColumn.CREATED_AT])),
    )


def _recovery_action_from_row(row: sqlite3.Row) -> RecoveryActionRecord:
    return RecoveryActionRecord(
        action_record_id=str(row[_RecoveryActionColumn.ACTION_RECORD_ID]),
        action_id=str(row[_RecoveryActionColumn.ACTION_ID]),
        recovery_attempt_id=str(row[_RecoveryActionColumn.RECOVERY_ATTEMPT_ID]),
        batch_id=str(row[_RecoveryActionColumn.BATCH_ID]),
        sequence=int(row[_RecoveryActionColumn.SEQUENCE]),
        action_type=str(row[_RecoveryActionColumn.ACTION_TYPE]),
        status=str(row[_RecoveryActionColumn.STATUS]),
        resource_key=_optional_str(row[_RecoveryActionColumn.RESOURCE_KEY]),
        reason=_optional_str(row[_RecoveryActionColumn.REASON]),
        payload=_frozen_payload(str(row[_RecoveryActionColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_RecoveryActionColumn.CREATED_AT])),
    )


def _artifact_cleanup_from_row(row: sqlite3.Row) -> ArtifactCleanupRecord:
    return ArtifactCleanupRecord(
        cleanup_record_id=str(row[_ArtifactCleanupColumn.CLEANUP_RECORD_ID]),
        artifact_id=str(row[_ArtifactCleanupColumn.ARTIFACT_ID]),
        batch_id=str(row[_ArtifactCleanupColumn.BATCH_ID]),
        sequence=int(row[_ArtifactCleanupColumn.SEQUENCE]),
        trigger=str(row[_ArtifactCleanupColumn.TRIGGER]),
        status=str(row[_ArtifactCleanupColumn.STATUS]),
        resource_key=_optional_str(row[_ArtifactCleanupColumn.RESOURCE_KEY]),
        reason=_optional_str(row[_ArtifactCleanupColumn.REASON]),
        payload=_frozen_payload(str(row[_ArtifactCleanupColumn.PAYLOAD])),
        created_at=_parse_datetime(str(row[_ArtifactCleanupColumn.CREATED_AT])),
    )
