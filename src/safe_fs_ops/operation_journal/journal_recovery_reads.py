from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping

from safe_fs_ops.operation_journal.journal_errors import BatchNotFoundError, RecoveryContextError
from safe_fs_ops.operation_journal.journal_payloads import _validate_required
from safe_fs_ops.operation_journal.journal_rows import (
    _batch_from_row,
    _batch_row,
    _latest_recovering_recovery_row,
    _list_checkpoints_in_connection,
    _list_operations_in_connection,
    _list_recovery_actions_in_connection,
    _list_recovery_records_in_connection,
    _recovery_from_row,
)
from safe_fs_ops.operation_journal.journal_schema import (
    _RECOVERY_SELECT_LIST,
    _RECOVERY_TABLE,
    BatchPhase,
    _BatchColumn,
    _RecoveryColumn,
)
from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext, RecoveryRecord
from safe_fs_ops.sqlite_store import SqliteStore


class JournalRecoveryReadMixin:
    sqlite_store: SqliteStore

    def initialize(self) -> None:
        raise NotImplementedError

    def read_recovery_context(
        self,
        batch_id: str,
        *,
        _after_batch_read: Callable[[], None] | None = None,
    ) -> JournaledFilesystemRecoveryContext:
        _validate_required("batch_id", batch_id)
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
            connection.execute("BEGIN")
            row = _batch_row(connection, batch_id)
            if row is None:
                raise BatchNotFoundError(f"batch {batch_id!r} does not exist")
            if _after_batch_read is not None:
                _after_batch_read()
            operations = tuple(_list_operations_in_connection(connection, batch_id))
            checkpoints = tuple(_list_checkpoints_in_connection(connection, batch_id))
            recovery_records = tuple(_list_recovery_records_in_connection(connection, batch_id))
            recovery_attempt = None
            if str(row[_BatchColumn.PHASE]) == BatchPhase.RECOVERING:
                recovery_attempt_row = _latest_recovering_recovery_row(connection, batch_id)
                if recovery_attempt_row is None:
                    raise RecoveryContextError(
                        f"batch {batch_id!r} is recovering but has no active recovery attempt record"
                    )
                recovery_attempt = _recovery_from_row(recovery_attempt_row)
            recovery_actions = tuple(_list_recovery_actions_in_connection(connection, batch_id))
            proof_contexts = _read_proof_contexts(
                connection,
                batch_id=batch_id,
                recovery_records=recovery_records,
            )
            return JournaledFilesystemRecoveryContext(
                batch=_batch_from_row(row),
                operations=operations,
                checkpoints=checkpoints,
                recovery_records=recovery_records,
                recovery_attempt=recovery_attempt,
                recovery_actions=recovery_actions,
                proof_contexts=proof_contexts,
            )

    def list_recovery_records(self, batch_id: str) -> list[RecoveryRecord]:
        self.initialize()
        with self.sqlite_store.read_connection() as connection:
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


def _read_proof_contexts(
    connection: sqlite3.Connection,
    *,
    batch_id: str,
    recovery_records: tuple[RecoveryRecord, ...],
) -> dict[str, JournaledFilesystemRecoveryContext]:
    proof_contexts: dict[str, JournaledFilesystemRecoveryContext] = {}
    for proof_batch_id in _proof_batch_ids_from_recovery_records(recovery_records, batch_id=batch_id):
        row = _batch_row(connection, proof_batch_id)
        if row is None:
            continue
        proof_contexts[proof_batch_id] = JournaledFilesystemRecoveryContext(
            batch=_batch_from_row(row),
            operations=tuple(_list_operations_in_connection(connection, proof_batch_id)),
            checkpoints=tuple(_list_checkpoints_in_connection(connection, proof_batch_id)),
            recovery_records=(),
            recovery_attempt=None,
            recovery_actions=(),
            proof_contexts={},
        )
    return proof_contexts


def _proof_batch_ids_from_recovery_records(
    recovery_records: tuple[RecoveryRecord, ...],
    *,
    batch_id: str,
) -> tuple[str, ...]:
    proof_batch_ids: list[str] = []
    for record in recovery_records:
        payload = record.payload
        if not isinstance(payload, Mapping):
            continue
        recursive_group = payload.get("recursive_group")
        if not isinstance(recursive_group, Mapping):
            continue
        prior_created_steps = recursive_group.get("prior_created_steps")
        if not isinstance(prior_created_steps, tuple | list):
            continue
        for step in prior_created_steps:
            if not isinstance(step, Mapping):
                continue
            proof = step.get("journaled_creation_proof")
            if not isinstance(proof, Mapping):
                continue
            proof_batch_id = proof.get("batch_id")
            if proof_batch_id is None:
                continue
            proof_batch_id_text = str(proof_batch_id)
            if proof_batch_id_text == batch_id or proof_batch_id_text in proof_batch_ids:
                continue
            proof_batch_ids.append(proof_batch_id_text)
    return tuple(proof_batch_ids)
