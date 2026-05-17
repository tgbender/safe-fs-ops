from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from safe_fs_ops.operation_journal.models import FileStateCommit, JournalTarget


class TargetJournal:
    def __init__(self, state_store: Any) -> None:
        self.state_store = state_store

    def begin_attempt(
        self,
        *,
        run_id: int,
        target: JournalTarget,
    ) -> Any:
        with self.state_store.transaction() as connection:
            return self.state_store.record_target_attempt(
                run_id=run_id,
                target_type=target.target_type,
                target_id=target.target_id,
                subject=target.subject,
                address=target.address,
                owner_id=target.owner_id,
                phase="attempting",
                connection=connection,
            )

    def commit_file_success(
        self,
        attempt: Any,
        commit: FileStateCommit,
    ) -> None:
        with self.state_store.transaction() as connection:
            if (
                commit.record_baseline
                and self.state_store.original_baseline(commit.path, connection=connection) is None
            ):
                self.state_store.record_baseline(
                    run_id=attempt.run_id,
                    path=commit.path,
                    content_text=commit.original_text or "",
                    format=commit.format,
                    original_exists=commit.original_exists,
                    connection=connection,
                )
            self.state_store.record_checkpoint(
                run_id=attempt.run_id,
                path=commit.path,
                content_text=commit.content_text,
                format=commit.format,
                original_exists=commit.original_exists,
                connection=connection,
            )
            self.state_store.record_snapshot(
                run_id=attempt.run_id,
                path=commit.path,
                content_hash=commit.content_hash,
                size=commit.size,
                mtime_ns=commit.mtime_ns,
                format=commit.format,
                spec_hash=commit.spec_hash,
                connection=connection,
            )
            operations = [_mutable_json_object(operation) for operation in commit.operations]
            if operations:
                self.state_store.record_change_batch(
                    run_id=attempt.run_id,
                    path=commit.path,
                    operations=operations,
                    original_exists=commit.original_exists,
                    format=commit.format,
                    connection=connection,
                )
            self.state_store.record_event(
                run_id=attempt.run_id,
                event_type=commit.event_type,
                path=commit.path,
                changed=True,
                summary=commit.summary,
                connection=connection,
            )
            self.state_store.update_target_attempt_ids(
                {attempt.id},
                phase="succeeded",
                connection=connection,
            )

    def commit_failure(
        self,
        attempt: Any,
        *,
        error: str,
        path: Path | None = None,
        changed: bool = False,
    ) -> None:
        with self.state_store.transaction() as connection:
            self.state_store.record_event(
                run_id=attempt.run_id,
                event_type="error",
                path=path,
                changed=changed,
                summary=error,
                connection=connection,
            )
            self.state_store.update_target_attempt_ids(
                {attempt.id},
                phase="failed",
                error=error,
                connection=connection,
            )


def _mutable_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _mutable_json_value(item) for key, item in value.items()}


def _mutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _mutable_json_object(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_mutable_json_value(item) for item in value]
    return value
