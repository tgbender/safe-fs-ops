from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError, capture_backup
from safe_fs_ops.operation_journal import ArtifactCleanupTrigger, OperationJournalStore, file_resource_key
from safe_fs_ops.operation_journal.file_backup_artifacts import (
    BackupArtifactCleanupError,
    backup_artifacts_root,
    cleanup_backup_artifacts,
    execute_backup_artifact_cleanup_candidate,
    plan_backup_artifact_cleanup_candidates,
)
from safe_fs_ops.operation_journal.filesystem_mutation_checkpoints import (
    backup_content_path,
    backup_content_path_for_state_path,
    capture_backup_artifact,
    prepare_mutation_checkpoints,
)
from safe_fs_ops.operation_journal.models import CheckpointRecord
from safe_fs_ops.workspace_state import LeaseStore

pytestmark = pytest.mark.safe_fs_ops


def test_capture_backup_artifact_does_not_bypass_primary_capture_refusal(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    journal = OperationJournalStore(state_path)
    snapshot = capture_backup(target).snapshot

    def refuse_primary_capture(path: Path | str, **_: object):
        raise UnsafePathError(f"backup refused for symlink path: {path}")

    with pytest.raises(UnsafePathError, match="symlink"):
        capture_backup_artifact(
            journal,
            target,
            before_snapshot=snapshot,
            batch_id="batch-1",
            operation_id="op-1",
            capture_backup_operation=refuse_primary_capture,
        )


def test_prepare_mutation_checkpoints_records_backup_cleanup_intent_before_backup_write(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    journal = OperationJournalStore(state_path)
    lease = LeaseStore(state_path).acquire("workspace", owner="runner-a", ttl=timedelta(seconds=30))
    batch = journal.create_batch(
        idempotency_key="write-config",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=file_resource_key(target),
        claim_owner="owner-a",
        payload={"operation": "write_text", "path": str(target)},
    )
    operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="write_text",
        resource_key=file_resource_key(target),
        payload={"operation": "write_text", "path": str(target)},
    )[1]
    snapshot = capture_backup(target).snapshot

    def crash_before_backup_write(path: Path | str, **_: object):
        assert path == target
        backup_intent_checkpoints = [
            checkpoint
            for checkpoint in journal.list_checkpoints(batch.batch_id)
            if checkpoint.checkpoint_type == "backup_artifact_intent"
        ]
        backup_checkpoints = [
            checkpoint
            for checkpoint in journal.list_checkpoints(batch.batch_id)
            if checkpoint.checkpoint_type == "backup"
        ]
        cleanup_records = journal.list_artifact_cleanup_records(batch.batch_id)
        assert len(backup_intent_checkpoints) == 1
        assert backup_checkpoints == []
        assert len(cleanup_records) == 1
        assert cleanup_records[0].trigger == ArtifactCleanupTrigger.DEFERRED_CLEANUP
        assert cleanup_records[0].status == "planned"
        assert cleanup_records[0].payload["checkpoint_id"] == backup_intent_checkpoints[0].checkpoint_id
        raise RuntimeError("simulated crash before backup write")

    with pytest.raises(RuntimeError, match="simulated crash"):
        prepare_mutation_checkpoints(
            journal,
            lease=lease,
            batch_id=batch.batch_id,
            operation_id=operation.operation_id,
            resource_key=file_resource_key(target),
            path=target,
            before_snapshot=snapshot,
            operation_type="write_text",
            recovery_payload={"desired": {"path": str(target)}},
            capture_file_rollback_proof=True,
            operation_time=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            capture_backup_operation=crash_before_backup_write,
        )


def test_cleanup_backup_artifacts_rejects_paths_outside_artifacts_root(tmp_path: Path) -> None:
    outside_path = tmp_path / "outside-backup.bin"
    outside_path.write_text("keep\n", encoding="utf-8")
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-1",
        batch_id="batch-1",
        sequence=1,
        operation_id="op-1",
        resource_key=file_resource_key(tmp_path / "config.txt"),
        checkpoint_type="backup",
        payload={
            "path": str(tmp_path / "config.txt"),
            "content_path": str(outside_path),
            "content_hash": hashlib.sha256(outside_path.read_bytes()).hexdigest(),
            "size": outside_path.stat().st_size,
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(BackupArtifactCleanupError, match="unsafe_artifact_path"):
        cleanup_backup_artifacts((checkpoint,), state_path=tmp_path / "state.db")

    assert outside_path.read_text(encoding="utf-8") == "keep\n"


def test_cleanup_backup_artifacts_rejects_in_root_wrong_subtree_path(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    wrong_subtree = state_path.parent / f"{state_path.name}.artifacts" / "wrong-subtree" / "backup.bin"
    wrong_subtree.parent.mkdir(parents=True, exist_ok=True)
    wrong_subtree.write_text("keep\n", encoding="utf-8")
    checkpoint = _backup_checkpoint(
        target=target,
        batch_id="batch-1",
        operation_id="op-1",
        content_path=wrong_subtree,
    )

    with pytest.raises(BackupArtifactCleanupError, match="unsafe_artifact_path"):
        cleanup_backup_artifacts((checkpoint,), state_path=state_path)

    assert wrong_subtree.read_text(encoding="utf-8") == "keep\n"
    expected_path = backup_content_path(journal, batch_id="batch-1", operation_id="op-1", path=target)
    assert wrong_subtree != expected_path


def test_backup_content_path_uses_short_deterministic_artifact_name() -> None:
    state_path = PureWindowsPath("C:/pytest-temp") / ("s" * 80) / "state.sqlite"
    target = PureWindowsPath("C:/pytest-temp") / ("nested-" + "p" * 80) / ("settings" + ".toml" * 8)
    first = backup_content_path_for_state_path(
        state_path,
        batch_id="batch-" + "b" * 64,
        operation_id="operation-" + "o" * 64,
        path=target,
    )
    second = backup_content_path_for_state_path(
        state_path,
        batch_id="batch-" + "b" * 64,
        operation_id="operation-" + "o" * 64,
        path=target,
    )

    assert first == second
    assert first.parent == state_path.parent / f"{state_path.name}.artifacts" / "file-backups"
    assert first.name.endswith(".bak")
    assert len(first.stem) == 32
    int(first.stem, 16)
    assert "operation-" not in first.name
    assert ".toml" not in first.name


def test_backup_content_path_keeps_windows_temp_write_path_bounded_for_long_inputs() -> None:
    state_path = PureWindowsPath("C:/") / ("s" * 80) / ("t" * 70) / "state.db"
    target = PureWindowsPath("C:/workspace") / ("nested-" + "p" * 120) / ("settings" + ".toml" * 8)
    content_path = backup_content_path_for_state_path(
        state_path,
        batch_id="batch-" + "b" * 64,
        operation_id="operation-" + "o" * 64,
        path=target,
    )
    atomic_temp_path = content_path.parent / f".{content_path.name}.0123456789abcdef.tmp"

    assert len(str(atomic_temp_path)) <= 259


def test_backup_content_path_distinguishes_batch_operation_and_source_path() -> None:
    state_path = PureWindowsPath("C:/repo/state.db")
    target = PureWindowsPath("C:/repo/settings.toml")

    paths = {
        backup_content_path_for_state_path(state_path, batch_id="batch-1", operation_id="op-1", path=target),
        backup_content_path_for_state_path(state_path, batch_id="batch-2", operation_id="op-1", path=target),
        backup_content_path_for_state_path(state_path, batch_id="batch-1", operation_id="op-2", path=target),
        backup_content_path_for_state_path(
            state_path,
            batch_id="batch-1",
            operation_id="op-1",
            path=PureWindowsPath("C:/repo/other.toml"),
        ),
    }

    assert len(paths) == 4


def test_cleanup_backup_artifacts_accepts_legacy_long_artifact_path(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    target = tmp_path / "config.toml"
    content_path = _legacy_backup_content_path(
        state_path,
        batch_id="batch-1",
        operation_id="operation-1",
        path=target,
    )
    content_path.parent.mkdir(parents=True, exist_ok=True)
    content_path.write_text("old\n", encoding="utf-8")
    checkpoint = _backup_checkpoint(
        target=target,
        batch_id="batch-1",
        operation_id="operation-1",
        content_path=content_path,
    )

    deleted = cleanup_backup_artifacts((checkpoint,), state_path=state_path)

    assert deleted == (content_path,)
    assert not content_path.exists()


def test_cleanup_backup_artifacts_skips_missing_file_backup_without_artifact(tmp_path: Path) -> None:
    delete_calls: list[Path] = []
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-missing-backup",
        batch_id="batch-1",
        sequence=1,
        operation_id="op-1",
        resource_key=file_resource_key(tmp_path / "config.txt"),
        checkpoint_type="backup",
        payload={
            "path": str(tmp_path / "config.txt"),
            "existed": False,
            "file_type": "missing",
            "content_path": None,
            "snapshot": {
                "path": str(tmp_path / "config.txt"),
                "exists": False,
                "file_type": "missing",
            },
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    deleted = cleanup_backup_artifacts(
        (checkpoint,),
        state_path=tmp_path / "state.db",
        delete_operation=lambda path: delete_calls.append(path),
    )

    assert deleted == ()
    assert delete_calls == []
    assert plan_backup_artifact_cleanup_candidates((checkpoint,), state_path=tmp_path / "state.db") == ()


def test_cleanup_backup_artifacts_reports_regular_file_backup_missing_content_path_as_debt(
    tmp_path: Path,
) -> None:
    delete_calls: list[Path] = []
    target = tmp_path / "config.txt"
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-file-backup-missing-content-path",
        batch_id="batch-1",
        sequence=1,
        operation_id="op-1",
        resource_key=file_resource_key(target),
        checkpoint_type="backup",
        payload={
            "path": str(target),
            "existed": True,
            "file_type": "file",
            "content_path": None,
            "snapshot": {
                "path": str(target),
                "exists": True,
                "file_type": "file",
            },
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(BackupArtifactCleanupError, match="malformed_backup_checkpoint") as excinfo:
        cleanup_backup_artifacts(
            (checkpoint,),
            state_path=tmp_path / "state.db",
            delete_operation=lambda path: delete_calls.append(path),
        )

    assert delete_calls == []
    debt = excinfo.value.debts[0]
    assert debt.reason_code == "malformed_backup_checkpoint"
    assert "content_path" in debt.detail


@pytest.mark.parametrize(
    ("missing_field", "payload", "operation_id"),
    [
        ("path", {"content_path": "backup.bin"}, "op-1"),
        ("operation_id", {"path": "config.txt", "content_path": "backup.bin"}, None),
    ],
)
def test_cleanup_backup_artifacts_reports_malformed_backup_checkpoint_debt(
    tmp_path: Path,
    missing_field: str,
    payload: dict[str, object],
    operation_id: str | None,
) -> None:
    delete_calls: list[Path] = []
    checkpoint = CheckpointRecord(
        checkpoint_id=f"checkpoint-{missing_field}",
        batch_id="batch-1",
        sequence=1,
        operation_id=operation_id,
        resource_key=file_resource_key(tmp_path / "config.txt"),
        checkpoint_type="backup",
        payload=payload,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(BackupArtifactCleanupError, match="malformed_backup_checkpoint") as excinfo:
        cleanup_backup_artifacts(
            (checkpoint,),
            state_path=tmp_path / "state.db",
            delete_operation=lambda path: delete_calls.append(path),
        )

    assert delete_calls == []
    debt = excinfo.value.debts[0]
    assert debt.batch_id == "batch-1"
    assert debt.reason_code == "malformed_backup_checkpoint"
    assert missing_field in debt.detail


@pytest.mark.parametrize("symlink_target", ["artifacts_root", "artifact_parent"])
def test_cleanup_backup_artifacts_rejects_symlinked_artifact_chain(
    tmp_path: Path,
    symlink_target: str,
) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    expected_path = backup_content_path(journal, batch_id="batch-1", operation_id="op-1", path=target)
    outside_root = tmp_path / "outside-artifacts"

    if symlink_target == "artifacts_root":
        outside_root.mkdir(parents=True, exist_ok=True)
        _try_symlink(
            backup_artifacts_root(state_path),
            outside_root,
        )
    else:
        outside_parent = outside_root / "nested-parent"
        outside_parent.mkdir(parents=True, exist_ok=True)
        _try_symlink(expected_path.parent, outside_parent)

    realized_path = expected_path
    realized_path.parent.mkdir(parents=True, exist_ok=True)
    realized_path.write_text("keep\n", encoding="utf-8")
    checkpoint = _backup_checkpoint(
        target=target,
        batch_id="batch-1",
        operation_id="op-1",
        content_path=realized_path,
    )

    with pytest.raises(BackupArtifactCleanupError, match="unsafe_artifact_path"):
        cleanup_backup_artifacts((checkpoint,), state_path=state_path)

    assert realized_path.read_text(encoding="utf-8") == "keep\n"


def test_cleanup_backup_artifacts_rejects_replaced_content(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    content_path = backup_content_path(journal, batch_id="batch-1", operation_id="op-1", path=target)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    original = b"original backup\n"
    replacement = b"foreign replacement\n"
    content_path.write_bytes(replacement)
    checkpoint = _backup_checkpoint(
        target=target,
        batch_id="batch-1",
        operation_id="op-1",
        content_path=content_path,
        expected_content=original,
    )

    with pytest.raises(BackupArtifactCleanupError, match="backup_artifact_content_mismatch") as excinfo:
        cleanup_backup_artifacts((checkpoint,), state_path=state_path)

    assert excinfo.value.debts[0].reason_code == "backup_artifact_content_mismatch"
    assert content_path.read_bytes() == replacement


def test_cleanup_candidate_revalidates_content_after_pre_delete_hook(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    content_path = backup_content_path(journal, batch_id="batch-1", operation_id="op-1", path=target)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    original = b"original backup\n"
    replacement = b"foreign replacement\n"
    content_path.write_bytes(original)
    checkpoint = _backup_checkpoint(
        target=target,
        batch_id="batch-1",
        operation_id="op-1",
        content_path=content_path,
        expected_content=original,
    )
    [candidate] = plan_backup_artifact_cleanup_candidates((checkpoint,), state_path=state_path)
    deleted: list[Path] = []

    result = execute_backup_artifact_cleanup_candidate(
        candidate,
        state_path=state_path,
        delete_operation=deleted.append,
        before_delete=lambda: content_path.write_bytes(replacement),
    )

    assert result.status == "manual_intervention_required"
    assert result.reason_code == "backup_artifact_content_mismatch"
    assert deleted == []
    assert content_path.read_bytes() == replacement


def test_cleanup_backup_artifacts_rejects_intent_path_with_foreign_content(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    journal = OperationJournalStore(state_path)
    target = tmp_path / "config.txt"
    target.write_text("old\n", encoding="utf-8")
    content_path = backup_content_path(journal, batch_id="batch-1", operation_id="op-1", path=target)
    content_path.parent.mkdir(parents=True, exist_ok=True)
    expected = b"original backup\n"
    foreign = b"foreign data\n"
    content_path.write_bytes(foreign)
    checkpoint = CheckpointRecord(
        checkpoint_id="checkpoint-intent",
        batch_id="batch-1",
        sequence=1,
        operation_id="op-1",
        resource_key=file_resource_key(target),
        checkpoint_type="backup_artifact_intent",
        payload={
            "path": str(target),
            "content_path": str(content_path),
            "content_hash": hashlib.sha256(expected).hexdigest(),
            "size": len(expected),
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(BackupArtifactCleanupError, match="backup_artifact_content_mismatch"):
        cleanup_backup_artifacts((checkpoint,), state_path=state_path)

    assert content_path.read_bytes() == foreign


def _backup_checkpoint(
    *,
    target: Path,
    batch_id: str,
    operation_id: str,
    content_path: Path,
    expected_content: bytes | None = None,
) -> CheckpointRecord:
    content = content_path.read_bytes() if expected_content is None else expected_content
    return CheckpointRecord(
        checkpoint_id=f"checkpoint:{batch_id}:{operation_id}",
        batch_id=batch_id,
        sequence=1,
        operation_id=operation_id,
        resource_key=file_resource_key(target),
        checkpoint_type="backup",
        payload={
            "path": str(target),
            "content_path": str(content_path),
            "content_hash": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        },
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _legacy_backup_content_path(
    state_path: Path,
    *,
    batch_id: str,
    operation_id: str,
    path: Path,
) -> Path:
    extension = "".join(path.suffixes)
    artifact_name = operation_id if not extension else f"{operation_id}{extension}.bak"
    if extension == "":
        artifact_name = f"{operation_id}.bak"
    return (
        state_path.parent
        / f"{state_path.name}.artifacts"
        / "file-backups"
        / batch_id
        / uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex
        / artifact_name
    )


def _try_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_path.symlink_to(target_path, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
