from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from recovery_runner_helpers import (
    _acquire_and_claim,
    _coordinator,
    _current_recovery_attempt_id,
    _start_recovering_batch,
)
from workspace_api_helpers import portable_snapshot

from safe_fs_ops.filesystem_ops import (
    ContentAddressedPutResult,
    ContentAddressedStore,
    ContentRef,
    RestoreConflictError,
    TreeBackup,
)
from safe_fs_ops.filesystem_ops import (
    backup_tree as create_tree_backup,
)
from safe_fs_ops.operation_journal import (
    BatchIdempotencyMismatchError,
    BatchPhase,
    JournaledFilesystemBatchStateError,
    JournaledFilesystemMutationError,
    JournaledFilesystemRecoveryError,
    OperationJournalStore,
    RecoveryActionRecord,
    RecoveryAuthority,
    TreeResourceKeyMismatchError,
    directory_resource_key,
    tree_resource_key,
)
from safe_fs_ops.operation_journal.recovery_runner import RecoveryActionManualInterventionRequired
from safe_fs_ops.operation_journal.tree_backup_recovery import (
    RESTORE_TREE_BACKUP_ACTION,
    default_tree_backup_store_path,
    execute_tree_backup_artifact_cleanup_candidate,
    plan_tree_backup_artifact_cleanup_candidates,
    restore_tree_backup_recovery_action,
    run_restore_tree_backup_operation,
    tree_backup_checkpoint_fingerprint,
    tree_backup_checkpoint_payload,
    tree_backup_recovery_action_plans,
)
from safe_fs_ops.workspace_state import LeaseLostError, LeaseStore

pytestmark = pytest.mark.safe_fs_ops


@dataclass(frozen=True, slots=True)
class FakeTreeEntry:
    relative_path: str
    file_type: str
    content_ref: ContentRef | None = None
    symlink_target: str | None = None


@dataclass(frozen=True, slots=True)
class FakeTreeBackup:
    root_path: Path
    entries: tuple[FakeTreeEntry, ...]


class _ConcurrentInstallBeforePutStore(ContentAddressedStore):
    def put_file_with_status(self, path: Path | str) -> ContentAddressedPutResult:
        ContentAddressedStore(self.root).put_bytes(Path(path).read_bytes())
        return super().put_file_with_status(path)


def test_backup_tree_records_tree_backup_checkpoint_without_mutating_source(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    target = source / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    store_path = tmp_path / "tree-cas"

    def fake_backup_tree(path: Path | str, *, artifact_store: ContentAddressedStore) -> FakeTreeBackup:
        root = Path(path)
        content_ref = artifact_store.put_file(root / "config.txt")
        return FakeTreeBackup(
            root_path=root,
            entries=(
                FakeTreeEntry("config.txt", "file", content_ref=content_ref),
                FakeTreeEntry("latest", "symlink", symlink_target="config.txt"),
            ),
        )

    result = coordinator.backup_tree(
        source,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="backup-tree:workspace",
        store_path=store_path,
        backup_tree_operation=fake_backup_tree,
        now=now,
    )

    assert target.read_text(encoding="utf-8") == "original\n"
    assert result.batch.phase == BatchPhase.SUCCEEDED
    checkpoints = journal.list_checkpoints(result.batch.batch_id)
    checkpoint = next(checkpoint for checkpoint in checkpoints if checkpoint.checkpoint_type == "tree_backup")
    assert checkpoint.checkpoint_type == "tree_backup"
    assert checkpoint.payload["root_path"] == str(source)
    assert checkpoint.payload["store_path"] == str(store_path)
    entries = checkpoint.payload["entries"]
    assert isinstance(entries, tuple)
    file_entry = entries[0]
    expected_digest = hashlib.sha256(target.read_bytes()).hexdigest()
    assert file_entry["content_ref"]["digest"] == expected_digest
    assert file_entry["content_ref"]["path"] == str(store_path / "sha256" / expected_digest[:2] / expected_digest)
    assert entries[1]["symlink_target"] == "config.txt"
    assert plan_tree_backup_artifact_cleanup_candidates(checkpoints) == ()


def test_backup_tree_checkpoints_partial_artifacts_as_refs_are_created(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    first = source / "first.txt"
    second = source / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    store_path = tmp_path / "tree-cas"

    def checkpointing_backup_tree(
        path: Path | str,
        relative_paths: tuple[Path | str, ...],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        assert relative_paths == ("first.txt", "second.txt")
        root = Path(path)
        first_ref = artifact_store.put_file(root / "first.txt")
        [batch] = journal.list_batches()
        checkpoints = journal.list_checkpoints(batch.batch_id)
        assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == ["tree_backup_partial_artifacts"]
        assert checkpoints[0].payload["content_refs"][0]["digest"] == first_ref.digest

        second_ref = artifact_store.put_file(root / "second.txt")
        checkpoints = journal.list_checkpoints(batch.batch_id)
        assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == [
            "tree_backup_partial_artifacts",
            "tree_backup_partial_artifacts",
        ]
        assert checkpoints[1].payload["content_refs"][0]["digest"] == second_ref.digest
        raise RuntimeError("stop after immediate partial checkpoints")

    with pytest.raises(JournaledFilesystemMutationError, match="stop after immediate"):
        coordinator.backup_tree(
            source,
            ["first.txt", "second.txt"],
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:immediate-partials",
            store_path=store_path,
            backup_tree_operation=checkpointing_backup_tree,
            now=now,
        )

    [batch] = journal.list_batches()
    assert batch.phase == BatchPhase.FAILED
    checkpoints = journal.list_checkpoints(batch.batch_id)
    assert [checkpoint.checkpoint_type for checkpoint in checkpoints] == [
        "tree_backup_partial_artifacts",
        "tree_backup_partial_artifacts",
    ]
    candidates = plan_tree_backup_artifact_cleanup_candidates(checkpoints)
    assert [candidate.digest for candidate in candidates] == [
        hashlib.sha256(first.read_bytes()).hexdigest(),
        hashlib.sha256(second.read_bytes()).hexdigest(),
    ]


def test_backup_tree_does_not_checkpoint_artifact_created_by_concurrent_shared_store(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    target = source / "config.txt"
    target.write_text("shared\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    store = _ConcurrentInstallBeforePutStore(tmp_path / "tree-cas")
    refs: list[ContentRef] = []

    def write_artifact_then_fail(
        path: Path | str,
        relative_paths: tuple[Path | str, ...],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        assert relative_paths == ("config.txt",)
        result = artifact_store.put_file_with_status(Path(path) / "config.txt")
        assert result.created is False
        refs.append(result.ref)
        raise RuntimeError("stop after shared object install")

    with pytest.raises(JournaledFilesystemMutationError, match="stop after shared"):
        coordinator.backup_tree(
            source,
            ["config.txt"],
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:shared-digest-race",
            artifact_store=store,
            backup_tree_operation=write_artifact_then_fail,
            now=now,
        )

    [batch] = journal.list_batches()
    checkpoints = journal.list_checkpoints(batch.batch_id)
    assert batch.phase == BatchPhase.FAILED
    assert checkpoints == []
    assert plan_tree_backup_artifact_cleanup_candidates(checkpoints) == ()
    assert refs[0].path.exists()


def test_tree_backup_artifact_cleanup_revalidates_content_after_pre_delete_hook(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    target = source / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    store_path = tmp_path / "tree-cas"

    def write_artifact_then_fail(
        path: Path | str,
        relative_paths: tuple[Path | str, ...],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        assert relative_paths == ("config.txt",)
        artifact_store.put_file(Path(path) / "config.txt")
        raise RuntimeError("stop after partial artifact")

    with pytest.raises(JournaledFilesystemMutationError, match="stop after partial"):
        coordinator.backup_tree(
            source,
            ["config.txt"],
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:partial-revalidate",
            store_path=store_path,
            backup_tree_operation=write_artifact_then_fail,
            now=now,
        )

    [batch] = journal.list_batches()
    [candidate] = plan_tree_backup_artifact_cleanup_candidates(journal.list_checkpoints(batch.batch_id))
    replacement = b"foreign replacement\n"
    deleted: list[Path] = []

    result = execute_tree_backup_artifact_cleanup_candidate(
        candidate,
        delete_operation=deleted.append,
        before_delete=lambda: candidate.content_path.write_bytes(replacement),
    )

    assert result.status == "manual_intervention_required"
    assert result.reason_code == "tree_backup_artifact_content_mismatch"
    assert deleted == []
    assert candidate.content_path.read_bytes() == replacement


def test_backup_tree_lease_loss_during_failure_recording_preserves_partial_artifact_debt(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    target = source / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    takeover_now = now + timedelta(seconds=31)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)

    class LeaseLostOnMarkFailedJournal(OperationJournalStore):
        def mark_failed(self, *args, **kwargs):
            LeaseStore(state_path).acquire("workspace", owner="runner-b", ttl=timedelta(seconds=30), now=takeover_now)
            raise LeaseLostError("lease 'workspace' is no longer current")

    journal = LeaseLostOnMarkFailedJournal(state_path)
    coordinator = _coordinator(state_path, journal=journal)
    store_path = tmp_path / "tree-cas"

    def write_artifact_then_fail(
        path: Path | str,
        relative_paths: tuple[Path | str, ...],
        *,
        artifact_store: ContentAddressedStore,
    ) -> object:
        assert relative_paths == ("config.txt",)
        artifact_store.put_file(Path(path) / "config.txt")
        raise RuntimeError("capture failed after partial artifact")

    with pytest.raises(JournaledFilesystemMutationError, match="capture failed after partial artifact"):
        coordinator.backup_tree(
            source,
            ["config.txt"],
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:lease-loss-failure-recording",
            store_path=store_path,
            backup_tree_operation=write_artifact_then_fail,
            now=now,
        )

    [batch] = journal.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    [recovery] = journal.list_recovery_records(batch.batch_id)
    assert recovery.reason == "tree backup failed"
    partial_artifacts = recovery.payload["partial_artifacts"]
    assert partial_artifacts["content_refs"][0]["digest"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_backup_tree_marks_batch_failed_when_capture_fails(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    (source / "config.txt").write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    def failing_backup_tree(path: Path | str, *, artifact_store: ContentAddressedStore) -> object:
        raise RuntimeError(f"cannot capture {path}")

    with pytest.raises(JournaledFilesystemMutationError, match="cannot capture"):
        coordinator.backup_tree(
            source,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:workspace",
            store_path=tmp_path / "tree-cas",
            backup_tree_operation=failing_backup_tree,
            now=now,
        )

    batches = journal.list_batches()
    assert len(batches) == 1
    assert batches[0].phase == BatchPhase.FAILED
    assert batches[0].status_message == f"cannot capture {source}"
    assert batches[0].status_payload["operation"] == "backup_tree"
    assert batches[0].status_payload["resource_key"] == resource_key


def test_backup_tree_rejects_backup_root_that_does_not_match_resource_key(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    other = tmp_path / "other"
    source.mkdir()
    other.mkdir()
    (source / "config.txt").write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    def wrong_root_backup_tree(path: Path | str, *, artifact_store: ContentAddressedStore) -> FakeTreeBackup:
        del path, artifact_store
        return FakeTreeBackup(root_path=other, entries=())

    with pytest.raises(JournaledFilesystemMutationError, match="does not match resource_key"):
        coordinator.backup_tree(
            source,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:wrong-root",
            store_path=tmp_path / "tree-cas",
            backup_tree_operation=wrong_root_backup_tree,
            now=now,
        )

    [batch] = journal.list_batches()
    assert batch.phase == BatchPhase.FAILED


def test_tree_resource_key_mismatch_uses_tree_specific_error(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    other = tmp_path / "other"
    source.mkdir()
    other.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(other)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    coordinator = _coordinator(state_path, snapshot=portable_snapshot)

    with pytest.raises(TreeResourceKeyMismatchError):
        coordinator.backup_tree(
            source,
            ["config.txt"],
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="backup-tree:mismatch",
            now=now,
        )


@pytest.mark.parametrize("operation", ["backup_tree", "restore_tree_backup"])
def test_tree_operation_retry_with_newer_lease_marks_abandoned_attempt_for_recovery(
    tmp_path: Path,
    operation: str,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.txt").write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    first_lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    store_path = tmp_path / "tree-cas"
    store = ContentAddressedStore(store_path)
    backup = create_tree_backup(source, ["config.txt"], artifact_store=store)
    intent_payload = (
        {
            "operation": "backup_tree",
            "path": str(source),
            "relative_paths": ["config.txt"],
            "resource_key": resource_key,
            "store_path": str(store_path),
        }
        if operation == "backup_tree"
        else {
            "operation": "restore_tree_backup",
            "destination_root": str(source),
            "resource_key": resource_key,
            "store_path": str(store_path),
            "conflict_policy": "no_replace",
            "backup_fingerprint": tree_backup_checkpoint_fingerprint(backup, store_path=store_path),
        }
    )
    batch = journal.create_batch(
        idempotency_key=f"{operation}:abandoned",
        lease=first_lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=resource_key,
        claim_owner="owner-a",
        payload=intent_payload,
        now=now,
    )
    journal.start_batch_operation(
        batch.batch_id,
        lease=first_lease,
        operation_type=operation,
        resource_key=resource_key,
        payload=intent_payload,
        now=now,
    )
    takeover_now = now + timedelta(seconds=31)
    takeover_lease = _acquire_and_claim(state_path, resource_key, now=takeover_now)
    coordinator = _coordinator(state_path, journal=journal, snapshot=portable_snapshot)

    with pytest.raises(JournaledFilesystemBatchStateError):
        if operation == "backup_tree":
            coordinator.backup_tree(
                source,
                ["config.txt"],
                resource_key=resource_key,
                lease=takeover_lease,
                owner="owner-a",
                run_id="run-1",
                idempotency_key=f"{operation}:abandoned",
                artifact_store=store,
                now=takeover_now,
            )
        else:
            coordinator.restore_tree_backup(
                backup,
                destination_root=source,
                artifact_store=store,
                resource_key=resource_key,
                lease=takeover_lease,
                owner="owner-a",
                run_id="run-1",
                idempotency_key=f"{operation}:abandoned",
                now=takeover_now,
            )

    updated = journal.get_batch(batch.batch_id)
    assert updated is not None
    assert updated.phase == BatchPhase.RECOVERY_DESIRED
    assert journal.list_recovery_records(batch.batch_id)[0].reason == (
        "retry found abandoned batch outside planned phase"
    )


def test_tree_backup_recovery_plans_no_restore_for_read_only_backup_batch(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    target = source / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    store_path = tmp_path / "tree-cas"

    def fake_backup_tree(path: Path | str, *, artifact_store: ContentAddressedStore) -> FakeTreeBackup:
        root = Path(path)
        return FakeTreeBackup(
            root_path=root,
            entries=(FakeTreeEntry("config.txt", "file", content_ref=artifact_store.put_file(root / "config.txt")),),
        )

    coordinator = _coordinator(state_path, journal=journal)
    result = coordinator.backup_tree(
        source,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="backup-tree:workspace",
        store_path=store_path,
        backup_tree_operation=fake_backup_tree,
        now=now,
    )
    target.write_text("mutated\n", encoding="utf-8")
    _start_recovery_for_succeeded_batch(
        journal,
        lease=lease,
        batch_id=result.batch.batch_id,
        now=now + timedelta(seconds=1),
    )

    context = journal.read_recovery_context(result.batch.batch_id)
    plans = tree_backup_recovery_action_plans(context)
    assert plans == ()

    recovery_result = coordinator.recover_batch(
        result.batch.batch_id,
        lease=lease,
        recover=lambda recovery_context: coordinator.run_recovery_actions(
            recovery_context,
            lease=lease,
            now=now + timedelta(seconds=2),
        ),
        now=now + timedelta(seconds=2),
    )

    assert target.read_text(encoding="utf-8") == "mutated\n"
    assert recovery_result.callback_result == ()


def test_failed_restore_tree_backup_recovery_requires_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a\n", encoding="utf-8")
    (source / "b.txt").write_text("b\n", encoding="utf-8")
    destination = tmp_path / "restore"
    destination.mkdir()
    (destination / "b.txt").write_text("keep\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "tree-cas")
    backup = create_tree_backup(source, ["a.txt", "b.txt"], artifact_store=store)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(destination)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal, snapshot=portable_snapshot)

    with pytest.raises(JournaledFilesystemMutationError, match="destination already exists") as raised:
        coordinator.restore_tree_backup(
            backup,
            destination_root=destination,
            artifact_store=store,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="restore-tree:partial",
            now=now,
        )

    assert (destination / "a.txt").exists() is False
    assert (destination / "b.txt").read_text(encoding="utf-8") == "keep\n"
    batch_id = raised.value.batch_id
    failed_batch = journal.get_batch(batch_id)
    assert failed_batch is not None
    assert failed_batch.phase == BatchPhase.RECOVERY_DESIRED

    with pytest.raises(JournaledFilesystemRecoveryError, match="recovery callback failed"):
        coordinator.recover_batch(
            batch_id,
            lease=lease,
            recover=lambda context: coordinator.run_recovery_actions(
                context,
                lease=lease,
                now=now + timedelta(seconds=1),
            ),
            now=now + timedelta(seconds=1),
        )

    recovered_batch = journal.get_batch(batch_id)
    assert recovered_batch is not None
    assert recovered_batch.phase == BatchPhase.RECOVERY_FAILED
    actions = journal.list_recovery_actions(batch_id)
    assert [action.status for action in actions] == ["planned", "attempting", "manual_intervention_required"]
    assert actions[-1].payload["manual_intervention"]["reason_code"] == "missing_tree_restore_rollback_proof"
    assert (destination / "a.txt").exists() is False
    assert (destination / "b.txt").read_text(encoding="utf-8") == "keep\n"


def test_restore_tree_backup_idempotency_includes_backup_identity(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source_a = tmp_path / "source-a"
    source_b = tmp_path / "source-b"
    source_a.mkdir()
    source_b.mkdir()
    (source_a / "config.txt").write_text("a\n", encoding="utf-8")
    (source_b / "config.txt").write_text("b\n", encoding="utf-8")
    destination = tmp_path / "restore"
    store = ContentAddressedStore(tmp_path / "tree-cas")
    backup_a = create_tree_backup(source_a, ["config.txt"], artifact_store=store)
    backup_b = create_tree_backup(source_b, ["config.txt"], artifact_store=store)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(destination)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    coordinator = _coordinator(state_path, journal=journal, snapshot=portable_snapshot)

    coordinator.restore_tree_backup(
        backup_a,
        destination_root=destination,
        artifact_store=store,
        resource_key=resource_key,
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        idempotency_key="restore-tree:same-key",
        now=now,
    )

    with pytest.raises(BatchIdempotencyMismatchError, match="payload"):
        coordinator.restore_tree_backup(
            backup_b,
            destination_root=destination,
            artifact_store=store,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="restore-tree:same-key",
            now=now + timedelta(seconds=1),
        )

    assert (destination / "config.txt").read_text(encoding="utf-8") == "a\n"


def test_restore_tree_backup_after_snapshot_failure_records_recovery_desired(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.txt").write_text("original\n", encoding="utf-8")
    destination = tmp_path / "restore"
    store = ContentAddressedStore(tmp_path / "tree-cas")
    backup = create_tree_backup(source, ["config.txt"], artifact_store=store)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(destination)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    snapshot_calls = 0

    def fail_after_restore(path: Path | str):
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 2:
            raise RuntimeError("injected after snapshot failure")
        return portable_snapshot(path)

    coordinator = _coordinator(state_path, journal=journal, snapshot=fail_after_restore)

    with pytest.raises(RuntimeError, match="injected after snapshot failure"):
        coordinator.restore_tree_backup(
            backup,
            destination_root=destination,
            artifact_store=store,
            resource_key=resource_key,
            lease=lease,
            owner="owner-a",
            run_id="run-1",
            idempotency_key="restore-tree:after-snapshot-fails",
            now=now,
        )

    assert (destination / "config.txt").read_text(encoding="utf-8") == "original\n"
    [batch] = journal.list_batches()
    assert batch.phase == BatchPhase.RECOVERY_DESIRED
    recovery = journal.list_recovery_records(batch.batch_id)[-1]
    assert recovery.reason == "filesystem post-mutation journal failed"
    assert recovery.payload["failure"]["stage"] == "after_snapshot"


def test_restore_tree_backup_uses_default_store_path_when_payload_omits_store(tmp_path: Path) -> None:
    target = tmp_path / "workspace"
    target.mkdir()
    expected_store_path = target.parent / ".safe-fs-ops-tree-backups" / "cas"
    seen_store_paths: list[Path] = []

    def fake_restore_tree_backup(
        backup: object,
        *,
        artifact_store: ContentAddressedStore,
        destination_root: Path,
    ) -> None:
        assert isinstance(backup, TreeBackup)
        assert backup.root == target
        assert destination_root == target
        seen_store_paths.append(artifact_store.root)

    action = _recovery_action_record(
        resource_key=tree_resource_key(target),
        payload={
            "backup": {
                "root_path": str(target),
                "entries": [],
            }
        },
    )

    restore_tree_backup_recovery_action(
        _authorized_context(tmp_path, action),
        action,
        restore_tree_backup_operation=fake_restore_tree_backup,
    )

    assert seen_store_paths == [expected_store_path]


def test_restore_tree_backup_recovery_action_refuses_without_authority_before_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "workspace"
    target.mkdir()
    action = _recovery_action_record(
        resource_key=tree_resource_key(target),
        payload={
            "backup": {
                "root_path": str(target),
                "store_path": str(tmp_path / "cas"),
                "entries": [],
            },
            "destination_root": str(target),
        },
    )
    restore_calls = 0

    def fake_restore_tree_backup(*_args: object, **_kwargs: object) -> None:
        nonlocal restore_calls
        restore_calls += 1

    with pytest.raises(RecoveryActionManualInterventionRequired, match="active recovery authority") as raised:
        restore_tree_backup_recovery_action(
            _empty_context(tmp_path),
            action,
            restore_tree_backup_operation=fake_restore_tree_backup,
        )

    assert raised.value.payload["reason_code"] == "missing_recovery_authority"
    assert restore_calls == 0


def test_restore_tree_backup_recovery_checks_authority_between_entries(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a\n", encoding="utf-8")
    (source / "b.txt").write_text("b\n", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    store = ContentAddressedStore(tmp_path / "tree-cas")
    backup = create_tree_backup(source, ["a.txt", "b.txt"], artifact_store=store)
    check_calls = 0

    def require_current() -> None:
        nonlocal check_calls
        check_calls += 1
        if check_calls >= 5:
            raise LeaseLostError("lease 'workspace' is no longer current")

    action = _recovery_action_record(
        resource_key=tree_resource_key(destination),
        payload={
            "backup": tree_backup_checkpoint_payload(backup, store_path=store.root),
            "destination_root": str(destination),
            "store_path": str(store.root),
        },
    )
    context = _authorized_context(
        tmp_path,
        action,
        authority=RecoveryAuthority(require_current, recovery_attempt_id="recovery-attempt"),
    )

    with pytest.raises(LeaseLostError, match="no longer current"):
        restore_tree_backup_recovery_action(context, action)

    assert (destination / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (destination / "b.txt").exists() is False


def test_restore_tree_backup_recovery_checks_action_authority_between_entries(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a\n", encoding="utf-8")
    (source / "b.txt").write_text("b\n", encoding="utf-8")
    destination = tmp_path / "destination"
    destination.mkdir()
    store = ContentAddressedStore(tmp_path / "tree-cas")
    backup = create_tree_backup(source, ["a.txt", "b.txt"], artifact_store=store)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(destination)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    recovery_attempt_id = _current_recovery_attempt_id(journal, batch.batch_id)

    def terminalize_after_first_entry(
        _backup: object,
        *,
        destination_root: Path,
        recovery_authority,
        **_kwargs: object,
    ) -> None:
        recovery_authority.require_current()
        (destination_root / "a.txt").write_text("a\n", encoding="utf-8")
        journal.record_recovery_action_skipped(
            batch_id=batch.batch_id,
            lease=lease,
            recovery_attempt_id=recovery_attempt_id,
            action_id="action-restore-tree",
            reason="terminalized elsewhere",
            payload={"event_type": "skipped"},
            now=now + timedelta(seconds=3),
        )
        recovery_authority.require_current()
        (destination_root / "b.txt").write_text("b\n", encoding="utf-8")

    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={
            RESTORE_TREE_BACKUP_ACTION: lambda context, action: restore_tree_backup_recovery_action(
                context,
                action,
                restore_tree_backup_operation=terminalize_after_first_entry,
            )
        },
    )
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type=RESTORE_TREE_BACKUP_ACTION,
        resource_key=resource_key,
        payload={
            "backup": tree_backup_checkpoint_payload(backup, store_path=store.root),
            "destination_root": str(destination),
            "store_path": str(store.root),
        },
        action_id="action-restore-tree",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="no longer active"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    assert (destination / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (destination / "b.txt").exists() is False
    assert [record.status for record in journal.list_recovery_actions(batch.batch_id)] == [
        "planned",
        "attempting",
        "skipped",
    ]


def test_restore_tree_backup_recovery_action_rejects_destination_resource_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    action = _recovery_action_record(
        resource_key=tree_resource_key(destination),
        payload={
            "backup": {
                "root_path": str(source),
                "store_path": str(tmp_path / "cas"),
                "entries": [],
            }
        },
    )

    with pytest.raises(RecoveryActionManualInterventionRequired) as raised:
        restore_tree_backup_recovery_action(_authorized_context(tmp_path, action), action)

    assert raised.value.payload["reason_code"] == "tree_backup_resource_mismatch"


def test_run_restore_tree_backup_operation_passes_keyword_only_destination_root(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    destination_root = tmp_path / "restore"
    seen: list[tuple[Path, ContentAddressedStore, str]] = []

    def fake_restore_tree_backup(
        backup: object,
        *,
        destination_root: Path,
        artifact_store: ContentAddressedStore,
        conflict_policy: str,
    ) -> str:
        assert backup == "backup"
        seen.append((destination_root, artifact_store, conflict_policy))
        return "restored"

    result = run_restore_tree_backup_operation(
        fake_restore_tree_backup,
        "backup",
        destination_root=destination_root,
        store=store,
        conflict_policy="replace",
    )

    assert result == "restored"
    assert seen == [(destination_root, store, "replace")]


@pytest.mark.parametrize(
    ("error", "reason_code"),
    [
        (RestoreConflictError("current tree changed"), "restore_conflict"),
        (ValueError("content object failed verification"), "tree_backup_content_mismatch"),
    ],
)
def test_recovery_runner_maps_tree_backup_restore_failures_to_manual_intervention(
    tmp_path: Path,
    error: Exception,
    reason_code: str,
) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    store_path = default_tree_backup_store_path(state_path)

    def fake_restore_tree_backup(*_args: object, **_kwargs: object) -> None:
        raise error

    coordinator = _coordinator(
        state_path,
        journal=journal,
        recovery_action_handlers={
            RESTORE_TREE_BACKUP_ACTION: lambda context, action: restore_tree_backup_recovery_action(
                context,
                action,
                restore_tree_backup_operation=fake_restore_tree_backup,
            )
        },
    )
    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type=RESTORE_TREE_BACKUP_ACTION,
        resource_key=resource_key,
        payload={
            "backup": {
                "root_path": str(source),
                "store_path": str(store_path),
                "entries": [],
            }
        },
        action_id="action-restore-tree",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="manual intervention"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
    exception_payload = recovery_actions[-1].payload["exception_payload"]
    assert exception_payload["reason_code"] == reason_code
    assert exception_payload["store_path"] == str(store_path)
    assert exception_payload["root_path"] == str(source)


def test_recovery_runner_maps_malformed_tree_backup_payload_to_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    store_path = default_tree_backup_store_path(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type=RESTORE_TREE_BACKUP_ACTION,
        resource_key=resource_key,
        payload={
            "allow_overwrite": "yes",
            "backup": {
                "root_path": str(source),
                "store_path": str(store_path),
                "entries": [],
            },
        },
        action_id="action-restore-tree",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="manual intervention"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
    exception_payload = recovery_actions[-1].payload["exception_payload"]
    assert exception_payload["reason_code"] == "malformed_tree_backup_recovery_payload"


def test_recovery_runner_maps_malformed_tree_backup_entry_to_manual_intervention(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    source = tmp_path / "workspace"
    source.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    resource_key = tree_resource_key(source)
    lease = _acquire_and_claim(state_path, resource_key, now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=resource_key, now=now)
    store_path = default_tree_backup_store_path(state_path)
    coordinator = _coordinator(state_path, journal=journal)

    journal.record_recovery_action_planned(
        batch_id=batch.batch_id,
        lease=lease,
        recovery_attempt_id=_current_recovery_attempt_id(journal, batch.batch_id),
        action_type=RESTORE_TREE_BACKUP_ACTION,
        resource_key=resource_key,
        payload={
            "backup": {
                "root_path": str(source),
                "store_path": str(store_path),
                "entries": [{"relative_path": "x.txt"}],
            },
        },
        action_id="action-restore-tree",
        now=now + timedelta(seconds=1),
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="manual intervention"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch.batch_id),
            lease=lease,
            now=now + timedelta(seconds=2),
        )

    recovery_actions = journal.list_recovery_actions(batch.batch_id)
    assert [record.status for record in recovery_actions] == [
        "planned",
        "attempting",
        "manual_intervention_required",
    ]
    exception_payload = recovery_actions[-1].payload["exception_payload"]
    assert exception_payload["reason_code"] == "malformed_tree_backup_recovery_payload"


def _start_recovery_for_succeeded_batch(
    journal: OperationJournalStore,
    *,
    lease,
    batch_id: str,
    now: datetime,
) -> None:
    journal.record_recovery_desired(
        batch_id,
        lease=lease,
        reason="restore tree backup",
        payload={"batch_id": batch_id},
        recovery_id=f"{batch_id}-desired",
        now=now,
    )


def _empty_context(tmp_path: Path):
    state_path = tmp_path / "empty-context.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, directory_resource_key(tmp_path), now=now)
    journal = OperationJournalStore(state_path)
    batch = _start_recovering_batch(journal, lease=lease, resource_key=directory_resource_key(tmp_path), now=now)
    return journal.read_recovery_context(batch.batch_id)


def _authorized_context(
    tmp_path: Path,
    action: RecoveryActionRecord,
    *,
    authority: RecoveryAuthority | None = None,
):
    return replace(
        _empty_context(tmp_path),
        recovery_actions=(action,),
        recovery_authority=authority or RecoveryAuthority(lambda: None, recovery_attempt_id="recovery-attempt"),
    )


def _recovery_action_record(*, payload: dict[str, object], resource_key: str = "tree:test") -> RecoveryActionRecord:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return RecoveryActionRecord(
        action_record_id="record-1",
        action_id="action-restore-tree",
        recovery_attempt_id="recovery-attempt",
        batch_id="batch-1",
        sequence=1,
        action_type=RESTORE_TREE_BACKUP_ACTION,
        status="planned",
        resource_key=resource_key,
        reason=None,
        payload=payload,
        created_at=now,
    )
