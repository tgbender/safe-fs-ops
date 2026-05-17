from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    BackupContentMismatchError,
    ContentAddressedStore,
    ContentRef,
    ResourceSnapshot,
    RestoreConflictError,
    TreeBackup,
    TreeBackupEntry,
    TreeBackupEntryKind,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    backup_tree,
    restore_tree_backup,
    tree_backup_from_journal_payload,
)
from safe_fs_ops.operation_journal.tree_backup_recovery import tree_backup_checkpoint_payload

pytestmark = pytest.mark.safe_fs_ops


def test_backup_tree_captures_explicit_files_and_restores_to_destination_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    nested = source_root / "settings"
    nested.mkdir(parents=True)
    target = nested / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)
    store = ContentAddressedStore(tmp_path / "objects")

    backup = backup_tree(source_root, ["settings/config.txt"], artifact_store=store)

    assert isinstance(backup, TreeBackup)
    assert backup.root == source_root
    [entry] = backup.entries
    kind: TreeBackupEntryKind = entry.kind
    assert kind == "file"
    assert entry.relative_path == Path("settings/config.txt")
    assert entry.content_address is not None
    assert store.verify(entry.content_address) is True

    destination_root = tmp_path / "restore"
    restored = restore_tree_backup(backup, destination_root, artifact_store=store)

    restored_target = destination_root / "settings" / "config.txt"
    assert restored_target.read_text(encoding="utf-8") == "original\n"
    assert restored[0].path == restored_target
    assert restored[0].content_hash == entry.content_hash
    if os.name != "nt":
        assert (restored_target.lstat().st_mode & 0o777) == 0o640


def test_tree_backup_from_journal_payload_restores_checkpoint_payload(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "config.txt").write_text("original\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    payload = tree_backup_checkpoint_payload(backup, store_path=store.root)

    restored_backup = tree_backup_from_journal_payload(payload)

    assert restored_backup == backup
    restore_tree_backup(restored_backup, tmp_path / "restore", artifact_store=store)
    assert (tmp_path / "restore" / "config.txt").read_text(encoding="utf-8") == "original\n"


def test_tree_backup_from_journal_payload_accepts_backup_tree_result_payload(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "config.txt").write_text("original\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    checkpoint_payload = tree_backup_checkpoint_payload(backup, store_path=store.root)

    restored_backup = tree_backup_from_journal_payload(
        {
            "operation_id": "operation-1",
            "resource_key": "tree:source",
            "tree_backup": checkpoint_payload,
        }
    )

    assert restored_backup == backup


def test_restore_tree_backup_restores_recorded_file_mtime(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = source_root / "config.txt"
    source.write_text("original\n", encoding="utf-8")
    os.utime(source, ns=(1_700_000_000_123_456_000, 1_700_000_000_123_456_000))
    recorded_mtime_ns = source.lstat().st_mtime_ns
    store = ContentAddressedStore(tmp_path / "objects")

    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    restored = restore_tree_backup(backup, tmp_path / "restore", artifact_store=store)

    restored_target = tmp_path / "restore" / "config.txt"
    assert backup.entries[0].snapshot.mtime_ns == recorded_mtime_ns
    assert restored_target.lstat().st_mtime_ns == recorded_mtime_ns
    assert restored[0].mtime_ns == recorded_mtime_ns


def test_backup_tree_records_symlink_target_without_following(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    real = source_root / "real.txt"
    real.write_text("real\n", encoding="utf-8")
    link = source_root / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(tmp_path / "objects")

    backup = backup_tree(source_root, ["link.txt"], artifact_store=store)

    [entry] = backup.entries
    assert entry.kind == "symlink"
    assert entry.symlink_target == os.readlink(link)
    assert entry.content_address is None
    if os.name == "nt":
        pytest.skip("safe symlink restore is unsupported on Windows")
    if os.symlink not in os.supports_dir_fd:
        pytest.skip("descriptor-relative symlink creation unavailable")

    destination_root = tmp_path / "restore"
    restore_tree_backup(backup, destination_root, artifact_store=store)

    restored_link = destination_root / "link.txt"
    assert restored_link.is_symlink()
    assert os.readlink(restored_link) == os.readlink(link)
    assert not (destination_root / "real.txt").exists()


def test_restore_tree_backup_replace_refuses_symlink_without_safe_primitive(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("safe symlink replacement is unsupported on Windows")
    if os.symlink not in os.supports_dir_fd:
        pytest.skip("descriptor-relative symlink creation unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    first_target = source_root / "first.txt"
    first_target.write_text("first\n", encoding="utf-8")
    source_link = source_root / "link.txt"
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    second_target = destination_root / "second.txt"
    second_target.write_text("second\n", encoding="utf-8")
    destination_link = destination_root / "link.txt"
    try:
        source_link.symlink_to(first_target)
        destination_link.symlink_to(second_target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["link.txt"], artifact_store=store)

    with pytest.raises(UnsupportedFilesystemMutationError, match="safe symlink replacement is unavailable"):
        restore_tree_backup(backup, destination_root, artifact_store=store, conflict_policy="replace")

    assert destination_link.is_symlink()
    assert os.readlink(destination_link) == str(second_target)
    assert second_target.read_text(encoding="utf-8") == "second\n"


def test_restore_tree_backup_replace_preflights_unsupported_symlink_before_file_restore(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("safe symlink restore is unsupported on Windows")
    if os.symlink not in os.supports_dir_fd:
        pytest.skip("descriptor-relative symlink creation unavailable")
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "first.txt").write_text("first\n", encoding="utf-8")
    first_target = source_root / "first-target.txt"
    first_target.write_text("first target\n", encoding="utf-8")
    source_link = source_root / "link.txt"
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    (destination_root / "first.txt").write_text("keep\n", encoding="utf-8")
    second_target = destination_root / "second-target.txt"
    second_target.write_text("second target\n", encoding="utf-8")
    destination_link = destination_root / "link.txt"
    try:
        source_link.symlink_to(first_target)
        destination_link.symlink_to(second_target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["first.txt", "link.txt"], artifact_store=store)

    with pytest.raises(UnsupportedFilesystemMutationError, match="safe symlink replacement is unavailable"):
        restore_tree_backup(backup, destination_root, artifact_store=store, conflict_policy="replace")

    assert (destination_root / "first.txt").read_text(encoding="utf-8") == "keep\n"
    assert destination_link.is_symlink()
    assert os.readlink(destination_link) == str(second_target)


def test_restore_tree_backup_windows_preflights_unsupported_symlink_before_file_restore(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows symlink restore preflight is Windows-specific")
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = source_root / "first.txt"
    first.write_text("first\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    file_backup = backup_tree(source_root, ["first.txt"], artifact_store=store)
    symlink_path = source_root / "link.txt"
    symlink_entry = TreeBackupEntry(
        relative_path=Path("link.txt"),
        kind="symlink",
        snapshot=ResourceSnapshot(
            path=symlink_path,
            exists=True,
            file_type="symlink",
            content_hash=None,
            size=11,
            mtime_ns=None,
            symlink_target="target.txt",
        ),
        symlink_target="target.txt",
    )
    backup = TreeBackup(root=source_root, entries=(*file_backup.entries, symlink_entry))
    destination_root = tmp_path / "restore"

    with pytest.raises(UnsupportedFilesystemMutationError, match="safe symlink creation is unavailable on Windows"):
        restore_tree_backup(backup, destination_root, artifact_store=store)

    assert destination_root.exists() is False


@pytest.mark.parametrize("relative_path", ["/absolute.txt", "../escape.txt", "nested/../../escape.txt", "."])
def test_backup_tree_rejects_non_explicit_relative_paths(tmp_path: Path, relative_path: str) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    store = ContentAddressedStore(tmp_path / "objects")

    with pytest.raises(UnsafePathError, match="relative|escape"):
        backup_tree(source_root, [relative_path], artifact_store=store)


def test_backup_tree_rejects_missing_paths_and_directories(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    directory = source_root / "directory"
    directory.mkdir(parents=True)
    store = ContentAddressedStore(tmp_path / "objects")

    with pytest.raises(FileNotFoundError):
        backup_tree(source_root, ["missing.txt"], artifact_store=store)
    with pytest.raises(UnsafePathError, match="directory"):
        backup_tree(source_root, ["directory"], artifact_store=store)


def test_backup_tree_rejects_paths_beneath_symlink_parent(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    outside = tmp_path / "outside"
    source_root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
    link_parent = source_root / "link-parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(tmp_path / "objects")

    with pytest.raises(UnsafePathError, match="redirects"):
        backup_tree(source_root, ["link-parent/secret.txt"], artifact_store=store)


def test_restore_tree_backup_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "config.txt").write_text("original\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    conflict = destination_root / "config.txt"
    conflict.write_text("keep\n", encoding="utf-8")

    with pytest.raises(RestoreConflictError, match="destination already exists"):
        restore_tree_backup(backup, destination_root, artifact_store=store)

    assert conflict.read_text(encoding="utf-8") == "keep\n"


def test_restore_tree_backup_rejects_tampered_content_object(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "config.txt").write_text("original\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    [entry] = backup.entries
    assert entry.content_address is not None
    entry.content_address.path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(BackupContentMismatchError, match="content address mismatch"):
        restore_tree_backup(backup, tmp_path / "restore", artifact_store=store)

    assert not (tmp_path / "restore" / "config.txt").exists()


def test_restore_tree_backup_preflights_all_artifacts_before_creating_destination_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "first.txt").write_text("first\n", encoding="utf-8")
    (source_root / "second.txt").write_text("second\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["first.txt", "second.txt"], artifact_store=store)
    second_entry = backup.entries[1]
    assert second_entry.content_address is not None
    second_entry.content_address.path.write_text("tampered\n", encoding="utf-8")
    destination_root = tmp_path / "restore"

    with pytest.raises(BackupContentMismatchError, match="content address mismatch"):
        restore_tree_backup(backup, destination_root, artifact_store=store)

    assert not destination_root.exists()


def test_restore_tree_backup_preflights_conflicts_before_restoring_any_entry(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "first.txt").write_text("first\n", encoding="utf-8")
    (source_root / "second.txt").write_text("second\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["first.txt", "second.txt"], artifact_store=store)
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    (destination_root / "second.txt").write_text("keep\n", encoding="utf-8")

    with pytest.raises(RestoreConflictError, match="destination already exists"):
        restore_tree_backup(backup, destination_root, artifact_store=store)

    assert not (destination_root / "first.txt").exists()
    assert (destination_root / "second.txt").read_text(encoding="utf-8") == "keep\n"


def test_restore_tree_backup_preflights_file_over_symlink_before_earlier_replace(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "first.txt").write_text("first\n", encoding="utf-8")
    (source_root / "second.txt").write_text("second\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["first.txt", "second.txt"], artifact_store=store)
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    (destination_root / "first.txt").write_text("keep\n", encoding="utf-8")
    symlink_target = destination_root / "target.txt"
    symlink_target.write_text("target\n", encoding="utf-8")
    second_link = destination_root / "second.txt"
    try:
        second_link.symlink_to(symlink_target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="replace symlink with file"):
        restore_tree_backup(backup, destination_root, artifact_store=store, conflict_policy="replace")

    assert (destination_root / "first.txt").read_text(encoding="utf-8") == "keep\n"
    assert second_link.is_symlink()
    assert os.path.normcase(os.readlink(second_link).removeprefix("\\\\?\\")) == os.path.normcase(str(symlink_target))


def test_restore_tree_backup_replace_preserves_existing_file_when_content_is_tampered(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "config.txt").write_text("original\n", encoding="utf-8")
    store = ContentAddressedStore(tmp_path / "objects")
    backup = backup_tree(source_root, ["config.txt"], artifact_store=store)
    [entry] = backup.entries
    assert entry.content_address is not None
    entry.content_address.path.write_text("tampered\n", encoding="utf-8")
    destination_root = tmp_path / "restore"
    destination_root.mkdir()
    destination = destination_root / "config.txt"
    destination.write_text("keep\n", encoding="utf-8")

    with pytest.raises(BackupContentMismatchError, match="content address mismatch"):
        restore_tree_backup(backup, destination_root, artifact_store=store, conflict_policy="replace")

    assert destination.read_text(encoding="utf-8") == "keep\n"


def test_tree_backup_rejects_prefix_colliding_entries(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    content_path = tmp_path / "objects" / "content"
    content_path.parent.mkdir()
    content_path.write_text("content\n", encoding="utf-8")
    content_ref = ContentRef(
        digest="ed7002b439e9ac845f2233f779a5a57f51bdb407e3bf80184e9d569e8f73a163",
        size=8,
        path=content_path,
    )

    def entry(relative_path: str) -> TreeBackupEntry:
        return TreeBackupEntry(
            relative_path=Path(relative_path),
            kind="file",
            snapshot=ResourceSnapshot(
                path=root / relative_path,
                exists=True,
                file_type="file",
                content_hash=content_ref.digest,
                size=content_ref.size,
                mtime_ns=None,
                symlink_target=None,
            ),
            content_address=content_ref,
            content_hash=content_ref.digest,
            size=content_ref.size,
            permissions=0o644,
        )

    with pytest.raises(ValueError, match="conflicts with prefix path"):
        TreeBackup(root=root, entries=(entry("a"), entry("a/b")))
