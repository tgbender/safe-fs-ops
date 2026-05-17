from __future__ import annotations

from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    BackupContentMismatchError,
    ContentAddressedStore,
    DurabilityMode,
    FileBackup,
    ResourceSnapshot,
    RestoreConflictError,
    UnsafePathError,
    capture_backup,
    restore_backup,
    snapshot_resource,
)
from safe_fs_ops.filesystem_ops.backups import _read_regular_file_backup_bytes_portable

pytestmark = pytest.mark.safe_fs_ops


def _descriptor_relative_snapshot_supported() -> bool:
    import os

    return (
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def _descriptor_relative_mutations_supported() -> bool:
    import os

    return (
        os.open in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and (os.replace in os.supports_dir_fd or (os.name != "nt" and os.rename in os.supports_dir_fd))
        and os.mkdir in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


requires_backup_restore_support = pytest.mark.skipif(
    not (_descriptor_relative_snapshot_supported() and _descriptor_relative_mutations_supported()),
    reason="descriptor-relative snapshot/mutation support is unavailable on this platform",
)


@requires_backup_restore_support
def test_capture_and_restore_regular_file_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    backup = capture_backup(target)
    target.write_text("mutated\n", encoding="utf-8")
    expected_current = snapshot_resource(target)

    restored = restore_backup(
        backup,
        expected_current=expected_current,
        durability=DurabilityMode.NONE,
    )

    assert restored.file_type == "file"
    assert target.read_text(encoding="utf-8") == "original\n"
    assert (target.lstat().st_mode & 0o777) == 0o640
    assert restored.content_hash == backup.content_hash


@requires_backup_restore_support
def test_capture_and_restore_regular_file_with_content_path(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    content_path = tmp_path / "backup-content.bin"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    backup = capture_backup(target, content_path=content_path)
    assert backup.content_bytes is None
    assert backup.content_path == content_path

    target.write_text("mutated\n", encoding="utf-8")
    restored = restore_backup(
        backup,
        expected_current=snapshot_resource(target),
        durability=DurabilityMode.NONE,
    )

    assert restored.file_type == "file"
    assert target.read_text(encoding="utf-8") == "original\n"
    assert (target.lstat().st_mode & 0o777) == 0o640


@requires_backup_restore_support
def test_capture_and_restore_regular_file_with_content_addressed_store(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    store = ContentAddressedStore(tmp_path / "objects")
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    backup = capture_backup(target, artifact_store=store)
    assert backup.content_bytes is None
    assert backup.content_path is None
    assert backup.content_address is not None
    assert store.verify(backup.content_address) is True

    target.write_text("mutated\n", encoding="utf-8")
    restored = restore_backup(
        backup,
        expected_current=snapshot_resource(target),
        artifact_store=store,
        durability=DurabilityMode.NONE,
    )

    assert restored.file_type == "file"
    assert target.read_text(encoding="utf-8") == "original\n"
    assert (target.lstat().st_mode & 0o777) == 0o640


@requires_backup_restore_support
def test_restore_content_addressed_backup_rejects_tampered_object(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    store = ContentAddressedStore(tmp_path / "objects")
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target, artifact_store=store)
    assert backup.content_address is not None
    backup.content_address.path.write_text("tampered\n", encoding="utf-8")
    target.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(BackupContentMismatchError, match="content address mismatch"):
        restore_backup(
            backup,
            expected_current=snapshot_resource(target),
            artifact_store=store,
            durability=DurabilityMode.NONE,
        )

    assert target.read_text(encoding="utf-8") == "mutated\n"


def test_capture_backup_rejects_content_path_with_artifact_store(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot both be provided"):
        capture_backup(
            target,
            content_path=tmp_path / "backup-content.bin",
            artifact_store=ContentAddressedStore(tmp_path / "objects"),
        )


def test_portable_backup_reader_reads_large_file_with_identity_checks(tmp_path: Path) -> None:
    target = tmp_path / "large.bin"
    content = (b"0123456789abcdef" * 70000) + b"\n"
    target.write_bytes(content)
    stat_result = target.lstat()
    snapshot = ResourceSnapshot(
        path=target,
        exists=True,
        file_type="file",
        content_hash=None,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )

    captured, permissions = _read_regular_file_backup_bytes_portable(
        target,
        expected_snapshot=snapshot,
    )

    assert captured == content
    assert permissions is not None


@requires_backup_restore_support
def test_capture_backup_rejects_content_path_alias_of_source(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    with pytest.raises(ValueError, match="must not alias"):
        capture_backup(target, content_path=tmp_path / "." / "config.txt")

    assert target.read_text(encoding="utf-8") == "original\n"
    assert (target.lstat().st_mode & 0o777) == 0o640


@requires_backup_restore_support
def test_restore_rejects_out_of_line_content_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    content_path = tmp_path / "backup-content.bin"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target, content_path=content_path)
    assert backup.content_path == content_path

    content_path.write_text("tampered\n", encoding="utf-8")
    target.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(BackupContentMismatchError, match="hash mismatch"):
        restore_backup(
            backup,
            expected_current=snapshot_resource(target),
            durability=DurabilityMode.NONE,
        )


@requires_backup_restore_support
def test_restore_regular_file_is_idempotent_when_run_twice(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target)
    target.write_text("mutated\n", encoding="utf-8")
    restore_backup(backup, expected_current=snapshot_resource(target), durability=DurabilityMode.NONE)
    restored_again = restore_backup(backup, durability=DurabilityMode.NONE)

    assert target.read_text(encoding="utf-8") == "original\n"
    assert restored_again.content_hash == backup.content_hash


def test_file_backup_requires_permissions(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    snapshot = ResourceSnapshot(
        path=target,
        exists=True,
        file_type="file",
        content_hash="0" * 64,
        size=9,
        mtime_ns=1,
        symlink_target=None,
    )

    with pytest.raises(ValueError, match="permissions"):
        FileBackup(
            path=target,
            existed=True,
            file_type="file",
            content_bytes=b"original\n",
            content_path=None,
            content_hash=snapshot.content_hash,
            size=snapshot.size,
            permissions=None,
            snapshot=snapshot,
        )


@requires_backup_restore_support
def test_restore_missing_backup_deletes_only_when_current_state_is_expected(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"

    backup = capture_backup(target)
    target.write_text("created\n", encoding="utf-8")
    expected_current = snapshot_resource(target)

    restored = restore_backup(
        backup,
        expected_current=expected_current,
        durability=DurabilityMode.NONE,
    )
    restored_again = restore_backup(backup, durability=DurabilityMode.NONE)

    assert restored.file_type == "missing"
    assert restored_again.file_type == "missing"
    assert target.exists() is False


@requires_backup_restore_support
def test_restore_rejects_final_permission_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    backup = capture_backup(target)
    assert backup.permissions is not None

    target.write_text("mutated\n", encoding="utf-8")

    def _wrong_permissions_reader(path: Path) -> int | None:
        if path == target:
            return (backup.permissions ^ 0o111) & 0o777
        return None

    with pytest.raises(BackupContentMismatchError, match="permissions"):
        restore_backup(
            backup,
            expected_current=snapshot_resource(target),
            durability=DurabilityMode.NONE,
            _current_permissions_reader=_wrong_permissions_reader,
        )


@requires_backup_restore_support
def test_restore_refuses_external_content_changes_by_default(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target)
    target.write_text("first mutation\n", encoding="utf-8")
    expected_current = snapshot_resource(target)
    target.write_text("external mutation\n", encoding="utf-8")

    with pytest.raises(RestoreConflictError, match="current state changed unexpectedly"):
        restore_backup(backup, expected_current=expected_current, durability=DurabilityMode.NONE)

    assert target.read_text(encoding="utf-8") == "external mutation\n"


@requires_backup_restore_support
def test_restore_refuses_when_expected_current_file_was_deleted(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target)
    target.write_text("mutated\n", encoding="utf-8")
    expected_current = snapshot_resource(target)
    target.unlink()

    with pytest.raises(RestoreConflictError, match="current state changed unexpectedly"):
        restore_backup(backup, expected_current=expected_current, durability=DurabilityMode.NONE)

    assert target.exists() is False


@requires_backup_restore_support
def test_restore_rejects_backup_hash_mismatch(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(target)
    corrupted = FileBackup(
        path=backup.path,
        existed=backup.existed,
        file_type=backup.file_type,
        content_bytes=b"corrupted\n",
        content_path=None,
        content_hash=backup.content_hash,
        size=backup.size,
        permissions=backup.permissions,
        snapshot=backup.snapshot,
    )

    with pytest.raises(BackupContentMismatchError, match="hash mismatch"):
        restore_backup(corrupted, durability=DurabilityMode.NONE)


@requires_backup_restore_support
def test_capture_and_restore_reject_symlink_targets(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("real\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        capture_backup(link)

    missing_backup = capture_backup(tmp_path / "missing.txt")
    link.unlink()
    try:
        (tmp_path / "missing.txt").symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        restore_backup(missing_backup, allow_overwrite=True, durability=DurabilityMode.NONE)
