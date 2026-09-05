from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    DurabilityMode,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    capture_directory_to_quarantine,
    cleanup_captured_directory,
    restore_captured_directory,
)
from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token

pytestmark = pytest.mark.safe_fs_ops


def test_capture_moves_preexisting_directory_into_quarantine_and_records_identity(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")
    original_identity = DirectoryIdentity.from_stat(source.stat())

    capture_token = directory_capture_token(
        source, device=original_identity.device, inode=original_identity.inode, create=True
    )
    captured = _capture_or_skip(source, quarantine)

    assert captured.capture_token is not None
    assert captured == CapturedDirectoryRecord(
        original_path=source.resolve(strict=False),
        quarantine_path=quarantine.resolve(strict=False),
        original_identity=original_identity,
        captured_identity=original_identity,
        capture_token=capture_token,
    )
    assert source.exists() is False
    assert quarantine.is_dir()
    assert (quarantine / "data.txt").read_text(encoding="utf-8") == "payload\n"


def test_capture_refuses_when_quarantine_destination_exists(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    quarantine.mkdir()

    with pytest.raises(FileExistsError):
        capture_directory_to_quarantine(source, quarantine_path=quarantine)

    assert source.is_dir()
    assert quarantine.is_dir()


def test_capture_does_not_overwrite_quarantine_created_after_precheck(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")

    def destination_appears(source_path: Path, destination: Path, *, operation: str) -> None:
        destination.mkdir()
        raise FileExistsError(destination)

    with pytest.raises(FileExistsError):
        capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=destination_appears)

    assert source.is_dir()
    assert (source / "data.txt").read_text(encoding="utf-8") == "payload\n"
    assert quarantine.is_dir()
    assert (quarantine / "data.txt").exists() is False


def test_capture_refuses_symlink_source(tmp_path: Path) -> None:
    real_source = tmp_path / "real-state"
    link_source = tmp_path / "state-link"
    quarantine = tmp_path / "quarantine" / "txn-1"
    real_source.mkdir()
    quarantine.parent.mkdir()
    try:
        link_source.symlink_to(real_source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        capture_directory_to_quarantine(link_source, quarantine_path=quarantine)

    assert real_source.is_dir()
    assert quarantine.exists() is False


def test_capture_refuses_symlink_substituted_at_quarantine_during_post_rename_verification(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    hidden_destination = tmp_path / "quarantine" / "hidden"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    hidden_destination.mkdir()
    try:
        hidden_link_target = tmp_path / "quarantine" / "hidden-link-target"
        hidden_link_target.mkdir()
        os.rmdir(hidden_destination)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    def rename_to_hidden_and_substitute_symlink(source_path: Path, destination: Path, *, operation: str) -> None:
        del operation
        source_path.rename(hidden_destination)
        destination.symlink_to(hidden_link_target, target_is_directory=True)

    with pytest.raises(UnsafePathError, match="not a directory"):
        capture_directory_to_quarantine(
            source,
            quarantine_path=quarantine,
            _rename_no_replace=rename_to_hidden_and_substitute_symlink,
        )


def test_restore_moves_captured_directory_back_when_original_path_is_absent(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")

    captured = _capture_or_skip(source, quarantine)
    restored_identity = restore_captured_directory(captured)

    assert restored_identity == captured.captured_identity
    assert source.is_dir()
    assert quarantine.exists() is False
    assert (source / "data.txt").read_text(encoding="utf-8") == "payload\n"


def test_restore_refuses_symlink_substituted_at_original_path_during_post_rename_verification(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    hidden_destination = tmp_path / "workspace" / "hidden"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)
    hidden_destination.mkdir()
    try:
        hidden_link_target = tmp_path / "workspace" / "hidden-link-target"
        hidden_link_target.mkdir()
        os.rmdir(hidden_destination)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")

    def rename_to_hidden_and_substitute_symlink(source_path: Path, destination: Path, *, operation: str) -> None:
        del operation
        source_path.rename(hidden_destination)
        destination.symlink_to(hidden_link_target, target_is_directory=True)

    with pytest.raises(UnsafePathError, match="not a directory"):
        restore_captured_directory(captured, _rename_no_replace=rename_to_hidden_and_substitute_symlink)


def test_restore_refuses_when_original_path_is_occupied(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()

    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)
    source.mkdir()

    with pytest.raises(FileExistsError):
        restore_captured_directory(captured)

    assert source.is_dir()
    assert quarantine.is_dir()


def test_restore_does_not_overwrite_original_path_created_after_precheck(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)

    def destination_appears(source_path: Path, destination: Path, *, operation: str) -> None:
        destination.mkdir()
        raise FileExistsError(destination)

    with pytest.raises(FileExistsError):
        restore_captured_directory(captured, _rename_no_replace=destination_appears)

    assert source.is_dir()
    assert (source / "data.txt").exists() is False
    assert quarantine.is_dir()
    assert (quarantine / "data.txt").read_text(encoding="utf-8") == "payload\n"


def test_cleanup_refuses_replacement_directory_even_on_supported_platform(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    moved_quarantine = tmp_path / "quarantine" / "moved"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()

    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)
    quarantine.rename(moved_quarantine)
    quarantine.mkdir()

    with pytest.raises(UnsafePathError, match="no longer matches"):
        cleanup_captured_directory(
            captured,
            durability=DurabilityMode.NONE,
            _platform="win32",
        )

    assert moved_quarantine.is_dir()
    assert quarantine.is_dir()


def test_cleanup_removes_empty_captured_quarantine(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()

    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)

    cleanup_captured_directory(
        captured,
        durability=DurabilityMode.NONE,
        _identity_remove_directory=_portable_identity_remove,
    )

    assert quarantine.exists() is False


def test_cleanup_refuses_non_empty_captured_directory_tree_without_safe_recursive_deleter(tmp_path: Path) -> None:
    source = tmp_path / "workspace" / "state"
    quarantine = tmp_path / "quarantine" / "txn-1"
    source.parent.mkdir()
    quarantine.parent.mkdir()
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")
    nested = source / "nested"
    nested.mkdir()
    (nested / "more.txt").write_text("more\n", encoding="utf-8")
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine, _rename_no_replace=_plain_rename)

    with pytest.raises(UnsupportedFilesystemMutationError, match="safe recursive deleter"):
        cleanup_captured_directory(captured, durability=DurabilityMode.NONE, _platform="win32")

    assert quarantine.is_dir()
    assert (quarantine / "data.txt").read_text(encoding="utf-8") == "payload\n"
    assert (quarantine / "nested" / "more.txt").read_text(encoding="utf-8") == "more\n"


def _capture_or_skip(source: Path, quarantine: Path) -> CapturedDirectoryRecord:
    try:
        return capture_directory_to_quarantine(source, quarantine_path=quarantine)
    except UnsupportedFilesystemMutationError as exc:
        if "no-replace directory rename is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise


def _plain_rename(source: Path, destination: Path, *, operation: str) -> None:
    os.rename(source, destination)


def _portable_identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
    target = Path(path)
    current_identity = DirectoryIdentity.from_stat(target.stat())
    if current_identity != expected_identity:
        raise UnsafePathError("identity mismatch")
    target.rmdir()
