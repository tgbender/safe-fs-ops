from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    RenameRecord,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    rename_no_replace,
    restore_inverse_rename,
)

pytestmark = pytest.mark.safe_fs_ops


def test_rename_no_replace_moves_file_and_restore_inverse_moves_it_back(tmp_path: Path) -> None:
    source = tmp_path / "config.txt"
    destination = tmp_path / "renamed.txt"
    source.write_text("payload\n", encoding="utf-8")
    before = source.lstat()

    record = _rename_or_skip(source, destination)

    assert isinstance(record, RenameRecord)
    assert record.source_path == source.resolve(strict=False)
    assert record.destination_path == destination.resolve(strict=False)
    assert record.file_type == "file"
    assert record.device == before.st_dev
    assert record.inode == before.st_ino
    assert source.exists() is False
    assert destination.read_text(encoding="utf-8") == "payload\n"

    inverse = restore_inverse_rename(record)

    assert inverse.source_path == destination
    assert inverse.destination_path == source
    assert source.read_text(encoding="utf-8") == "payload\n"
    assert destination.exists() is False


def test_rename_no_replace_moves_directory_without_merging(tmp_path: Path) -> None:
    source = tmp_path / "state"
    destination = tmp_path / "captured"
    source.mkdir()
    (source / "data.txt").write_text("payload\n", encoding="utf-8")

    record = _rename_or_skip(source, destination)

    assert record.file_type == "directory"
    assert source.exists() is False
    assert (destination / "data.txt").read_text(encoding="utf-8") == "payload\n"


def test_rename_no_replace_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    destination.write_text("destination\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        rename_no_replace(source, destination)

    assert source.read_text(encoding="utf-8") == "source\n"
    assert destination.read_text(encoding="utf-8") == "destination\n"


def test_rename_no_replace_refuses_symlink_source(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    link = tmp_path / "link.txt"
    destination = tmp_path / "destination.txt"
    real.write_text("real\n", encoding="utf-8")
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        rename_no_replace(link, destination)

    assert real.read_text(encoding="utf-8") == "real\n"
    assert destination.exists() is False


def test_rename_no_replace_does_not_report_success_when_destination_identity_differs(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    hidden = tmp_path / "hidden.txt"
    source.write_text("source\n", encoding="utf-8")

    def move_elsewhere_and_create_replacement(source_path: Path, destination_path: Path, *, operation: str) -> None:
        del operation
        os.rename(source_path, hidden)
        destination_path.write_text("replacement\n", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="destination identity changed"):
        rename_no_replace(source, destination, _rename_no_replace=move_elsewhere_and_create_replacement)

    assert hidden.read_text(encoding="utf-8") == "source\n"
    assert destination.read_text(encoding="utf-8") == "replacement\n"


def test_restore_inverse_rename_refuses_when_recorded_destination_was_replaced(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    record = _rename_or_skip(source, destination)
    destination.unlink()
    destination.write_text("replacement\n", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="no longer matches recorded identity"):
        restore_inverse_rename(record)

    assert source.exists() is False
    assert destination.read_text(encoding="utf-8") == "replacement\n"


def test_rename_no_replace_backend_file_exists_leaves_source_in_place(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")

    def destination_appears(source_path: Path, destination_path: Path, *, operation: str) -> None:
        del source_path, operation
        destination_path.write_text("destination\n", encoding="utf-8")
        raise FileExistsError(destination_path)

    with pytest.raises(FileExistsError):
        rename_no_replace(source, destination, _rename_no_replace=destination_appears)

    assert source.read_text(encoding="utf-8") == "source\n"
    assert destination.read_text(encoding="utf-8") == "destination\n"


def _rename_or_skip(source: Path, destination: Path) -> RenameRecord:
    try:
        return rename_no_replace(source, destination)
    except UnsupportedFilesystemMutationError as exc:
        if "no-replace rename is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise
