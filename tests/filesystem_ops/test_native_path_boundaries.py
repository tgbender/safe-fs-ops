from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    DurabilityMode,
    capture_directory_to_quarantine,
    rename_no_replace,
    restore_captured_directory,
    restore_inverse_rename,
)
from safe_fs_ops.filesystem_ops._windows_identity_rmdir import remove_empty_directory_by_identity_windows
from safe_fs_ops.filesystem_ops._windows_primitives import WindowsFilePrimitiveApi, replace_file_windows
from safe_fs_ops.filesystem_ops.no_replace_rename import rename_path_no_replace

pytestmark = pytest.mark.safe_fs_ops


@pytest.mark.parametrize("name", ["ascii", "a\U0001f600b", "\U0001f600\U0001f600"])
@pytest.mark.parametrize("directory", [False, True])
def test_native_rename_preserves_unicode_names_and_inverse(tmp_path: Path, name: str, directory: bool) -> None:
    source, destination = tmp_path / "source", tmp_path / name
    if directory:
        source.mkdir()
    content = source / "data.txt" if directory else source
    content.write_text("original", encoding="utf-8")
    identity = (source.stat().st_dev, source.stat().st_ino)

    record = rename_no_replace(source, destination)

    assert {p.name for p in tmp_path.iterdir()} == {name}
    assert (destination.stat().st_dev, destination.stat().st_ino) == identity
    moved_content = destination / "data.txt" if directory else destination
    assert moved_content.read_text(encoding="utf-8") == "original"
    restore_inverse_rename(record)
    assert {p.name for p in tmp_path.iterdir()} == {"source"}
    assert content.read_text(encoding="utf-8") == "original"


def test_unicode_rename_does_not_collide_with_shortened_name(tmp_path: Path) -> None:
    source, destination, neighbor = tmp_path / "source", tmp_path / "a\U0001f600b", tmp_path / "a\U0001f600"
    source.write_text("original", encoding="utf-8")
    neighbor.write_text("neighbor", encoding="utf-8")

    rename_no_replace(source, destination)

    assert destination.read_text(encoding="utf-8") == "original"
    assert neighbor.read_text(encoding="utf-8") == "neighbor"
    assert not source.exists()
    source.write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError):
        rename_no_replace(source, destination)
    assert source.read_text(encoding="utf-8") == "second"
    assert destination.read_text(encoding="utf-8") == "original"


def test_capture_and_restore_preserve_unicode_paths(tmp_path: Path) -> None:
    source, quarantine = tmp_path / "original\U0001f600", tmp_path / "quarantine\U0001f600end"
    source.mkdir()
    (source / "data.txt").write_text("original", encoding="utf-8")

    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)

    assert {p.name for p in tmp_path.iterdir()} == {quarantine.name}
    assert (quarantine / "data.txt").read_text(encoding="utf-8") == "original"
    restore_captured_directory(record)
    assert {p.name for p in tmp_path.iterdir()} == {source.name}
    assert (source / "data.txt").read_text(encoding="utf-8") == "original"


@pytest.mark.platform_windows
@pytest.mark.skipif(os.name != "nt", reason="Windows permits lone UTF-16 surrogates in filenames")
def test_windows_rename_preserves_lone_surrogate(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "name\ud800end"
    source.write_text("original", encoding="utf-8")
    record = rename_no_replace(source, destination)
    assert {p.name for p in tmp_path.iterdir()} == {destination.name}
    restore_inverse_rename(record)
    assert source.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("invalid_argument", ["source", "destination"])
def test_native_rename_rejects_nul_without_mutation(tmp_path: Path, invalid_argument: str) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("original", encoding="utf-8")
    arguments = {"source": source, "destination": destination}
    arguments[invalid_argument] = Path(str(arguments[invalid_argument]) + "\0suffix")

    with pytest.raises(ValueError, match="NUL"):
        rename_path_no_replace(**arguments, operation="test rename")

    assert {p.name for p in tmp_path.iterdir()} == {source.name}
    assert source.read_text(encoding="utf-8") == "original"


@pytest.mark.platform_windows
@pytest.mark.skipif(os.name != "nt", reason="native Windows primitives")
@pytest.mark.parametrize("invalid_argument", ["source", "destination"])
@pytest.mark.parametrize("method", ["wrapper", "replace", "move"])
def test_windows_replace_rejects_nul_without_mutation(tmp_path: Path, invalid_argument: str, method: str) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_text("replacement", encoding="utf-8")
    destination.write_text("original", encoding="utf-8")
    arguments = {"source": source, "destination": destination}
    arguments[invalid_argument] = Path(str(arguments[invalid_argument]) + "\0suffix")
    api = WindowsFilePrimitiveApi.load()

    with pytest.raises(ValueError, match="NUL"):
        if method == "wrapper":
            replace_file_windows(**arguments, api=api)
        elif method == "replace":
            api.replace_existing_file(**arguments)
        else:
            api.move_file_replace(**arguments, write_through=True)

    assert source.read_text(encoding="utf-8") == "replacement"
    assert destination.read_text(encoding="utf-8") == "original"
    assert {p.name for p in tmp_path.iterdir()} == {source.name, destination.name}


@pytest.mark.platform_windows
@pytest.mark.skipif(os.name != "nt", reason="native Windows primitives")
def test_windows_attributes_reject_nul_path(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_text("original", encoding="utf-8")
    api = WindowsFilePrimitiveApi.load()
    with pytest.raises(ValueError, match="NUL"):
        api.get_file_attributes(Path(str(source) + "\0suffix"))
    assert source.read_text(encoding="utf-8") == "original"


@pytest.mark.platform_windows
@pytest.mark.skipif(os.name != "nt", reason="native Windows primitives")
def test_windows_identity_remove_rejects_nul_without_deleting_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    before = source.stat()
    with pytest.raises(ValueError, match="NUL"):
        remove_empty_directory_by_identity_windows(
            Path(str(source) + "\0suffix"),
            expected_identity=(before.st_dev, before.st_ino),
            durability=DurabilityMode.NONE,
        )
    assert source.is_dir()
    assert (source.stat().st_dev, source.stat().st_ino) == (before.st_dev, before.st_ino)
