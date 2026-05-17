from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    DeleteHooks,
    DurabilityMode,
    MakeDirectoryHooks,
    ParentChangedAfterMutationError,
    PathSafety,
    UnsafePathError,
)
from safe_fs_ops.filesystem_ops._windows_operations import (
    atomic_write_bytes_windows,
    atomic_write_text_windows,
    delete_file_windows,
    make_directory_windows,
)
from safe_fs_ops.filesystem_ops._windows_primitives import ReplaceFilePartialFailureError

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.platform_windows]


def test_atomic_write_text_windows_writes_with_same_directory_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.txt"
    replaced_paths: list[tuple[Path, Path]] = []

    def record_replace(source: Path, destination: Path, *, write_through: bool) -> None:
        replaced_paths.append((source, destination))
        assert write_through is True
        os.replace(source, destination)

    atomic_write_text_windows(target, "enabled = true\n", _replace_file=record_replace)

    assert target.read_text(encoding="utf-8") == "enabled = true\n"
    assert len(replaced_paths) == 1
    assert replaced_paths[0][0].parent == target.parent
    assert replaced_paths[0][0].name.startswith(f".{target.name}.")
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_bytes_windows_rejects_windows_reparse_target(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"

    def path_inspector(path: Path) -> PathSafety:
        safety = _inspect(path)
        if path == target:
            return _with_reparse(safety)
        return safety

    with pytest.raises(UnsafePathError, match="Windows reparse point"):
        atomic_write_bytes_windows(target, b"new", _path_inspector=path_inspector)

    assert target.exists() is False


def test_atomic_write_bytes_windows_reports_parent_replaced_after_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"

    def replace_parent_after_validation(source: Path, destination: Path, *, write_through: bool) -> None:
        assert destination == target
        assert write_through is True
        parent.rename(moved_parent)
        parent.mkdir()
        os.replace(moved_parent / source.name, destination)

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        atomic_write_bytes_windows(target, b"new", _replace_file=replace_parent_after_validation)

    assert target.read_bytes() == b"new"
    assert (moved_parent / target.name).exists() is False


def test_atomic_write_bytes_windows_preserves_temp_file_after_replacefile_partial_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"old")
    replace_calls: list[tuple[Path, Path, bool]] = []

    def fail_after_partial_replace(source: Path, destination: Path, *, write_through: bool) -> None:
        replace_calls.append((source, destination, write_through))
        raise ReplaceFilePartialFailureError(
            1176,
            "replace existing file failed: simulated partial failure",
            replacement_path=source,
            destination_path=destination,
        )

    with pytest.raises(ReplaceFilePartialFailureError):
        atomic_write_bytes_windows(
            target,
            b"new",
            _file_flush=lambda _descriptor: None,
            _replace_file=fail_after_partial_replace,
            _token_hex=lambda _nbytes: "abcd",
        )

    assert len(replace_calls) == 1
    temp_path, destination, write_through = replace_calls[0]
    assert destination == target
    assert write_through is True
    assert temp_path.exists()
    assert temp_path.read_bytes() == b"new"
    assert target.read_bytes() == b"old"


def test_atomic_write_bytes_windows_disables_write_through_replace_for_none_durability(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    replace_calls: list[bool] = []

    def record_replace(source: Path, destination: Path, *, write_through: bool) -> None:
        replace_calls.append(write_through)
        os.replace(source, destination)

    atomic_write_bytes_windows(
        target,
        b"new",
        durability=DurabilityMode.NONE,
        _replace_file=record_replace,
    )

    assert replace_calls == [False]
    assert target.read_bytes() == b"new"


def test_delete_file_windows_removes_regular_file_and_respects_missing_ok(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("delete me", encoding="utf-8")

    delete_file_windows(target)
    delete_file_windows(target, missing_ok=True)

    assert target.exists() is False


def test_delete_file_windows_rejects_symlink_paths(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        delete_file_windows(link)

    assert real.read_text(encoding="utf-8") == "real"


def test_delete_file_windows_reports_parent_replaced_after_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"
    target.write_text("delete me", encoding="utf-8")

    def replace_parent_after_validation(destination: Path) -> None:
        assert destination == target
        parent.rename(moved_parent)
        parent.mkdir()
        target.write_text("replacement", encoding="utf-8")

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        delete_file_windows(target, hooks=DeleteHooks(after_unlink_validation=replace_parent_after_validation))

    assert target.exists() is False
    assert (moved_parent / target.name).read_text(encoding="utf-8") == "delete me"


def test_make_directory_windows_creates_parents_and_flushes_each_created_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "settings" / "nested" / "state"
    flushes: list[Path] = []

    make_directory_windows(
        target,
        parents=True,
        durability=DurabilityMode.FSYNC,
        _directory_flush=flushes.append,
    )

    assert target.is_dir()
    assert flushes == [
        workspace,
        workspace / "settings",
        workspace / "settings" / "nested",
        target,
    ]


def test_make_directory_windows_rejects_windows_reparse_parent(tmp_path: Path) -> None:
    parent = tmp_path / "redirect"
    target = parent / "state"

    def path_inspector(path: Path) -> PathSafety:
        safety = _inspect(path)
        if path == parent:
            return _with_reparse(
                PathSafety(
                    path=path,
                    exists=True,
                    file_type="directory",
                    is_mount=False,
                    is_windows_reparse_point=False,
                    hardlink_count=1,
                    size=0,
                )
            )
        return safety

    with pytest.raises(UnsafePathError, match="Windows reparse point"):
        make_directory_windows(target, parents=True, _path_inspector=path_inspector)

    assert parent.exists() is False


def test_make_directory_windows_reports_parent_replaced_after_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "state"

    def move_parent_after_validation(directory: Path) -> None:
        assert directory == target
        parent.rename(moved_parent)
        parent.mkdir()

    with pytest.raises(UnsafePathError, match="parent path redirects elsewhere"):
        make_directory_windows(
            target,
            hooks=MakeDirectoryHooks(before_descriptor_mkdir=move_parent_after_validation),
        )

    assert target.exists() is False
    assert moved_parent.is_dir()
    assert (moved_parent / target.name).exists() is False


def _inspect(path: Path) -> PathSafety:
    if path.exists():
        if path.is_symlink():
            file_type = "symlink"
        elif path.is_file():
            file_type = "file"
        elif path.is_dir():
            file_type = "directory"
        else:
            file_type = "other"
        stat_result = path.lstat()
        return PathSafety(
            path=path,
            exists=True,
            file_type=file_type,
            is_mount=False,
            is_windows_reparse_point=False,
            hardlink_count=stat_result.st_nlink,
            size=stat_result.st_size,
        )
    return PathSafety(
        path=path,
        exists=False,
        file_type="missing",
        is_mount=False,
        is_windows_reparse_point=False,
        hardlink_count=0,
        size=None,
    )


def _with_reparse(safety: PathSafety) -> PathSafety:
    return PathSafety(
        path=safety.path,
        exists=safety.exists,
        file_type=safety.file_type,
        is_mount=safety.is_mount,
        is_windows_reparse_point=True,
        hardlink_count=safety.hardlink_count,
        size=safety.size,
    )
