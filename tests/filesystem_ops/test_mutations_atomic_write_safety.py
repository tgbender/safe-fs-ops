from __future__ import annotations

import os
from pathlib import Path

import pytest
from mutation_helpers import _unsupported_platform_support, requires_descriptor_relative_replace

from safe_fs_ops.filesystem_ops import (
    AtomicWriteHooks,
    ParentChangedAfterMutationError,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    atomic_write_bytes,
)

pytestmark = pytest.mark.safe_fs_ops


def test_atomic_write_refuses_symlink_final_path(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        atomic_write_bytes(link, b"new")

    assert real.read_text(encoding="utf-8") == "real"


def test_atomic_write_rejects_custom_replace_before_parent_side_effects(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "nested" / "config.txt"

    def replace_callback(_source: Path, _destination: Path) -> None:
        raise AssertionError("replace callback must not run")

    with pytest.raises(UnsupportedFilesystemMutationError, match="custom replace"):
        atomic_write_bytes(
            target,
            b"new",
            replace=replace_callback,
            _platform_support=_unsupported_platform_support(),
        )

    assert (tmp_path / "missing").exists() is False


def test_atomic_write_refuses_symlink_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(link_dir / "config.txt", b"new")

    assert list(real_dir.iterdir()) == []


@requires_descriptor_relative_replace
def test_atomic_write_revalidates_missing_parent_after_pre_mkdir_hook(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = workspace / "settings"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = parent / "config.txt"

    def swap_missing_parent_to_symlink(parent_path: Path) -> None:
        assert parent_path == parent
        try:
            parent.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(
            target,
            b"new",
            hooks=AtomicWriteHooks(before_parent_mkdir=swap_missing_parent_to_symlink),
        )

    assert (outside / target.name).exists() is False
    assert list(outside.iterdir()) == []


@requires_descriptor_relative_replace
def test_atomic_write_relative_path_uses_entry_cwd_after_hook_chdir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    (other_cwd / "config.txt").write_text("other", encoding="utf-8")
    target = workspace / "config.txt"
    original_cwd = Path.cwd()

    def chdir_before_parent_open(parent_path: Path) -> None:
        assert parent_path == workspace
        os.chdir(other_cwd)

    try:
        os.chdir(workspace)
        atomic_write_bytes(
            "config.txt",
            b"new",
            hooks=AtomicWriteHooks(before_open_parent=chdir_before_parent_open),
        )
    finally:
        os.chdir(original_cwd)

    assert target.read_bytes() == b"new"
    assert (other_cwd / "config.txt").read_text(encoding="utf-8") == "other"


@requires_descriptor_relative_replace
def test_atomic_write_does_not_follow_swapped_missing_ancestor_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    missing_ancestor = workspace / "settings"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = missing_ancestor / "nested" / "config.txt"

    def swap_missing_ancestor_to_symlink(parent_path: Path) -> None:
        assert parent_path == target.parent
        try:
            missing_ancestor.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(
            target,
            b"new",
            hooks=AtomicWriteHooks(before_parent_mkdir=swap_missing_ancestor_to_symlink),
        )

    assert (outside / "nested").exists() is False
    assert list(outside.iterdir()) == []


@requires_descriptor_relative_replace
def test_atomic_write_does_not_follow_swapped_existing_ancestor_before_parent_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ancestor = workspace / "settings"
    parent = ancestor / "nested"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside_parent = outside / "nested"
    outside_parent.mkdir(parents=True)
    moved_ancestor = tmp_path / "moved-settings"
    target = parent / "config.txt"

    def swap_existing_ancestor_to_symlink(parent_path: Path) -> None:
        assert parent_path == parent
        ancestor.rename(moved_ancestor)
        try:
            ancestor.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(
            target,
            b"new",
            hooks=AtomicWriteHooks(before_open_parent=swap_existing_ancestor_to_symlink),
        )

    assert (outside_parent / target.name).exists() is False
    assert list(outside_parent.iterdir()) == []
    assert list((moved_ancestor / "nested").iterdir()) == []


@requires_descriptor_relative_replace
def test_atomic_write_revalidates_parent_immediately_before_temp_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = parent / "config.txt"

    def swap_parent_to_symlink(_target: Path) -> None:
        parent.rmdir()
        try:
            parent.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(target, b"new", hooks=AtomicWriteHooks(before_temp_file=swap_parent_to_symlink))

    assert (outside / target.name).exists() is False
    assert list(outside.glob(f".{target.name}.*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_revalidates_final_path_immediately_before_replace(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("old", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    def swap_target_to_symlink(_source: Path, destination: Path) -> None:
        destination.unlink()
        try:
            destination.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        atomic_write_bytes(target, b"new", hooks=AtomicWriteHooks(before_replace=swap_target_to_symlink))

    assert outside.read_text(encoding="utf-8") == "outside"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_revalidates_parent_immediately_before_replace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"

    def swap_parent_to_symlink(_source: Path, _destination: Path) -> None:
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory swap unavailable while temp file is open: {exc}")
        try:
            parent.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        atomic_write_bytes(target, b"new", hooks=AtomicWriteHooks(before_replace=swap_parent_to_symlink))

    assert (outside / target.name).exists() is False
    assert list(outside.glob(f".{target.name}.*.tmp")) == []
    assert list(moved_parent.glob(f".{target.name}.*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_reports_parent_replaced_after_final_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"
    target.write_text("old", encoding="utf-8")

    def replace_parent_after_validation(source: Path, destination: Path) -> None:
        assert source.parent == parent
        assert destination == target
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory swap unavailable before descriptor replace: {exc}")
        parent.mkdir()

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        atomic_write_bytes(
            target,
            b"new",
            hooks=AtomicWriteHooks(after_replace_validation=replace_parent_after_validation),
        )

    assert (parent / target.name).exists() is False
    assert (moved_parent / target.name).read_bytes() == b"new"
    assert list(moved_parent.glob(f".{target.name}.*.tmp")) == []
