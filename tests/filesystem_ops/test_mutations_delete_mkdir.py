from __future__ import annotations

import os
from pathlib import Path

import pytest
from mutation_helpers import (
    _CallRecorder,
    _descriptor_relative_mkdir_supported,
    _descriptor_relative_mutations_supported,
    _DirectoryFsyncRecorder,
    _FailingFsync,
    _path_identity,
    _unsupported_platform_support,
    requires_descriptor_relative_mkdir,
    requires_descriptor_relative_replace,
    requires_descriptor_relative_rmdir,
    requires_descriptor_relative_unlink,
)

from safe_fs_ops.filesystem_ops import (
    DeleteHooks,
    DurabilityMode,
    MakeDirectoryHooks,
    ParentChangedAfterMutationError,
    RemoveDirectoryHooks,
    UnsafePathError,
    atomic_write_bytes,
    atomic_write_text,
    delete_file,
    make_directory,
    remove_empty_directory,
)

pytestmark = pytest.mark.safe_fs_ops


@requires_descriptor_relative_unlink
def test_delete_file_removes_regular_file_and_ignores_missing_when_requested(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("delete me", encoding="utf-8")

    delete_file(target)
    delete_file(target, missing_ok=True)

    assert not target.exists()


@requires_descriptor_relative_unlink
def test_delete_file_none_skips_directory_fsync(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("delete me", encoding="utf-8")
    fsyncs = _CallRecorder()

    delete_file(target, durability=DurabilityMode.NONE, _directory_fsync=fsyncs)

    assert not target.exists()
    assert fsyncs.calls == 0


@requires_descriptor_relative_unlink
def test_delete_file_best_effort_swallows_directory_fsync_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("delete me", encoding="utf-8")
    directory_fsync = _FailingFsync("directory fsync failed")

    delete_file(
        target,
        durability=DurabilityMode.BEST_EFFORT,
        _directory_fsync=directory_fsync,
    )

    assert not target.exists()
    assert directory_fsync.calls == 1


def test_delete_file_refuses_directory_and_symlink(tmp_path: Path) -> None:
    directory = tmp_path / "directory"
    directory.mkdir()

    with pytest.raises(UnsafePathError, match="non-regular file"):
        delete_file(directory)

    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        delete_file(link)

    assert real.read_text(encoding="utf-8") == "real"


@requires_descriptor_relative_unlink
def test_delete_file_does_not_follow_swapped_parent_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"
    target.write_text("delete me", encoding="utf-8")
    outside_target = outside / target.name
    outside_target.write_text("outside", encoding="utf-8")

    def swap_parent_to_symlink(_target: Path) -> None:
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory swap unavailable while parent is open: {exc}")
        try:
            parent.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        delete_file(target, hooks=DeleteHooks(before_unlink=swap_parent_to_symlink))

    assert outside_target.read_text(encoding="utf-8") == "outside"
    assert (moved_parent / target.name).read_text(encoding="utf-8") == "delete me"


@requires_descriptor_relative_unlink
def test_delete_file_reports_parent_replaced_after_final_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "config.txt"
    target.write_text("delete me", encoding="utf-8")

    def replace_parent_after_validation(destination: Path) -> None:
        assert destination == target
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory swap unavailable before descriptor unlink: {exc}")
        parent.mkdir()
        target.write_text("lexical replacement", encoding="utf-8")

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        delete_file(target, hooks=DeleteHooks(after_unlink_validation=replace_parent_after_validation))

    assert target.read_text(encoding="utf-8") == "lexical replacement"
    assert (moved_parent / target.name).exists() is False


@requires_descriptor_relative_unlink
def test_delete_file_does_not_follow_swapped_existing_ancestor_before_parent_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ancestor = workspace / "settings"
    parent = ancestor / "nested"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside_parent = outside / "nested"
    outside_parent.mkdir(parents=True)
    moved_ancestor = tmp_path / "moved-settings"
    target = parent / "config.txt"
    target.write_text("delete me", encoding="utf-8")
    outside_target = outside_parent / target.name
    outside_target.write_text("outside", encoding="utf-8")

    def swap_existing_ancestor_to_symlink(parent_path: Path) -> None:
        assert parent_path == parent
        ancestor.rename(moved_ancestor)
        try:
            ancestor.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        delete_file(target, hooks=DeleteHooks(before_open_parent=swap_existing_ancestor_to_symlink))

    assert outside_target.read_text(encoding="utf-8") == "outside"
    assert (moved_ancestor / "nested" / target.name).read_text(encoding="utf-8") == "delete me"


@requires_descriptor_relative_unlink
def test_delete_file_relative_path_uses_entry_cwd_after_hook_chdir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    target = workspace / "config.txt"
    target.write_text("delete me", encoding="utf-8")
    other_target = other_cwd / "config.txt"
    other_target.write_text("other", encoding="utf-8")
    original_cwd = Path.cwd()

    def chdir_before_parent_open(parent_path: Path) -> None:
        assert parent_path == workspace
        os.chdir(other_cwd)

    try:
        os.chdir(workspace)
        delete_file("config.txt", hooks=DeleteHooks(before_open_parent=chdir_before_parent_open))
    finally:
        os.chdir(original_cwd)

    assert target.exists() is False
    assert other_target.read_text(encoding="utf-8") == "other"


def test_mutations_fail_conservatively_without_descriptor_relative_support(tmp_path: Path) -> None:
    replace_supported = _descriptor_relative_mutations_supported(needs_replace=True)
    unlink_supported = _descriptor_relative_mutations_supported(needs_replace=False)
    if replace_supported and unlink_supported:
        pytest.skip("descriptor-relative mutations are available on this platform")

    target = tmp_path / "config.txt"
    target.write_text("old", encoding="utf-8")

    if not replace_supported:
        with pytest.raises(UnsafePathError, match="descriptor-relative"):
            atomic_write_bytes(target, b"new")
    if not unlink_supported:
        with pytest.raises(UnsafePathError, match="descriptor-relative"):
            delete_file(target)

    assert target.read_text(encoding="utf-8") == "old"


def test_recursive_parent_creation_fails_before_side_effects_without_descriptor_support(tmp_path: Path) -> None:
    if _descriptor_relative_mkdir_supported():
        pytest.skip("descriptor-relative mkdir is available on this platform")

    target = tmp_path / "missing" / "nested" / "config.txt"

    with pytest.raises(UnsafePathError, match="descriptor-relative"):
        atomic_write_bytes(target, b"new")
    with pytest.raises(UnsafePathError, match="descriptor-relative"):
        make_directory(target.parent, parents=True)

    assert (tmp_path / "missing").exists() is False


@requires_descriptor_relative_mkdir
def test_make_directory_guard_behavior(tmp_path: Path) -> None:
    target = tmp_path / "state"

    make_directory(target)
    make_directory(target, exist_ok=True)

    assert target.is_dir()
    with pytest.raises(FileExistsError):
        make_directory(target)

    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(UnsafePathError, match="not a directory"):
        make_directory(file_path, exist_ok=True)


@requires_descriptor_relative_rmdir
def test_remove_empty_directory_removes_only_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()

    remove_empty_directory(target)
    remove_empty_directory(target, missing_ok=True)

    assert target.exists() is False


@requires_descriptor_relative_rmdir
def test_remove_empty_directory_missing_ok_returns_after_observed_missing_without_removing_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()

    def remove_before_parent_open(parent_path: Path) -> None:
        assert parent_path == tmp_path
        target.rmdir()

    def recreate_after_missing_observed(directory: Path) -> None:
        assert directory == target
        directory.mkdir()

    remove_empty_directory(
        target,
        missing_ok=True,
        hooks=RemoveDirectoryHooks(
            before_open_parent=remove_before_parent_open,
            missing_ok_missing_observed=recreate_after_missing_observed,
        ),
    )

    assert target.is_dir()


@requires_descriptor_relative_rmdir
def test_remove_empty_directory_refuses_non_empty_directory_and_symlink(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()
    (target / "nested.txt").write_text("data", encoding="utf-8")

    with pytest.raises(OSError):
        remove_empty_directory(target)

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="symlink"):
        remove_empty_directory(link_dir)


@requires_descriptor_relative_rmdir
def test_remove_empty_directory_reports_parent_replaced_after_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "state"
    target.mkdir()

    def replace_parent_after_validation(directory: Path) -> None:
        assert directory == target
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory swap unavailable before descriptor rmdir: {exc}")
        parent.mkdir()
        target.mkdir()

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        remove_empty_directory(
            target, hooks=RemoveDirectoryHooks(after_rmdir_validation=replace_parent_after_validation)
        )

    assert target.is_dir()
    assert (moved_parent / target.name).exists() is False


@requires_descriptor_relative_mkdir
def test_make_directory_none_skips_fsync_hooks(tmp_path: Path) -> None:
    target = tmp_path / "state"
    fsyncs = _CallRecorder()

    make_directory(target, durability=DurabilityMode.NONE, _directory_fsync=fsyncs)

    assert target.is_dir()
    assert fsyncs.calls == 0


@requires_descriptor_relative_mkdir
def test_make_directory_best_effort_swallows_fsync_failures(tmp_path: Path) -> None:
    target = tmp_path / "state"
    fsyncs = _FailingFsync("directory fsync failed")

    make_directory(target, durability=DurabilityMode.BEST_EFFORT, _directory_fsync=fsyncs)

    assert target.is_dir()
    assert fsyncs.calls == 1


@requires_descriptor_relative_mkdir
def test_make_directory_parents_fsyncs_each_created_parent_before_descending(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "settings" / "nested" / "state"
    fsyncs = _DirectoryFsyncRecorder()

    def assert_previous_parent_fsynced(directory: Path) -> None:
        if directory == workspace / "settings" / "nested":
            assert _path_identity(workspace) in fsyncs.identities
        if directory == target:
            assert _path_identity(workspace / "settings") in fsyncs.identities

    make_directory(
        target,
        parents=True,
        hooks=MakeDirectoryHooks(before_descriptor_mkdir=assert_previous_parent_fsynced),
        _directory_fsync=fsyncs,
    )

    assert target.is_dir()
    assert fsyncs.identities[:4] == [
        _path_identity(workspace),
        _path_identity(workspace / "settings"),
        _path_identity(workspace / "settings" / "nested"),
        _path_identity(target),
    ]


@requires_descriptor_relative_mkdir
def test_make_directory_relative_path_uses_entry_cwd_after_hook_chdir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    original_cwd = Path.cwd()

    def chdir_before_mkdir(directory: Path) -> None:
        assert directory == workspace / "state"
        os.chdir(other_cwd)

    try:
        os.chdir(workspace)
        make_directory("state", hooks=MakeDirectoryHooks(before_mkdir=chdir_before_mkdir))
    finally:
        os.chdir(original_cwd)

    assert (workspace / "state").is_dir()
    assert (other_cwd / "state").exists() is False


@requires_descriptor_relative_mkdir
def test_make_directory_reports_parent_moved_after_descriptor_validation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "state"

    def move_parent_after_descriptor_validation(directory: Path) -> None:
        assert directory == target
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory move unavailable while parent is open: {exc}")
        parent.mkdir()

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        make_directory(
            target,
            hooks=MakeDirectoryHooks(before_descriptor_mkdir=move_parent_after_descriptor_validation),
        )

    assert target.exists() is False
    assert (moved_parent / target.name).is_dir()


@requires_descriptor_relative_mkdir
def test_make_directory_parents_reports_moved_incremental_parent_after_descriptor_validation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    parent = workspace / "settings"
    parent.mkdir(parents=True)
    moved_parent = tmp_path / "moved-settings"
    target = parent / "nested" / "state"

    def move_parent_after_descriptor_validation(directory: Path) -> None:
        if directory != parent / "nested":
            return
        try:
            parent.rename(moved_parent)
        except OSError as exc:
            pytest.skip(f"directory move unavailable while parent is open: {exc}")
        parent.mkdir()

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        make_directory(
            target,
            parents=True,
            hooks=MakeDirectoryHooks(before_descriptor_mkdir=move_parent_after_descriptor_validation),
        )

    assert target.exists() is False
    assert (moved_parent / "nested").is_dir()
    assert (moved_parent / "nested" / "state").exists() is False


def test_make_directory_without_parents_fails_before_side_effects_when_descriptor_mkdir_missing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"

    with pytest.raises(UnsafePathError, match="descriptor-relative directory creation"):
        make_directory(target, _platform_support=_unsupported_platform_support())

    assert target.exists() is False


def test_make_directory_refuses_symlink_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        make_directory(link_dir / "state", parents=True)

    assert list(real_dir.iterdir()) == []


@requires_descriptor_relative_mkdir
def test_make_directory_does_not_follow_swapped_missing_ancestor_symlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    missing_ancestor = workspace / "state"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = missing_ancestor / "nested" / "cache"

    def swap_missing_ancestor_to_symlink(directory: Path) -> None:
        assert directory == target
        try:
            missing_ancestor.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        make_directory(
            target,
            parents=True,
            hooks=MakeDirectoryHooks(before_mkdir=swap_missing_ancestor_to_symlink),
        )

    assert (outside / "nested").exists() is False
    assert list(outside.iterdir()) == []


@requires_descriptor_relative_replace
def test_mutations_stay_inside_explicit_temp_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / ".config" / "tool" / "settings.toml"

    atomic_write_text(target, "enabled = true\n", newline="")
    make_directory(workspace / ".cache")
    delete_file(target)

    assert workspace.exists()
    assert not target.exists()
    assert (workspace / ".cache").is_dir()
