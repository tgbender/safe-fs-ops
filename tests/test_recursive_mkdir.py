from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops.models import PathSafety
from safe_fs_ops.filesystem_ops.paths import UnsafePathError
from safe_fs_ops.operation_journal.filesystem_support import directory_resource_key
from safe_fs_ops.recursive_mkdir import plan_directory_creation

pytestmark = pytest.mark.safe_fs_ops


def test_plan_directory_creation_single_leaf_step_when_parent_exists(tmp_path: Path) -> None:
    parent = tmp_path / "workspace"
    parent.mkdir()
    target = parent / "state"

    plan = plan_directory_creation(target)

    assert plan.target == target
    assert plan.target_existed_before is False
    assert [step.path for step in plan.steps] == [target]
    assert [step.resource_key for step in plan.steps] == [directory_resource_key(target)]
    assert all(step.existed_before is False for step in plan.steps)
    assert all(step.cleanup_eligible is False for step in plan.steps)
    assert target.exists() is False


def test_plan_directory_creation_orders_missing_parents_shallow_to_deep(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "settings" / "nested" / "state"

    plan = plan_directory_creation(target, parents=True)

    assert [step.path for step in plan.steps] == [
        workspace / "settings",
        workspace / "settings" / "nested",
        target,
    ]
    assert all(step.cleanup_eligible is False for step in plan.steps), (
        "planned steps become cleanup-eligible only after execution creates directories"
    )
    assert target.exists() is False
    assert (workspace / "settings").exists() is False


def test_plan_directory_creation_without_parents_rejects_missing_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "settings" / "state"

    with pytest.raises(FileNotFoundError) as exc_info:
        plan_directory_creation(target, parents=False)

    assert exc_info.value.args == (target.parent,)
    assert target.exists() is False
    assert (workspace / "settings").exists() is False


def test_plan_directory_creation_existing_target_dir_with_exist_ok_is_noop(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()

    plan = plan_directory_creation(target, parents=True, exist_ok=True)

    assert plan.target == target
    assert plan.target_existed_before is True
    assert plan.steps == ()


def test_plan_directory_creation_existing_target_dir_without_exist_ok_rejects(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()

    with pytest.raises(FileExistsError) as exc_info:
        plan_directory_creation(target)

    assert exc_info.value.args == (target,)


def test_plan_directory_creation_existing_file_target_rejects(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.write_text("not a directory", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="target is not a directory"):
        plan_directory_creation(target, parents=True, exist_ok=True)


def test_plan_directory_creation_rejects_symlink_in_chain_without_mutating(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    target = link_parent / "state"

    with pytest.raises(UnsafePathError, match="symlink"):
        plan_directory_creation(target, parents=True)

    assert target.exists() is False
    assert list(real_parent.iterdir()) == []


def test_plan_directory_creation_rejects_mount_point_in_chain_without_mutating(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mount_parent = workspace / "mount-parent"
    target = mount_parent / "state"

    def inspect_with_mount(path: Path) -> PathSafety:
        checked_path = Path(path)
        exists = checked_path in (workspace, mount_parent)
        return PathSafety(
            path=checked_path,
            exists=exists,
            file_type="directory" if exists else "missing",
            is_mount=checked_path == mount_parent,
            is_windows_reparse_point=False,
            hardlink_count=0,
            size=None,
        )

    with pytest.raises(UnsafePathError, match="mount point"):
        plan_directory_creation(target, parents=True, path_inspector=inspect_with_mount)

    assert target.exists() is False


def test_plan_directory_creation_canonicalizes_relative_dotdot_without_creating_directories(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    base = workspace / "base"
    parent = workspace / "parent"
    base.mkdir(parents=True)
    parent.mkdir()
    original_cwd = Path.cwd()

    try:
        os.chdir(base)
        plan = plan_directory_creation(Path("..") / "parent" / "state")
    finally:
        os.chdir(original_cwd)

    assert plan.target == parent / "state"
    assert [step.path for step in plan.steps] == [parent / "state"]
    assert (parent / "state").exists() is False


def test_plan_directory_creation_rejects_boundary_escape_after_canonicalization(tmp_path: Path) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    sibling = tmp_path / "sibling"
    nested.mkdir(parents=True)
    sibling.mkdir()
    original_cwd = Path.cwd()

    try:
        os.chdir(nested)
        with pytest.raises(UnsafePathError, match="escapes root boundary"):
            plan_directory_creation(Path("..") / ".." / "sibling" / "state", parents=True, root_boundary=root)
    finally:
        os.chdir(original_cwd)

    assert (sibling / "state").exists() is False
