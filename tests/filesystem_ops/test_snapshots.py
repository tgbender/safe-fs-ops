from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError, snapshot_resource
from safe_fs_ops.filesystem_ops.snapshots import (
    SnapshotHooks,
    UnsupportedFilesystemSnapshotError,
    _SnapshotPlatformSupport,
)

pytestmark = pytest.mark.safe_fs_ops


def _descriptor_relative_snapshots_supported() -> bool:
    return (
        os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and os.readlink in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


requires_descriptor_relative_snapshot = pytest.mark.skipif(
    not _descriptor_relative_snapshots_supported(),
    reason="descriptor-relative no-follow snapshots are unavailable on this platform",
)


def _unsupported_platform_support() -> _SnapshotPlatformSupport:
    return _SnapshotPlatformSupport(
        open_dir_fd=False,
        stat_dir_fd=False,
        stat_follow_symlinks=False,
        readlink_dir_fd=False,
        nofollow_directory_open=False,
        nofollow_file_open=False,
    )


def _replace_parent_with_empty_directory(parent: Path, moved_parent: Path) -> None:
    parent.rename(moved_parent)
    parent.mkdir()


@requires_descriptor_relative_snapshot
def test_snapshot_missing_resource(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    snapshot = snapshot_resource(target)

    assert snapshot.path == target
    assert snapshot.exists is False
    assert snapshot.file_type == "missing"
    assert snapshot.content_hash is None
    assert snapshot.size is None
    assert snapshot.mtime_ns is None
    assert snapshot.symlink_target is None


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_missing_resource_when_parent_replaced_before_completion(tmp_path: Path) -> None:
    parent = tmp_path / "settings"
    parent.mkdir()
    moved_parent = tmp_path / "moved-settings"
    target = parent / "missing.txt"

    def replace_parent_after_missing_classification(path: Path) -> None:
        assert path == target
        _replace_parent_with_empty_directory(parent, moved_parent)

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        snapshot_resource(target, hooks=SnapshotHooks(before_complete=replace_parent_after_missing_classification))

    assert moved_parent.is_dir()
    assert parent.is_dir()


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_missing_resource_that_appears_before_completion(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    def create_target_before_snapshot_returns(path: Path) -> None:
        assert path == target
        target.write_text("created\n", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="classified missing path appeared"):
        snapshot_resource(target, hooks=SnapshotHooks(before_complete=create_target_before_snapshot_returns))

    assert target.read_text(encoding="utf-8") == "created\n"


@requires_descriptor_relative_snapshot
def test_snapshot_regular_file_hash_size_and_mtime(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    content = b"alpha\nbeta\n"
    target.write_bytes(content)

    snapshot = snapshot_resource(target)

    assert snapshot.exists is True
    assert snapshot.file_type == "file"
    assert snapshot.content_hash == hashlib.sha256(content).hexdigest()
    assert snapshot.size == len(content)
    assert snapshot.mtime_ns == target.lstat().st_mtime_ns
    assert snapshot.symlink_target is None


@requires_descriptor_relative_snapshot
@pytest.mark.parametrize("child_kind", ["file", "directory", "symlink", "other"])
def test_snapshot_rejects_existing_resource_when_parent_replaced_before_completion(
    tmp_path: Path,
    child_kind: str,
) -> None:
    parent = tmp_path / "settings"
    parent.mkdir()
    moved_parent = tmp_path / "moved-settings"
    target = parent / "child"

    if child_kind == "file":
        target.write_text("content", encoding="utf-8")
    elif child_kind == "directory":
        target.mkdir()
    elif child_kind == "symlink":
        real = tmp_path / "real.txt"
        real.write_text("real", encoding="utf-8")
        try:
            target.symlink_to(real)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation unavailable")
        os.mkfifo(target)

    def replace_parent_before_snapshot_returns(path: Path) -> None:
        assert path == target
        _replace_parent_with_empty_directory(parent, moved_parent)

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        snapshot_resource(target, hooks=SnapshotHooks(before_complete=replace_parent_before_snapshot_returns))

    assert moved_parent.is_dir()
    assert parent.is_dir()


def test_snapshot_regular_file_fails_conservatively_without_descriptor_support(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"alpha\n")

    with pytest.raises(UnsupportedFilesystemSnapshotError, match="descriptor-relative no-follow"):
        snapshot_resource(target, _platform_support=_unsupported_platform_support())


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_regular_file_replaced_between_lstat_and_open(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"claimed\n")
    replacement_content = b"replacement\n"

    def replace_target_after_classification(path: Path) -> None:
        assert path == target
        target.unlink()
        target.write_bytes(replacement_content)

    with pytest.raises(UnsafePathError, match="classified file changed"):
        snapshot_resource(target, hooks=SnapshotHooks(before_open_file=replace_target_after_classification))

    assert target.read_bytes() == replacement_content


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_regular_file_same_inode_mutation_between_lstat_and_hash(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"old\n")
    replacement_content = b"new content with different size\n"

    def overwrite_target_after_classification(path: Path) -> None:
        assert path == target
        path.write_bytes(replacement_content)

    with pytest.raises(UnsafePathError, match="classified file changed before snapshot completed"):
        snapshot_resource(target, hooks=SnapshotHooks(before_open_file=overwrite_target_after_classification))

    assert target.read_bytes() == replacement_content


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_regular_file_same_size_mutation_with_restored_mtime(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    original_content = b"alpha\n"
    replacement_content = b"bravo\n"
    target.write_bytes(original_content)
    original_stat = target.lstat()

    def overwrite_target_and_restore_mtime(path: Path) -> None:
        assert path == target
        path.write_bytes(replacement_content)
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        if path.lstat().st_ctime_ns == original_stat.st_ctime_ns:
            pytest.skip("filesystem did not expose a ctime change for the overwrite")

    with pytest.raises(UnsafePathError, match="classified file changed before snapshot completed"):
        snapshot_resource(target, hooks=SnapshotHooks(before_open_file=overwrite_target_and_restore_mtime))

    assert target.read_bytes() == replacement_content
    assert target.lstat().st_size == original_stat.st_size
    assert target.lstat().st_mtime_ns == original_stat.st_mtime_ns


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_regular_file_replaced_before_completion(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"old\n")
    replacement_content = b"replacement\n"

    def replace_target_before_snapshot_returns(path: Path) -> None:
        assert path == target
        target.unlink()
        target.write_bytes(replacement_content)

    with pytest.raises(UnsafePathError, match="classified path changed before snapshot completed"):
        snapshot_resource(target, hooks=SnapshotHooks(before_complete=replace_target_before_snapshot_returns))

    assert target.read_bytes() == replacement_content


@requires_descriptor_relative_snapshot
def test_snapshot_does_not_hash_symlink_swapped_after_lstat(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"claimed\n")
    outside = tmp_path / "outside.txt"
    outside_content = b"outside\n"
    outside.write_bytes(outside_content)

    def swap_target_to_symlink(path: Path) -> None:
        assert path == target
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="redirects elsewhere"):
        snapshot_resource(target, hooks=SnapshotHooks(before_open_file=swap_target_to_symlink))

    assert outside.read_bytes() == outside_content


@requires_descriptor_relative_snapshot
def test_snapshot_directory_does_not_hash_contents(tmp_path: Path) -> None:
    directory = tmp_path / "settings"
    directory.mkdir()
    (directory / "child.txt").write_text("child", encoding="utf-8")

    snapshot = snapshot_resource(directory)

    assert snapshot.exists is True
    assert snapshot.file_type == "directory"
    assert snapshot.content_hash is None
    assert snapshot.size == directory.lstat().st_size
    assert snapshot.mtime_ns == directory.lstat().st_mtime_ns
    assert snapshot.symlink_target is None


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_directory_replaced_after_lstat(tmp_path: Path) -> None:
    directory = tmp_path / "settings"
    directory.mkdir()

    def replace_directory_after_lstat(path: Path) -> None:
        assert path == directory
        directory.rmdir()
        directory.write_text("not a directory", encoding="utf-8")

    with pytest.raises(UnsafePathError, match="classified path changed"):
        snapshot_resource(directory, hooks=SnapshotHooks(after_lstat=replace_directory_after_lstat))

    assert directory.read_text(encoding="utf-8") == "not a directory"


@requires_descriptor_relative_snapshot
def test_snapshot_symlink_records_link_not_target(tmp_path: Path) -> None:
    real = tmp_path / "real.txt"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(real)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    snapshot = snapshot_resource(link)

    assert snapshot.exists is True
    assert snapshot.file_type == "symlink"
    assert snapshot.content_hash is None
    assert snapshot.size == link.lstat().st_size
    assert snapshot.mtime_ns == link.lstat().st_mtime_ns
    assert snapshot.symlink_target == os.readlink(link)


@requires_descriptor_relative_snapshot
def test_snapshot_rejects_symlink_swapped_after_lstat_before_readlink(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    first.write_text("first", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("second", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(first)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    def swap_symlink_after_lstat(path: Path) -> None:
        assert path == link
        link.unlink()
        try:
            link.symlink_to(second)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="classified path changed"):
        snapshot_resource(link, hooks=SnapshotHooks(after_lstat=swap_symlink_after_lstat))

    assert Path(os.readlink(link)).name == second.name


@requires_descriptor_relative_snapshot
@pytest.mark.parametrize("child_kind", ["directory", "symlink", "missing"])
def test_snapshot_refuses_non_file_paths_beneath_symlink_parent(tmp_path: Path, child_kind: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link_parent = tmp_path / "link-parent"
    try:
        link_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    if child_kind == "directory":
        (outside / "child").mkdir()
    elif child_kind == "symlink":
        real = outside / "real.txt"
        real.write_text("real", encoding="utf-8")
        try:
            (outside / "child").symlink_to(real)
        except OSError as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        snapshot_resource(link_parent / "child")
