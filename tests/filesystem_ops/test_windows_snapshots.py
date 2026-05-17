from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError
from safe_fs_ops.filesystem_ops._windows_snapshots import snapshot_resource_windows
from safe_fs_ops.filesystem_ops.models import PathSafety

pytestmark = pytest.mark.safe_fs_ops


def test_snapshot_resource_windows_reports_missing_resource(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    snapshot = snapshot_resource_windows(target, _windows_reparse_checker=lambda _path: False)

    assert snapshot.path == target
    assert snapshot.exists is False
    assert snapshot.file_type == "missing"
    assert snapshot.content_hash is None


def test_snapshot_resource_windows_hashes_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    content = b"alpha\nbeta\n"
    target.write_bytes(content)

    snapshot = snapshot_resource_windows(target, _windows_reparse_checker=lambda _path: False)

    assert snapshot.exists is True
    assert snapshot.file_type == "file"
    assert snapshot.content_hash == hashlib.sha256(content).hexdigest()
    assert snapshot.size == len(content)


def test_snapshot_resource_windows_rejects_runtime_reparse_target(tmp_path: Path) -> None:
    target = tmp_path / "redirect"
    target.mkdir()

    def mark_runtime_reparse(path: Path) -> bool:
        return path == target

    with pytest.raises(UnsafePathError, match="redirects elsewhere"):
        snapshot_resource_windows(target, _windows_reparse_checker=mark_runtime_reparse)


def test_snapshot_resource_windows_rejects_reparse_parent_chain(tmp_path: Path) -> None:
    parent = tmp_path / "redirect"
    target = parent / "child.txt"
    parent.mkdir()

    def inspect_path(path: Path) -> PathSafety:
        if path == parent:
            return PathSafety(
                path=path,
                exists=True,
                file_type="directory",
                is_mount=False,
                is_windows_reparse_point=True,
                hardlink_count=1,
                size=0,
            )
        if path.exists():
            file_type = "directory" if path.is_dir() else "file"
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

    with pytest.raises(UnsafePathError, match="redirects elsewhere"):
        snapshot_resource_windows(
            target,
            _path_inspector=inspect_path,
            _windows_reparse_checker=lambda _path: False,
        )
