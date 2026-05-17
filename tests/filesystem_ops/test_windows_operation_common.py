from __future__ import annotations

from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import PathSafety, UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops._windows_operation_common import safe_inspect

pytestmark = pytest.mark.safe_fs_ops


def test_safe_inspect_propagates_windows_reparse_checker_failures() -> None:
    target = Path("C:/workspace/config.txt")

    def inspect_existing_file(path: Path) -> PathSafety:
        assert path == target
        return PathSafety(
            path=path,
            exists=True,
            file_type="file",
            is_mount=False,
            is_windows_reparse_point=False,
            hardlink_count=1,
            size=10,
        )

    def fail_windows_reparse_check(path: Path) -> bool:
        assert path == target
        raise UnsupportedFilesystemMutationError("reparse probe failed")

    with pytest.raises(UnsupportedFilesystemMutationError, match="reparse probe failed"):
        safe_inspect(
            target,
            path_inspector=inspect_existing_file,
            windows_reparse_checker=fail_windows_reparse_check,
        )


def test_safe_inspect_preserves_existing_reparse_classification_without_runtime_probe() -> None:
    target = Path("C:/workspace/link")

    def inspect_existing_reparse(path: Path) -> PathSafety:
        assert path == target
        return PathSafety(
            path=path,
            exists=True,
            file_type="directory",
            is_mount=False,
            is_windows_reparse_point=True,
            hardlink_count=1,
            size=0,
        )

    def fail_if_called(_path: Path) -> bool:
        raise AssertionError("runtime reparse probe should not run when classification already marked reparse")

    safety = safe_inspect(
        target,
        path_inspector=inspect_existing_reparse,
        windows_reparse_checker=fail_if_called,
    )

    assert safety.is_windows_reparse_point is True
