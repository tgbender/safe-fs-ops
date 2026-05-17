from __future__ import annotations

import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops._windows_identity_rmdir import windows_identity_safe_remove_available
from safe_fs_ops.filesystem_ops.mutations import _MutationPlatformSupport

pytestmark = pytest.mark.safe_fs_ops


def _descriptor_relative_mutations_supported(*, needs_replace: bool = True) -> bool:
    required = [
        os.open in os.supports_dir_fd,
        os.unlink in os.supports_dir_fd,
        os.stat in os.supports_dir_fd,
        os.stat in os.supports_follow_symlinks,
        hasattr(os, "O_DIRECTORY"),
        hasattr(os, "O_NOFOLLOW"),
    ]
    if needs_replace:
        required.extend(
            [
                _descriptor_relative_replace_supported(),
                os.mkdir in os.supports_dir_fd,
            ]
        )
    return all(required)


def _descriptor_relative_replace_supported() -> bool:
    return (os.replace in os.supports_dir_fd) or (os.name != "nt" and os.rename in os.supports_dir_fd)


def _descriptor_relative_rename_replace_fallback_supported() -> bool:
    return (
        os.name != "nt"
        and os.rename in os.supports_dir_fd
        and os.replace not in os.supports_dir_fd
        and _descriptor_relative_mutations_supported(needs_replace=False)
        and os.mkdir in os.supports_dir_fd
    )


def _descriptor_relative_mkdir_supported() -> bool:
    return (
        os.mkdir in os.supports_dir_fd
        and os.open in os.supports_dir_fd
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def _descriptor_relative_rmdir_supported() -> bool:
    return (
        os.rmdir in os.supports_dir_fd
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def _windows_identity_safe_rmdir_supported() -> bool:
    return os.name == "nt" and windows_identity_safe_remove_available()


requires_descriptor_relative_replace = pytest.mark.skipif(
    not _descriptor_relative_mutations_supported(needs_replace=True),
    reason="descriptor-relative replace is unavailable on this platform",
)

requires_descriptor_relative_rename_replace_fallback = pytest.mark.skipif(
    not _descriptor_relative_rename_replace_fallback_supported(),
    reason="descriptor-relative rename fallback is unavailable on this platform",
)

requires_descriptor_relative_unlink = pytest.mark.skipif(
    not _descriptor_relative_mutations_supported(needs_replace=False),
    reason="descriptor-relative unlink is unavailable on this platform",
)

requires_descriptor_relative_mkdir = pytest.mark.skipif(
    not _descriptor_relative_mkdir_supported(),
    reason="descriptor-relative mkdir is unavailable on this platform",
)

requires_descriptor_relative_rmdir = pytest.mark.skipif(
    not _descriptor_relative_rmdir_supported(),
    reason="descriptor-relative rmdir is unavailable on this platform",
)

requires_windows_identity_safe_rmdir = pytest.mark.skipif(
    not _windows_identity_safe_rmdir_supported(),
    reason="native Windows identity-safe rmdir is unavailable on this platform",
)


def _unsupported_platform_support() -> _MutationPlatformSupport:
    return _MutationPlatformSupport(
        open_dir_fd=False,
        unlink_dir_fd=False,
        stat_dir_fd=False,
        stat_follow_symlinks=False,
        replace_dir_fd=False,
        rename_dir_fd=False,
        mkdir_dir_fd=False,
        rmdir_dir_fd=False,
        nofollow_directory_open=False,
    )


def _rename_only_platform_support() -> _MutationPlatformSupport:
    return _MutationPlatformSupport(
        open_dir_fd=True,
        unlink_dir_fd=True,
        stat_dir_fd=True,
        stat_follow_symlinks=True,
        replace_dir_fd=False,
        rename_dir_fd=True,
        mkdir_dir_fd=True,
        rmdir_dir_fd=True,
        nofollow_directory_open=True,
    )


def _path_identity(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    return (stat_result.st_dev, stat_result.st_ino)


class _DirectoryFsyncRecorder:
    def __init__(self) -> None:
        self.identities: list[tuple[int, int]] = []

    def __call__(self, descriptor: int) -> None:
        stat_result = os.fstat(descriptor)
        self.identities.append((stat_result.st_dev, stat_result.st_ino))


class _FileFsyncRecorder:
    def __init__(self) -> None:
        self.modes: list[int] = []

    def __call__(self, descriptor: int) -> None:
        self.modes.append(os.fstat(descriptor).st_mode & 0o777)


class _CallRecorder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, _descriptor: int) -> None:
        self.calls += 1


class _FailingFsync:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, _descriptor: int) -> None:
        self.calls += 1
        raise OSError(self.message)


__all__ = [name for name in globals() if not name.startswith("__")]
