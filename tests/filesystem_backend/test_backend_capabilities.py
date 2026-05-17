from __future__ import annotations

import pytest

from safe_fs_ops.filesystem_ops.backend import capabilities_from_platform_support
from safe_fs_ops.filesystem_ops.mutation_support import _MutationPlatformSupport

pytestmark = [pytest.mark.safe_fs_ops, pytest.mark.safe_fs_ops_backend]


def test_write_text_is_unsupported_for_native_windows_without_replace_dir_fd() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="windows_native",
        platform="windows",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=True,
        windows_file_api_available=True,
    )

    assert capabilities.write_text.supported is True
    assert capabilities.delete_file.supported is True
    assert capabilities.make_directory.supported is True
    assert capabilities.remove_directory.supported is True
    assert capabilities.remove_directory.required_features == ("windows_delete_disposition_by_handle",)
    assert capabilities.write_text.reason is None
    assert capabilities.write_text.required_features == (
        "replace_file_w_or_move_file_ex_w",
        "flush_file_buffers",
        "windows_reparse_guard",
    )


def test_write_text_is_supported_for_posix_rename_fallback() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="posix_descriptor_relative",
        platform="posix",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=False,
    )

    assert capabilities.write_text.supported is True
    assert capabilities.write_text.reason is None
    assert capabilities.remove_directory.supported is True
    assert capabilities.remove_directory.reason is not None
    assert "safe-ish transaction-owned cleanup" in capabilities.remove_directory.reason
    assert "rename_dir_fd" in capabilities.write_text.required_features


def test_remove_directory_requires_same_features_as_primitive() -> None:
    support = _MutationPlatformSupport(
        open_dir_fd=True,
        unlink_dir_fd=True,
        stat_dir_fd=True,
        stat_follow_symlinks=False,
        replace_dir_fd=True,
        rename_dir_fd=True,
        mkdir_dir_fd=True,
        rmdir_dir_fd=True,
        nofollow_directory_open=True,
    )

    capabilities = capabilities_from_platform_support(
        backend_name="posix_descriptor_relative",
        platform="posix",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=False,
    )

    assert capabilities.remove_directory.supported is False
    assert capabilities.remove_directory.reason is not None
    assert "stat_follow_symlinks" in capabilities.remove_directory.reason
    assert capabilities.remove_directory.required_features == (
        "rmdir_dir_fd",
        "open_dir_fd",
        "stat_dir_fd",
        "stat_follow_symlinks",
        "nofollow_directory_open",
        "safe_ish_transaction_owned_rmdir",
    )


def test_remove_directory_is_supported_for_posix_descriptor_relative_safe_ish_cleanup() -> None:
    support = _MutationPlatformSupport(
        open_dir_fd=True,
        unlink_dir_fd=True,
        stat_dir_fd=True,
        stat_follow_symlinks=True,
        replace_dir_fd=True,
        rename_dir_fd=True,
        mkdir_dir_fd=True,
        rmdir_dir_fd=True,
        nofollow_directory_open=True,
    )

    capabilities = capabilities_from_platform_support(
        backend_name="posix_descriptor_relative",
        platform="posix",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=False,
    )

    assert capabilities.remove_directory.supported is True
    assert capabilities.remove_directory.reason is not None
    assert "safe-ish transaction-owned cleanup" in capabilities.remove_directory.reason
    assert capabilities.remove_directory.required_features == (
        "rmdir_dir_fd",
        "open_dir_fd",
        "stat_dir_fd",
        "stat_follow_symlinks",
        "nofollow_directory_open",
        "safe_ish_transaction_owned_rmdir",
    )


def test_remove_directory_reports_unsupported_when_native_windows_handle_delete_is_unavailable() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="windows_native",
        platform="windows",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=False,
        windows_file_api_available=True,
    )

    assert capabilities.remove_directory.supported is False
    assert capabilities.remove_directory.reason is not None
    assert "Windows handle delete disposition" in capabilities.remove_directory.reason
    assert capabilities.remove_directory.required_features == ("windows_delete_disposition_by_handle",)


def test_remove_directory_reports_supported_for_win32_platform_with_native_windows_handle_delete() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="windows_native",
        platform="win32",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=True,
        windows_file_api_available=True,
    )

    assert capabilities.remove_directory.supported is True
    assert capabilities.remove_directory.reason is None
    assert capabilities.remove_directory.required_features == ("windows_delete_disposition_by_handle",)


def test_write_text_reports_unsupported_when_windows_file_primitives_are_unavailable() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="windows_native",
        platform="windows",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=True,
        windows_file_api_available=False,
    )

    assert capabilities.write_text.supported is False
    assert capabilities.write_text.reason == "native Windows file replacement and flush primitives are unavailable"
    assert capabilities.delete_file.supported is False
    assert capabilities.make_directory.supported is False


def test_windows_delete_and_mkdir_report_unsupported_when_windows_reparse_primitive_is_unavailable() -> None:
    support = _MutationPlatformSupport(
        open_dir_fd=True,
        unlink_dir_fd=True,
        stat_dir_fd=True,
        stat_follow_symlinks=True,
        replace_dir_fd=True,
        rename_dir_fd=True,
        mkdir_dir_fd=True,
        rmdir_dir_fd=True,
        nofollow_directory_open=True,
    )

    capabilities = capabilities_from_platform_support(
        backend_name="windows_native",
        platform="windows",
        platform_support=support,
        is_default_backend=True,
        identity_safe_rmdir_available=True,
        windows_file_api_available=False,
    )

    assert capabilities.delete_file.supported is False
    assert capabilities.delete_file.reason == "native Windows file attribute primitives are unavailable"
    assert capabilities.make_directory.supported is False
    assert capabilities.make_directory.reason == "native Windows file attribute primitives are unavailable"


def test_explicit_posix_descriptor_relative_query_on_windows_uses_posix_semantics() -> None:
    support = _MutationPlatformSupport(
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

    capabilities = capabilities_from_platform_support(
        backend_name="posix_descriptor_relative",
        platform="windows",
        platform_support=support,
        is_default_backend=False,
        identity_safe_rmdir_available=False,
        windows_file_api_available=True,
    )

    assert capabilities.write_text.supported is False
    assert capabilities.write_text.reason is not None
    assert "replace_dir_fd" in capabilities.write_text.reason
    assert capabilities.delete_file.supported is True
    assert capabilities.make_directory.supported is True
    assert capabilities.remove_directory.supported is True
    assert "safe-ish transaction-owned cleanup" in (capabilities.remove_directory.reason or "")
