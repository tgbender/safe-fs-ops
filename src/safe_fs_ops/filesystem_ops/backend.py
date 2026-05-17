from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

from safe_fs_ops.filesystem_ops._windows_identity_rmdir import windows_identity_safe_remove_available
from safe_fs_ops.filesystem_ops._windows_primitives import windows_file_primitives_available
from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    UnsupportedFilesystemMutationError,
    _MutationPlatformSupport,
)

CapabilityState = Literal["supported", "unsupported", "unknown"]


@dataclass(frozen=True, slots=True)
class FilesystemOperationCapability:
    operation: str
    state: CapabilityState
    reason: str | None = None
    required_features: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return self.state == "supported"


@dataclass(frozen=True, slots=True)
class FilesystemBackendCapabilities:
    backend_name: str
    platform: str
    write_text: FilesystemOperationCapability
    delete_file: FilesystemOperationCapability
    make_directory: FilesystemOperationCapability
    remove_directory: FilesystemOperationCapability
    is_default_backend: bool

    def support_for(self, operation: str) -> FilesystemOperationCapability:
        try:
            return {
                "write_text": self.write_text,
                "delete_file": self.delete_file,
                "make_directory": self.make_directory,
                "remove_directory": self.remove_directory,
            }[operation]
        except KeyError as exc:
            raise ValueError(f"unknown backend operation: {operation!r}") from exc

    def require(self, operation: str) -> None:
        capability = self.support_for(operation)
        if capability.state != "unsupported":
            return
        raise UnsupportedFilesystemBackendError(
            operation=operation,
            backend_name=self.backend_name,
            platform=self.platform,
            capability=capability,
        )


class UnsupportedFilesystemBackendError(UnsupportedFilesystemMutationError):
    def __init__(
        self,
        *,
        operation: str,
        backend_name: str,
        platform: str,
        capability: FilesystemOperationCapability,
    ) -> None:
        if capability.reason is None:
            message = f"{operation} is unsupported by backend {backend_name!r} on {platform!r}"
        else:
            message = f"{operation} is unsupported by backend {backend_name!r} on {platform!r}: {capability.reason}"
        super().__init__(message)
        self.operation = operation
        self.backend_name = backend_name
        self.platform = platform
        self.capability = capability


def detect_default_backend_capabilities() -> FilesystemBackendCapabilities:
    platform = _platform_name()
    return capabilities_from_platform_support(
        backend_name=_default_backend_name(platform),
        platform=platform,
        platform_support=_DEFAULT_PLATFORM_SUPPORT,
        is_default_backend=True,
    )


def custom_backend_capabilities() -> FilesystemBackendCapabilities:
    unknown = FilesystemOperationCapability(
        operation="custom",
        state="unknown",
        reason="capabilities are unknown for injected custom filesystem operations",
    )
    return FilesystemBackendCapabilities(
        backend_name="custom",
        platform=_platform_name(),
        write_text=FilesystemOperationCapability(
            operation="write_text",
            state=unknown.state,
            reason=unknown.reason,
        ),
        delete_file=FilesystemOperationCapability(
            operation="delete_file",
            state=unknown.state,
            reason=unknown.reason,
        ),
        make_directory=FilesystemOperationCapability(
            operation="make_directory",
            state=unknown.state,
            reason=unknown.reason,
        ),
        remove_directory=FilesystemOperationCapability(
            operation="remove_directory",
            state=unknown.state,
            reason=unknown.reason,
        ),
        is_default_backend=False,
    )


def capabilities_from_platform_support(
    *,
    backend_name: str,
    platform: str,
    platform_support: _MutationPlatformSupport,
    is_default_backend: bool,
    identity_safe_rmdir_available: bool | None = None,
    windows_file_api_available: bool | None = None,
) -> FilesystemBackendCapabilities:
    uses_windows_native_capabilities = _uses_windows_native_capabilities(
        backend_name=backend_name,
        platform=platform,
        is_default_backend=is_default_backend,
    )
    if uses_windows_native_capabilities:
        return _windows_backend_capabilities(
            backend_name=backend_name,
            platform=platform,
            is_default_backend=is_default_backend,
            identity_safe_rmdir_available=(
                _detect_identity_safe_rmdir_available(platform)
                if identity_safe_rmdir_available is None
                else identity_safe_rmdir_available
            ),
            windows_file_api_available=(
                windows_file_primitives_available()
                if windows_file_api_available is None
                else windows_file_api_available
            ),
        )
    mutation_reason = _missing_features_reason(
        platform_support=platform_support,
        required_features=_descriptor_relative_mutation_features(needs_replace=False),
    )
    mkdir_reason = _missing_features_reason(
        platform_support=platform_support,
        required_features=_descriptor_relative_mkdir_features(),
    )
    rmdir_reason = _missing_features_reason(
        platform_support=platform_support,
        required_features=_descriptor_relative_rmdir_features(),
    )
    return FilesystemBackendCapabilities(
        backend_name=backend_name,
        platform=platform,
        write_text=_write_text_capability(platform=platform, platform_support=platform_support),
        delete_file=_capability(
            operation="delete_file",
            required_features=_descriptor_relative_mutation_features(needs_replace=False),
            reason=mutation_reason,
        ),
        make_directory=_capability(
            operation="make_directory",
            required_features=_descriptor_relative_mkdir_features(),
            reason=mkdir_reason,
        ),
        remove_directory=_remove_directory_capability(
            platform=platform,
            uses_windows_native_capabilities=uses_windows_native_capabilities,
            rmdir_reason=rmdir_reason,
            identity_safe_rmdir_available=(
                _detect_identity_safe_rmdir_available(platform)
                if identity_safe_rmdir_available is None
                else identity_safe_rmdir_available
            ),
        ),
        is_default_backend=is_default_backend,
    )


def _windows_backend_capabilities(
    *,
    backend_name: str,
    platform: str,
    is_default_backend: bool,
    identity_safe_rmdir_available: bool,
    windows_file_api_available: bool,
) -> FilesystemBackendCapabilities:
    write_required_features = (
        "replace_file_w_or_move_file_ex_w",
        "flush_file_buffers",
        "windows_reparse_guard",
    )
    reparse_required_features = ("windows_reparse_guard",)
    write_reason = None
    if not windows_file_api_available:
        write_reason = "native Windows file replacement and flush primitives are unavailable"
    supported_state: CapabilityState = "supported" if windows_file_api_available else "unsupported"
    reparse_reason = None
    if not windows_file_api_available:
        reparse_reason = "native Windows file attribute primitives are unavailable"
    return FilesystemBackendCapabilities(
        backend_name=backend_name,
        platform=platform,
        write_text=FilesystemOperationCapability(
            operation="write_text",
            state=supported_state,
            reason=write_reason,
            required_features=write_required_features,
        ),
        delete_file=FilesystemOperationCapability(
            operation="delete_file",
            state=supported_state,
            reason=reparse_reason,
            required_features=(*reparse_required_features, "regular_file_only_delete"),
        ),
        make_directory=FilesystemOperationCapability(
            operation="make_directory",
            state=supported_state,
            reason=reparse_reason,
            required_features=(*reparse_required_features, "validated_parent_chain"),
        ),
        remove_directory=_remove_directory_capability(
            platform=platform,
            uses_windows_native_capabilities=True,
            rmdir_reason=None,
            identity_safe_rmdir_available=identity_safe_rmdir_available,
        ),
        is_default_backend=is_default_backend,
    )


def _capability(
    *,
    operation: str,
    required_features: tuple[str, ...],
    reason: str | None,
) -> FilesystemOperationCapability:
    return FilesystemOperationCapability(
        operation=operation,
        state="supported" if reason is None else "unsupported",
        reason=reason,
        required_features=required_features,
    )


def _write_text_capability(
    *,
    platform: str,
    platform_support: _MutationPlatformSupport,
) -> FilesystemOperationCapability:
    required_features = _descriptor_relative_mutation_features(needs_replace=False)
    missing = [feature for feature in required_features if not getattr(platform_support, feature)]
    replacement_required = _write_text_replacement_requirement(platform=platform, platform_support=platform_support)
    if replacement_required is not None:
        missing.append(replacement_required)
    if missing:
        return FilesystemOperationCapability(
            operation="write_text",
            state="unsupported",
            reason=f"missing required descriptor-relative features: {', '.join(missing)}",
            required_features=(*required_features, replacement_required)
            if replacement_required is not None
            else required_features,
        )
    return FilesystemOperationCapability(
        operation="write_text",
        state="supported",
        required_features=_write_text_supported_features(platform=platform, platform_support=platform_support),
    )


def _remove_directory_capability(
    *,
    platform: str,
    uses_windows_native_capabilities: bool,
    rmdir_reason: str | None,
    identity_safe_rmdir_available: bool,
) -> FilesystemOperationCapability:
    if uses_windows_native_capabilities:
        required_features: tuple[str, ...] = ("windows_delete_disposition_by_handle",)
        if identity_safe_rmdir_available:
            return FilesystemOperationCapability(
                operation="remove_directory",
                state="supported",
                required_features=required_features,
            )
        return FilesystemOperationCapability(
            operation="remove_directory",
            state="unsupported",
            reason="native Windows handle delete disposition API is unavailable",
            required_features=required_features,
        )

    required_features = (*_descriptor_relative_rmdir_features(), "safe_ish_transaction_owned_rmdir")
    if rmdir_reason is not None:
        return FilesystemOperationCapability(
            operation="remove_directory",
            state="unsupported",
            reason=rmdir_reason,
            required_features=required_features,
        )
    if identity_safe_rmdir_available:
        return FilesystemOperationCapability(
            operation="remove_directory",
            state="supported",
            reason=("supports safe-ish transaction-owned cleanup and atomic identity-conditional directory removal"),
            required_features=(*required_features, "identity_conditional_rmdir"),
        )
    return FilesystemOperationCapability(
        operation="remove_directory",
        state="supported",
        reason=(
            "supports safe-ish transaction-owned cleanup via descriptor-relative rmdir; "
            "atomic identity-conditional directory removal is unavailable through Python filesystem APIs"
        ),
        required_features=required_features,
    )


def _detect_identity_safe_rmdir_available(platform: str) -> bool:
    return _is_windows_platform(platform) and windows_identity_safe_remove_available()


def _missing_features_reason(
    *,
    platform_support: _MutationPlatformSupport,
    required_features: tuple[str, ...],
) -> str | None:
    missing = [feature for feature in required_features if not getattr(platform_support, feature)]
    if not missing:
        return None
    joined = ", ".join(missing)
    return f"missing required descriptor-relative features: {joined}"


def _descriptor_relative_mutation_features(*, needs_replace: bool) -> tuple[str, ...]:
    features = (
        "open_dir_fd",
        "unlink_dir_fd",
        "stat_dir_fd",
        "stat_follow_symlinks",
        "nofollow_directory_open",
    )
    if needs_replace:
        return (*features, "replace_dir_fd", "mkdir_dir_fd")
    return features


def _descriptor_relative_mkdir_features() -> tuple[str, ...]:
    return (
        "mkdir_dir_fd",
        "open_dir_fd",
        "nofollow_directory_open",
    )


def _descriptor_relative_rmdir_features() -> tuple[str, ...]:
    return (
        "rmdir_dir_fd",
        "open_dir_fd",
        "stat_dir_fd",
        "stat_follow_symlinks",
        "nofollow_directory_open",
    )


def _write_text_replacement_requirement(
    *,
    platform: str,
    platform_support: _MutationPlatformSupport,
) -> str | None:
    if platform_support.replace_dir_fd:
        return None
    if not _is_windows_platform(platform) and platform_support.rename_dir_fd:
        return None
    return "replace_dir_fd" if _is_windows_platform(platform) else "replace_dir_fd_or_rename_dir_fd"


def _write_text_supported_features(
    *,
    platform: str,
    platform_support: _MutationPlatformSupport,
) -> tuple[str, ...]:
    replacement_feature = "replace_dir_fd"
    if not platform_support.replace_dir_fd and not _is_windows_platform(platform) and platform_support.rename_dir_fd:
        replacement_feature = "rename_dir_fd"
    return (*_descriptor_relative_mutation_features(needs_replace=False), "mkdir_dir_fd", replacement_feature)


def _is_windows_platform(platform: str) -> bool:
    return platform in {"win32", "windows"}


def _uses_windows_native_capabilities(
    *,
    backend_name: str,
    platform: str,
    is_default_backend: bool,
) -> bool:
    if not _is_windows_platform(platform):
        return False
    if backend_name == "windows_native":
        return True
    return is_default_backend and backend_name == _default_backend_name(platform)


def _default_backend_name(platform: str) -> str:
    if _is_windows_platform(platform):
        return "windows_native"
    return "posix_descriptor_relative"


def _platform_name() -> str:
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if os.name == "posix":
        return "posix"
    return os.name
