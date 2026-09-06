import ctypes
import errno
from pathlib import Path

import pytest
from test_no_replace_rename_backends import _Kernel32Fake, _NtdllFake, _ParentFdRecorder, _validate_parent_noop

from safe_fs_ops.filesystem_ops import UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops._windows_identity_rmdir import (
    _flush_parent_handle_for_durability,
    _set_delete_disposition,
)
from safe_fs_ops.filesystem_ops.mutation_support import DurabilityMode
from safe_fs_ops.filesystem_ops.no_replace_rename import _rename_path_no_replace, _windows_no_replace_rename


@pytest.mark.parametrize("platform", ["linux", "darwin"])
@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EXDEV, errno.EACCES])
def test_posix_rename_failure_closes_both_parents(platform: str, error: int) -> None:
    parents = _ParentFdRecorder()
    expected = UnsupportedFilesystemMutationError if error in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP} else OSError
    with pytest.raises(expected) as raised:
        _rename_path_no_replace(
            Path("source/item"),
            Path("destination/item"),
            operation="rename",
            os_name="posix",
            platform_name=platform,
            linux_rename=lambda *args: -1,
            macos_rename=lambda *args: -1,
            open_parent_fd=parents.open,
            close_fd=parents.close,
            validate_parent_fd=_validate_parent_noop,
            validate_parent_fd_after_mutation=lambda *args: pytest.fail("validated success after failed mutation"),
            errno_reader=lambda: error,
        )
    if expected is OSError:
        assert raised.value.errno == error
    assert parents.closed == [101, 100]


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_failed_second_parent_open_closes_first_without_rename(platform: str) -> None:
    parents = _ParentFdRecorder()

    def open_parent(path: Path, operation: str) -> int:
        if parents.opened:
            raise PermissionError("destination parent denied")
        return parents.open(path, operation)

    with pytest.raises(PermissionError):
        _rename_path_no_replace(
            Path("source/item"),
            Path("destination/item"),
            operation="rename",
            os_name="posix",
            platform_name=platform,
            linux_rename=lambda *args: pytest.fail("rename called"),
            macos_rename=lambda *args: pytest.fail("rename called"),
            open_parent_fd=open_parent,
            close_fd=parents.close,
        )
    assert parents.closed == [100]


def test_windows_failed_destination_open_closes_source_without_rename() -> None:
    class DeniedParent(_Kernel32Fake):
        def CreateFileW(self, *args: object) -> int:
            if self.create_calls:
                return ctypes.c_void_p(-1).value
            return super().CreateFileW(*args)

    kernel = DeniedParent(last_error=5)
    native = _NtdllFake()
    with pytest.raises(PermissionError):
        _windows_no_replace_rename(
            Path("source/item"),
            Path("destination/item"),
            "rename",
            kernel32=kernel,
            ntdll=native,
            availability_check=lambda: True,
            last_error_reader=kernel.get_last_error,
        )
    assert native.rename_calls == []
    assert kernel.closed_handles == [100]


@pytest.mark.parametrize("status", [0xC0000022, 0xC0000034, 0xC0000001])
def test_windows_native_rename_error_closes_both_handles(status: int) -> None:
    kernel = _Kernel32Fake()
    with pytest.raises(OSError):
        _windows_no_replace_rename(
            Path("source/item"),
            Path("destination/item"),
            "rename",
            kernel32=kernel,
            ntdll=_NtdllFake(status=status),
            availability_check=lambda: True,
        )
    assert kernel.closed_handles == [101, 100]


@pytest.mark.parametrize("fallback_error", [0, 5])
def test_windows_delete_falls_back_only_for_unsupported_extended_disposition(fallback_error: int) -> None:
    class DeleteApi:
        error = 87
        calls: list[int]

        def __init__(self) -> None:
            self.calls = []

        def SetFileInformationByHandle(self, handle, information_class, buffer, size):
            assert handle == 123
            self.calls.append(information_class)
            if information_class == 21:
                assert ctypes.string_at(buffer, 4) == (0x13).to_bytes(4, "little")
                assert size >= 4  # Host wintypes differ outside native Windows; SDK layout has its own test.
                return False
            assert information_class == 4
            assert size == 1 and ctypes.string_at(buffer, 1) == b"\x01"
            self.error = fallback_error
            return not self.error

    api = DeleteApi()
    if fallback_error:
        with pytest.raises(OSError, match="mark directory for deletion"):
            _set_delete_disposition(api, 123, path=Path("target"), last_error_reader=lambda: api.error)
    else:
        _set_delete_disposition(api, 123, path=Path("target"), last_error_reader=lambda: api.error)
    assert api.calls == [21, 4]


def test_windows_delete_access_denied_does_not_try_legacy_disposition() -> None:
    calls = []

    class DeniedApi:
        def SetFileInformationByHandle(self, handle, information_class, buffer, size):
            calls.append(information_class)
            return False

    with pytest.raises(OSError):
        _set_delete_disposition(DeniedApi(), 123, path=Path("target"), last_error_reader=lambda: 5)
    assert calls == [21]


@pytest.mark.parametrize("durability", list(DurabilityMode))
def test_windows_parent_flush_failure_obeys_durability(durability: DurabilityMode) -> None:
    calls = []

    class FlushApi:
        def FlushFileBuffers(self, handle):
            calls.append(handle)
            return False

    def flush():
        _flush_parent_handle_for_durability(
            FlushApi(), 123, Path("parent"), durability=durability, last_error_reader=lambda: 5
        )

    if durability is DurabilityMode.FSYNC:
        with pytest.raises(OSError, match="flush parent directory"):
            flush()
    else:
        flush()
    assert calls == ([] if durability is DurabilityMode.NONE else [123])
