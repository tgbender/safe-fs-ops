from __future__ import annotations

import ctypes
import errno
import os
import sys
from ctypes import wintypes
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsupportedFilesystemMutationError, rename_no_replace
from safe_fs_ops.filesystem_ops.no_replace_rename import (
    _FILE_RENAME_INFO_FILENAME_OFFSET,
    _FILE_RENAME_INFORMATION,
    _RENAME_EXCL,
    _RENAME_NOREPLACE,
    _STATUS_OBJECT_NAME_COLLISION,
    _FileRenameInfoHeader,
    _rename_path_no_replace,
    _windows_no_replace_rename,
)

pytestmark = pytest.mark.safe_fs_ops


def test_linux_backend_uses_descriptor_relative_parent_fds() -> None:
    opener = _ParentFdRecorder()
    calls: list[tuple[int, int, bytes, int, bytes, int]] = []

    def syscall(
        number: int,
        source_fd: int,
        source_name: bytes,
        destination_fd: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        calls.append((number, source_fd, source_name, destination_fd, destination_name, flags))
        return 0

    _rename_path_no_replace(
        Path("/workspace/source/state"),
        Path("/workspace/quarantine/state"),
        operation="rename",
        os_name="posix",
        platform_name="linux",
        linux_rename=syscall,
        open_parent_fd=opener.open,
        validate_parent_fd=_validate_parent_noop,
        validate_parent_fd_after_mutation=_validate_parent_noop,
        close_fd=opener.close,
    )

    assert opener.opened == [
        (Path("/workspace/source"), "rename"),
        (Path("/workspace/quarantine"), "rename"),
    ]
    assert opener.closed == [101, 100]
    assert len(calls) == 1
    _, source_fd, source_name, destination_fd, destination_name, flags = calls[0]
    assert (source_fd, source_name) == (100, b"state")
    assert (destination_fd, destination_name) == (101, b"state")
    assert flags == _RENAME_NOREPLACE


def test_macos_backend_selection_uses_renameatx_np_exclusive_flag() -> None:
    opener = _ParentFdRecorder()
    calls: list[tuple[int, bytes, int, bytes, int]] = []

    def renameatx_np(
        source_fd: int,
        source_name: bytes,
        destination_fd: int,
        destination_name: bytes,
        flags: int,
    ) -> int:
        calls.append((source_fd, source_name, destination_fd, destination_name, flags))
        return 0

    _rename_path_no_replace(
        Path("/workspace/source/state"),
        Path("/workspace/quarantine/state"),
        operation="capture directory",
        os_name="posix",
        platform_name="darwin",
        macos_rename=renameatx_np,
        open_parent_fd=opener.open,
        validate_parent_fd=_validate_parent_noop,
        validate_parent_fd_after_mutation=_validate_parent_noop,
        close_fd=opener.close,
    )

    assert calls == [(100, b"state", 101, b"state", _RENAME_EXCL)]
    assert opener.closed == [101, 100]


def test_macos_backend_maps_existing_destination_to_file_exists() -> None:
    opener = _ParentFdRecorder()

    def renameatx_np(
        _source_fd: int,
        _source_name: bytes,
        _destination_fd: int,
        _destination_name: bytes,
        _flags: int,
    ) -> int:
        return -1

    with pytest.raises(FileExistsError):
        _rename_path_no_replace(
            Path("/workspace/source/state"),
            Path("/workspace/quarantine/state"),
            operation="capture directory",
            os_name="posix",
            platform_name="darwin",
            macos_rename=renameatx_np,
            open_parent_fd=opener.open,
            validate_parent_fd=_validate_parent_noop,
            validate_parent_fd_after_mutation=_validate_parent_noop,
            close_fd=opener.close,
            errno_reader=lambda: errno.EEXIST,
        )

    assert opener.closed == [101, 100]


def test_windows_backend_selection_uses_handle_rename_backend() -> None:
    calls: list[tuple[Path, Path, str]] = []

    def windows_rename(source: Path, destination: Path, operation: str) -> None:
        calls.append((source, destination, operation))

    source = Path("C:/workspace/source/state")
    destination = Path("C:/workspace/quarantine/state")

    _rename_path_no_replace(
        source,
        destination,
        operation="capture directory",
        os_name="nt",
        platform_name="win32",
        windows_rename=windows_rename,
    )

    assert calls == [(source, destination, "capture directory")]


def test_windows_handle_backend_sets_relative_destination_and_disallows_replacement() -> None:
    kernel32 = _Kernel32Fake()
    ntdll = _NtdllFake()
    source = Path("C:/workspace/source/state")
    destination = Path("C:/workspace/quarantine/state")

    _windows_no_replace_rename(
        source,
        destination,
        "capture directory",
        kernel32=kernel32,
        ntdll=ntdll,
        last_error_reader=kernel32.get_last_error,
        availability_check=lambda: True,
    )

    assert [call.path for call in kernel32.create_calls] == [str(source), str(destination.parent)]
    assert len(ntdll.rename_calls) == 1
    rename_call = ntdll.rename_calls[0]
    assert rename_call.source_handle == 100
    assert rename_call.file_information_class == _FILE_RENAME_INFORMATION
    assert rename_call.replace_if_exists is False
    assert rename_call.root_directory == 101
    assert rename_call.file_name == destination.name
    assert kernel32.closed_handles == [101, 100]


def test_windows_handle_backend_maps_existing_destination_to_file_exists() -> None:
    kernel32 = _Kernel32Fake()
    ntdll = _NtdllFake(status=_STATUS_OBJECT_NAME_COLLISION)

    with pytest.raises(FileExistsError):
        _windows_no_replace_rename(
            Path("C:/workspace/source/state"),
            Path("C:/workspace/quarantine/state"),
            "capture directory",
            kernel32=kernel32,
            ntdll=ntdll,
            last_error_reader=kernel32.get_last_error,
            availability_check=lambda: True,
        )

    assert kernel32.closed_handles == [101, 100]


@pytest.mark.platform_windows
def test_windows_native_no_replace_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    if os.name != "nt":
        pytest.skip("requires Windows")
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    destination.write_text("destination\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        rename_no_replace(source, destination)

    assert source.read_text(encoding="utf-8") == "source\n"
    assert destination.read_text(encoding="utf-8") == "destination\n"


def test_linux_backend_revalidates_parent_fds_before_and_after_rename() -> None:
    opener = _ParentFdRecorder()
    before_validations: list[tuple[int, Path, str]] = []
    after_validations: list[tuple[int, Path, str]] = []

    def syscall(
        _number: int,
        _source_fd: int,
        _source_name: bytes,
        _destination_fd: int,
        _destination_name: bytes,
        _flags: int,
    ) -> int:
        return 0

    def validate_before(descriptor: int, path: Path, operation: str) -> None:
        before_validations.append((descriptor, path, operation))

    def validate_after(descriptor: int, path: Path, operation: str) -> None:
        after_validations.append((descriptor, path, operation))

    source = Path("/workspace/source/state")
    destination = Path("/workspace/quarantine/state")

    _rename_path_no_replace(
        source,
        destination,
        operation="rename",
        os_name="posix",
        platform_name="linux",
        linux_rename=syscall,
        open_parent_fd=opener.open,
        validate_parent_fd=validate_before,
        validate_parent_fd_after_mutation=validate_after,
        close_fd=opener.close,
    )

    assert before_validations == [
        (100, source.parent, "rename"),
        (101, destination.parent, "rename"),
    ]
    assert after_validations == before_validations


@pytest.mark.platform_macos
def test_macos_native_no_replace_refuses_existing_destination_without_overwrite(tmp_path: Path) -> None:
    if sys.platform != "darwin":
        pytest.skip("requires macOS")
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source\n", encoding="utf-8")
    destination.write_text("destination\n", encoding="utf-8")

    try:
        rename_no_replace(source, destination)
    except UnsupportedFilesystemMutationError as exc:
        pytest.skip(str(exc))
    except FileExistsError:
        pass
    else:
        raise AssertionError("rename_no_replace replaced an existing destination")

    assert source.read_text(encoding="utf-8") == "source\n"
    assert destination.read_text(encoding="utf-8") == "destination\n"


class _ParentFdRecorder:
    def __init__(self) -> None:
        self.opened: list[tuple[Path, str]] = []
        self.closed: list[int] = []
        self._next_fd = 100

    def open(self, path: Path, operation: str) -> int:
        descriptor = self._next_fd
        self._next_fd += 1
        self.opened.append((path, operation))
        return descriptor

    def close(self, descriptor: int) -> None:
        self.closed.append(descriptor)


def _validate_parent_noop(_descriptor: int, _path: Path, _operation: str) -> None:
    return


class _CreateFileCall:
    def __init__(self, path: str, desired_access: int, share_mode: int, flags: int) -> None:
        self.path = path
        self.desired_access = desired_access
        self.share_mode = share_mode
        self.flags = flags


class _RenameCall:
    def __init__(
        self,
        *,
        source_handle: int,
        file_information_class: int,
        buffer_size: int,
        replace_if_exists: bool,
        root_directory: int,
        file_name: str,
    ) -> None:
        self.source_handle = source_handle
        self.file_information_class = file_information_class
        self.buffer_size = buffer_size
        self.replace_if_exists = replace_if_exists
        self.root_directory = root_directory
        self.file_name = file_name


class _Kernel32Fake:
    def __init__(self, *, last_error: int = 0) -> None:
        self.last_error = last_error
        self.create_calls: list[_CreateFileCall] = []
        self.closed_handles: list[int] = []
        self._next_handle = 100

    def CreateFileW(
        self,
        path: str,
        desired_access: int,
        share_mode: int,
        _security_attributes: object,
        _creation_disposition: int,
        flags: int,
        _template_file: object,
    ) -> int:
        handle = self._next_handle
        self._next_handle += 1
        self.create_calls.append(_CreateFileCall(path, desired_access, share_mode, flags))
        return handle

    def CloseHandle(self, handle: object) -> bool:
        self.closed_handles.append(_int_handle(handle))
        return True

    def get_last_error(self) -> int:
        return self.last_error


class _NtdllFake:
    def __init__(self, *, status: int = 0) -> None:
        self.status = status
        self.rename_calls: list[_RenameCall] = []

    def NtSetInformationFile(
        self,
        source_handle: int,
        _io_status_block: object,
        file_information: object,
        buffer_size: int,
        file_information_class: int,
    ) -> int:
        header = ctypes.cast(file_information, ctypes.POINTER(_FileRenameInfoHeader)).contents
        address = ctypes.cast(file_information, ctypes.c_void_p).value
        assert address is not None
        file_name_length = int(header.FileNameLength // ctypes.sizeof(wintypes.WCHAR))
        file_name = ctypes.wstring_at(address + _FILE_RENAME_INFO_FILENAME_OFFSET, file_name_length)
        self.rename_calls.append(
            _RenameCall(
                source_handle=source_handle,
                file_information_class=file_information_class,
                buffer_size=buffer_size,
                replace_if_exists=bool(header.ReplaceIfExists),
                root_directory=_int_handle(header.RootDirectory),
                file_name=file_name,
            )
        )
        return self.status


def _int_handle(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    assert isinstance(value, int)
    return value
