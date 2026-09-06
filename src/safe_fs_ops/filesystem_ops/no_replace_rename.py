from __future__ import annotations

import ctypes
import errno
import os
import platform
import sys
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol, cast

from safe_fs_ops.filesystem_ops.mutation_support import (
    _DEFAULT_PLATFORM_SUPPORT,
    UnsupportedFilesystemMutationError,
    _ensure_directory_descriptor_matches_path,
    _ensure_directory_descriptor_matches_path_after_mutation,
    _open_directory_for_mutation,
)
from safe_fs_ops.filesystem_ops.paths import require_no_nul


class DirectoryNoReplaceRename(Protocol):
    def __call__(self, source: Path, destination: Path, *, operation: str) -> None:
        pass


ParentDirectoryOpener = Callable[[Path, str], int]
ParentDirectoryValidator = Callable[[int, Path, str], None]
FileDescriptorCloser = Callable[[int], None]
ErrnoReader = Callable[[], int]
WindowsLastErrorReader = Callable[[], int]
LinuxRenameAt2Syscall = Callable[[int, int, bytes, int, bytes, int], int]
MacOSRenameAtxNp = Callable[[int, bytes, int, bytes, int], int]
WindowsNoReplaceRename = Callable[[Path, Path, str], None]

_RENAME_NOREPLACE: Final = 1
_RENAME_EXCL: Final = 0x00000004
_RENAMEAT2_SYSCALLS: Final = {
    "aarch64": 276,
    "amd64": 316,
    "armv7l": 382,
    "i386": 353,
    "i686": 353,
    "x86_64": 316,
}
_UNSUPPORTED_RENAME_ERRORS: Final = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", getattr(errno, "EOPNOTSUPP", 95)),
    getattr(errno, "EOPNOTSUPP", 95),
}

_ERROR_FILE_NOT_FOUND: Final = 2
_ERROR_PATH_NOT_FOUND: Final = 3
_ERROR_ACCESS_DENIED: Final = 5
_ERROR_FILE_EXISTS: Final = 80
_ERROR_DIR_NOT_EMPTY: Final = 145
_ERROR_ALREADY_EXISTS: Final = 183

_DELETE: Final = 0x00010000
_FILE_READ_ATTRIBUTES: Final = 0x0080
_FILE_TRAVERSE: Final = 0x0020
_FILE_SHARE_READ: Final = 0x00000001
_FILE_SHARE_WRITE: Final = 0x00000002
_FILE_SHARE_DELETE: Final = 0x00000004
_OPEN_EXISTING: Final = 3
_FILE_FLAG_BACKUP_SEMANTICS: Final = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT: Final = 0x00200000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
assert _INVALID_HANDLE_VALUE is not None
_INVALID_HANDLE_VALUE_INT: Final = _INVALID_HANDLE_VALUE
_FILE_RENAME_INFORMATION: Final = 10
_STATUS_SUCCESS: Final = 0x00000000
_STATUS_ACCESS_DENIED: Final = 0xC0000022
_STATUS_OBJECT_NAME_NOT_FOUND: Final = 0xC0000034
_STATUS_OBJECT_NAME_COLLISION: Final = 0xC0000035
_STATUS_OBJECT_PATH_NOT_FOUND: Final = 0xC000003A
_STATUS_DIRECTORY_NOT_EMPTY: Final = 0xC0000101


class _FileRenameInfoHeader(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("ReplaceIfExists", wintypes.BOOLEAN),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.DWORD),
    ]


class _FileRenameInfoTemplate(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("ReplaceIfExists", wintypes.BOOLEAN),
        ("RootDirectory", wintypes.HANDLE),
        ("FileNameLength", wintypes.ULONG),
        ("FileName", ctypes.c_uint16 * 1),
    ]


class _IoStatusBlock(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("Status", ctypes.c_void_p),
        ("Information", ctypes.c_void_p),
    ]


_FILE_RENAME_INFO_FILENAME_OFFSET: Final = _FileRenameInfoTemplate.FileName.offset


def rename_path_no_replace(source: Path, destination: Path, *, operation: str) -> None:
    _rename_path_no_replace(
        source,
        destination,
        operation=operation,
        os_name=os.name,
        platform_name=sys.platform,
    )


def rename_directory_no_replace(source: Path, destination: Path, *, operation: str) -> None:
    try:
        rename_path_no_replace(source, destination, operation=operation)
    except UnsupportedFilesystemMutationError as exc:
        raise _unsupported_no_replace_directory_rename(operation, str(exc)) from exc


def _rename_path_no_replace(
    source: Path,
    destination: Path,
    *,
    operation: str,
    os_name: str,
    platform_name: str,
    linux_rename: LinuxRenameAt2Syscall | None = None,
    macos_rename: MacOSRenameAtxNp | None = None,
    windows_rename: WindowsNoReplaceRename | None = None,
    open_parent_fd: ParentDirectoryOpener | None = None,
    validate_parent_fd: ParentDirectoryValidator | None = None,
    validate_parent_fd_after_mutation: ParentDirectoryValidator | None = None,
    close_fd: FileDescriptorCloser = os.close,
    errno_reader: ErrnoReader = ctypes.get_errno,
) -> None:
    _require_leaf_name(source)
    _require_leaf_name(destination)
    if os_name == "nt":
        rename = _windows_no_replace_rename if windows_rename is None else windows_rename
        rename(source, destination, operation)
        return
    parent_opener = _open_parent_directory_fd if open_parent_fd is None else open_parent_fd
    if platform_name.startswith("linux"):
        _linux_renameat2_no_replace(
            source,
            destination,
            operation=operation,
            rename=linux_rename,
            open_parent_fd=parent_opener,
            validate_parent_fd=validate_parent_fd,
            validate_parent_fd_after_mutation=validate_parent_fd_after_mutation,
            close_fd=close_fd,
            errno_reader=errno_reader,
        )
        return
    if platform_name == "darwin":
        _macos_renameatx_np_no_replace(
            source,
            destination,
            operation=operation,
            rename=macos_rename,
            open_parent_fd=parent_opener,
            validate_parent_fd=validate_parent_fd,
            validate_parent_fd_after_mutation=validate_parent_fd_after_mutation,
            close_fd=close_fd,
            errno_reader=errno_reader,
        )
        return
    raise _unsupported_no_replace_rename(operation, f"platform {platform_name!r} has no no-replace rename backend")


def _linux_renameat2_no_replace(
    source: Path,
    destination: Path,
    *,
    operation: str,
    rename: LinuxRenameAt2Syscall | None = None,
    open_parent_fd: ParentDirectoryOpener | None = None,
    validate_parent_fd: ParentDirectoryValidator | None = None,
    validate_parent_fd_after_mutation: ParentDirectoryValidator | None = None,
    close_fd: FileDescriptorCloser = os.close,
    errno_reader: ErrnoReader = ctypes.get_errno,
) -> None:
    syscall_number = _renameat2_syscall_number()
    if syscall_number is None:
        raise _unsupported_no_replace_rename(operation, f"Linux machine {platform.machine()!r} is not recognized")

    syscall = _linux_renameat2_syscall() if rename is None else rename
    parent_opener = _open_parent_directory_fd if open_parent_fd is None else open_parent_fd
    validate_parent = _validate_parent_directory_fd if validate_parent_fd is None else validate_parent_fd
    validate_parent_after = (
        _validate_parent_directory_fd_after_mutation
        if validate_parent_fd_after_mutation is None
        else validate_parent_fd_after_mutation
    )
    source_parent_fd = parent_opener(source.parent, operation)
    try:
        destination_parent_fd = parent_opener(destination.parent, operation)
        try:
            validate_parent(source_parent_fd, source.parent, operation)
            validate_parent(destination_parent_fd, destination.parent, operation)
            result = int(
                syscall(
                    syscall_number,
                    source_parent_fd,
                    os.fsencode(source.name),
                    destination_parent_fd,
                    os.fsencode(destination.name),
                    _RENAME_NOREPLACE,
                )
            )
            _raise_posix_rename_error_if_failed(
                result,
                errno_reader,
                destination=destination,
                operation=operation,
                primitive="renameat2(RENAME_NOREPLACE)",
            )
            validate_parent_after(source_parent_fd, source.parent, operation)
            validate_parent_after(destination_parent_fd, destination.parent, operation)
        finally:
            close_fd(destination_parent_fd)
    finally:
        close_fd(source_parent_fd)


def _macos_renameatx_np_no_replace(
    source: Path,
    destination: Path,
    *,
    operation: str,
    rename: MacOSRenameAtxNp | None = None,
    open_parent_fd: ParentDirectoryOpener | None = None,
    validate_parent_fd: ParentDirectoryValidator | None = None,
    validate_parent_fd_after_mutation: ParentDirectoryValidator | None = None,
    close_fd: FileDescriptorCloser = os.close,
    errno_reader: ErrnoReader = ctypes.get_errno,
) -> None:
    renameatx_np = _macos_renameatx_np(operation) if rename is None else rename
    parent_opener = _open_parent_directory_fd if open_parent_fd is None else open_parent_fd
    validate_parent = _validate_parent_directory_fd if validate_parent_fd is None else validate_parent_fd
    validate_parent_after = (
        _validate_parent_directory_fd_after_mutation
        if validate_parent_fd_after_mutation is None
        else validate_parent_fd_after_mutation
    )
    source_parent_fd = parent_opener(source.parent, operation)
    try:
        destination_parent_fd = parent_opener(destination.parent, operation)
        try:
            validate_parent(source_parent_fd, source.parent, operation)
            validate_parent(destination_parent_fd, destination.parent, operation)
            result = int(
                renameatx_np(
                    source_parent_fd,
                    os.fsencode(source.name),
                    destination_parent_fd,
                    os.fsencode(destination.name),
                    _RENAME_EXCL,
                )
            )
            _raise_posix_rename_error_if_failed(
                result,
                errno_reader,
                destination=destination,
                operation=operation,
                primitive="renameatx_np(RENAME_EXCL)",
            )
            validate_parent_after(source_parent_fd, source.parent, operation)
            validate_parent_after(destination_parent_fd, destination.parent, operation)
        finally:
            close_fd(destination_parent_fd)
    finally:
        close_fd(source_parent_fd)


def _windows_no_replace_rename(
    source: Path,
    destination: Path,
    operation: str,
    *,
    kernel32: Any | None = None,
    ntdll: Any | None = None,
    last_error_reader: WindowsLastErrorReader | None = None,
    availability_check: Callable[[], bool] | None = None,
) -> None:
    _require_leaf_name(source)
    _require_leaf_name(destination)
    check_available = windows_no_replace_rename_available if availability_check is None else availability_check
    if not check_available():
        raise _unsupported_no_replace_rename(operation, "Windows handle rename API is unavailable")
    win32 = _kernel32() if kernel32 is None else kernel32
    native_api = _ntdll() if ntdll is None else ntdll
    read_last_error = _ctypes_last_error if last_error_reader is None else last_error_reader
    source_handle = _open_windows_source_handle(win32, source, operation=operation, read_last_error=read_last_error)
    try:
        destination_parent_handle = _open_windows_destination_parent_handle(
            win32,
            destination.parent,
            operation=operation,
            read_last_error=read_last_error,
        )
        try:
            rename_info, rename_info_size = _file_rename_info(destination.name, destination_parent_handle)
            io_status = _IoStatusBlock()
            # Win32 FileRenameInfo rejects non-null RootDirectory on current Windows;
            # the native file-information call supports the destination parent handle.
            status = int(
                native_api.NtSetInformationFile(
                    source_handle,
                    ctypes.byref(io_status),
                    ctypes.pointer(rename_info),
                    rename_info_size,
                    _FILE_RENAME_INFORMATION,
                )
            )
            if _ntstatus_code(status) == _STATUS_SUCCESS:
                return
            _raise_windows_ntstatus_rename_error(
                status,
                destination,
                operation=f"{operation} no-replace rename",
            )
        finally:
            _close_windows_handle(win32, destination_parent_handle)
    finally:
        _close_windows_handle(win32, source_handle)


def windows_no_replace_rename_available() -> bool:
    return hasattr(ctypes, "WinDLL")


def _linux_renameat2_syscall() -> LinuxRenameAt2Syscall:
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.argtypes = [ctypes.c_long]  # Fixed argument; the remaining syscall arguments are variadic.
    syscall.restype = ctypes.c_long
    return cast(LinuxRenameAt2Syscall, syscall)


def _macos_renameatx_np(operation: str) -> MacOSRenameAtxNp:
    libc = ctypes.CDLL(None, use_errno=True)
    renameatx_np = getattr(libc, "renameatx_np", None)
    if renameatx_np is None:
        raise _unsupported_no_replace_rename(operation, "renameatx_np is unavailable")
    renameatx_np.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameatx_np.restype = ctypes.c_int
    return cast(MacOSRenameAtxNp, renameatx_np)


def _open_parent_directory_fd(path: Path, operation: str) -> int:
    descriptor = _open_directory_for_mutation(
        path,
        operation=operation,
        needs_replace=False,
        platform_support=_DEFAULT_PLATFORM_SUPPORT,
    )
    try:
        _validate_parent_directory_fd(descriptor, path, operation)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validate_parent_directory_fd(descriptor: int, path: Path, operation: str) -> None:
    try:
        _ensure_directory_descriptor_matches_path(descriptor, path, operation=operation)
    except OSError as exc:
        raise _unsupported_no_replace_rename(
            operation,
            f"parent directory could not be opened descriptor-relative: {path}",
        ) from exc


def _validate_parent_directory_fd_after_mutation(descriptor: int, path: Path, operation: str) -> None:
    _ensure_directory_descriptor_matches_path_after_mutation(descriptor, path, operation=operation)


def _raise_posix_rename_error_if_failed(
    result: int,
    errno_reader: ErrnoReader,
    *,
    destination: Path,
    operation: str,
    primitive: str,
) -> None:
    if result == 0:
        return
    error = errno_reader()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), destination) from None
    if error in _UNSUPPORTED_RENAME_ERRORS:
        error_name = errno.errorcode.get(error, str(error))
        raise _unsupported_no_replace_rename(operation, f"{primitive} failed with {error_name}")
    raise OSError(error, os.strerror(error), destination) from None


def _kernel32() -> Any:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise _unsupported_no_replace_rename("rename", "Windows handle rename API is unavailable")
    kernel32 = win_dll("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _ntdll() -> Any:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise _unsupported_no_replace_rename("rename", "Windows native rename API is unavailable")
    ntdll = win_dll("ntdll")
    ntdll.NtSetInformationFile.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.ULONG,
    ]
    ntdll.NtSetInformationFile.restype = ctypes.c_long
    return ntdll


def _open_windows_source_handle(
    kernel32: Any,
    path: Path,
    *,
    operation: str,
    read_last_error: WindowsLastErrorReader,
) -> int:
    return _open_windows_handle(
        kernel32,
        path,
        desired_access=_DELETE | _FILE_READ_ATTRIBUTES,
        flags=_FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        operation=f"{operation} open source",
        read_last_error=read_last_error,
    )


def _open_windows_destination_parent_handle(
    kernel32: Any,
    path: Path,
    *,
    operation: str,
    read_last_error: WindowsLastErrorReader,
) -> int:
    return _open_windows_handle(
        kernel32,
        path,
        desired_access=_FILE_READ_ATTRIBUTES | _FILE_TRAVERSE,
        flags=_FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        operation=f"{operation} open destination parent",
        read_last_error=read_last_error,
    )


def _open_windows_handle(
    kernel32: Any,
    path: Path,
    *,
    desired_access: int,
    flags: int,
    operation: str,
    read_last_error: WindowsLastErrorReader,
) -> int:
    require_no_nul(path)
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        flags,
        None,
    )
    if _is_invalid_handle(handle):
        _raise_windows_rename_error(read_last_error(), path, operation=operation)
    return _handle_value(handle)


def _file_rename_info(file_name: str, root_directory_handle: int) -> tuple[Any, int]:
    require_no_nul(file_name)
    # WCHAR counts UTF-16 code units, not Python characters. Preserve lone
    # surrogates too: Windows filenames can contain them.
    encoded_name = file_name.encode("utf-16-le", errors="surrogatepass")
    wchar_count = len(encoded_name) // 2 + 1
    name_type = ctypes.c_uint16 * wchar_count

    class _FileRenameInfo(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("ReplaceIfExists", wintypes.BOOLEAN),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", name_type),
        ]

    rename_info = _FileRenameInfo()
    rename_info.ReplaceIfExists = False
    rename_info.RootDirectory = root_directory_handle
    rename_info.FileNameLength = len(encoded_name)
    rename_info.FileName = name_type.from_buffer_copy(encoded_name + b"\0\0")
    return rename_info, ctypes.sizeof(rename_info)


def _close_windows_handle(kernel32: Any, handle: int) -> None:
    kernel32.CloseHandle(handle)


def _raise_windows_rename_error(error: int, path: Path, *, operation: str) -> None:
    message = f"{operation} failed: {_format_windows_error(error)}"
    if error in {_ERROR_FILE_EXISTS, _ERROR_ALREADY_EXISTS, _ERROR_DIR_NOT_EMPTY}:
        raise FileExistsError(error, message, str(path))
    if error in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
        raise FileNotFoundError(error, message, str(path))
    if error == _ERROR_ACCESS_DENIED:
        raise PermissionError(error, message, str(path))
    raise OSError(error, message, str(path))


def _raise_windows_ntstatus_rename_error(status: int, path: Path, *, operation: str) -> None:
    code = _ntstatus_code(status)
    message = f"{operation} failed with NTSTATUS 0x{code:08X}"
    if code in {_STATUS_OBJECT_NAME_COLLISION, _STATUS_DIRECTORY_NOT_EMPTY}:
        raise FileExistsError(code, message, str(path))
    if code in {_STATUS_OBJECT_NAME_NOT_FOUND, _STATUS_OBJECT_PATH_NOT_FOUND}:
        raise FileNotFoundError(code, message, str(path))
    if code == _STATUS_ACCESS_DENIED:
        raise PermissionError(code, message, str(path))
    raise OSError(code, message, str(path))


def _ntstatus_code(status: int) -> int:
    return status & 0xFFFFFFFF


def _ctypes_last_error() -> int:
    get_last_error = getattr(ctypes, "get_last_error", None)
    if get_last_error is None:
        return 0
    return int(get_last_error())


def _format_windows_error(error: int) -> str:
    format_error = getattr(ctypes, "FormatError", None)
    if format_error is None:
        return f"Windows error {error}"
    try:
        return str(format_error(error))
    except OSError:
        return f"Windows error {error}"


def _is_invalid_handle(handle: object) -> bool:
    return _handle_value(handle) == _INVALID_HANDLE_VALUE_INT


def _handle_value(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    raise TypeError(f"expected Windows handle value, got {type(handle).__name__}")


def _renameat2_syscall_number() -> int | None:
    return _RENAMEAT2_SYSCALLS.get(platform.machine().lower())


def _require_leaf_name(path: Path) -> None:
    require_no_nul(path)
    if path.name in {"", os.curdir, os.pardir}:
        raise ValueError(f"rename path must have a leaf name: {path}")


def _unsupported_no_replace_rename(operation: str, reason: str) -> UnsupportedFilesystemMutationError:
    return UnsupportedFilesystemMutationError(f"{operation} refused because no-replace rename is unavailable: {reason}")


def _unsupported_no_replace_directory_rename(operation: str, reason: str) -> UnsupportedFilesystemMutationError:
    return UnsupportedFilesystemMutationError(
        f"{operation} refused because no-replace directory rename is unavailable: {reason}"
    )


__all__ = ["DirectoryNoReplaceRename", "rename_directory_no_replace", "rename_path_no_replace"]
