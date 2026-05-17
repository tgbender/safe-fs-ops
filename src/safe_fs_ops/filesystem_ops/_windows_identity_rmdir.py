from __future__ import annotations

import ctypes
import importlib
import os
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path
from typing import Any, ClassVar, cast

from safe_fs_ops.filesystem_ops.mutation_support import (
    DurabilityMode,
    ParentChangedAfterMutationError,
    UnsupportedFilesystemMutationError,
)
from safe_fs_ops.filesystem_ops.paths import UnsafePathError

_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_INVALID_PARAMETER = 87

_GENERIC_WRITE = 0x40000000
_DELETE = 0x00010000
_FILE_READ_ATTRIBUTES = 0x0080
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_FILE_SHARE_DELETE = 0x00000004
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
assert _INVALID_HANDLE_VALUE is not None
_INVALID_HANDLE_VALUE_INT = _INVALID_HANDLE_VALUE
_DUPLICATE_SAME_ACCESS = 0x00000002

_FILE_DISPOSITION_INFO = 4
_FILE_DISPOSITION_INFO_EX = 21
_FILE_DISPOSITION_DELETE = 0x00000001
_FILE_DISPOSITION_POSIX_SEMANTICS = 0x00000002
_FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE = 0x00000010
_CTYPES_WIN_DLL_ATTRIBUTE = "WinDLL"
_CTYPES_GET_LAST_ERROR_ATTRIBUTE = "get_last_error"


class _ByHandleFileInformation(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("dwFileAttributes", wintypes.DWORD),
        ("ftCreationTime", wintypes.FILETIME),
        ("ftLastAccessTime", wintypes.FILETIME),
        ("ftLastWriteTime", wintypes.FILETIME),
        ("dwVolumeSerialNumber", wintypes.DWORD),
        ("nFileSizeHigh", wintypes.DWORD),
        ("nFileSizeLow", wintypes.DWORD),
        ("nNumberOfLinks", wintypes.DWORD),
        ("nFileIndexHigh", wintypes.DWORD),
        ("nFileIndexLow", wintypes.DWORD),
    ]


class _FileDispositionInfo(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [("DeleteFile", ctypes.c_ubyte)]


class _FileDispositionInfoEx(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [("Flags", wintypes.DWORD)]


WindowsDirectoryIdentity = tuple[int, int]
AfterIdentityValidationHook = Callable[[Path], None]
OpenOsfHandle = Callable[[int, int], int]
AvailabilityCheck = Callable[[], bool]
Kernel32Factory = Callable[[], Any]
FileInformationReader = Callable[[Any, int, Path], tuple[int, WindowsDirectoryIdentity]]
RemovedChecker = Callable[[Path], None]


def windows_identity_safe_remove_available() -> bool:
    return hasattr(ctypes, "WinDLL")


def remove_empty_directory_by_identity_windows(
    path: Path,
    *,
    expected_identity: WindowsDirectoryIdentity,
    durability: DurabilityMode,
    after_identity_validation: AfterIdentityValidationHook | None = None,
    _availability_check: AvailabilityCheck = windows_identity_safe_remove_available,
    _kernel32_factory: Kernel32Factory | None = None,
    _file_information_reader: FileInformationReader | None = None,
    _removed_checker: RemovedChecker | None = None,
) -> None:
    if not _availability_check():
        raise UnsupportedFilesystemMutationError(
            "identity-safe rmdir refused because native Windows handle delete disposition API is unavailable"
        )

    kernel32 = _kernel32() if _kernel32_factory is None else _kernel32_factory()
    file_information_reader = _file_information_reader or _read_file_information
    removed_checker = _ensure_removed_or_replaced if _removed_checker is None else _removed_checker
    parent_handle = _open_parent_handle(kernel32, path.parent, for_flush=durability is not DurabilityMode.NONE)
    try:
        parent_identity = _safe_directory_identity(
            kernel32,
            parent_handle,
            path=path.parent,
            file_information_reader=file_information_reader,
        )
        delete_requested = False
        handle = _open_directory_handle(kernel32, path)
        try:
            actual_identity = _safe_directory_identity(
                kernel32,
                handle,
                path=path,
                file_information_reader=file_information_reader,
            )
            if actual_identity != expected_identity:
                raise UnsafePathError(f"identity-safe rmdir refused because directory identity changed: {path}")
            if after_identity_validation is not None:
                after_identity_validation(path)
            _set_delete_disposition(kernel32, handle, path=path)
            delete_requested = True
        finally:
            _close_handle(kernel32, handle)

        if delete_requested:
            _ensure_parent_path_still_matches_handle(
                kernel32,
                path.parent,
                expected_identity=parent_identity,
                file_information_reader=file_information_reader,
            )
            removed_checker(path)
            _ensure_parent_path_still_matches_handle(
                kernel32,
                path.parent,
                expected_identity=parent_identity,
                file_information_reader=file_information_reader,
            )
            _flush_parent_handle_for_durability(kernel32, parent_handle, path.parent, durability=durability)
            _ensure_parent_path_still_matches_handle(
                kernel32,
                path.parent,
                expected_identity=parent_identity,
                file_information_reader=file_information_reader,
            )
    finally:
        _close_handle(kernel32, parent_handle)


def _read_file_information(
    kernel32: Any,
    handle: int,
    path: Path,
) -> tuple[int, WindowsDirectoryIdentity]:
    return _file_information(kernel32, handle, path=path)


def _kernel32() -> Any:
    win_dll = getattr(ctypes, _CTYPES_WIN_DLL_ATTRIBUTE)
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
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.DuplicateHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.DuplicateHandle.restype = wintypes.BOOL
    kernel32.SetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _open_directory_handle(kernel32: Any, path: Path) -> int:
    handle = kernel32.CreateFileW(
        str(path),
        _DELETE | _FILE_READ_ATTRIBUTES,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if _is_invalid_handle(handle):
        _raise_last_error(path=path, operation="open directory")
    return _handle_value(handle)


def _file_information(kernel32: Any, handle: int, *, path: Path) -> tuple[int, WindowsDirectoryIdentity]:
    information = _ByHandleFileInformation()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        _raise_last_error(path=path, operation="read directory identity")
    return int(information.dwFileAttributes), _python_stat_identity(kernel32, handle, path=path)


def _safe_directory_identity(
    kernel32: Any,
    handle: int,
    *,
    path: Path,
    file_information_reader: FileInformationReader,
) -> WindowsDirectoryIdentity:
    attributes, identity = file_information_reader(kernel32, handle, path)
    if not attributes & _FILE_ATTRIBUTE_DIRECTORY:
        raise UnsafePathError(f"identity-safe rmdir refused because path is not a directory: {path}")
    if attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise UnsafePathError(f"identity-safe rmdir refused for Windows reparse point: {path}")
    return identity


def _python_stat_identity(
    kernel32: Any,
    handle: int,
    *,
    path: Path,
    open_osfhandle: OpenOsfHandle | None = None,
) -> WindowsDirectoryIdentity:
    duplicated_handle = wintypes.HANDLE()
    current_process = kernel32.GetCurrentProcess()
    if not kernel32.DuplicateHandle(
        current_process,
        handle,
        current_process,
        ctypes.byref(duplicated_handle),
        0,
        False,
        _DUPLICATE_SAME_ACCESS,
    ):
        _raise_last_error(path=path, operation="duplicate directory handle")

    if open_osfhandle is None:
        msvcrt = importlib.import_module("msvcrt")
        open_osfhandle = msvcrt.open_osfhandle
    duplicated_handle_value = _handle_value(duplicated_handle)
    try:
        file_descriptor = open_osfhandle(duplicated_handle_value, os.O_RDONLY)
    except Exception:
        _close_handle(kernel32, duplicated_handle_value)
        raise
    try:
        stat_result = os.fstat(file_descriptor)
        return int(stat_result.st_dev), int(stat_result.st_ino)
    finally:
        os.close(file_descriptor)


def _set_delete_disposition(kernel32: Any, handle: int, *, path: Path) -> None:
    disposition_ex = _FileDispositionInfoEx(
        _FILE_DISPOSITION_DELETE | _FILE_DISPOSITION_POSIX_SEMANTICS | _FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE
    )
    if kernel32.SetFileInformationByHandle(
        handle,
        _FILE_DISPOSITION_INFO_EX,
        ctypes.byref(disposition_ex),
        ctypes.sizeof(disposition_ex),
    ):
        return
    if _get_last_error() != _ERROR_INVALID_PARAMETER:
        _raise_last_error(path=path, operation="mark directory for deletion")

    disposition = _FileDispositionInfo(True)
    if not kernel32.SetFileInformationByHandle(
        handle,
        _FILE_DISPOSITION_INFO,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        _raise_last_error(path=path, operation="mark directory for deletion")


def _ensure_parent_path_still_matches_handle(
    kernel32: Any,
    parent: Path,
    *,
    expected_identity: WindowsDirectoryIdentity,
    file_information_reader: FileInformationReader,
) -> None:
    try:
        current_handle = _open_parent_handle(kernel32, parent, for_flush=False)
    except FileNotFoundError as exc:
        raise ParentChangedAfterMutationError(
            f"identity-safe rmdir refused to report success because parent path changed after mutation: {parent}"
        ) from exc
    try:
        current_identity = _safe_directory_identity(
            kernel32,
            current_handle,
            path=parent,
            file_information_reader=file_information_reader,
        )
    finally:
        _close_handle(kernel32, current_handle)
    if current_identity != expected_identity:
        raise ParentChangedAfterMutationError(
            f"identity-safe rmdir refused to report success because parent path changed after mutation: {parent}"
        )


def _flush_parent_handle_for_durability(
    kernel32: Any,
    handle: int,
    parent: Path,
    *,
    durability: DurabilityMode,
) -> None:
    if durability is DurabilityMode.NONE:
        return
    if kernel32.FlushFileBuffers(handle):
        return
    if durability is DurabilityMode.FSYNC:
        _raise_last_error(path=parent, operation="flush parent directory")


def _open_parent_handle(kernel32: Any, path: Path, *, for_flush: bool) -> int:
    desired_access = _FILE_READ_ATTRIBUTES
    if for_flush:
        desired_access |= _GENERIC_WRITE
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if _is_invalid_handle(handle):
        _raise_last_error(path=path, operation="open parent directory")
    return _handle_value(handle)


def _close_handle(kernel32: Any, handle: int) -> None:
    kernel32.CloseHandle(handle)


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


def _raise_last_error(*, path: Path, operation: str) -> None:
    error = _get_last_error()
    if error in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
        raise FileNotFoundError(path)
    raise OSError(error, f"identity-safe rmdir failed to {operation}: {path}")


def _get_last_error() -> int:
    get_last_error = getattr(ctypes, _CTYPES_GET_LAST_ERROR_ATTRIBUTE)
    return cast(Callable[[], int], get_last_error)()


def _ensure_removed_or_replaced(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    else:
        raise UnsafePathError(f"identity-safe rmdir refused to report success because path was replaced: {path}")


__all__ = [
    "WindowsDirectoryIdentity",
    "remove_empty_directory_by_identity_windows",
    "windows_identity_safe_remove_available",
]
