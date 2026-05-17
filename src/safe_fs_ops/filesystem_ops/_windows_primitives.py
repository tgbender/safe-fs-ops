from __future__ import annotations

import ctypes
import importlib
import os
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from safe_fs_ops.filesystem_ops.mutation_support import UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops.paths import UnsafePathError

_ERROR_FILE_NOT_FOUND = 2
_ERROR_PATH_NOT_FOUND = 3
_ERROR_ACCESS_DENIED = 5
_ERROR_CALL_NOT_IMPLEMENTED = 120
_ERROR_INVALID_FUNCTION = 1
_ERROR_UNABLE_TO_REMOVE_REPLACED = 1175
_ERROR_UNABLE_TO_MOVE_REPLACEMENT = 1176
_ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 = 1177

_INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_REPLACEFILE_REQUIRE_METADATA_MERGE = 0x00000000
_REPLACEFILE_PARTIAL_FAILURE_ERRORS = {
    _ERROR_UNABLE_TO_REMOVE_REPLACED,
    _ERROR_UNABLE_TO_MOVE_REPLACEMENT,
    _ERROR_UNABLE_TO_MOVE_REPLACEMENT_2,
}
_MOVEFILE_REPLACE_EXISTING = 0x00000001
_MOVEFILE_WRITE_THROUGH = 0x00000008


class _Kernel32(Protocol):
    GetFileAttributesW: Any
    ReplaceFileW: Any
    MoveFileExW: Any
    FlushFileBuffers: Any


LastErrorReader = Callable[[], int]
ErrorFormatter = Callable[[int], str]
GetOsfHandle = Callable[[int], int]
OsReplace = Callable[[Path, Path], None]


def _ctypes_last_error() -> int:
    get_last_error = getattr(ctypes, "get_last_error", None)
    if get_last_error is None:
        return 0
    return int(get_last_error())


def _ctypes_format_error(error: int) -> str:
    format_error = getattr(ctypes, "FormatError", None)
    if format_error is None:
        return f"Windows error {error}"
    return str(format_error(error))


@dataclass(frozen=True, slots=True)
class WindowsFilePrimitiveApi:
    kernel32: _Kernel32
    last_error_reader: LastErrorReader = _ctypes_last_error
    error_formatter: ErrorFormatter = _ctypes_format_error

    @classmethod
    def load(cls) -> WindowsFilePrimitiveApi:
        win_dll = getattr(ctypes, "WinDLL", None)
        if win_dll is None:
            raise UnsupportedFilesystemMutationError(
                "Windows filesystem primitives are unavailable because ctypes.WinDLL is missing"
            )
        kernel32 = win_dll("kernel32", use_last_error=True)
        _configure_kernel32(kernel32)
        return cls(kernel32=kernel32)

    def get_file_attributes(self, path: Path) -> int:
        attributes = int(self.kernel32.GetFileAttributesW(str(path)))
        if attributes == _INVALID_FILE_ATTRIBUTES:
            _raise_windows_error(self, path=path, operation="read file attributes")
        return attributes

    def is_reparse_point(self, path: Path) -> bool:
        try:
            attributes = self.get_file_attributes(path)
        except FileNotFoundError:
            return False
        return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)

    def replace_existing_file(self, source: Path, destination: Path) -> None:
        ok = self.kernel32.ReplaceFileW(
            str(destination),
            str(source),
            None,
            # Fail closed if Windows cannot merge attributes, streams, or ACLs
            # from the replaced file onto the replacement.
            _REPLACEFILE_REQUIRE_METADATA_MERGE,
            None,
            None,
        )
        if not ok:
            error = self.last_error_reader()
            if error in _REPLACEFILE_PARTIAL_FAILURE_ERRORS:
                raise _replace_file_partial_failure_error(self, error, source=source, destination=destination)
            _raise_windows_error(self, path=destination, operation="replace existing file", error=error)

    def move_file_replace(self, source: Path, destination: Path, *, write_through: bool) -> None:
        move_file_ex = getattr(self.kernel32, "MoveFileExW", None)
        if move_file_ex is None:
            raise UnsupportedFilesystemMutationError("MoveFileExW is unavailable")
        flags = _MOVEFILE_REPLACE_EXISTING
        if write_through:
            flags |= _MOVEFILE_WRITE_THROUGH
        if move_file_ex(str(source), str(destination), flags):
            return
        error = self.last_error_reader()
        if error in {_ERROR_CALL_NOT_IMPLEMENTED, _ERROR_INVALID_FUNCTION}:
            raise UnsupportedFilesystemMutationError(
                f"MoveFileExW is unavailable: {_format_windows_error(self, error)}"
            )
        _raise_windows_error(self, path=destination, operation="move replacement file", error=error)

    def flush_file_handle(self, handle: int, *, path: Path | None, operation: str) -> None:
        if self.kernel32.FlushFileBuffers(handle):
            return
        _raise_windows_error(self, path=path, operation=operation)


def windows_file_primitives_available() -> bool:
    return getattr(ctypes, "WinDLL", None) is not None


def default_windows_file_api() -> WindowsFilePrimitiveApi:
    return WindowsFilePrimitiveApi.load()


def get_file_attributes_windows(
    path: Path | str,
    *,
    api: WindowsFilePrimitiveApi | None = None,
) -> int:
    return (api or default_windows_file_api()).get_file_attributes(Path(path))


def is_windows_reparse_point(
    path: Path | str,
    *,
    api: WindowsFilePrimitiveApi | None = None,
) -> bool:
    return (api or default_windows_file_api()).is_reparse_point(Path(path))


def replace_file_windows(
    source: Path | str,
    destination: Path | str,
    *,
    api: WindowsFilePrimitiveApi | None = None,
    write_through: bool = True,
    os_replace: OsReplace = os.replace,
) -> None:
    """Replace a file using Windows-native primitives.

    ``write_through`` only applies to the missing-target ``MoveFileExW`` path.
    Existing targets use ``ReplaceFileW`` without ``REPLACEFILE_WRITE_THROUGH``
    because Windows documents that flag as unsupported for ReplaceFile.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    primitive_api = api or default_windows_file_api()
    try:
        destination_attributes = primitive_api.get_file_attributes(destination_path)
    except FileNotFoundError:
        _move_missing_target(
            source_path,
            destination_path,
            api=primitive_api,
            write_through=write_through,
            os_replace=os_replace,
        )
        return
    if destination_attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
        raise UnsafePathError(f"replace file refused for Windows reparse point: {destination_path}")
    primitive_api.replace_existing_file(source_path, destination_path)


def flush_file_descriptor_windows(
    file_descriptor: int,
    *,
    api: WindowsFilePrimitiveApi | None = None,
    get_osfhandle: GetOsfHandle | None = None,
    path: Path | str | None = None,
    operation: str = "flush file",
    _msvcrt_loader: Callable[[str], Any] = importlib.import_module,
) -> None:
    if get_osfhandle is None:
        try:
            get_osfhandle = _msvcrt_loader("msvcrt").get_osfhandle
        except ImportError as exc:
            raise UnsupportedFilesystemMutationError(
                "Windows file handle flush is unavailable because msvcrt is missing"
            ) from exc
    handle = int(get_osfhandle(file_descriptor))
    flush_file_handle_windows(handle, api=api, path=path, operation=operation)


def flush_file_handle_windows(
    handle: int,
    *,
    api: WindowsFilePrimitiveApi | None = None,
    path: Path | str | None = None,
    operation: str = "flush file handle",
) -> None:
    checked_path = None if path is None else Path(path)
    (api or default_windows_file_api()).flush_file_handle(handle, path=checked_path, operation=operation)


def _move_missing_target(
    source: Path,
    destination: Path,
    *,
    api: WindowsFilePrimitiveApi,
    write_through: bool,
    os_replace: OsReplace,
) -> None:
    try:
        api.move_file_replace(source, destination, write_through=write_through)
    except UnsupportedFilesystemMutationError:
        if write_through:
            raise
        os_replace(source, destination)


def _configure_kernel32(kernel32: Any) -> None:
    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD
    kernel32.ReplaceFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    kernel32.ReplaceFileW.restype = wintypes.BOOL
    kernel32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.MoveFileExW.restype = wintypes.BOOL
    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL


def _raise_windows_error(
    api: WindowsFilePrimitiveApi,
    *,
    path: Path | None,
    operation: str,
    error: int | None = None,
) -> None:
    error_code = api.last_error_reader() if error is None else error
    filename = None if path is None else str(path)
    message = f"{operation} failed: {_format_windows_error(api, error_code)}"
    if error_code in {_ERROR_FILE_NOT_FOUND, _ERROR_PATH_NOT_FOUND}:
        raise FileNotFoundError(error_code, message, filename)
    if error_code == _ERROR_ACCESS_DENIED:
        raise PermissionError(error_code, message, filename)
    raise OSError(error_code, message, filename)


class ReplaceFilePartialFailureError(OSError):
    """Raised when ReplaceFileW reports a partial, state-ambiguous failure."""

    def __init__(
        self,
        error_code: int,
        message: str,
        *,
        replacement_path: Path,
        destination_path: Path,
    ) -> None:
        super().__init__(error_code, message, str(destination_path))
        self.replacement_path = replacement_path
        self.destination_path = destination_path


def _replace_file_partial_failure_error(
    api: WindowsFilePrimitiveApi,
    error: int,
    *,
    source: Path,
    destination: Path,
) -> ReplaceFilePartialFailureError:
    message = f"replace existing file failed with partial file state: {_format_windows_error(api, error)}"
    return ReplaceFilePartialFailureError(
        error,
        message,
        replacement_path=source,
        destination_path=destination,
    )


def _format_windows_error(api: WindowsFilePrimitiveApi, error: int) -> str:
    if error:
        try:
            return api.error_formatter(error)
        except OSError:
            pass
    return f"Windows error {error}"


__all__ = [
    "ReplaceFilePartialFailureError",
    "WindowsFilePrimitiveApi",
    "default_windows_file_api",
    "flush_file_descriptor_windows",
    "flush_file_handle_windows",
    "get_file_attributes_windows",
    "is_windows_reparse_point",
    "replace_file_windows",
    "windows_file_primitives_available",
]
