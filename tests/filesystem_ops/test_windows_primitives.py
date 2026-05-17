from __future__ import annotations

from ctypes import wintypes
from pathlib import Path
from typing import Any, ClassVar

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError, UnsupportedFilesystemMutationError
from safe_fs_ops.filesystem_ops._windows_primitives import (
    _FILE_ATTRIBUTE_REPARSE_POINT,
    _INVALID_FILE_ATTRIBUTES,
    _MOVEFILE_REPLACE_EXISTING,
    _MOVEFILE_WRITE_THROUGH,
    ReplaceFilePartialFailureError,
    WindowsFilePrimitiveApi,
    _configure_kernel32,
    flush_file_descriptor_windows,
    get_file_attributes_windows,
    is_windows_reparse_point,
    replace_file_windows,
)

pytestmark = pytest.mark.safe_fs_ops


def test_replace_file_windows_uses_replacefilew_when_target_exists() -> None:
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(attributes={str(destination): 0x80})
    api = _api(kernel32)

    replace_file_windows(source, destination, api=api, write_through=True)

    assert kernel32.replace_calls == [
        (
            str(destination),
            str(source),
            None,
            0,
            None,
            None,
        )
    ]
    assert kernel32.move_calls == []


@pytest.mark.parametrize("error_code", [1175, 1176, 1177])
def test_replace_file_windows_reports_replacefile_partial_failures(error_code: int) -> None:
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(
        attributes={str(destination): 0x80},
        replace_result=False,
        replace_error=error_code,
    )
    api = _api(kernel32)

    with pytest.raises(ReplaceFilePartialFailureError) as exc_info:
        replace_file_windows(source, destination, api=api)

    assert exc_info.value.errno == error_code
    assert exc_info.value.replacement_path == source
    assert exc_info.value.destination_path == destination


def test_replace_file_windows_rejects_existing_reparse_target_before_replacefilew() -> None:
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(attributes={str(destination): _FILE_ATTRIBUTE_REPARSE_POINT})
    api = _api(kernel32)

    with pytest.raises(UnsafePathError, match="Windows reparse point"):
        replace_file_windows(source, destination, api=api)

    assert kernel32.replace_calls == []
    assert kernel32.move_calls == []


def test_replace_file_windows_uses_movefileexw_when_target_is_missing() -> None:
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(attributes={}, last_error=2)
    api = _api(kernel32)

    replace_file_windows(source, destination, api=api)

    assert kernel32.replace_calls == []
    assert kernel32.move_calls == [
        (
            str(source),
            str(destination),
            _MOVEFILE_REPLACE_EXISTING | _MOVEFILE_WRITE_THROUGH,
        )
    ]


def test_replace_file_windows_can_fallback_to_os_replace_without_write_through_when_movefileexw_is_unavailable() -> (
    None
):
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(attributes={}, last_error=2, move_file_ex_available=False)
    api = _api(kernel32)
    fallback = _OsReplaceRecorder()

    replace_file_windows(source, destination, api=api, write_through=False, os_replace=fallback)

    assert fallback.calls == [(source, destination)]
    assert kernel32.replace_calls == []
    assert kernel32.move_calls == []


def test_replace_file_windows_rejects_write_through_fallback_when_movefileexw_is_unavailable() -> None:
    source = Path("C:/workspace/.config.tmp")
    destination = Path("C:/workspace/config.txt")
    kernel32 = _Kernel32Fake(attributes={}, last_error=2, move_file_ex_available=False)
    api = _api(kernel32)
    fallback = _OsReplaceRecorder()

    with pytest.raises(UnsupportedFilesystemMutationError, match="MoveFileExW is unavailable"):
        replace_file_windows(source, destination, api=api, write_through=True, os_replace=fallback)

    assert fallback.calls == []
    assert kernel32.replace_calls == []
    assert kernel32.move_calls == []


def test_replace_file_windows_does_not_fallback_for_movefileex_access_denied() -> None:
    kernel32 = _Kernel32Fake(attributes={}, last_error=2, move_result=False, move_error=5)
    api = _api(kernel32)
    fallback = _OsReplaceRecorder()

    with pytest.raises(PermissionError, match="move replacement file failed"):
        replace_file_windows(
            Path("C:/workspace/.config.tmp"),
            Path("C:/workspace/config.txt"),
            api=api,
            os_replace=fallback,
        )

    assert fallback.calls == []


def test_get_file_attributes_maps_missing_path_to_file_not_found() -> None:
    kernel32 = _Kernel32Fake(attributes={}, last_error=2)
    api = _api(kernel32)

    with pytest.raises(FileNotFoundError):
        get_file_attributes_windows(Path("C:/workspace/missing.txt"), api=api)


def test_is_windows_reparse_point_detects_reparse_attribute_and_treats_missing_as_false() -> None:
    reparse = Path("C:/workspace/link")
    plain = Path("C:/workspace/plain")
    missing = Path("C:/workspace/missing")
    kernel32 = _Kernel32Fake(
        attributes={
            str(reparse): _FILE_ATTRIBUTE_REPARSE_POINT,
            str(plain): 0x80,
        },
        last_error=2,
    )
    api = _api(kernel32)

    assert is_windows_reparse_point(reparse, api=api) is True
    assert is_windows_reparse_point(plain, api=api) is False
    assert is_windows_reparse_point(missing, api=api) is False


def test_flush_file_descriptor_windows_uses_os_handle() -> None:
    kernel32 = _Kernel32Fake(attributes={})
    api = _api(kernel32)

    flush_file_descriptor_windows(
        7,
        api=api,
        get_osfhandle=lambda descriptor: descriptor + 100,
        path=Path("C:/workspace/config.txt"),
    )

    assert kernel32.flushed_handles == [107]


def test_flush_file_descriptor_windows_maps_access_denied() -> None:
    kernel32 = _Kernel32Fake(attributes={}, flush_result=False, flush_error=5)
    api = _api(kernel32)

    with pytest.raises(PermissionError, match="flush file failed"):
        flush_file_descriptor_windows(
            7,
            api=api,
            get_osfhandle=lambda descriptor: descriptor + 100,
            path=Path("C:/workspace/config.txt"),
        )


def test_flush_file_descriptor_windows_reports_missing_msvcrt_when_no_handle_converter() -> None:
    kernel32 = _Kernel32Fake(attributes={})
    api = _api(kernel32)

    with pytest.raises(UnsupportedFilesystemMutationError, match="msvcrt is missing"):
        flush_file_descriptor_windows(
            7,
            api=api,
            _msvcrt_loader=lambda _name: (_ for _ in ()).throw(ImportError("no msvcrt")),
        )


def test_configure_kernel32_sets_ctypes_signatures() -> None:
    kernel32 = _Kernel32ForConfiguration()

    _configure_kernel32(kernel32)

    assert kernel32.GetFileAttributesW.argtypes == [wintypes.LPCWSTR]
    assert kernel32.GetFileAttributesW.restype is wintypes.DWORD
    assert kernel32.ReplaceFileW.argtypes == [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    assert kernel32.ReplaceFileW.restype is wintypes.BOOL
    assert kernel32.MoveFileExW.argtypes == [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    assert kernel32.MoveFileExW.restype is wintypes.BOOL
    assert kernel32.FlushFileBuffers.argtypes == [wintypes.HANDLE]
    assert kernel32.FlushFileBuffers.restype is wintypes.BOOL


def _api(kernel32: _Kernel32Fake) -> WindowsFilePrimitiveApi:
    return WindowsFilePrimitiveApi(
        kernel32=kernel32,
        last_error_reader=kernel32.get_last_error,
        error_formatter=lambda error: f"error {error}",
    )


class _Kernel32Fake:
    def __init__(
        self,
        *,
        attributes: dict[str, int],
        last_error: int = 0,
        move_result: bool = True,
        move_error: int = 0,
        move_file_ex_available: bool = True,
        replace_result: bool = True,
        replace_error: int = 0,
        flush_result: bool = True,
        flush_error: int = 0,
    ) -> None:
        self.attributes = attributes
        self.last_error = last_error
        self.move_result = move_result
        self.move_error = move_error
        self.move_file_ex_available = move_file_ex_available
        self.replace_result = replace_result
        self.replace_error = replace_error
        self.flush_result = flush_result
        self.flush_error = flush_error
        self.replace_calls: list[tuple[str, str, None, int, None, None]] = []
        self.move_calls: list[tuple[str, str, int]] = []
        self.flushed_handles: list[int] = []

    def GetFileAttributesW(self, file_name: str) -> int:
        try:
            return self.attributes[file_name]
        except KeyError:
            return _INVALID_FILE_ATTRIBUTES

    def ReplaceFileW(
        self,
        replaced_file_name: str,
        replacement_file_name: str,
        backup_file_name: None,
        replace_flags: int,
        exclude: None,
        reserved: None,
    ) -> bool:
        self.replace_calls.append(
            (
                replaced_file_name,
                replacement_file_name,
                backup_file_name,
                replace_flags,
                exclude,
                reserved,
            )
        )
        if not self.replace_result:
            self.last_error = self.replace_error
        return self.replace_result

    def MoveFileExW(self, existing_file_name: str, new_file_name: str, flags: int) -> bool:
        if not self.move_file_ex_available:
            self.last_error = 120
            return False
        self.move_calls.append((existing_file_name, new_file_name, flags))
        if not self.move_result:
            self.last_error = self.move_error
        return self.move_result

    def FlushFileBuffers(self, handle: int) -> bool:
        self.flushed_handles.append(handle)
        if not self.flush_result:
            self.last_error = self.flush_error
        return self.flush_result

    def get_last_error(self) -> int:
        return self.last_error


class _OsReplaceRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, source: Path, destination: Path) -> None:
        self.calls.append((source, destination))


class _CtypesFunction:
    argtypes: ClassVar[list[Any] | None] = None
    restype: ClassVar[Any | None] = None

    def __call__(self, *_args: object) -> int:
        return 1


class _Kernel32ForConfiguration:
    def __init__(self) -> None:
        self.GetFileAttributesW = _CtypesFunction()
        self.ReplaceFileW = _CtypesFunction()
        self.MoveFileExW = _CtypesFunction()
        self.FlushFileBuffers = _CtypesFunction()
