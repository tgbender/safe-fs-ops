"""Darwin's descriptor-based extended-attribute ABI (not Linux's ABI)."""

from __future__ import annotations

import ctypes
import os


class DarwinXattrs:
    def __init__(self) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        self._get = libc.fgetxattr
        self._get.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_int,
        ]
        self._get.restype = ctypes.c_ssize_t
        self._set = libc.fsetxattr
        self._set.argtypes = self._get.argtypes
        self._set.restype = ctypes.c_int
        self._remove = libc.fremovexattr
        self._remove.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        self._remove.restype = ctypes.c_int

    def get(self, descriptor: int, name: str) -> bytes:
        # A valid tag is exactly 32 bytes. Reject oversized attributes as well.
        buffer = ctypes.create_string_buffer(33)
        count = self._get(descriptor, name.encode(), buffer, len(buffer), 0, 0)
        if count == -1:
            self._raise_errno()
        return buffer.raw[:count]

    def set(self, descriptor: int, name: str, value: bytes, flags: int) -> None:
        buffer = ctypes.create_string_buffer(value)
        if self._set(descriptor, name.encode(), buffer, len(value), 0, flags) == -1:
            self._raise_errno()

    @staticmethod
    def _raise_errno() -> None:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))

    def remove(self, descriptor: int, name: str) -> None:
        if self._remove(descriptor, name.encode(), 0) == -1:
            self._raise_errno()
