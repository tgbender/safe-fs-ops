from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest
from mutation_helpers import (
    _rename_only_platform_support,
    _unsupported_platform_support,
    requires_descriptor_relative_rmdir,
    requires_windows_identity_safe_rmdir,
)

from safe_fs_ops.filesystem_ops import (
    DirectoryIdentity,
    DurabilityMode,
    IdentitySafeRemoveDirectoryUnavailableError,
    ParentChangedAfterMutationError,
    RemoveDirectoryHooks,
    UnsafePathError,
    identity_safe_remove_directory_unavailable_reason,
    remove_empty_directory,
    remove_existing_empty_directory_by_identity,
)
from safe_fs_ops.filesystem_ops._windows_identity_rmdir import (
    _FILE_ATTRIBUTE_DIRECTORY,
    _python_stat_identity,
    remove_empty_directory_by_identity_windows,
)
from safe_fs_ops.workspace import SafeWorkspace

pytestmark = pytest.mark.safe_fs_ops


def test_identity_safe_remove_reports_unavailable_even_with_descriptor_relative_rmdir_support(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()

    with pytest.raises(IdentitySafeRemoveDirectoryUnavailableError, match="identity-conditional"):
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(target.stat()),
            _platform_support=_rename_only_platform_support(),
            _platform="linux",
        )

    assert target.is_dir()


def test_identity_safe_remove_unavailable_reason_treats_backend_windows_as_windows() -> None:
    reason = identity_safe_remove_directory_unavailable_reason(
        platform_support=_unsupported_platform_support(),
        platform="windows",
    )

    assert "Windows handle delete disposition" in reason
    assert "rmdir_dir_fd" not in reason


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_deletes_matching_existing_empty_directory(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()

    remove_existing_empty_directory_by_identity(
        target,
        expected_identity=DirectoryIdentity.from_stat(target.stat()),
        durability=DurabilityMode.NONE,
    )

    assert target.exists() is False


@requires_windows_identity_safe_rmdir
def test_default_workspace_remove_directory_uses_windows_identity_safe_primitive(
    tmp_path: Path,
) -> None:
    workspace = SafeWorkspace.open(tmp_path / "state.db", owner="owner-a", durability=DurabilityMode.NONE)
    target = tmp_path / "state"
    target.mkdir()

    assert workspace.filesystem_backend.remove_directory.supported is True

    workspace._remove_directory(target)

    assert target.exists() is False


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_cannot_delete_target_moved_outside_requested_path(
    tmp_path: Path,
) -> None:
    requested_parent = tmp_path / "requested"
    outside_parent = tmp_path / "outside"
    target = requested_parent / "state"
    moved_target = outside_parent / "state"
    requested_parent.mkdir()
    outside_parent.mkdir()
    target.mkdir()
    move_error: OSError | None = None

    def move_after_validation(directory: Path) -> None:
        nonlocal move_error
        assert directory == target
        try:
            target.rename(moved_target)
        except OSError as exc:
            move_error = exc

    operation_error: Exception | None = None
    try:
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(target.stat()),
            durability=DurabilityMode.NONE,
            hooks=RemoveDirectoryHooks(after_rmdir_validation=move_after_validation),
        )
    except (FileNotFoundError, OSError, ParentChangedAfterMutationError, UnsafePathError) as exc:
        operation_error = exc

    if move_error is None:
        assert operation_error is not None
        assert moved_target.is_dir()
        assert target.exists() is False
    else:
        assert operation_error is None
        assert target.exists() is False
        assert moved_target.exists() is False


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_succeeds_with_default_fsync_durability(tmp_path: Path) -> None:
    target = tmp_path / "state"
    target.mkdir()

    remove_existing_empty_directory_by_identity(
        target,
        expected_identity=DirectoryIdentity.from_stat(target.stat()),
    )

    assert target.exists() is False


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_refuses_identity_mismatch(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    target = tmp_path / "state"
    expected.mkdir()
    target.mkdir()

    with pytest.raises(UnsafePathError, match="directory identity changed"):
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(expected.stat()),
            durability=DurabilityMode.NONE,
        )

    assert target.is_dir()


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_refuses_to_report_success_when_path_is_replaced_after_validation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    moved_original = tmp_path / "moved-state"
    target.mkdir()
    replacement_identity: DirectoryIdentity | None = None

    def replace_after_validation(directory: Path) -> None:
        nonlocal replacement_identity
        assert directory == target
        try:
            target.rename(moved_original)
        except OSError as exc:
            pytest.skip(f"directory replacement unavailable while identity-safe handle is open: {exc}")
        target.mkdir()
        replacement_identity = DirectoryIdentity.from_stat(target.stat())

    with pytest.raises(UnsafePathError, match="path was replaced"):
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(target.stat()),
            durability=DurabilityMode.NONE,
            hooks=RemoveDirectoryHooks(after_rmdir_validation=replace_after_validation),
        )

    assert replacement_identity is not None
    assert DirectoryIdentity.from_stat(target.stat()) == replacement_identity
    assert moved_original.exists() is False


@requires_windows_identity_safe_rmdir
def test_identity_safe_remove_detects_broken_symlink_replacement_after_validation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    moved_original = tmp_path / "moved-state"
    missing_destination = tmp_path / "missing-destination"
    target.mkdir()
    symlink_created = False

    def replace_after_validation(directory: Path) -> None:
        nonlocal symlink_created
        assert directory == target
        try:
            target.rename(moved_original)
        except OSError as exc:
            pytest.skip(f"directory replacement unavailable while identity-safe handle is open: {exc}")
        try:
            os.symlink(missing_destination, target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"directory symlink replacement unavailable: {exc}")
        symlink_created = True

    with pytest.raises(UnsafePathError, match="path was replaced"):
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(target.stat()),
            durability=DurabilityMode.NONE,
            hooks=RemoveDirectoryHooks(after_rmdir_validation=replace_after_validation),
        )

    assert symlink_created is True
    assert target.exists() is False
    assert target.is_symlink() is True
    assert moved_original.exists() is False


def test_duplicate_handle_is_closed_when_open_osfhandle_raises(tmp_path: Path) -> None:
    kernel32 = _DuplicateHandleKernel32()

    def fail_open_osfhandle(_handle: int, _flags: int) -> int:
        raise OSError("open_osfhandle failed")

    with pytest.raises(OSError, match="open_osfhandle failed"):
        _python_stat_identity(kernel32, 111, path=tmp_path, open_osfhandle=fail_open_osfhandle)

    assert kernel32.closed_handles == [222]


def test_identity_safe_remove_revalidates_parent_chain_after_before_open_parent_hook(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    target = parent / "state"
    parent.mkdir()
    target.mkdir()
    original_parent = tmp_path / "original-parent"
    outside = tmp_path / "outside"
    outside.mkdir()
    primitive_called = False

    def replace_parent_with_redirect(path: Path) -> None:
        assert path == parent
        parent.rename(original_parent)
        try:
            os.symlink(outside, parent, target_is_directory=True)
        except OSError:
            parent.write_text("not a directory", encoding="utf-8")

    def fail_if_primitive_called(
        _path: Path,
        *,
        expected_identity: tuple[int, int],
        durability: DurabilityMode,
        after_identity_validation: object | None = None,
    ) -> None:
        nonlocal primitive_called
        primitive_called = True

    with pytest.raises(UnsafePathError):
        remove_existing_empty_directory_by_identity(
            target,
            expected_identity=DirectoryIdentity.from_stat(target.stat()),
            durability=DurabilityMode.NONE,
            hooks=RemoveDirectoryHooks(before_open_parent=replace_parent_with_redirect),
            _platform="win32",
            _windows_identity_safe_remove_available=lambda: True,
            _windows_remove=fail_if_primitive_called,
        )

    assert primitive_called is False
    assert (original_parent / "state").is_dir()


def test_windows_identity_safe_remove_refuses_success_when_parent_replaced_after_delete() -> None:
    kernel32 = _ParentReplacementKernel32()
    removed_checks: list[Path] = []

    def file_information(
        kernel: _ParentReplacementKernel32,
        handle: int,
        _path: Path,
    ) -> tuple[int, tuple[int, int]]:
        return _FILE_ATTRIBUTE_DIRECTORY, kernel.identities[handle]

    def record_removed_check(path: Path) -> None:
        removed_checks.append(path)

    with pytest.raises(ParentChangedAfterMutationError, match="parent path changed after mutation"):
        remove_empty_directory_by_identity_windows(
            Path("C:/workspace/parent/state"),
            expected_identity=_ParentReplacementKernel32.TARGET_IDENTITY,
            durability=DurabilityMode.FSYNC,
            _availability_check=lambda: True,
            _kernel32_factory=lambda: kernel32,
            _file_information_reader=file_information,
            _removed_checker=record_removed_check,
        )

    assert kernel32.delete_requested is True
    assert kernel32.flushed_handles == []
    assert removed_checks == []


@requires_descriptor_relative_rmdir
def test_name_based_remove_empty_directory_can_remove_replacement_after_validation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    replacement_created = False

    def replace_child_after_validation(directory: Path) -> None:
        nonlocal replacement_created
        assert directory == target
        target.rmdir()
        target.mkdir()
        replacement_created = True

    remove_empty_directory(
        target,
        hooks=RemoveDirectoryHooks(after_rmdir_validation=replace_child_after_validation),
    )

    assert replacement_created is True
    assert target.exists() is False


class _DuplicateHandleKernel32:
    def __init__(self) -> None:
        self.closed_handles: list[int] = []

    def GetCurrentProcess(self) -> int:
        return 999

    def DuplicateHandle(
        self,
        _source_process: int,
        _source_handle: int,
        _target_process: int,
        duplicated_handle_pointer: object,
        _desired_access: int,
        _inherit_handle: bool,
        _options: int,
    ) -> bool:
        ctypes.cast(duplicated_handle_pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = 222
        return True

    def CloseHandle(self, handle: object) -> bool:
        self.closed_handles.append(int(handle))
        return True


class _ParentReplacementKernel32:
    ORIGINAL_PARENT_IDENTITY = (11, 101)
    REPLACED_PARENT_IDENTITY = (11, 202)
    TARGET_IDENTITY = (11, 303)

    def __init__(self) -> None:
        self.identities: dict[int, tuple[int, int]] = {}
        self.delete_requested = False
        self.flushed_handles: list[int] = []
        self._next_handle = 1000

    def CreateFileW(
        self,
        path: str,
        _desired_access: int,
        _share_mode: int,
        _security_attributes: object,
        _creation_disposition: int,
        _flags: int,
        _template_file: object,
    ) -> int:
        self._next_handle += 1
        handle = self._next_handle
        if path.endswith("/state") or path.endswith("\\state"):
            self.identities[handle] = self.TARGET_IDENTITY
        elif self.delete_requested:
            self.identities[handle] = self.REPLACED_PARENT_IDENTITY
        else:
            self.identities[handle] = self.ORIGINAL_PARENT_IDENTITY
        return handle

    def SetFileInformationByHandle(
        self,
        _handle: int,
        _file_information_class: int,
        _file_information: object,
        _buffer_size: int,
    ) -> bool:
        self.delete_requested = True
        return True

    def FlushFileBuffers(self, handle: int) -> bool:
        self.flushed_handles.append(handle)
        return True

    def CloseHandle(self, _handle: object) -> bool:
        return True
