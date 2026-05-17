from __future__ import annotations

from pathlib import Path

import pytest
from mutation_helpers import (
    _CallRecorder,
    _DirectoryFsyncRecorder,
    _FailingFsync,
    _FileFsyncRecorder,
    _path_identity,
    _rename_only_platform_support,
    requires_descriptor_relative_rename_replace_fallback,
    requires_descriptor_relative_replace,
)

from safe_fs_ops.filesystem_ops import AtomicWriteHooks, DurabilityMode, atomic_write_bytes, atomic_write_text
from safe_fs_ops.filesystem_ops.mutations import (
    _durability_wrapped_fsync,
    _fsync_directory_descriptor,
    _fsync_file,
)

pytestmark = pytest.mark.safe_fs_ops


def test_fsync_helpers_propagate_os_errors() -> None:
    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    with pytest.raises(OSError, match="injected fsync failure"):
        _fsync_file(0, _fsync=fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        _fsync_directory_descriptor(0, _fsync=fail_fsync)


def test_durability_wrapped_fsync_rejects_invalid_mode() -> None:
    with pytest.raises(TypeError, match="durability must be a DurabilityMode"):
        _durability_wrapped_fsync("sync", _fsync_file)


@requires_descriptor_relative_rename_replace_fallback
def test_atomic_write_bytes_falls_back_to_same_directory_rename_when_replace_dir_fd_is_missing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config.bin"
    target.write_bytes(b"old bytes")

    atomic_write_bytes(
        target,
        b"new bytes",
        _platform_support=_rename_only_platform_support(),
    )

    assert target.read_bytes() == b"new bytes"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_bytes_leaves_final_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.bin"

    atomic_write_bytes(target, b"new bytes")

    assert target.read_bytes() == b"new bytes"
    assert list(target.parent.glob("*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_fsyncs_file_after_mode_changes(tmp_path: Path) -> None:
    target = tmp_path / "config.bin"
    fsyncs = _FileFsyncRecorder()

    atomic_write_bytes(target, b"new bytes", permissions=0o640, _file_fsync=fsyncs)

    assert target.read_bytes() == b"new bytes"
    assert fsyncs.modes == [0o640]


@requires_descriptor_relative_replace
def test_atomic_write_reports_file_fsync_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.bin"

    def fail_file_fsync(_descriptor: int) -> None:
        raise OSError("file fsync failed")

    with pytest.raises(OSError, match="file fsync failed"):
        atomic_write_bytes(
            target,
            b"new bytes",
            durability=DurabilityMode.FSYNC,
            _file_fsync=fail_file_fsync,
        )

    assert target.exists() is False
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


@requires_descriptor_relative_replace
def test_atomic_write_reports_directory_fsync_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.bin"
    calls = 0

    def fail_directory_fsync(_descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")

    with pytest.raises(OSError, match="directory fsync failed"):
        atomic_write_bytes(
            target,
            b"new bytes",
            durability=DurabilityMode.FSYNC,
            _directory_fsync=fail_directory_fsync,
        )

    assert target.read_bytes() == b"new bytes"


def test_atomic_write_rejects_invalid_durability_before_side_effects(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "config.bin"

    with pytest.raises(TypeError, match="durability must be a DurabilityMode"):
        atomic_write_bytes(target, b"new bytes", durability=object())

    assert target.exists() is False
    assert (tmp_path / "nested").exists() is False


@requires_descriptor_relative_replace
def test_atomic_write_none_skips_fsync_hooks(tmp_path: Path) -> None:
    target = tmp_path / "config.bin"
    file_fsync = _CallRecorder()
    directory_fsync = _CallRecorder()

    atomic_write_bytes(
        target,
        b"new bytes",
        durability=DurabilityMode.NONE,
        _file_fsync=file_fsync,
        _directory_fsync=directory_fsync,
    )

    assert target.read_bytes() == b"new bytes"
    assert file_fsync.calls == 0
    assert directory_fsync.calls == 0


@requires_descriptor_relative_replace
def test_atomic_write_best_effort_swallows_fsync_failures(tmp_path: Path) -> None:
    target = tmp_path / "config.bin"
    file_fsync = _FailingFsync("file fsync failed")
    directory_fsync = _FailingFsync("directory fsync failed")

    atomic_write_bytes(
        target,
        b"new bytes",
        durability=DurabilityMode.BEST_EFFORT,
        _file_fsync=file_fsync,
        _directory_fsync=directory_fsync,
    )

    assert target.read_bytes() == b"new bytes"
    assert file_fsync.calls == 1
    assert directory_fsync.calls >= 1


@requires_descriptor_relative_replace
def test_atomic_write_fsyncs_each_created_parent_directory_entry(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "settings" / "nested" / "state" / "config.bin"
    fsyncs = _DirectoryFsyncRecorder()

    atomic_write_bytes(target, b"new bytes", _directory_fsync=fsyncs)

    assert target.read_bytes() == b"new bytes"
    assert fsyncs.identities[:3] == [
        _path_identity(workspace),
        _path_identity(workspace / "settings"),
        _path_identity(workspace / "settings" / "nested"),
    ]


@requires_descriptor_relative_replace
def test_atomic_write_text_preserves_newlines_when_translation_disabled(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"

    atomic_write_text(target, "one\ntwo\n", newline="")

    assert target.read_bytes() == b"one\ntwo\n"


@requires_descriptor_relative_replace
def test_atomic_write_cleans_temp_file_on_injected_replace_failure(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("old", encoding="utf-8")

    def fail_before_replace(source: Path, destination: Path) -> None:
        assert source.exists()
        assert destination == target
        raise OSError("injected replace failure")

    with pytest.raises(OSError, match="injected replace failure"):
        atomic_write_bytes(target, b"new", hooks=AtomicWriteHooks(before_replace=fail_before_replace))

    assert target.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
