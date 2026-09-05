import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    DirectoryIdentity,
    UnsafePathError,
    UnsupportedFilesystemMutationError,
    capture_directory_to_quarantine,
    cleanup_captured_directory,
    restore_captured_directory,
)
from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token


@pytest.mark.skipif(os.name == "nt", reason="probe for POSIX inode reuse")
@pytest.mark.parametrize("action", ["restore", "cleanup"])
def test_recovery_refuses_a_replacement_directory_with_reused_inode(tmp_path: Path, action: str):
    source = tmp_path / "original"
    quarantine = tmp_path / "quarantine"
    source.mkdir()
    (source / "original.txt").write_text("original contents")
    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    shutil.rmtree(quarantine)
    for _ in range(256):
        quarantine.mkdir()
        if DirectoryIdentity.from_stat(quarantine.stat()) == record.captured_identity:
            break
        quarantine.rmdir()
    else:
        pytest.skip("filesystem did not reuse the inode in this run")
    if action == "restore":
        (quarantine / "unrelated.txt").write_text("belongs to a replacement directory")
    with pytest.raises(UnsafePathError):
        if action == "restore":
            restore_captured_directory(record)
        else:
            cleanup_captured_directory(record, _identity_remove_directory=_remove_empty_directory)
    assert not source.exists()
    assert quarantine.is_dir()
    if action == "restore":
        assert (quarantine / "unrelated.txt").read_text() == "belongs to a replacement directory"


def _remove_empty_directory(path: Path, *, expected_identity: DirectoryIdentity) -> None:
    assert DirectoryIdentity.from_stat(path.stat()) == expected_identity
    path.rmdir()


@pytest.mark.parametrize("action", ["restore", "cleanup"])
@pytest.mark.parametrize("proof", ["missing", "different"])
def test_recovery_requires_matching_capture_token(tmp_path: Path, action: str, proof: str):
    source = tmp_path / "source"
    quarantine = tmp_path / "quarantine"
    source.mkdir()
    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    assert record.capture_token is not None
    record = replace(record, capture_token=None if proof == "missing" else "0" * 32)
    with pytest.raises(UnsafePathError, match="capture-token evidence"):
        if action == "restore":
            restore_captured_directory(record)
        else:
            cleanup_captured_directory(record, _identity_remove_directory=_remove_empty_directory)
    assert quarantine.is_dir()
    assert not source.exists()


def test_captured_directory_contents_can_change_before_restore(tmp_path: Path):
    source = tmp_path / "source"
    quarantine = tmp_path / "quarantine"
    source.mkdir()
    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    (quarantine / "new.txt").write_text("added after capture")
    restore_captured_directory(record)
    assert (source / "new.txt").read_text() == "added after capture"
    assert not quarantine.exists()


def test_capture_refuses_unavailable_capture_token_before_moving(tmp_path: Path):
    source = tmp_path / "source"
    quarantine = tmp_path / "quarantine"
    source.mkdir()
    (source / "original.txt").write_text("untouched")

    def unavailable(path: Path, *, device: int, inode: int, create: bool) -> None:
        return None

    with pytest.raises(UnsupportedFilesystemMutationError, match="capture-token evidence"):
        capture_directory_to_quarantine(source, quarantine_path=quarantine, _capture_token=unavailable)
    assert (source / "original.txt").read_text() == "untouched"
    assert not quarantine.exists()


@pytest.mark.parametrize("action", ["restore", "cleanup"])
@pytest.mark.parametrize("tag", [None, b"invalid"])
def test_recovery_never_recreates_missing_or_malformed_directory_tag(tmp_path: Path, action: str, tag: bytes | None):
    source = tmp_path / "source"
    quarantine = tmp_path / "quarantine"
    source.mkdir()
    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    if os.name == "nt":
        stream = Path(str(quarantine) + ":safe_fs_ops.capture")
        if tag is None:
            stream.unlink()
        else:
            stream.write_bytes(tag)
    elif sys.platform == "darwin":
        from safe_fs_ops.filesystem_ops.darwin_xattrs import DarwinXattrs

        api = DarwinXattrs()
        descriptor = os.open(quarantine, os.O_RDONLY)
        try:
            if tag is None:
                api.remove(descriptor, "user.safe_fs_ops.capture")
            else:
                api.set(descriptor, "user.safe_fs_ops.capture", tag, 4)
        finally:
            os.close(descriptor)
    elif tag is None:
        os.removexattr(quarantine, "user.safe_fs_ops.capture")
    else:
        os.setxattr(quarantine, "user.safe_fs_ops.capture", tag)
    with pytest.raises(UnsafePathError, match="capture-token evidence"):
        if action == "restore":
            restore_captured_directory(record)
        else:
            cleanup_captured_directory(record, _identity_remove_directory=_remove_empty_directory)
    assert quarantine.is_dir()
    assert not source.exists()
    assert (
        directory_capture_token(
            quarantine, device=record.captured_identity.device, inode=record.captured_identity.inode
        )
        is None
    )
