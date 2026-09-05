import os
import shutil
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    DirectoryIdentity,
    UnsafePathError,
    capture_directory_to_quarantine,
    cleanup_captured_directory,
    restore_captured_directory,
)


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
            cleanup_captured_directory(record)
    assert not source.exists()
    assert quarantine.is_dir()
    if action == "restore":
        assert (quarantine / "unrelated.txt").read_text() == "belongs to a replacement directory"
