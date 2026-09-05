import os
import sys
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import capture_directory_to_quarantine, restore_captured_directory
from safe_fs_ops.filesystem_ops.darwin_xattrs import DarwinXattrs
from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token

pytestmark = [pytest.mark.platform_macos, pytest.mark.skipif(sys.platform != "darwin", reason="native Darwin ABI")]


def test_darwin_tag_is_persistent_and_exclusive(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    identity = source.stat()
    assert directory_capture_token(source, device=identity.st_dev, inode=identity.st_ino) is None
    record = capture_directory_to_quarantine(source, quarantine_path=tmp_path / "quarantine")
    assert len(record.capture_token) == 32
    assert list(record.quarantine_path.iterdir()) == []
    assert (
        directory_capture_token(record.quarantine_path, device=identity.st_dev, inode=identity.st_ino, create=True)
        == record.capture_token
    )
    (record.quarantine_path / "child").write_text("preserve")
    restore_captured_directory(record)
    assert (source / "child").read_text() == "preserve"
    assert directory_capture_token(source, device=identity.st_dev, inode=identity.st_ino) == record.capture_token


def test_darwin_xattr_create_does_not_overwrite(tmp_path: Path):
    api = DarwinXattrs()
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        api.set(descriptor, "user.safe_fs_ops.capture", b"a" * 32, 2)
        with pytest.raises(FileExistsError):
            api.set(descriptor, "user.safe_fs_ops.capture", b"b" * 32, 2)
        assert api.get(descriptor, "user.safe_fs_ops.capture") == b"a" * 32
        with pytest.raises(OSError):
            api.get(-1, "user.safe_fs_ops.capture")
    finally:
        os.close(descriptor)
