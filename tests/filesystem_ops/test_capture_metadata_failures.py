import errno
import os
from functools import partial
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsupportedFilesystemMutationError, capture_directory_to_quarantine
from safe_fs_ops.filesystem_ops.directory_capture_token import _read_metadata, directory_capture_token


@pytest.mark.parametrize("error", [errno.ENOTSUP, errno.EACCES, errno.EIO])
def test_metadata_error_refuses_capture_before_rename(tmp_path: Path, error: int) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_bytes(b"preserve")
    quarantine = tmp_path / "quarantine"

    def unavailable(*args, **kwargs):
        raise OSError(error, "metadata unavailable")

    with pytest.raises(UnsupportedFilesystemMutationError, match="capture-token evidence"):
        capture_directory_to_quarantine(
            source,
            quarantine_path=quarantine,
            _capture_token=partial(directory_capture_token, _metadata_reader=unavailable),
            _rename_no_replace=lambda *args, **kwargs: pytest.fail("renamed without metadata proof"),
        )
    assert (source / "data").read_bytes() == b"preserve"
    assert not quarantine.exists()


def test_metadata_flush_failure_closes_descriptor_and_refuses_capture(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data").write_bytes(b"preserve")
    quarantine = tmp_path / "quarantine"
    descriptors = []

    def failed_sync(descriptor: int) -> None:
        os.fstat(descriptor)  # Exercise an actual open stream/directory, not a fake descriptor.
        descriptors.append(descriptor)
        raise OSError(errno.EIO, "metadata flush failed")

    with pytest.raises(UnsupportedFilesystemMutationError, match="capture-token evidence"):
        capture_directory_to_quarantine(
            source,
            quarantine_path=quarantine,
            _capture_token=partial(directory_capture_token, _metadata_reader=partial(_read_metadata, sync=failed_sync)),
            _rename_no_replace=lambda *args, **kwargs: pytest.fail("renamed before durable proof"),
        )
    assert len(descriptors) == 1
    with pytest.raises(OSError) as raised:
        os.fstat(descriptors[0])
    assert raised.value.errno == errno.EBADF
    assert (source / "data").read_bytes() == b"preserve"
    assert not quarantine.exists()

    # A retry must flush existing metadata too; a failed previous sync is not proof.
    with pytest.raises(UnsupportedFilesystemMutationError, match="capture-token evidence"):
        capture_directory_to_quarantine(
            source,
            quarantine_path=quarantine,
            _capture_token=partial(directory_capture_token, _metadata_reader=partial(_read_metadata, sync=failed_sync)),
        )
    assert len(descriptors) == 2
    assert not quarantine.exists()

    # A tag may have been written, but capture did not move the directory.
    # A normal retry must produce a usable capture record and retain the data.
    record = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    assert record.capture_token
    assert (quarantine / "data").read_bytes() == b"preserve"
    assert not source.exists()
