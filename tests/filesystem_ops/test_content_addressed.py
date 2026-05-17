from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import ContentAddressedStore, ContentRef
from safe_fs_ops.filesystem_ops.paths import UnsafePathError

pytestmark = pytest.mark.safe_fs_ops


def test_put_bytes_stores_sha256_object_and_verifies(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    content = b"backup content\n"

    ref = store.put_bytes(content)

    expected_digest = hashlib.sha256(content).hexdigest()
    assert ref == ContentRef(digest=expected_digest, size=len(content), path=ref.path)
    assert ref.path == tmp_path / "objects" / "sha256" / expected_digest[:2] / expected_digest
    assert ref.path.read_bytes() == content
    assert store.verify(ref) is True


def test_put_with_status_reports_only_actual_object_creation(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    source = tmp_path / "source.txt"
    source.write_bytes(b"shared content\n")

    first = store.put_file_with_status(source)
    second = store.put_file_with_status(source)
    third = store.put_bytes_with_status(source.read_bytes())

    assert first.created is True
    assert second.created is False
    assert third.created is False
    assert second.ref == first.ref
    assert third.ref == first.ref


def test_put_bytes_creates_multi_level_store_root_incrementally(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "state.db.artifacts" / "tree-backups" / "cas")

    ref = store.put_bytes(b"backup content\n")

    assert ref.path.is_file()
    assert store.verify(ref) is True


def test_put_file_streams_regular_file_and_rejects_non_regular_paths(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    source = tmp_path / "source.bin"
    source.write_bytes((b"0123456789abcdef" * 128) + b"\n")

    ref = store.put_file(source)

    assert ref.digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert ref.size == source.stat().st_size
    assert store.verify(ref) is True

    with pytest.raises(ValueError, match="regular files"):
        store.put_file(tmp_path)


def test_copy_to_verifies_content_and_refuses_overwrite_by_default(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    ref = store.put_bytes(b"original\n")
    destination = tmp_path / "restored.txt"

    store.copy_to(ref, destination, permissions=0o640)

    assert destination.read_bytes() == b"original\n"
    if os.name != "nt":
        assert (destination.lstat().st_mode & 0o777) == 0o640
    with pytest.raises(FileExistsError):
        store.copy_to(ref, destination)


def test_copy_to_applies_recorded_mtime_ns(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    ref = store.put_bytes(b"original\n")
    destination = tmp_path / "restored.txt"
    mtime_ns = 1_700_000_000_123_456_000

    store.copy_to(ref, destination, mtime_ns=mtime_ns)

    assert destination.lstat().st_mtime_ns == mtime_ns


def test_copy_to_rejects_tampered_object_before_destination_replace(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    ref = store.put_bytes(b"original\n")
    destination = tmp_path / "restored.txt"
    destination.write_bytes(b"keep me\n")
    ref.path.write_bytes(b"tampered\n")

    assert store.verify(ref) is False
    with pytest.raises(ValueError, match="failed verification"):
        store.copy_to(ref, destination, no_replace=False)

    assert destination.read_bytes() == b"keep me\n"


def test_copy_to_refuses_redirecting_parent_chain(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "objects")
    ref = store.put_bytes(b"original\n")
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="parent path redirects"):
        store.copy_to(ref, link_parent / "restored.txt")

    assert not (real_parent / "restored.txt").exists()


def test_store_refuses_redirecting_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-objects"
    real_root.mkdir()
    link_root = tmp_path / "objects"
    try:
        link_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(link_root)

    with pytest.raises(UnsafePathError, match="store path redirects"):
        store.put_bytes(b"original\n")

    assert list(real_root.iterdir()) == []


def test_store_refuses_redirecting_object_parent_without_mutating_target(tmp_path: Path) -> None:
    real_parent = tmp_path / "outside"
    real_parent.mkdir()
    store_root = tmp_path / "objects"
    store_root.mkdir()
    link_parent = store_root / "sha256"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(store_root)

    with pytest.raises(UnsafePathError, match="store path redirects"):
        store.put_bytes(b"x")

    assert list(real_parent.iterdir()) == []


def test_store_refuses_redirecting_digest_prefix_without_mutating_target(tmp_path: Path) -> None:
    real_parent = tmp_path / "outside"
    real_parent.mkdir()
    store_root = tmp_path / "objects"
    prefix_parent = store_root / "sha256"
    prefix_parent.mkdir(parents=True)
    link_prefix = prefix_parent / hashlib.sha256(b"x").hexdigest()[:2]
    try:
        link_prefix.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    store = ContentAddressedStore(store_root)

    with pytest.raises(UnsafePathError, match="store path redirects"):
        store.put_bytes(b"x")

    assert list(real_parent.iterdir()) == []
