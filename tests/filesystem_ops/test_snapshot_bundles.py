from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import SnapshotBundle, SnapshotBundleEntry, snapshot_bundle

pytestmark = pytest.mark.safe_fs_ops


def test_snapshot_bundle_records_requested_paths_without_hashing_by_default(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_bytes(b"alpha\n")
    missing = tmp_path / "missing.txt"

    bundle = snapshot_bundle([target, missing])

    assert isinstance(bundle, SnapshotBundle)
    assert bundle.requested_paths == (target.resolve(strict=False), missing.resolve(strict=False))
    assert bundle.include_children is False
    assert bundle.hash_policy == "metadata-only"
    by_path = bundle.by_path()
    assert isinstance(by_path[target], SnapshotBundleEntry)
    assert by_path[target].requested is True
    assert by_path[target].snapshot.exists is True
    assert by_path[target].snapshot.file_type == "file"
    assert by_path[target].snapshot.content_hash is None
    assert by_path[missing].snapshot.exists is False


def test_snapshot_bundle_small_files_policy_hashes_only_files_within_limit(tmp_path: Path) -> None:
    small = tmp_path / "small.txt"
    large = tmp_path / "large.txt"
    small_content = b"small\n"
    large_content = b"larger\n"
    small.write_bytes(small_content)
    large.write_bytes(large_content)

    bundle = snapshot_bundle(
        [small, large],
        hash_policy="small-files",
        small_file_max_bytes=len(small_content),
    )

    by_path = bundle.by_path()
    assert by_path[small].snapshot.content_hash == hashlib.sha256(small_content).hexdigest()
    assert by_path[large].snapshot.content_hash is None


def test_snapshot_bundle_include_children_records_directory_tree(tmp_path: Path) -> None:
    root = tmp_path / "state"
    nested = root / "nested"
    root.mkdir()
    nested.mkdir()
    file_path = nested / "data.txt"
    file_path.write_text("payload\n", encoding="utf-8")

    bundle = snapshot_bundle(root, include_children=True, hash_policy="metadata-only")

    by_path = bundle.by_path()
    assert tuple(by_path) == (root, nested, file_path)
    assert by_path[root].requested is True
    assert by_path[nested].requested is False
    assert by_path[file_path].snapshot.file_type == "file"
    assert by_path[file_path].snapshot.content_hash is None


def test_snapshot_bundle_include_children_does_not_traverse_symlinked_directory(tmp_path: Path) -> None:
    root = tmp_path / "state"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside\n", encoding="utf-8")
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    bundle = snapshot_bundle(root, include_children=True, hash_policy="metadata-only")

    by_path = bundle.by_path()
    assert outside / "secret.txt" not in by_path
    if link in by_path:
        assert by_path[link].snapshot.file_type == "symlink"


def test_snapshot_bundle_rejects_unknown_hash_policy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported hash_policy"):
        snapshot_bundle(tmp_path / "config.txt", hash_policy="full")  # type: ignore[arg-type]


def test_snapshot_bundle_rejects_negative_small_file_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="small_file_max_bytes"):
        snapshot_bundle(tmp_path / "config.txt", small_file_max_bytes=-1)
