from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_IMODE, S_ISDIR, S_ISLNK, S_ISREG

import pytest

from safe_fs_ops.filesystem_ops import (
    BackupContentMismatchError,
    DurabilityMode,
    ResourceSnapshot,
    RestoreConflictError,
    capture_backup,
    restore_backup,
)

pytestmark = pytest.mark.safe_fs_ops


@dataclass
class _InjectedBackend:
    snapshot_calls: list[Path] = field(default_factory=list)
    read_calls: list[Path] = field(default_factory=list)
    write_calls: list[tuple[Path, bytes, int | None, DurabilityMode]] = field(default_factory=list)
    delete_calls: list[tuple[Path, DurabilityMode]] = field(default_factory=list)

    def snapshot(self, path: Path) -> ResourceSnapshot:
        self.snapshot_calls.append(path)
        return _snapshot_for_test(path)

    def read_backup_content(
        self,
        path: Path,
        *,
        expected_snapshot: ResourceSnapshot,
    ) -> tuple[bytes, int | None]:
        self.read_calls.append(path)
        if expected_snapshot.file_type != "file":
            raise ValueError("expected_snapshot must describe a regular file")
        content = path.read_bytes()
        return content, S_IMODE(path.lstat().st_mode)

    def write_backup_content(
        self,
        path: Path,
        content: bytes,
        *,
        permissions: int | None,
        durability: DurabilityMode,
    ) -> None:
        self.write_calls.append((path, content, permissions, durability))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        if permissions is not None:
            path.chmod(permissions)

    def delete_backup_target(self, path: Path, *, durability: DurabilityMode) -> None:
        self.delete_calls.append((path, durability))
        path.unlink()

    def current_permissions(self, path: Path) -> int | None:
        return S_IMODE(path.lstat().st_mode)


def test_restore_missing_backup_removes_created_file_with_injected_backend(tmp_path: Path) -> None:
    backend = _InjectedBackend()
    target = tmp_path / "created.txt"

    backup = capture_backup(target, _snapshot_reader=backend.snapshot)
    target.write_text("created\n", encoding="utf-8")
    expected_current = backend.snapshot(target)

    restored = restore_backup(
        backup,
        expected_current=expected_current,
        durability=DurabilityMode.NONE,
        _snapshot_reader=backend.snapshot,
        _backup_file_deleter=backend.delete_backup_target,
    )

    assert restored.file_type == "missing"
    assert target.exists() is False
    assert backend.delete_calls == [(target.resolve(strict=False), DurabilityMode.NONE)]


def test_restore_overwritten_file_uses_injected_writer(tmp_path: Path) -> None:
    backend = _InjectedBackend()
    target = tmp_path / "config.txt"
    target.write_bytes(b"original\n")
    target.chmod(0o640)

    backup = capture_backup(
        target,
        _snapshot_reader=backend.snapshot,
        _backup_content_reader=backend.read_backup_content,
    )
    target.write_bytes(b"mutated\n")
    expected_current = backend.snapshot(target)

    restored = restore_backup(
        backup,
        expected_current=expected_current,
        durability=DurabilityMode.NONE,
        _snapshot_reader=backend.snapshot,
        _backup_content_reader=backend.read_backup_content,
        _backup_content_writer=backend.write_backup_content,
        _current_permissions_reader=backend.current_permissions,
    )

    assert restored.file_type == "file"
    assert target.read_bytes() == b"original\n"
    assert backup.permissions is not None
    assert backend.write_calls == [
        (target.resolve(strict=False), b"original\n", backup.permissions, DurabilityMode.NONE)
    ]


def test_restore_deleted_file_requires_expected_current_with_injected_backend(tmp_path: Path) -> None:
    backend = _InjectedBackend()
    target = tmp_path / "config.txt"
    target.write_bytes(b"original\n")
    target.chmod(0o640)

    backup = capture_backup(
        target,
        _snapshot_reader=backend.snapshot,
        _backup_content_reader=backend.read_backup_content,
    )
    target.write_bytes(b"mutated\n")
    expected_current = backend.snapshot(target)
    target.unlink()

    with pytest.raises(RestoreConflictError, match="current state changed unexpectedly"):
        restore_backup(
            backup,
            expected_current=expected_current,
            durability=DurabilityMode.NONE,
            _snapshot_reader=backend.snapshot,
            _backup_content_reader=backend.read_backup_content,
            _backup_content_writer=backend.write_backup_content,
            _current_permissions_reader=backend.current_permissions,
        )

    assert target.exists() is False
    assert backend.write_calls == []


def test_restore_rejects_expected_current_mismatch_with_injected_backend(tmp_path: Path) -> None:
    backend = _InjectedBackend()
    target = tmp_path / "config.txt"
    target.write_text("original\n", encoding="utf-8")

    backup = capture_backup(
        target,
        _snapshot_reader=backend.snapshot,
        _backup_content_reader=backend.read_backup_content,
    )
    target.write_text("first mutation\n", encoding="utf-8")
    expected_current = backend.snapshot(target)
    target.write_text("second mutation\n", encoding="utf-8")

    with pytest.raises(RestoreConflictError, match="current state changed unexpectedly"):
        restore_backup(
            backup,
            expected_current=expected_current,
            durability=DurabilityMode.NONE,
            _snapshot_reader=backend.snapshot,
            _backup_content_reader=backend.read_backup_content,
            _backup_content_writer=backend.write_backup_content,
            _current_permissions_reader=backend.current_permissions,
        )

    assert backend.write_calls == []


@pytest.mark.parametrize("artifact_state", ["missing", "tampered"])
def test_restore_rejects_missing_or_tampered_backup_artifact(
    tmp_path: Path,
    artifact_state: str,
) -> None:
    backend = _InjectedBackend()
    target = tmp_path / "config.txt"
    content_path = tmp_path / "backup-content.bin"
    target.write_text("original\n", encoding="utf-8")
    target.chmod(0o640)

    backup = capture_backup(
        target,
        content_path=content_path,
        durability=DurabilityMode.NONE,
        _snapshot_reader=backend.snapshot,
        _backup_content_reader=backend.read_backup_content,
        _backup_content_writer=backend.write_backup_content,
    )
    target.write_text("mutated\n", encoding="utf-8")

    if artifact_state == "missing":
        content_path.unlink()
        error_match = "not a regular file"
    else:
        content_path.write_text("tampered\n", encoding="utf-8")
        error_match = "hash mismatch"

    with pytest.raises(BackupContentMismatchError, match=error_match):
        restore_backup(
            backup,
            expected_current=backend.snapshot(target),
            durability=DurabilityMode.NONE,
            _snapshot_reader=backend.snapshot,
            _backup_content_reader=backend.read_backup_content,
            _backup_content_writer=backend.write_backup_content,
            _current_permissions_reader=backend.current_permissions,
        )


def _snapshot_for_test(path: Path) -> ResourceSnapshot:
    target = path.resolve(strict=False)
    try:
        stat_result = target.lstat()
    except FileNotFoundError:
        return ResourceSnapshot(
            path=target,
            exists=False,
            file_type="missing",
            content_hash=None,
            size=None,
            mtime_ns=None,
            symlink_target=None,
            device=None,
            inode=None,
        )

    if S_ISREG(stat_result.st_mode):
        content = target.read_bytes()
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="file",
            content_hash=hashlib.sha256(content).hexdigest(),
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=None,
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
        )
    if S_ISDIR(stat_result.st_mode):
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="directory",
            content_hash=None,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=None,
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
        )
    if S_ISLNK(stat_result.st_mode):
        return ResourceSnapshot(
            path=target,
            exists=True,
            file_type="symlink",
            content_hash=None,
            size=stat_result.st_size,
            mtime_ns=stat_result.st_mtime_ns,
            symlink_target=str(target.readlink()),
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
        )
    return ResourceSnapshot(
        path=target,
        exists=True,
        file_type="other",
        content_hash=None,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        symlink_target=None,
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
    )
