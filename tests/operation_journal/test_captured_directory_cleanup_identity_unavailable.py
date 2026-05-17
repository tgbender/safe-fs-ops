from __future__ import annotations

from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import (
    CapturedDirectoryRecord,
    DirectoryIdentity,
    IdentitySafeRemoveDirectoryUnavailableError,
    UnsupportedFilesystemMutationError,
)
from safe_fs_ops.operation_journal.captured_directory_recovery_actions import (
    CapturedDirectoryCleanupCandidate,
    execute_captured_directory_cleanup_candidate,
)

pytestmark = pytest.mark.safe_fs_ops


def test_captured_directory_cleanup_classifies_typed_identity_remove_unavailable(
    tmp_path: Path,
) -> None:
    candidate = _empty_captured_directory_candidate(tmp_path)

    def identity_remove_unavailable(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        del path, expected_identity
        raise IdentitySafeRemoveDirectoryUnavailableError("identity-safe remove unavailable")

    result = execute_captured_directory_cleanup_candidate(
        candidate,
        identity_cleanup_operation=identity_remove_unavailable,
    )

    assert result.status == "manual_intervention_required"
    assert result.reason_code == "identity_safe_remove_directory_unavailable"


def test_captured_directory_cleanup_keeps_plain_unsupported_identity_remove_unsafe(
    tmp_path: Path,
) -> None:
    candidate = _empty_captured_directory_candidate(tmp_path)

    def unsafe_identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        del path, expected_identity
        raise UnsupportedFilesystemMutationError(
            "identity-safe rmdir refused because the directory identity changed before removal"
        )

    result = execute_captured_directory_cleanup_candidate(
        candidate,
        identity_cleanup_operation=unsafe_identity_remove,
    )

    assert result.status == "manual_intervention_required"
    assert result.reason_code == "captured_directory_cleanup_unsafe"


def _empty_captured_directory_candidate(tmp_path: Path) -> CapturedDirectoryCleanupCandidate:
    original = tmp_path / "repo" / ".git"
    quarantine = tmp_path / "repo" / ".safe" / "git-captured"
    quarantine.mkdir(parents=True)
    identity = DirectoryIdentity.from_stat(quarantine.stat())
    return CapturedDirectoryCleanupCandidate(
        artifact_id="captured-directory:test",
        batch_id="batch-1",
        checkpoint_id="checkpoint-1",
        resource_key="directory:repo/.git",
        quarantine_path=quarantine,
        record=CapturedDirectoryRecord(
            original_path=original,
            quarantine_path=quarantine,
            original_identity=identity,
            captured_identity=identity,
        ),
    )
