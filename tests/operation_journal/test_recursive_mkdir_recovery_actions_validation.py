from __future__ import annotations

from pathlib import Path

import pytest
from journaled_filesystem_helpers import _portable_snapshot as portable_snapshot
from recursive_mkdir_recovery_actions_helpers import (
    directory_identity_payload as _directory_identity_payload,
)
from recursive_mkdir_recovery_actions_helpers import (
    journaled_creation_proof_payload as _journaled_creation_proof_payload,
)
from recursive_mkdir_recovery_actions_helpers import (
    recovery_action as _recovery_action,
)
from recursive_mkdir_recovery_actions_helpers import (
    verified_recovery_context as _verified_recovery_context,
)

from safe_fs_ops.filesystem_ops import DirectoryIdentity
from safe_fs_ops.operation_journal import directory_resource_key
from safe_fs_ops.operation_journal.recovery_runner import RecoveryActionManualInterventionRequired
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import remove_created_directory_recovery_action

pytestmark = pytest.mark.safe_fs_ops


def test_remove_created_directory_recovery_action_requires_matching_identity_payload_path_and_resource_key(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    context = _verified_recovery_context(target)
    bad_identity = {
        **_directory_identity_payload(target),
        "path": str(tmp_path / "other"),
        "resource_key": directory_resource_key(tmp_path / "other"),
    }
    action = _recovery_action(
        payload={
            "path": str(target),
            "step_resource_key": directory_resource_key(target),
            "ownership_class": "created_by_transaction",
            "created_directory_identity": bad_identity,
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="ownership proof did not match"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            None,
            portable_snapshot,
            context,
            action,
        )


def test_remove_created_directory_recovery_action_requires_matching_journaled_proof_path_and_resource_key(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    context = _verified_recovery_context(target)
    action = _recovery_action(
        payload={
            "path": str(target),
            "step_resource_key": directory_resource_key(target),
            "ownership_class": "created_by_transaction",
            "created_directory_identity": _directory_identity_payload(target),
            "journaled_creation_proof": {
                **_journaled_creation_proof_payload(target),
                "resource_key": directory_resource_key(tmp_path / "other"),
            },
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="journaled ownership proof did not match"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            None,
            portable_snapshot,
            context,
            action,
        )


def test_remove_created_directory_recovery_action_requires_context_for_delete_capable_payload(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    action = _recovery_action(
        payload={
            "path": str(target),
            "step_resource_key": directory_resource_key(target),
            "ownership_class": "created_by_transaction",
            "created_directory_identity": _directory_identity_payload(target),
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="journaled ownership proof did not match"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            None,
            portable_snapshot,
            None,
            action,
        )


def test_remove_created_directory_recovery_action_uses_identity_conditional_remover_when_available(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    recorded_identity = _directory_identity_payload(target)
    expected_identity = DirectoryIdentity.from_stat(target.stat())
    context = _verified_recovery_context(target)
    action = _recovery_action(
        payload={
            "path": str(target),
            "step_resource_key": directory_resource_key(target),
            "ownership_class": "created_by_transaction",
            "created_directory_identity": recorded_identity,
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )
    calls: list[tuple[Path, DirectoryIdentity]] = []

    def identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        target_path = Path(path)
        calls.append((target_path, expected_identity))
        assert expected_identity == DirectoryIdentity(
            device=int(recorded_identity["device"]),
            inode=int(recorded_identity["inode"]),
        )
        target_path.rmdir()

    result = remove_created_directory_recovery_action(
        lambda path, *, missing_ok=False: Path(path).rmdir(),
        identity_remove,
        portable_snapshot,
        context,
        action,
    )

    assert result["removed"] is True
    assert calls == [(target, expected_identity)]
    assert target.exists() is False
