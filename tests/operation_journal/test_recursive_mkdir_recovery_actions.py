from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from journaled_filesystem_helpers import _portable_snapshot as portable_snapshot
from recovery_runner_helpers import fake_identity_remove_directory
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
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import remove_created_directory_recovery_action

pytestmark = pytest.mark.safe_fs_ops


def test_remove_created_directory_recovery_action_refuses_missing_ownership_proof(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()

    action = _recovery_action(
        payload={
            "path": str(target),
            "step_resource_key": directory_resource_key(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
        }
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="ownership proof is missing"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            lambda path, *, expected_identity: None,
            portable_snapshot,
            None,
            action,
        )

    assert target.is_dir()


def test_remove_created_directory_recovery_action_deletes_matching_empty_directory(
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
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )

    remove_calls: list[tuple[Path, DirectoryIdentity]] = []

    def track_identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        remove_calls.append((Path(path), expected_identity))

    result = remove_created_directory_recovery_action(
        lambda path, *, missing_ok=False: Path(path).rmdir(),
        track_identity_remove,
        portable_snapshot,
        context,
        action,
    )

    assert result["removed"] is True
    assert result["backend_guarantee"] == "identity_conditional_remove"
    assert remove_calls == [(target, DirectoryIdentity.from_stat(target.stat()))]


def test_remove_created_directory_recovery_action_passes_expected_identity_to_identity_remover(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    expected_identity = DirectoryIdentity.from_stat(target.stat())
    context = _verified_recovery_context(target)
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
    calls: list[tuple[Path, DirectoryIdentity]] = []

    def identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        calls.append((Path(path), expected_identity))
        Path(path).rmdir()

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


def test_remove_created_directory_recovery_action_refuses_without_authority_before_mutation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    context = replace(_verified_recovery_context(target), recovery_authority=None)
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
    remove_calls: list[Path] = []

    def identity_remove(path: Path | str, *, expected_identity: DirectoryIdentity) -> None:
        del expected_identity
        remove_calls.append(Path(path))
        Path(path).rmdir()

    with pytest.raises(RecoveryActionManualInterventionRequired, match="active recovery authority") as raised:
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            identity_remove,
            portable_snapshot,
            context,
            action,
        )

    assert raised.value.payload["reason_code"] == "missing_recovery_authority"
    assert remove_calls == []
    assert target.is_dir()


def test_remove_created_directory_recovery_action_requires_empty_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    (target / "note.txt").write_text("tampered\n", encoding="utf-8")
    context = _verified_recovery_context(target)
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

    with pytest.raises(RecoveryActionManualInterventionRequired, match="not empty"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            fake_identity_remove_directory,
            portable_snapshot,
            context,
            action,
        )

    assert target.is_dir()


def test_remove_created_directory_recovery_action_requires_manual_for_replaced_directory_identity(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    recorded_identity = _directory_identity_payload(target)
    context = _verified_recovery_context(target)
    _replace_directory_or_skip_if_identity_is_reused(target, recorded_identity)

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

    with pytest.raises(RecoveryActionManualInterventionRequired, match="identity changed"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            fake_identity_remove_directory,
            portable_snapshot,
            context,
            action,
        )

    assert target.is_dir()


def test_remove_created_directory_recovery_action_does_not_call_generic_remover_after_identity_precheck_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    recorded_identity = _directory_identity_payload(target)
    context = _verified_recovery_context(target)
    _replace_directory_or_skip_if_identity_is_reused(target, recorded_identity)
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
    remove_calls: list[Path] = []

    def generic_remove(path: Path | str, *, missing_ok: bool = False) -> None:
        del missing_ok
        remove_calls.append(Path(path))
        Path(path).rmdir()

    with pytest.raises(RecoveryActionManualInterventionRequired, match="identity changed"):
        remove_created_directory_recovery_action(
            generic_remove,
            None,
            portable_snapshot,
            context,
            action,
        )

    assert remove_calls == []
    assert target.is_dir()


def test_remove_created_directory_recovery_action_requires_manual_without_identity_remover(
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
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )
    remove_calls: list[Path] = []

    def generic_remove(path: Path | str, *, missing_ok: bool = False) -> None:
        del missing_ok
        remove_calls.append(Path(path))
        Path(path).rmdir()

    with pytest.raises(RecoveryActionManualInterventionRequired, match="no identity-conditional directory remover"):
        remove_created_directory_recovery_action(
            generic_remove,
            None,
            portable_snapshot,
            context,
            action,
        )

    assert remove_calls == []
    assert target.is_dir()


def test_remove_created_directory_recovery_action_refuses_hook_capable_generic_remover_without_identity_remover(
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
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_conditional_remove",
        }
    )
    remove_calls: list[Path] = []

    def hook_capable_remove(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
        del missing_ok
        target_path = Path(path)
        remove_calls.append(target_path)
        if hooks is not None:
            hooks.after_rmdir_validation(target_path)
        target_path.rmdir()

    with pytest.raises(RecoveryActionManualInterventionRequired, match="no identity-conditional directory remover"):
        remove_created_directory_recovery_action(
            hook_capable_remove,
            None,
            portable_snapshot,
            context,
            action,
        )

    assert remove_calls == []
    assert target.is_dir()


def test_remove_created_directory_recovery_action_rejects_legacy_prechecked_backend_guarantee(
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
            "journaled_creation_proof": _journaled_creation_proof_payload(target),
            "cleanup_policy": "owned_empty_directory_safe_ish",
            "backend_guarantee": "identity_checked_before_name_remove",
        }
    )
    with pytest.raises(RecoveryActionManualInterventionRequired, match="backend guarantee is missing or unsupported"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            None,
            portable_snapshot,
            context,
            action,
        )

    assert target.is_dir()


def _replace_directory_or_skip_if_identity_is_reused(
    path: Path,
    recorded_identity: dict[str, object],
) -> None:
    path.rmdir()
    path.mkdir()
    current_identity = DirectoryIdentity.from_stat(path.stat())
    if (
        recorded_identity.get("device") == current_identity.device
        and recorded_identity.get("inode") == current_identity.inode
    ):
        pytest.skip("filesystem reused directory identity for remove/recreate")


def test_remove_created_directory_recovery_action_skips_missing_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state"
    target.mkdir()
    context = _verified_recovery_context(target)
    recorded_identity = _directory_identity_payload(target)
    target.rmdir()
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

    with pytest.raises(RecoveryActionSkipped, match="already absent"):
        remove_created_directory_recovery_action(
            lambda path, *, missing_ok=False: Path(path).rmdir(),
            None,
            portable_snapshot,
            context,
            action,
        )
