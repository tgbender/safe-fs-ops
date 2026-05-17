from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from recovery_runner_helpers import _acquire_and_claim, _coordinator, fake_identity_remove_directory
from recursive_mkdir_recovery_helpers import (
    _after_directory_checkpoint_payload,
    _context_with_recovery_payload,
    _directory_identity_payload,
    _journaled_creation_proof_payload,
    _latest_recovery_payload,
    _start_recursive_mkdir_recovery_batch,
    portable_snapshot,
)

from safe_fs_ops.filesystem_ops.backend import capabilities_from_platform_support
from safe_fs_ops.filesystem_ops.mutation_support import _MutationPlatformSupport
from safe_fs_ops.operation_journal import OperationJournalStore, directory_resource_key
from safe_fs_ops.operation_journal.filesystem_support import JournaledFilesystemRecoveryError
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    REMOVE_CREATED_DIRECTORY_ACTION,
    recursive_mkdir_recovery_action_plans,
)
from safe_fs_ops.workspace import SafeWorkspace

pytestmark = pytest.mark.safe_fs_ops


def _hook_capable_remove_directory(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
    del missing_ok
    target = Path(path)
    if hooks is not None:
        hooks.after_rmdir_validation(target)
    target.rmdir()


def test_recursive_mkdir_cleanup_deletes_only_transaction_owned_directories_with_matching_identity(
    tmp_path: Path,
) -> None:
    journal, lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    coordinator = _coordinator(
        tmp_path / "state.db",
        journal=journal,
        remove_directory_operation=_hook_capable_remove_directory,
        identity_remove_directory_operation=fake_identity_remove_directory,
        snapshot=portable_snapshot,
    )

    results = coordinator.run_recovery_actions(
        journal.read_recovery_context(batch_id),
        lease=lease,
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
    )

    actions = journal.list_recovery_actions(batch_id)
    assert results[-1].status == "succeeded"
    assert created_parent.exists() is False
    assert created_root.exists() is False
    assert [record.status for record in actions] == [
        "planned",
        "planned",
        "attempting",
        "succeeded",
        "attempting",
        "succeeded",
    ]


def test_plan_recovery_actions_adds_missing_recursive_cleanup_alongside_existing_actions(
    tmp_path: Path,
) -> None:
    journal, lease, batch_id, _created_root, _created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    coordinator = _coordinator(tmp_path / "state.db", journal=journal)
    context = journal.read_recovery_context(batch_id)
    recovery_attempt_id = context.recovery_attempt_id
    assert recovery_attempt_id is not None

    journal.record_recovery_action_planned(
        batch_id=batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type="custom_success",
        resource_key=context.batch.resource_key,
        payload={"path": str(tmp_path / "other.txt")},
        action_id="action-custom",
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=2),
    )
    expected_cleanup_plans = recursive_mkdir_recovery_action_plans(journal.read_recovery_context(batch_id))
    first_cleanup = expected_cleanup_plans[0]
    journal.record_recovery_action_planned(
        batch_id=batch_id,
        lease=lease,
        recovery_attempt_id=recovery_attempt_id,
        action_type=str(first_cleanup["action_type"]),
        resource_key=context.batch.resource_key,
        payload=cast(Mapping[str, Any], first_cleanup["payload"]),
        action_id=str(first_cleanup["action_id"]),
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=3),
    )

    updated_context = coordinator._plan_recovery_actions(  # pyright: ignore[reportPrivateUsage]
        journal.read_recovery_context(batch_id),
        lease=lease,
        now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
    )

    latest_actions = {
        action.action_id: action for action in updated_context.recovery_actions if action.status == "planned"
    }
    expected_action_ids = {"action-custom"}
    expected_action_ids.update(str(plan["action_id"]) for plan in expected_cleanup_plans)
    assert set(latest_actions) == expected_action_ids
    first_cleanup_payload = cast(Mapping[str, Any], expected_cleanup_plans[0]["payload"])
    second_cleanup_payload = cast(Mapping[str, Any], expected_cleanup_plans[1]["payload"])
    assert latest_actions[str(expected_cleanup_plans[0]["action_id"])].action_type == REMOVE_CREATED_DIRECTORY_ACTION
    assert latest_actions[str(expected_cleanup_plans[0]["action_id"])].payload["step_resource_key"] == str(
        first_cleanup_payload["step_resource_key"]
    )
    assert latest_actions[str(expected_cleanup_plans[1]["action_id"])].payload["step_resource_key"] == str(
        second_cleanup_payload["step_resource_key"]
    )


def test_recursive_mkdir_cleanup_requires_identity_remove_even_when_generic_remover_accepts_hooks(
    tmp_path: Path,
) -> None:
    journal, lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    remove_calls: list[Path] = []

    def generic_remove(path: Path | str, *, missing_ok: bool = False, hooks: object = None) -> None:
        del missing_ok
        remove_calls.append(Path(path))
        if hooks is not None:
            hooks.after_rmdir_validation(Path(path))
        Path(path).rmdir()

    coordinator = _coordinator(
        tmp_path / "state.db",
        journal=journal,
        remove_directory_operation=generic_remove,
        identity_remove_directory_operation=None,
        snapshot=portable_snapshot,
    )

    with pytest.raises(JournaledFilesystemRecoveryError, match="requires manual intervention"):
        coordinator.run_recovery_actions(
            journal.read_recovery_context(batch_id),
            lease=lease,
            now=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=1),
        )

    assert remove_calls == []
    assert created_parent.is_dir()
    assert created_root.is_dir()
    actions = journal.list_recovery_actions(batch_id)
    assert [record.status for record in actions] == [
        "planned",
        "planned",
        "attempting",
        "manual_intervention_required",
    ]


def test_backend_capabilities_include_remove_directory_and_custom_backend_keeps_it_unknown(
    tmp_path: Path,
) -> None:
    support = _MutationPlatformSupport(
        open_dir_fd=True,
        unlink_dir_fd=True,
        stat_dir_fd=True,
        stat_follow_symlinks=True,
        replace_dir_fd=True,
        rename_dir_fd=True,
        mkdir_dir_fd=True,
        rmdir_dir_fd=False,
        nofollow_directory_open=True,
    )
    capabilities = capabilities_from_platform_support(
        backend_name="posix_descriptor_relative",
        platform="posix",
        platform_support=support,
        is_default_backend=True,
    )

    assert capabilities.support_for("remove_directory") == capabilities.remove_directory
    assert capabilities.remove_directory.supported is False
    assert capabilities.remove_directory.reason is not None
    assert "rmdir_dir_fd" in capabilities.remove_directory.reason

    workspace = SafeWorkspace.open(
        tmp_path / "custom-state.db",
        owner="owner-a",
        remove_directory_operation=lambda path, *, missing_ok=False: Path(path).rmdir(),
    )
    target = tmp_path / "custom-dir"
    target.mkdir()

    assert workspace.filesystem_backend.remove_directory.state == "unknown"
    workspace._remove_directory(target)
    assert target.exists() is False


def test_recursive_mkdir_cleanup_does_not_plan_preexisting_or_unknown_entries(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.db"
    root = tmp_path / "config"
    created = root / "state"
    leaf = created / "cache"
    root.mkdir()
    created.mkdir()
    leaf.mkdir()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    lease = _acquire_and_claim(state_path, directory_resource_key(leaf), now=now)
    journal = OperationJournalStore(state_path)
    batch = journal.create_batch(
        idempotency_key=f"recover:{directory_resource_key(leaf)}",
        lease=lease,
        owner="owner-a",
        run_id="run-1",
        resource_key=directory_resource_key(leaf),
        claim_owner="owner-a",
        payload={"operation": "make_directory"},
        batch_id="batch-1",
        now=now,
    )
    _, leaf_operation = journal.start_batch_operation(
        batch.batch_id,
        lease=lease,
        operation_type="make_directory",
        resource_key=directory_resource_key(leaf),
        payload={
            "operation": "make_directory",
            "path": str(leaf),
            "resource_key": directory_resource_key(leaf),
        },
        operation_id="operation-created-3",
        now=now,
    )
    journal.record_checkpoint(
        batch.batch_id,
        lease=lease,
        operation_id=leaf_operation.operation_id,
        resource_key=directory_resource_key(leaf),
        checkpoint_type="after",
        checkpoint_id="checkpoint-created-3",
        payload=_after_directory_checkpoint_payload(leaf),
        now=now + timedelta(microseconds=1),
    )
    journal.mark_failed(
        batch.batch_id,
        lease=lease,
        error="mkdir failed",
        observed_state={"resource_key": directory_resource_key(leaf)},
        now=now + timedelta(microseconds=2),
    )
    journal.record_recovery_desired(
        batch.batch_id,
        lease=lease,
        reason="filesystem mutation failed",
        payload={
            "recursive_group": {
                "prior_created_steps": [
                    {
                        "step_index": 1,
                        "step_path": str(root),
                        "step_resource_key": directory_resource_key(root),
                        "ownership_class": "preexisting_external",
                        "created_directory_identity": _directory_identity_payload(root),
                        "journaled_creation_proof": _journaled_creation_proof_payload(root, sequence=1),
                    },
                    {
                        "step_index": 2,
                        "step_path": str(created),
                        "step_resource_key": directory_resource_key(created),
                        "ownership_class": "unknown",
                        "created_directory_identity": _directory_identity_payload(created),
                        "journaled_creation_proof": _journaled_creation_proof_payload(created, sequence=2),
                    },
                    {
                        "step_index": 3,
                        "step_path": str(leaf),
                        "step_resource_key": directory_resource_key(leaf),
                        "ownership_class": "created_by_transaction",
                        "created_directory_identity": _directory_identity_payload(leaf),
                        "journaled_creation_proof": _journaled_creation_proof_payload(leaf, sequence=3),
                        "cleanup_policy": "owned_empty_directory_safe_ish",
                        "backend_guarantee": "identity_conditional_remove",
                    },
                ]
            }
        },
        recovery_id="recovery-desired",
        now=now + timedelta(microseconds=3),
    )
    journal.start_recovery(
        batch.batch_id,
        lease=lease,
        reason="recovery reserved",
        recovery_id="recovery-attempt",
        now=now + timedelta(microseconds=4),
    )

    actions = recursive_mkdir_recovery_action_plans(journal.read_recovery_context(batch.batch_id))
    assert len(actions) == 1
    assert cast(Mapping[str, Any], actions[0]["payload"])["path"] == str(leaf)


def test_recursive_mkdir_cleanup_requires_explicit_policy_backend_and_journaled_proof(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, _created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0].pop("cleanup_policy")
    steps[1].pop("journaled_creation_proof")

    actions = recursive_mkdir_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == "manual_intervention_required"
    assert cast(Mapping[str, Any], actions[0]["payload"])["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"
    assert created_parent.is_dir()


def test_recursive_mkdir_cleanup_does_not_plan_mismatched_identity_or_journaled_proof_payload(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, _created_root, _created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0]["created_directory_identity"] = {
        **cast(dict[str, object], steps[0]["created_directory_identity"]),
        "resource_key": directory_resource_key(tmp_path / "other"),
    }
    steps[1]["journaled_creation_proof"] = {
        **cast(dict[str, object], steps[1]["journaled_creation_proof"]),
        "path": str(tmp_path / "wrong"),
    }

    actions = recursive_mkdir_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == "manual_intervention_required"
    assert cast(Mapping[str, Any], actions[0]["payload"])["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"


def test_recursive_mkdir_cleanup_does_not_plan_when_checkpoint_identity_disagrees_with_owned_identity(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0]["created_directory_identity"] = {
        **cast(dict[str, object], steps[0]["created_directory_identity"]),
        "inode": cast(dict[str, object], steps[0]["created_directory_identity"])["inode"] + 1,
    }

    actions = recursive_mkdir_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == "manual_intervention_required"
    assert cast(Mapping[str, Any], actions[0]["payload"])["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"
    assert created_root.is_dir()


def test_recursive_mkdir_cleanup_does_not_plan_foreign_batch_journaled_proof(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0]["journaled_creation_proof"] = {
        **cast(dict[str, object], steps[0]["journaled_creation_proof"]),
        "batch_id": "foreign-batch",
    }

    actions = recursive_mkdir_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == "manual_intervention_required"
    assert cast(Mapping[str, Any], actions[0]["payload"])["reason_code"] == "incomplete_recursive_mkdir_cleanup_proof"
    assert created_root.is_dir()


def test_recursive_mkdir_cleanup_action_payload_exposes_policy_and_backend_guarantee(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, created_root, created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)

    actions = recursive_mkdir_recovery_action_plans(journal.read_recovery_context(batch_id))

    assert [cast(Mapping[str, Any], action["payload"])["path"] for action in actions] == [
        str(created_parent),
        str(created_root),
    ]
    for action in actions:
        payload = cast(Mapping[str, Any], action["payload"])
        assert payload["ownership_class"] == "created_by_transaction"
        assert payload["cleanup_policy"] == "owned_empty_directory_safe_ish"
        assert payload["backend_guarantee"] == "identity_conditional_remove"
        created_directory_identity = cast(Mapping[str, Any], payload["created_directory_identity"])
        journaled_creation_proof = cast(Mapping[str, Any], payload["journaled_creation_proof"])
        assert isinstance(created_directory_identity["device"], int)
        assert isinstance(created_directory_identity["inode"], int)
        assert journaled_creation_proof["operation_type"] == "make_directory"
        assert journaled_creation_proof["checkpoint_type"] == "after"
