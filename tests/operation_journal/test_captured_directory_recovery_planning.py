from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from recursive_mkdir_recovery_helpers import (
    _captured_directory_step_payload,
    _context_with_recovery_payload,
    _latest_recovery_payload,
    _start_recursive_mkdir_recovery_batch,
)

from safe_fs_ops.filesystem_ops import capture_directory_to_quarantine
from safe_fs_ops.operation_journal.recursive_mkdir_recovery import (
    RESTORE_CAPTURED_DIRECTORY_ACTION,
    captured_directory_recovery_action_plans,
)

pytestmark = pytest.mark.safe_fs_ops


def test_captured_directory_recovery_plans_restore_when_payload_proves_transaction_capture(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, created_root, _created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    quarantine = tmp_path / ".quarantine" / "captured-root"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(created_root, quarantine_path=quarantine)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0] = _captured_directory_step_payload(captured, step_index=1)

    actions = captured_directory_recovery_action_plans(_context_with_recovery_payload(context, payload))

    assert len(actions) == 1
    assert actions[0]["action_type"] == RESTORE_CAPTURED_DIRECTORY_ACTION
    action_payload = cast(dict[str, Any], actions[0]["payload"])
    assert action_payload["ownership_class"] == "captured_by_transaction"
    assert action_payload["path"] == str(created_root)
    assert cast(dict[str, Any], action_payload["captured_directory"])["quarantine_path"] == str(quarantine)


def test_captured_directory_recovery_does_not_plan_without_explicit_capture_ownership(
    tmp_path: Path,
) -> None:
    journal, _lease, batch_id, created_root, _created_parent = _start_recursive_mkdir_recovery_batch(tmp_path)
    quarantine = tmp_path / ".quarantine" / "captured-root"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(created_root, quarantine_path=quarantine)
    context = journal.read_recovery_context(batch_id)
    payload = _latest_recovery_payload(context)
    steps = cast(list[dict[str, object]], payload["recursive_group"]["prior_created_steps"])
    steps[0] = _captured_directory_step_payload(captured, step_index=1)
    steps[0]["ownership_class"] = "preexisting_external"

    assert captured_directory_recovery_action_plans(_context_with_recovery_payload(context, payload)) == ()
