from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from recovery_runner_helpers import _coordinator
from recursive_mkdir_recovery_helpers import _captured_directory_action_payload, _recovery_action

from safe_fs_ops.filesystem_ops import capture_directory_to_quarantine
from safe_fs_ops.operation_journal import (
    JournaledFilesystemRecoveryContext,
    OperationBatchRecord,
    RecoveryActionRecord,
    RecoveryAuthority,
    RecoveryRecord,
)
from safe_fs_ops.operation_journal.captured_directory_recovery import (
    RESTORE_CAPTURED_DIRECTORY_ACTION,
    restore_captured_directory_recovery_action,
)
from safe_fs_ops.operation_journal.recovery_runner import (
    RecoveryActionManualInterventionRequired,
    RecoveryActionSkipped,
)

pytestmark = pytest.mark.safe_fs_ops


def test_restore_captured_directory_recovery_action_restores_directory(tmp_path: Path) -> None:
    source = tmp_path / "config"
    source.mkdir()
    (source / "settings.json").write_text("{}", encoding="utf-8")
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )

    result = restore_captured_directory_recovery_action(_authorized_context(action), action)

    assert source.is_dir()
    assert quarantine.exists() is False
    assert result["restored"] is True
    assert result["path"] == str(source)


def test_restore_captured_directory_recovery_action_refuses_without_authority_before_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config"
    source.mkdir()
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="active recovery authority") as raised:
        restore_captured_directory_recovery_action(_empty_context(), action)

    assert raised.value.payload["reason_code"] == "missing_recovery_authority"
    assert source.exists() is False
    assert quarantine.is_dir()


def test_restore_captured_directory_recovery_action_refuses_without_active_action_before_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config"
    source.mkdir()
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )
    context = replace(
        _empty_context(),
        recovery_authority=RecoveryAuthority(lambda: None, recovery_attempt_id="recovery-attempt"),
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="active recovery action") as raised:
        restore_captured_directory_recovery_action(context, action)

    assert raised.value.payload["reason_code"] == "missing_recovery_action_authority"
    assert source.exists() is False
    assert quarantine.is_dir()


def test_restore_captured_directory_recovery_action_requires_manual_when_original_path_is_tampered(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config"
    source.mkdir()
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    source.mkdir()
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="manual intervention"):
        restore_captured_directory_recovery_action(_authorized_context(action), action)


def test_restore_captured_directory_recovery_action_requires_manual_when_quarantine_is_tampered(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config"
    source.mkdir()
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    quarantine.rmdir()
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )

    with pytest.raises(RecoveryActionManualInterventionRequired, match="manual intervention"):
        restore_captured_directory_recovery_action(_authorized_context(action), action)


def test_restore_captured_directory_recovery_action_skips_when_directory_is_already_restored(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config"
    source.mkdir()
    quarantine = tmp_path / ".quarantine" / "config-captured"
    quarantine.parent.mkdir()
    captured = capture_directory_to_quarantine(source, quarantine_path=quarantine)
    quarantine.rename(source)
    action = _recovery_action(
        action_type=RESTORE_CAPTURED_DIRECTORY_ACTION,
        payload=_captured_directory_action_payload(captured),
    )

    with pytest.raises(RecoveryActionSkipped, match="already restored"):
        restore_captured_directory_recovery_action(_authorized_context(action), action)


def test_journaled_filesystem_coordinator_registers_restore_captured_directory_action(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(tmp_path / "state.db")

    assert RESTORE_CAPTURED_DIRECTORY_ACTION in coordinator._recovery_action_handlers  # pyright: ignore[reportPrivateUsage]


def _empty_context() -> JournaledFilesystemRecoveryContext:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return JournaledFilesystemRecoveryContext(
        batch=cast(
            OperationBatchRecord,
            OperationBatchRecord(
                batch_id="batch-1",
                idempotency_key="recover:config",
                lease_name="workspace",
                lease_fencing_token=1,
                owner="owner-a",
                run_id="run-1",
                operation_run_id=None,
                operation_phase_id=None,
                resource_key=None,
                claim_owner=None,
                claim_scope=None,
                phase="recovering",
                payload={},
                status_message=None,
                status_payload={},
                created_at=now,
                updated_at=now,
            ),
        ),
        operations=(),
        checkpoints=(),
        recovery_records=(),
        recovery_attempt=RecoveryRecord(
            recovery_id="recovery-attempt",
            batch_id="batch-1",
            sequence=1,
            phase="recovering",
            reason="recovery reserved",
            payload={},
            created_at=now,
        ),
    )


def _authorized_context(action: RecoveryActionRecord) -> JournaledFilesystemRecoveryContext:
    return replace(
        _empty_context(),
        recovery_actions=(action,),
        recovery_authority=RecoveryAuthority(lambda: None, recovery_attempt_id="recovery-attempt"),
    )
