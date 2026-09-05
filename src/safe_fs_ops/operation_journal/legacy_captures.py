"""Inventory and proof lookup for explicitly adopted legacy directory captures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safe_fs_ops.filesystem_ops.directory_capture_token import directory_capture_token
from safe_fs_ops.filesystem_ops.paths import UnsafePathError, ensure_safe_parent_chain, inspect_path
from safe_fs_ops.operation_journal.models import JournaledFilesystemRecoveryContext

APPROVAL = "legacy_capture_approval"
ADOPTION = "captured_directory_adoption"


@dataclass(frozen=True, slots=True)
class LegacyCapture:
    batch_id: str
    capture_id: str
    original_path: Path
    quarantine_path: Path
    resource_key: str
    batch_phase: str
    status: str
    observed: tuple[int, int, int, int] | None
    record_json: str


def capture_id(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_record_json(payload).encode()).hexdigest()


def _record_json(payload: Mapping[str, Any]) -> str:
    nested = dict(payload.get("captured_directory", {}))
    if nested.get("capture_token") is None:
        nested.pop("capture_token", None)
    return json.dumps(
        {
            key: nested if key == "captured_directory" else payload.get(key)
            for key in ("path", "step_resource_key", "ownership_class", "captured_directory")
        },
        sort_keys=True,
        default=dict,
        separators=(",", ":"),
    )


def adoption_payload(
    context: JournaledFilesystemRecoveryContext, identifier: str, *, checkpoint_type: str = ADOPTION
) -> Mapping[str, Any] | None:
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type == checkpoint_type and checkpoint.payload.get("capture_id") == identifier:
            return checkpoint.payload
    return None


def resolve_legacy_payload(context: JournaledFilesystemRecoveryContext, payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("captured_directory")
    if not isinstance(nested, Mapping) or nested.get("capture_token") is not None:
        return dict(payload)
    proof = adoption_payload(context, capture_id(payload))
    if proof is None:
        return dict(payload)
    return {
        **payload,
        "captured_directory": {**nested, "capture_token": proof["capture_token"]},
        "captured_directory_cleanup": "retain",
    }


def adopted_restore_plans(context: JournaledFilesystemRecoveryContext) -> tuple[dict[str, object], ...]:
    plans: list[dict[str, object]] = []
    seen: set[str] = set()
    for checkpoint in reversed(context.checkpoints):
        if checkpoint.checkpoint_type != ADOPTION:
            continue
        identifier = str(checkpoint.payload["capture_id"])
        if identifier in seen:
            continue
        seen.add(identifier)
        payload = resolve_legacy_payload(context, checkpoint.payload["legacy_payload"])
        plans.append(
            {
                "action_id": f"restore-adopted-capture:{identifier}",
                "action_type": "restore_captured_directory",
                "resource_key": context.batch.resource_key,
                "payload": payload,
            }
        )
    return tuple(plans)


def observed_directory(path: Path) -> tuple[int, int, int, int]:
    ensure_safe_parent_chain(path, operation="inspect legacy capture")
    safety = inspect_path(path)
    if not safety.is_dir or safety.is_symlink or safety.is_windows_reparse_point or safety.is_mount:
        raise UnsafePathError(f"legacy capture is not an ordinary directory: {path}")
    value = path.lstat()
    return value.st_dev, value.st_ino, value.st_ctime_ns, value.st_mtime_ns


def inventory_legacy_captures(context: JournaledFilesystemRecoveryContext) -> tuple[LegacyCapture, ...]:
    # Import locally: the recovery handler also uses proof lookup from this module.
    from safe_fs_ops.operation_journal.captured_directory_recovery_actions import (
        _captured_directory_record_from_payload,
    )

    result: dict[str, LegacyCapture] = {}
    for payload in _legacy_payloads(context):
        nested = payload["captured_directory"]
        if nested.get("capture_token") is not None:
            continue
        # Validate historical paths and identity fields, without treating a placeholder as real proof.
        try:
            record = _captured_directory_record_from_payload(
                {**payload, "captured_directory": {**nested, "capture_token": "0" * 32}}
            )
        except (ValueError, RuntimeError):
            continue
        identifier = capture_id(payload)
        if identifier in result:
            continue
        observed = None
        status = "unavailable"
        proof = adoption_payload(context, identifier)
        try:
            observed = observed_directory(record.quarantine_path)
            status = (
                "needs_confirmation"
                if observed[:2] == (record.captured_identity.device, record.captured_identity.inode)
                else "identity_changed"
            )
            if (
                proof is not None
                and status == "needs_confirmation"
                and directory_capture_token(record.quarantine_path, device=observed[0], inode=observed[1])
                == proof["capture_token"]
            ):
                status = "adopted"
        except (OSError, UnsafePathError):
            if proof is not None:
                try:
                    original = observed_directory(record.original_path)
                    if original[:2] == (record.captured_identity.device, record.captured_identity.inode) and (
                        directory_capture_token(record.original_path, device=original[0], inode=original[1])
                        == proof["capture_token"]
                    ):
                        status = "restored"
                except (OSError, UnsafePathError):
                    pass
        result[identifier] = LegacyCapture(
            context.batch.batch_id,
            identifier,
            record.original_path,
            record.quarantine_path,
            str(payload["step_resource_key"]),
            context.batch.phase,
            status,
            observed,
            _record_json(payload),
        )
    return tuple(result.values())


def _walk(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("captured_directory"), Mapping):
            yield {**value, "path": value.get("path", value.get("step_path"))}
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _walk(child)


def _legacy_payloads(context: JournaledFilesystemRecoveryContext) -> Iterator[Mapping[str, Any]]:
    for group in (context.checkpoints, context.recovery_records, context.recovery_actions):
        for item in group:
            yield from _walk(item.payload)
    intent = context.batch.payload
    if intent.get("operation") != "capture_directory":
        return
    # Old processes could crash after moving but before recording the captured record.
    for checkpoint in reversed(context.checkpoints):
        before = checkpoint.payload
        if checkpoint.checkpoint_type != "before" or before.get("file_type") != "directory":
            continue
        if before.get("capture_token") is not None or before.get("path") != intent.get("path"):
            continue
        source, quarantine = intent.get("path"), intent.get("quarantine_path")
        key = checkpoint.resource_key
        identity = {
            "file_type": "directory",
            "resource_key": key,
            "device": before.get("device"),
            "inode": before.get("inode"),
        }
        yield {
            "path": source,
            "step_resource_key": key,
            "ownership_class": "captured_by_transaction",
            "captured_directory": {
                "original_path": source,
                "quarantine_path": quarantine,
                "ownership_class": "captured_by_transaction",
                "original_identity": {**identity, "path": source},
                "captured_identity": {**identity, "path": quarantine},
            },
        }
        return
