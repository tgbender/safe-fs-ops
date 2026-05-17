from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, cast


def _validate_required(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


def _payload_to_json(payload: Mapping[str, Any] | None) -> str:
    return json.dumps({} if payload is None else payload, sort_keys=True, separators=(",", ":"))


def _frozen_payload(payload_json: str) -> Mapping[str, Any]:
    loaded = json.loads(payload_json)
    if not isinstance(loaded, dict):
        raise ValueError("stored payload must be a JSON object")
    return cast("Mapping[str, Any]", _freeze_json(loaded))


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen_items = {str(key): _freeze_json(item) for key, item in value.items()}
        return MappingProxyType(frozen_items)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(_freeze_json(item) for item in value)
    return value


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _utcnow(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
