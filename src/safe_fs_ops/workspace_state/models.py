from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    resource_key: str
    owner: str
    scope: str | None
    details: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    name: str
    owner: str
    token: str
    fencing_token: int
    acquired_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    acquired: bool = False
