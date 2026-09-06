from __future__ import annotations

import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.models import LeaseRecord
from safe_fs_ops.workspace_state.orm import LeaseRow

_LEASE_TABLE = "workspace_leases"


class _LeaseColumn:
    NAME = "name"
    OWNER = "owner"
    TOKEN = "token"
    FENCING_TOKEN = "fencing_token"
    ACQUIRED_AT = "acquired_at"
    HEARTBEAT_AT = "heartbeat_at"
    EXPIRES_AT = "expires_at"


_LEASE_SELECT_COLUMNS = (
    _LeaseColumn.NAME,
    _LeaseColumn.OWNER,
    _LeaseColumn.TOKEN,
    _LeaseColumn.FENCING_TOKEN,
    _LeaseColumn.ACQUIRED_AT,
    _LeaseColumn.HEARTBEAT_AT,
    _LeaseColumn.EXPIRES_AT,
)
_LEASE_SELECT_LIST = ", ".join(_LEASE_SELECT_COLUMNS)

_LEASE_SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS {_LEASE_TABLE} (
        {_LeaseColumn.NAME} TEXT PRIMARY KEY,
        {_LeaseColumn.OWNER} TEXT NOT NULL,
        {_LeaseColumn.TOKEN} TEXT NOT NULL,
        {_LeaseColumn.FENCING_TOKEN} INTEGER NOT NULL,
        {_LeaseColumn.ACQUIRED_AT} TEXT NOT NULL,
        {_LeaseColumn.HEARTBEAT_AT} TEXT NOT NULL,
        {_LeaseColumn.EXPIRES_AT} TEXT NOT NULL
    )
    """,
)


class LeaseLostError(RuntimeError):
    pass


def lease_schema() -> tuple[str, ...]:
    return _LEASE_SCHEMA


def require_current_lease(
    connection: sqlite3.Connection,
    lease: LeaseRecord,
    *,
    now: datetime | None = None,
) -> None:
    if not _lease_is_current(connection, lease, now=_utcnow(now)):
        raise LeaseLostError(f"lease {lease.name!r} is no longer current")


class LeaseStore:
    """SQLite-backed cross-process leases with tokens and fencing counters."""

    def __init__(self, path: Path | str, *, sqlite_store: SqliteStore | None = None) -> None:
        self.sqlite_store = sqlite_store or SqliteStore(path)

    @property
    def path(self) -> Path | str:
        return self.sqlite_store.path

    def initialize(self) -> None:
        self.sqlite_store.initialize(_LEASE_SCHEMA)

    def acquire(
        self,
        name: str,
        *,
        owner: str,
        ttl: timedelta,
        token: str | None = None,
        now: datetime | None = None,
    ) -> LeaseRecord:
        if ttl.total_seconds() <= 0:
            raise ValueError("lease ttl must be positive")
        current_time = _utcnow(now)
        expires_at = current_time + ttl
        lease_token = token or secrets.token_hex(16)
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                lease_row = session.get(LeaseRow, name)
                if lease_row is not None and _parse_datetime(lease_row.expires_at) > current_time:
                    return _lease_from_orm(lease_row, acquired=False, expose_authority=False)
                if lease_row is None:
                    lease_row = LeaseRow(
                        name=name,
                        owner=owner,
                        token=lease_token,
                        fencing_token=1,
                        acquired_at=current_time.isoformat(),
                        heartbeat_at=current_time.isoformat(),
                        expires_at=expires_at.isoformat(),
                    )
                    session.add(lease_row)
                    try:
                        session.flush()
                    except IntegrityError:
                        session.rollback()
                        with self.sqlite_store.sqlalchemy_session() as read_session:
                            current = read_session.get(LeaseRow, name)
                            if current is None:
                                raise
                            return _lease_from_orm(current, acquired=False, expose_authority=False)
                else:
                    updated_result = session.execute(
                        update(LeaseRow)
                        .where(LeaseRow.name == name, LeaseRow.expires_at <= current_time.isoformat())
                        .values(
                            owner=owner,
                            token=lease_token,
                            fencing_token=LeaseRow.fencing_token + 1,
                            acquired_at=current_time.isoformat(),
                            heartbeat_at=current_time.isoformat(),
                            expires_at=expires_at.isoformat(),
                        )
                    )
                    if int(cast(Any, updated_result).rowcount) != 1:
                        current = session.get(LeaseRow, name)
                        if current is None:
                            raise RuntimeError("lease disappeared during acquisition")
                        return _lease_from_orm(current, acquired=False, expose_authority=False)
                    session.flush()
                    lease_row = session.get(LeaseRow, name)
                if lease_row is None:
                    raise RuntimeError("lease acquisition did not produce a row")
                return _lease_from_orm(lease_row, acquired=True, expose_authority=True)
        with self.sqlite_store.transaction() as connection:
            row = _lease_row(connection, name)
            if row is not None and _parse_datetime(str(row[_LeaseColumn.EXPIRES_AT])) > current_time:
                return _lease_from_row(row, acquired=False, expose_authority=False)
            if row is None:
                try:
                    connection.execute(
                        f"""
                        INSERT INTO {_LEASE_TABLE} (
                            {_LeaseColumn.NAME},
                            {_LeaseColumn.OWNER},
                            {_LeaseColumn.TOKEN},
                            {_LeaseColumn.FENCING_TOKEN},
                            {_LeaseColumn.ACQUIRED_AT},
                            {_LeaseColumn.HEARTBEAT_AT},
                            {_LeaseColumn.EXPIRES_AT}
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            name,
                            owner,
                            lease_token,
                            1,
                            current_time.isoformat(),
                            current_time.isoformat(),
                            expires_at.isoformat(),
                        ),
                    )
                except sqlite3.IntegrityError:
                    current_row = _lease_row(connection, name)
                    if current_row is None:
                        raise
                    return _lease_from_row(current_row, acquired=False, expose_authority=False)
            else:
                updated = connection.execute(
                    f"""
                    UPDATE {_LEASE_TABLE}
                    SET {_LeaseColumn.OWNER} = ?,
                        {_LeaseColumn.TOKEN} = ?,
                        {_LeaseColumn.FENCING_TOKEN} = {_LeaseColumn.FENCING_TOKEN} + 1,
                        {_LeaseColumn.ACQUIRED_AT} = ?,
                        {_LeaseColumn.HEARTBEAT_AT} = ?,
                        {_LeaseColumn.EXPIRES_AT} = ?
                    WHERE {_LeaseColumn.NAME} = ? AND {_LeaseColumn.EXPIRES_AT} <= ?
                    """,
                    (
                        owner,
                        lease_token,
                        current_time.isoformat(),
                        current_time.isoformat(),
                        expires_at.isoformat(),
                        name,
                        current_time.isoformat(),
                    ),
                )
                if updated.rowcount != 1:
                    current_row = _lease_row(connection, name)
                    if current_row is None:
                        raise RuntimeError("lease disappeared during acquisition")
                    return _lease_from_row(current_row, acquired=False, expose_authority=False)
            acquired = _lease_row(connection, name)
            if acquired is None:
                raise RuntimeError("lease acquisition did not produce a row")
            return _lease_from_row(acquired, acquired=True, expose_authority=True)

    def active(self, name: str, *, now: datetime | None = None) -> LeaseRecord | None:
        self.initialize()
        current_time = _utcnow(now)
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_session() as session:
                lease_row = session.get(LeaseRow, name)
            if lease_row is None:
                return None
            record = _lease_from_orm(lease_row, acquired=False, expose_authority=False)
            return record if record.expires_at > current_time else None
        with self.sqlite_store.read_connection() as connection:
            row = _lease_row(connection, name)
        if row is None:
            return None
        record = _lease_from_row(row, acquired=False, expose_authority=False)
        return record if record.expires_at > current_time else None

    def heartbeat(
        self,
        lease: LeaseRecord,
        *,
        ttl: timedelta,
        now: datetime | None = None,
    ) -> LeaseRecord | None:
        if ttl.total_seconds() <= 0:
            raise ValueError("lease ttl must be positive")
        current_time = _utcnow(now)
        expires_at = current_time + ttl
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                updated_result = session.execute(
                    update(LeaseRow)
                    .where(
                        LeaseRow.name == lease.name,
                        LeaseRow.owner == lease.owner,
                        LeaseRow.token == lease.token,
                        LeaseRow.fencing_token == lease.fencing_token,
                        LeaseRow.expires_at > current_time.isoformat(),
                    )
                    .values(
                        heartbeat_at=current_time.isoformat(),
                        expires_at=expires_at.isoformat(),
                    )
                )
                if int(cast(Any, updated_result).rowcount) != 1:
                    return None
                lease_row = session.get(LeaseRow, lease.name)
                if lease_row is None:
                    raise RuntimeError("lease disappeared during heartbeat")
                return _lease_from_orm(lease_row, acquired=False, expose_authority=True)
        with self.sqlite_store.transaction() as connection:
            updated = connection.execute(
                f"""
                UPDATE {_LEASE_TABLE}
                SET {_LeaseColumn.HEARTBEAT_AT} = ?, {_LeaseColumn.EXPIRES_AT} = ?
                WHERE {_LeaseColumn.NAME} = ?
                  AND {_LeaseColumn.OWNER} = ?
                  AND {_LeaseColumn.TOKEN} = ?
                  AND {_LeaseColumn.FENCING_TOKEN} = ?
                  AND {_LeaseColumn.EXPIRES_AT} > ?
                """,
                (
                    current_time.isoformat(),
                    expires_at.isoformat(),
                    lease.name,
                    lease.owner,
                    lease.token,
                    lease.fencing_token,
                    current_time.isoformat(),
                ),
            )
            if updated.rowcount != 1:
                return None
            row = _lease_row(connection, lease.name)
            if row is None:
                raise RuntimeError("lease disappeared during heartbeat")
            return _lease_from_row(row, acquired=False, expose_authority=True)

    def is_current(self, lease: LeaseRecord, *, now: datetime | None = None) -> bool:
        current_time = _utcnow(now)
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_session() as session:
                row = session.execute(
                    select(LeaseRow.name).where(
                        LeaseRow.name == lease.name,
                        LeaseRow.owner == lease.owner,
                        LeaseRow.token == lease.token,
                        LeaseRow.fencing_token == lease.fencing_token,
                        LeaseRow.expires_at > current_time.isoformat(),
                    )
                ).first()
            return row is not None
        with self.sqlite_store.read_connection() as connection:
            return _lease_is_current(connection, lease, now=current_time)

    def require_current(self, lease: LeaseRecord, *, now: datetime | None = None) -> None:
        if not self.is_current(lease, now=now):
            raise LeaseLostError(f"lease {lease.name!r} is no longer current")

    def release(self, lease: LeaseRecord, *, now: datetime | None = None) -> bool:
        current_time = _utcnow(now)
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                released_result = session.execute(
                    update(LeaseRow)
                    .where(
                        LeaseRow.name == lease.name,
                        LeaseRow.owner == lease.owner,
                        LeaseRow.token == lease.token,
                        LeaseRow.fencing_token == lease.fencing_token,
                        LeaseRow.expires_at > current_time.isoformat(),
                    )
                    .values(
                        heartbeat_at=current_time.isoformat(),
                        expires_at=current_time.isoformat(),
                        # Expiry alone cannot revoke a heartbeat that sampled
                        # its clock before release and reaches storage later.
                        token=secrets.token_hex(16),
                    )
                )
                return int(cast(Any, released_result).rowcount) == 1
        with self.sqlite_store.transaction() as connection:
            return _release_lease_in_connection(connection, lease, now=current_time)


def _lease_row(connection: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_LEASE_SELECT_LIST}
            FROM {_LEASE_TABLE}
            WHERE {_LeaseColumn.NAME} = ?
            """,
            (name,),
        ).fetchone(),
    )


def _lease_is_current(connection: sqlite3.Connection, lease: LeaseRecord, *, now: datetime) -> bool:
    row = connection.execute(
        f"""
        SELECT 1
        FROM {_LEASE_TABLE}
        WHERE {_LeaseColumn.NAME} = ?
          AND {_LeaseColumn.OWNER} = ?
          AND {_LeaseColumn.TOKEN} = ?
          AND {_LeaseColumn.FENCING_TOKEN} = ?
          AND {_LeaseColumn.EXPIRES_AT} > ?
        """,
        (
            lease.name,
            lease.owner,
            lease.token,
            lease.fencing_token,
            now.isoformat(),
        ),
    ).fetchone()
    return row is not None


def _release_lease_in_connection(connection: sqlite3.Connection, lease: LeaseRecord, *, now: datetime) -> bool:
    released = connection.execute(
        f"""
        UPDATE {_LEASE_TABLE}
        SET {_LeaseColumn.HEARTBEAT_AT} = ?,
            {_LeaseColumn.EXPIRES_AT} = ?,
            {_LeaseColumn.TOKEN} = ?
        WHERE {_LeaseColumn.NAME} = ?
          AND {_LeaseColumn.OWNER} = ?
          AND {_LeaseColumn.TOKEN} = ?
          AND {_LeaseColumn.FENCING_TOKEN} = ?
          AND {_LeaseColumn.EXPIRES_AT} > ?
        """,
        (
            now.isoformat(),
            now.isoformat(),
            secrets.token_hex(16),
            lease.name,
            lease.owner,
            lease.token,
            lease.fencing_token,
            now.isoformat(),
        ),
    )
    return released.rowcount == 1


def _lease_from_row(row: sqlite3.Row, *, acquired: bool, expose_authority: bool) -> LeaseRecord:
    return LeaseRecord(
        name=str(row[_LeaseColumn.NAME]),
        owner=str(row[_LeaseColumn.OWNER]),
        token=str(row[_LeaseColumn.TOKEN]) if expose_authority else "",
        fencing_token=int(row[_LeaseColumn.FENCING_TOKEN]) if expose_authority else 0,
        acquired_at=_parse_datetime(str(row[_LeaseColumn.ACQUIRED_AT])),
        heartbeat_at=_parse_datetime(str(row[_LeaseColumn.HEARTBEAT_AT])),
        expires_at=_parse_datetime(str(row[_LeaseColumn.EXPIRES_AT])),
        acquired=acquired,
    )


def _lease_from_orm(row: LeaseRow, *, acquired: bool, expose_authority: bool) -> LeaseRecord:
    return LeaseRecord(
        name=row.name,
        owner=row.owner,
        token=row.token if expose_authority else "",
        fencing_token=row.fencing_token if expose_authority else 0,
        acquired_at=_parse_datetime(row.acquired_at),
        heartbeat_at=_parse_datetime(row.heartbeat_at),
        expires_at=_parse_datetime(row.expires_at),
        acquired=acquired,
    )


def _utcnow(value: datetime | None = None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
