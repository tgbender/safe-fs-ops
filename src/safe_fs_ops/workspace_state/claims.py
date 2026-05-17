from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from safe_fs_ops.sqlite_store import SqliteStore
from safe_fs_ops.workspace_state.leases import LeaseLostError, lease_schema, require_current_lease
from safe_fs_ops.workspace_state.models import ClaimRecord, LeaseRecord
from safe_fs_ops.workspace_state.orm import ClaimRow, LeaseRow

_CLAIM_TABLE = "workspace_resource_claims"


class _ClaimColumn:
    RESOURCE_KEY = "resource_key"
    OWNER = "owner"
    SCOPE = "scope"
    DETAILS = "details"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


_CLAIM_SELECT_COLUMNS = (
    _ClaimColumn.RESOURCE_KEY,
    _ClaimColumn.OWNER,
    _ClaimColumn.SCOPE,
    _ClaimColumn.DETAILS,
    _ClaimColumn.CREATED_AT,
    _ClaimColumn.UPDATED_AT,
)
_CLAIM_SELECT_LIST = ", ".join(_CLAIM_SELECT_COLUMNS)

_CLAIM_SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS {_CLAIM_TABLE} (
        {_ClaimColumn.RESOURCE_KEY} TEXT PRIMARY KEY,
        {_ClaimColumn.OWNER} TEXT NOT NULL,
        {_ClaimColumn.SCOPE} TEXT,
        {_ClaimColumn.DETAILS} TEXT,
        {_ClaimColumn.CREATED_AT} TEXT NOT NULL,
        {_ClaimColumn.UPDATED_AT} TEXT NOT NULL
    )
    """,
    f"""
    CREATE INDEX IF NOT EXISTS idx_{_CLAIM_TABLE}_owner
    ON {_CLAIM_TABLE} ({_ClaimColumn.OWNER})
    """,
)


def lease_claim_details_payload(
    lease: LeaseRecord,
    *,
    claim_id: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "claim_id": claim_id,
        "lease_name": lease.name,
        "lease_owner": lease.owner,
        "lease_token_digest": _lease_token_digest(lease),
        "lease_fencing_token": lease.fencing_token,
    }
    if extra is not None:
        payload.update(dict(extra))
    return payload


class ClaimConflictError(RuntimeError):
    def __init__(self, message: str, *, existing: ClaimRecord) -> None:
        super().__init__(message)
        self.existing = existing


class ClaimStore:
    """SQLite-backed durable resource claims.

    Claims record durable ownership of resources over time. They do not grant
    active mutation authority. Claim writes still require a current lease so the
    durable ownership change composes with active workspace authority in one
    SQLite transaction.
    """

    def __init__(self, path: Path | str, *, sqlite_store: SqliteStore | None = None) -> None:
        self.sqlite_store = sqlite_store or SqliteStore(path)

    @property
    def path(self) -> Path | str:
        return self.sqlite_store.path

    def initialize(self) -> None:
        self.sqlite_store.initialize((*lease_schema(), *_CLAIM_SCHEMA))

    def create(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        details: str | None = None,
        now: datetime | None = None,
    ) -> ClaimRecord:
        _validate_required("resource_key", resource_key)
        _validate_required("owner", owner)
        current_time = _utcnow(now)
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                _require_current_lease_orm(session, lease, now=current_time)
                existing_row = session.get(ClaimRow, resource_key)
                if existing_row is not None:
                    raise ClaimConflictError(
                        f"resource {resource_key!r} is already claimed",
                        existing=_claim_from_orm(existing_row),
                    )
                claim_row = ClaimRow(
                    resource_key=resource_key,
                    owner=owner,
                    scope=scope,
                    details=details,
                    created_at=current_time.isoformat(),
                    updated_at=current_time.isoformat(),
                )
                session.add(claim_row)
                try:
                    session.flush()
                except IntegrityError as exc:
                    session.rollback()
                    existing_row = session.get(ClaimRow, resource_key)
                    if existing_row is None:
                        raise
                    raise ClaimConflictError(
                        f"resource {resource_key!r} is already claimed",
                        existing=_claim_from_orm(existing_row),
                    ) from exc
                created_row = session.get(ClaimRow, resource_key)
                if created_row is None:
                    raise RuntimeError("claim creation did not produce a row")
                return _claim_from_orm(created_row)
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            try:
                connection.execute(
                    f"""
                    INSERT INTO {_CLAIM_TABLE} (
                        {_ClaimColumn.RESOURCE_KEY},
                        {_ClaimColumn.OWNER},
                        {_ClaimColumn.SCOPE},
                        {_ClaimColumn.DETAILS},
                        {_ClaimColumn.CREATED_AT},
                        {_ClaimColumn.UPDATED_AT}
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resource_key,
                        owner,
                        scope,
                        details,
                        current_time.isoformat(),
                        current_time.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = _claim_row(connection, resource_key)
                if existing is None:
                    raise
                raise ClaimConflictError(
                    f"resource {resource_key!r} is already claimed",
                    existing=_claim_from_row(existing),
                ) from exc
            row = _claim_row(connection, resource_key)
            if row is None:
                raise RuntimeError("claim creation did not produce a row")
            return _claim_from_row(row)

    def upsert(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        details: str | None = None,
        now: datetime | None = None,
    ) -> ClaimRecord:
        _validate_required("resource_key", resource_key)
        _validate_required("owner", owner)
        current_time = _utcnow(now)
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                _require_current_lease_orm(session, lease, now=current_time)
                claim_row = session.get(ClaimRow, resource_key)
                if claim_row is None:
                    claim_row = ClaimRow(
                        resource_key=resource_key,
                        owner=owner,
                        scope=scope,
                        details=details,
                        created_at=current_time.isoformat(),
                        updated_at=current_time.isoformat(),
                    )
                    session.add(claim_row)
                    session.flush()
                else:
                    existing = _claim_from_orm(claim_row)
                    if _conflicts(existing, owner=owner, scope=scope):
                        raise ClaimConflictError(
                            f"resource {resource_key!r} is claimed by {existing.owner!r}",
                            existing=existing,
                        )
                    claim_row.details = details
                    claim_row.updated_at = current_time.isoformat()
                    session.flush()
                upserted_row = session.get(ClaimRow, resource_key)
                if upserted_row is None:
                    raise RuntimeError("claim upsert did not produce a row")
                return _claim_from_orm(upserted_row)
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            existing_row = _claim_row(connection, resource_key)
            if existing_row is None:
                connection.execute(
                    f"""
                    INSERT INTO {_CLAIM_TABLE} (
                        {_ClaimColumn.RESOURCE_KEY},
                        {_ClaimColumn.OWNER},
                        {_ClaimColumn.SCOPE},
                        {_ClaimColumn.DETAILS},
                        {_ClaimColumn.CREATED_AT},
                        {_ClaimColumn.UPDATED_AT}
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resource_key,
                        owner,
                        scope,
                        details,
                        current_time.isoformat(),
                        current_time.isoformat(),
                    ),
                )
            else:
                existing = _claim_from_row(existing_row)
                if _conflicts(existing, owner=owner, scope=scope):
                    raise ClaimConflictError(
                        f"resource {resource_key!r} is claimed by {existing.owner!r}",
                        existing=existing,
                    )
                else:
                    connection.execute(
                        f"""
                        UPDATE {_CLAIM_TABLE}
                        SET {_ClaimColumn.DETAILS} = ?,
                            {_ClaimColumn.UPDATED_AT} = ?
                        WHERE {_ClaimColumn.RESOURCE_KEY} = ?
                        """,
                        (details, current_time.isoformat(), resource_key),
                    )
            row = _claim_row(connection, resource_key)
            if row is None:
                raise RuntimeError("claim upsert did not produce a row")
            return _claim_from_row(row)

    def get(self, resource_key: str) -> ClaimRecord | None:
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_session() as session:
                claim_row = session.get(ClaimRow, resource_key)
            return None if claim_row is None else _claim_from_orm(claim_row)
        with self.sqlite_store.read_connection() as connection:
            row = _claim_row(connection, resource_key)
        return None if row is None else _claim_from_row(row)

    def conflict_for(
        self,
        resource_key: str,
        *,
        owner: str,
        scope: str | None = None,
    ) -> ClaimRecord | None:
        _validate_required("resource_key", resource_key)
        _validate_required("owner", owner)
        existing = self.get(resource_key)
        if existing is None or not _conflicts(existing, owner=owner, scope=scope):
            return None
        return existing

    def list_claims(self, *, owner: str | None = None) -> list[ClaimRecord]:
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            statement = select(ClaimRow).order_by(ClaimRow.resource_key)
            if owner is not None:
                statement = statement.where(ClaimRow.owner == owner)
            with self.sqlite_store.sqlalchemy_session() as session:
                rows = session.scalars(statement).all()
            return [_claim_from_orm(row) for row in rows]
        parameters: tuple[str, ...] = ()
        where_clause = ""
        if owner is not None:
            where_clause = f"WHERE {_ClaimColumn.OWNER} = ?"
            parameters = (owner,)
        with self.sqlite_store.read_connection() as connection:
            rows = connection.execute(
                f"""
                SELECT {_CLAIM_SELECT_LIST}
                FROM {_CLAIM_TABLE}
                {where_clause}
                ORDER BY {_ClaimColumn.RESOURCE_KEY}
                """,
                parameters,
            ).fetchall()
        return [_claim_from_row(row) for row in rows]

    def release(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        expected_claim: ClaimRecord | None = None,
        now: datetime | None = None,
    ) -> bool:
        _validate_required("resource_key", resource_key)
        _validate_required("owner", owner)
        current_time = _utcnow(now)
        if expected_claim is not None and expected_claim.resource_key != resource_key:
            raise ValueError("expected_claim.resource_key must match resource_key")
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                _require_current_lease_orm(session, lease, now=current_time)
                return _release_claim_orm(
                    session,
                    resource_key,
                    owner=owner,
                    scope=scope,
                    expected_claim=expected_claim,
                )
        with self.sqlite_store.transaction() as connection:
            require_current_lease(connection, lease, now=current_time)
            return _release_claim_in_connection(
                connection,
                resource_key,
                owner=owner,
                scope=scope,
                expected_claim=expected_claim,
            )

    def release_if_owner_scope_matches(
        self,
        resource_key: str,
        *,
        lease: LeaseRecord,
        owner: str,
        scope: str | None = None,
        expected_claim: ClaimRecord | None = None,
        now: datetime | None = None,
    ) -> bool:
        _validate_required("resource_key", resource_key)
        _validate_required("owner", owner)
        current_time = _utcnow(now)
        if expected_claim is None:
            raise ValueError("expected_claim is required for claim release")
        if expected_claim is not None and expected_claim.resource_key != resource_key:
            raise ValueError("expected_claim.resource_key must match resource_key")
        self.initialize()
        if self.sqlite_store.uses_sqlalchemy:
            with self.sqlite_store.sqlalchemy_transaction() as session:
                if not _lease_is_current_orm(session, lease, now=current_time) and not _claim_matches_lease_authority(
                    expected_claim,
                    lease,
                ):
                    raise ValueError("claim release requires current lease or lease-bound expected_claim")
                return _release_claim_orm(
                    session,
                    resource_key,
                    owner=owner,
                    scope=scope,
                    expected_claim=expected_claim,
                )
        with self.sqlite_store.transaction() as connection:
            if not _lease_is_current(connection, lease, now=current_time) and not _claim_matches_lease_authority(
                expected_claim,
                lease,
            ):
                raise ValueError("claim release requires current lease or lease-bound expected_claim")
            return _release_claim_in_connection(
                connection,
                resource_key,
                owner=owner,
                scope=scope,
                expected_claim=expected_claim,
            )


def _release_claim_in_connection(
    connection: sqlite3.Connection,
    resource_key: str,
    *,
    owner: str,
    scope: str | None,
    expected_claim: ClaimRecord | None = None,
) -> bool:
    if expected_claim is None:
        deleted = connection.execute(
            f"""
            DELETE FROM {_CLAIM_TABLE}
            WHERE {_ClaimColumn.RESOURCE_KEY} = ?
              AND {_ClaimColumn.OWNER} = ?
              AND (
                ({_ClaimColumn.SCOPE} IS NULL AND ? IS NULL)
                OR {_ClaimColumn.SCOPE} = ?
              )
            """,
            (resource_key, owner, scope, scope),
        )
        return deleted.rowcount == 1
    deleted = connection.execute(
        f"""
        DELETE FROM {_CLAIM_TABLE}
        WHERE {_ClaimColumn.RESOURCE_KEY} = ?
          AND {_ClaimColumn.OWNER} = ?
          AND (
            ({_ClaimColumn.SCOPE} IS NULL AND ? IS NULL)
            OR {_ClaimColumn.SCOPE} = ?
          )
          AND (
            ({_ClaimColumn.DETAILS} IS NULL AND ? IS NULL)
            OR {_ClaimColumn.DETAILS} = ?
          )
          AND {_ClaimColumn.CREATED_AT} = ?
          AND {_ClaimColumn.UPDATED_AT} = ?
        """,
        (
            resource_key,
            owner,
            scope,
            scope,
            expected_claim.details,
            expected_claim.details,
            expected_claim.created_at.isoformat(),
            expected_claim.updated_at.isoformat(),
        ),
    )
    return deleted.rowcount == 1


def _claim_matches_lease_authority(claim: ClaimRecord, lease: LeaseRecord) -> bool:
    details = _lease_bound_claim_details(claim.details)
    if details is None:
        return False
    claim_id = details.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        return False
    lease_fencing_token = details.get("lease_fencing_token")
    return (
        details.get("lease_name") == lease.name
        and details.get("lease_owner") == lease.owner
        and details.get("lease_token_digest") == _lease_token_digest(lease)
        and isinstance(lease_fencing_token, int)
        and lease_fencing_token == lease.fencing_token
    )


def _lease_token_digest(lease: LeaseRecord) -> str:
    return hashlib.sha256(lease.token.encode("utf-8")).hexdigest()


def _lease_is_current(connection: sqlite3.Connection, lease: LeaseRecord, *, now: datetime) -> bool:
    try:
        require_current_lease(connection, lease, now=now)
    except LeaseLostError:
        return False
    return True


def _lease_bound_claim_details(details: str | None) -> Mapping[str, object] | None:
    if details is None:
        return None
    try:
        payload = json.loads(details)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _release_claim_orm(
    session: Session,
    resource_key: str,
    *,
    owner: str,
    scope: str | None,
    expected_claim: ClaimRecord | None = None,
) -> bool:
    conditions: list[Any] = [
        ClaimRow.resource_key == resource_key,
        ClaimRow.owner == owner,
        ClaimRow.scope.is_(None) if scope is None else ClaimRow.scope == scope,
    ]
    if expected_claim is not None:
        conditions.extend(
            [
                (
                    ClaimRow.details.is_(None)
                    if expected_claim.details is None
                    else ClaimRow.details == expected_claim.details
                ),
                ClaimRow.created_at == expected_claim.created_at.isoformat(),
                ClaimRow.updated_at == expected_claim.updated_at.isoformat(),
            ]
        )
    deleted = session.execute(delete(ClaimRow).where(*conditions))
    return int(cast(Any, deleted).rowcount) == 1


def _claim_row(connection: sqlite3.Connection, resource_key: str) -> sqlite3.Row | None:
    return cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_CLAIM_SELECT_LIST}
            FROM {_CLAIM_TABLE}
            WHERE {_ClaimColumn.RESOURCE_KEY} = ?
            """,
            (resource_key,),
        ).fetchone(),
    )


def _claim_from_row(row: sqlite3.Row) -> ClaimRecord:
    return ClaimRecord(
        resource_key=str(row[_ClaimColumn.RESOURCE_KEY]),
        owner=str(row[_ClaimColumn.OWNER]),
        scope=_optional_str(row[_ClaimColumn.SCOPE]),
        details=_optional_str(row[_ClaimColumn.DETAILS]),
        created_at=_parse_datetime(str(row[_ClaimColumn.CREATED_AT])),
        updated_at=_parse_datetime(str(row[_ClaimColumn.UPDATED_AT])),
    )


def _claim_from_orm(row: ClaimRow) -> ClaimRecord:
    return ClaimRecord(
        resource_key=row.resource_key,
        owner=row.owner,
        scope=row.scope,
        details=row.details,
        created_at=_parse_datetime(row.created_at),
        updated_at=_parse_datetime(row.updated_at),
    )


def _require_current_lease_orm(session: Session, lease: LeaseRecord, *, now: datetime) -> None:
    if not _lease_is_current_orm(session, lease, now=now):
        from safe_fs_ops.workspace_state.leases import LeaseLostError

        raise LeaseLostError(f"lease {lease.name!r} is no longer current")


def _lease_is_current_orm(session: Session, lease: LeaseRecord, *, now: datetime) -> bool:
    row = session.execute(
        select(LeaseRow.name).where(
            LeaseRow.name == lease.name,
            LeaseRow.owner == lease.owner,
            LeaseRow.token == lease.token,
            LeaseRow.fencing_token == lease.fencing_token,
            LeaseRow.expires_at > now.isoformat(),
        )
    ).first()
    return row is not None


def _conflicts(existing: ClaimRecord, *, owner: str, scope: str | None) -> bool:
    return existing.owner != owner or existing.scope != scope


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _validate_required(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


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
