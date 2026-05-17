from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from sqlalchemy.engine import Connection, Engine, Row
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session, sessionmaker

from safe_fs_ops.filesystem_ops.paths import UnsafePathError, absolute_without_resolving, inspect_path
from safe_fs_ops.persistence import (
    SqliteEngineConfig,
    create_session_factory,
    create_sqlite_engine,
    immediate_session_scope,
    session_scope,
)

ConnectionFactory = Callable[[Path | str], sqlite3.Connection]
MigrationAction = Callable[[sqlite3.Connection], None]
SchemaValidation = Callable[[sqlite3.Connection, int], None]


_SCHEMA_METADATA_TABLE = "safe_fs_ops_schema_metadata"


class _SchemaMetadataColumn:
    SINGLETON_KEY = "singleton_key"
    SCHEMA_IDENTITY = "schema_identity"
    SCHEMA_VERSION = "schema_version"


_SCHEMA_METADATA_SELECT_COLUMNS = (
    _SchemaMetadataColumn.SINGLETON_KEY,
    _SchemaMetadataColumn.SCHEMA_IDENTITY,
    _SchemaMetadataColumn.SCHEMA_VERSION,
)
_SCHEMA_METADATA_SELECT_LIST = ", ".join(_SCHEMA_METADATA_SELECT_COLUMNS)
_SCHEMA_METADATA_SINGLETON_VALUE = 1


@dataclass(frozen=True)
class SchemaMigration:
    version: int
    statements: tuple[str, ...] = ()
    apply: MigrationAction | None = None

    def run(self, connection: sqlite3.Connection) -> None:
        for statement in self.statements:
            connection.execute(statement)
        if self.apply is not None:
            self.apply(connection)


@dataclass(frozen=True)
class SchemaDefinition:
    identity: str
    migrations: tuple[SchemaMigration, ...] = field(default_factory=tuple)
    validate: SchemaValidation | None = None

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("schema identity must not be empty")
        if not self.migrations:
            raise ValueError("schema definition must include at least one migration")
        expected_version = 1
        for migration in self.migrations:
            if migration.version != expected_version:
                raise ValueError("schema migrations must start at version 1 and increase by 1")
            expected_version += 1

    @property
    def current_version(self) -> int:
        return self.migrations[-1].version


@dataclass(frozen=True)
class SchemaStatus:
    identity: str
    version: int


class SchemaMigrationError(RuntimeError):
    """Base class for schema migration failures."""


class SchemaIdentityMismatchError(SchemaMigrationError):
    """Raised when a store contains a different logical schema."""


class SchemaVersionError(SchemaMigrationError):
    """Raised when a store version is incompatible with the requested schema."""


class SchemaValidationError(SchemaMigrationError):
    """Raised when legacy schema adoption validation is missing or fails."""


class SqliteStore:
    """Small SQLite wrapper that opens short-lived connections per operation."""

    def __init__(
        self,
        path: Path | str,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if str(path) == ":memory:":
            raise ValueError("SqliteStore(':memory:') is unsupported because it uses short-lived connections")
        self.path = absolute_without_resolving(path)
        _reject_redirecting_sqlite_path(self.path)
        self.connection_factory = connection_factory
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    def connect(self) -> sqlite3.Connection:
        return self._open()

    @property
    def uses_sqlalchemy(self) -> bool:
        return self.connection_factory is None

    @contextmanager
    def sqlalchemy_session(self) -> Iterator[Session]:
        with session_scope(self._sqlalchemy_session_factory()) as session:
            yield session

    @contextmanager
    def sqlalchemy_transaction(self) -> Iterator[Session]:
        with immediate_session_scope(self._sqlalchemy_session_factory()) as session:
            yield session

    def initialize(self, statements: Sequence[str]) -> None:
        with self.transaction() as connection:
            for statement in statements:
                connection.execute(statement)

    def adopt_existing_schema(self, definition: SchemaDefinition, *, version: int) -> SchemaStatus:
        with self.transaction() as connection:
            _ensure_schema_metadata_table(connection)
            status = _schema_status(connection)
            if status is not None:
                if status.identity != definition.identity:
                    raise SchemaIdentityMismatchError(
                        f"sqlite store contains schema {status.identity!r}, not {definition.identity!r}"
                    )
                if status.version != version:
                    raise SchemaVersionError(
                        f"sqlite store schema {definition.identity!r} is already recorded at version {status.version}, "
                        f"not requested version {version}"
                    )
                return status
            _validate_schema_version(version, current_version=definition.current_version)
            _validate_existing_schema_for_adoption(connection, definition, version=version)
            _insert_schema_status(connection, identity=definition.identity, version=version)
            return SchemaStatus(identity=definition.identity, version=version)

    def initialize_schema(self, definition: SchemaDefinition) -> SchemaStatus:
        with self.transaction() as connection:
            _ensure_schema_metadata_table(connection)
            status = _schema_status(connection)
            if status is None:
                for migration in definition.migrations:
                    migration.run(connection)
                _insert_schema_status(connection, identity=definition.identity, version=definition.current_version)
                return SchemaStatus(identity=definition.identity, version=definition.current_version)
            if status.identity != definition.identity:
                raise SchemaIdentityMismatchError(
                    f"sqlite store contains schema {status.identity!r}, not {definition.identity!r}"
                )
            if status.version > definition.current_version:
                raise SchemaVersionError(
                    f"sqlite store schema {definition.identity!r} is at version {status.version}, "
                    f"which is newer than supported version {definition.current_version}"
                )
            if status.version == definition.current_version:
                return status
            for migration in definition.migrations[status.version :]:
                migration.run(connection)
            _update_schema_status(connection, version=definition.current_version)
            return SchemaStatus(identity=definition.identity, version=definition.current_version)

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._open()
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _open(self) -> sqlite3.Connection:
        if str(self.path) != ":memory:":
            _reject_redirecting_sqlite_path(self.path)
            _ensure_sqlite_parent_directory(self.path)
            _reject_redirecting_sqlite_path(self.path)
        if self.connection_factory is not None:
            connection = self.connection_factory(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            if str(self.path) != ":memory:":
                connection.execute("PRAGMA journal_mode = WAL")
            return connection
        if self._engine is None:
            self._engine = create_sqlite_engine(SqliteEngineConfig(self.path, synchronous=None))
        return cast(sqlite3.Connection, _SqlAlchemyConnectionCompat(self._engine.connect()))

    def _sqlalchemy_session_factory(self) -> sessionmaker[Session]:
        if self.connection_factory is not None:
            raise RuntimeError("SQLAlchemy sessions are unavailable when a custom sqlite connection factory is used")
        if str(self.path) != ":memory:":
            _reject_redirecting_sqlite_path(self.path)
            _ensure_sqlite_parent_directory(self.path)
            _reject_redirecting_sqlite_path(self.path)
        if self._engine is None:
            self._engine = create_sqlite_engine(SqliteEngineConfig(self.path, synchronous=None))
        if self._session_factory is None:
            self._session_factory = create_session_factory(self._engine)
        return self._session_factory


class _SqlAlchemyConnectionCompat:
    """sqlite3-like connection facade backed by a short-lived SQLAlchemy connection."""

    def __init__(self, connection: Connection) -> None:
        self._connection = connection

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] | dict[str, object] = (),
        /,
    ) -> _SqlAlchemyResultCompat:
        try:
            return _SqlAlchemyResultCompat(self._connection.exec_driver_sql(str(sql), parameters))
        except DBAPIError as exc:
            if isinstance(exc.orig, sqlite3.Error):
                raise exc.orig from exc
            raise

    def commit(self) -> None:
        self._connection.commit()

    def rollback(self) -> None:
        self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


class _SqlAlchemyResultCompat:
    def __init__(self, result: Any) -> None:
        self._result = result

    @property
    def rowcount(self) -> int:
        return int(self._result.rowcount)

    def fetchone(self) -> _SqlAlchemyRowCompat | None:
        row = self._result.fetchone()
        return None if row is None else _SqlAlchemyRowCompat(row)

    def fetchall(self) -> list[_SqlAlchemyRowCompat]:
        return [_SqlAlchemyRowCompat(row) for row in self._result.fetchall()]


class _SqlAlchemyRowCompat:
    def __init__(self, row: Row[Any]) -> None:
        self._row = row

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return self._row._mapping[key]
        return self._row[key]


def _ensure_sqlite_parent_directory(path: Path) -> None:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    parts = absolute.relative_to(anchor).parts[:-1]
    current = anchor
    for part in parts:
        current = current / part
        safety = inspect_path(current)
        if safety.is_symlink:
            raise UnsafePathError(f"SQLite store refused because state path redirects through symlink: {current}")
        if safety.is_windows_reparse_point:
            raise UnsafePathError(
                f"SQLite store refused because state path redirects through Windows reparse point: {current}"
            )
        if safety.exists:
            if not safety.is_dir:
                raise UnsafePathError(f"SQLite store refused because state parent is not a directory: {current}")
            continue
        with suppress(FileExistsError):
            current.mkdir()
        _reject_redirecting_sqlite_path(current)


def _reject_redirecting_sqlite_path(path: Path) -> None:
    absolute = absolute_without_resolving(path)
    anchor = Path(absolute.anchor)
    for index, _part in enumerate(absolute.relative_to(anchor).parts):
        current = anchor / Path(*absolute.relative_to(anchor).parts[: index + 1])
        safety = inspect_path(current)
        if safety.is_symlink:
            raise UnsafePathError(f"SQLite store refused because state path redirects through symlink: {current}")
        if safety.is_windows_reparse_point:
            raise UnsafePathError(
                f"SQLite store refused because state path redirects through Windows reparse point: {current}"
            )
        if current != absolute and safety.exists and not safety.is_dir:
            raise UnsafePathError(f"SQLite store refused because state parent is not a directory: {current}")


def _ensure_schema_metadata_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_SCHEMA_METADATA_TABLE} (
            {_SchemaMetadataColumn.SINGLETON_KEY} INTEGER PRIMARY KEY
                CHECK ({_SchemaMetadataColumn.SINGLETON_KEY} = {_SCHEMA_METADATA_SINGLETON_VALUE}),
            {_SchemaMetadataColumn.SCHEMA_IDENTITY} TEXT NOT NULL,
            {_SchemaMetadataColumn.SCHEMA_VERSION} INTEGER NOT NULL
                CHECK ({_SchemaMetadataColumn.SCHEMA_VERSION} >= 1)
        )
        """
    )


def _schema_status(connection: sqlite3.Connection) -> SchemaStatus | None:
    row = cast(
        "sqlite3.Row | None",
        connection.execute(
            f"""
            SELECT {_SCHEMA_METADATA_SELECT_LIST}
            FROM {_SCHEMA_METADATA_TABLE}
            WHERE {_SchemaMetadataColumn.SINGLETON_KEY} = ?
            """,
            (_SCHEMA_METADATA_SINGLETON_VALUE,),
        ).fetchone(),
    )
    if row is None:
        return None
    return SchemaStatus(
        identity=str(row[_SchemaMetadataColumn.SCHEMA_IDENTITY]),
        version=int(row[_SchemaMetadataColumn.SCHEMA_VERSION]),
    )


def _insert_schema_status(connection: sqlite3.Connection, *, identity: str, version: int) -> None:
    connection.execute(
        f"""
        INSERT INTO {_SCHEMA_METADATA_TABLE} (
            {_SchemaMetadataColumn.SINGLETON_KEY},
            {_SchemaMetadataColumn.SCHEMA_IDENTITY},
            {_SchemaMetadataColumn.SCHEMA_VERSION}
        )
        VALUES (?, ?, ?)
        """,
        (_SCHEMA_METADATA_SINGLETON_VALUE, identity, version),
    )


def _update_schema_status(connection: sqlite3.Connection, *, version: int) -> None:
    updated = connection.execute(
        f"""
        UPDATE {_SCHEMA_METADATA_TABLE}
        SET {_SchemaMetadataColumn.SCHEMA_VERSION} = ?
        WHERE {_SchemaMetadataColumn.SINGLETON_KEY} = ?
        """,
        (version, _SCHEMA_METADATA_SINGLETON_VALUE),
    )
    if updated.rowcount != 1:
        raise RuntimeError("schema metadata row disappeared during migration")


def _validate_schema_version(version: int, *, current_version: int) -> None:
    if version < 1:
        raise SchemaVersionError(f"sqlite store schema version must be at least 1, not {version}")
    if version > current_version:
        raise SchemaVersionError(
            f"sqlite store schema version {version} is newer than supported version {current_version}"
        )


def _validate_existing_schema_for_adoption(
    connection: sqlite3.Connection,
    definition: SchemaDefinition,
    *,
    version: int,
) -> None:
    if definition.validate is None:
        raise SchemaValidationError(
            f"sqlite store adoption for schema {definition.identity!r} requires a validation callback"
        )
    try:
        definition.validate(connection, version)
    except SchemaMigrationError:
        raise
    except Exception as exc:
        raise SchemaValidationError(
            f"sqlite store adoption validation failed for schema {definition.identity!r}: {exc}"
        ) from exc
