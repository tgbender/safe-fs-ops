from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.event import listens_for
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

SqliteSynchronousMode = Literal["OFF", "NORMAL", "FULL", "EXTRA"]


@dataclass(frozen=True, slots=True)
class SqliteEngineConfig:
    path: Path | str
    busy_timeout_ms: int = 5_000
    enable_foreign_keys: bool = True
    enable_wal: bool = True
    synchronous: SqliteSynchronousMode | None = "NORMAL"
    connect_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be >= 0")
        if self.synchronous is not None and self.synchronous not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
            raise ValueError(f"unsupported SQLite synchronous mode: {self.synchronous!r}")


def create_sqlite_engine(config: SqliteEngineConfig) -> Engine:
    database = str(config.path)
    _reject_sqlite_uri_filename(database)
    connect_args: dict[str, Any] = {"timeout": config.busy_timeout_ms / 1_000}
    connect_args.update(dict(config.connect_args))
    engine = create_engine(
        URL.create("sqlite+pysqlite", database=database),
        poolclass=NullPool,
        connect_args=connect_args,
    )

    @listens_for(engine, "connect")
    def _configure_connection(dbapi_connection: Any, _: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA busy_timeout = {config.busy_timeout_ms:d}")
            if config.enable_foreign_keys:
                cursor.execute("PRAGMA foreign_keys = ON")
            if config.enable_wal and not _is_in_memory_database(database):
                cursor.execute("PRAGMA journal_mode = WAL")
            if config.synchronous is not None:
                cursor.execute(f"PRAGMA synchronous = {config.synchronous}")
        finally:
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def immediate_session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    engine = _session_factory_engine(session_factory)
    connection = engine.connect()
    session = session_factory(bind=connection)
    try:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        yield session
        session.flush()
        connection.commit()
    except BaseException:
        session.rollback()
        connection.rollback()
        raise
    finally:
        session.close()
        connection.close()


def _session_factory_engine(session_factory: sessionmaker[Session]) -> Engine:
    bind = session_factory.kw.get("bind")
    if not isinstance(bind, Engine):
        raise TypeError("session factory must be bound to a SQLAlchemy Engine")
    return bind


def _is_in_memory_database(database: str) -> bool:
    return database == ":memory:" or database.startswith("file::memory:")


def _reject_sqlite_uri_filename(database: str) -> None:
    if database.startswith("file:") and database != ":memory:":
        raise ValueError("SQLite URI-style database filenames are not supported; pass a filesystem path or ':memory:'")
