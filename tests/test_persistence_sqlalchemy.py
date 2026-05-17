from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool

from safe_fs_ops.persistence import (
    SqliteEngineConfig,
    create_session_factory,
    create_sqlite_engine,
    immediate_session_scope,
    session_scope,
)

pytestmark = pytest.mark.safe_fs_ops


def test_create_sqlite_engine_configures_foreign_keys_wal_busy_timeout_and_synchronous(tmp_path: Path) -> None:
    engine = create_sqlite_engine(
        SqliteEngineConfig(
            tmp_path / "state.db",
            busy_timeout_ms=2_500,
            synchronous="FULL",
        )
    )
    try:
        with engine.connect() as connection:
            foreign_keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
            journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
            busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
            synchronous = connection.exec_driver_sql("PRAGMA synchronous").scalar_one()

        assert foreign_keys == 1
        assert str(journal_mode).lower() == "wal"
        assert busy_timeout == 2_500
        assert synchronous == 2
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "database",
    [
        "file::memory:?cache=shared",
        "file:memdb1?mode=memory&cache=shared",
    ],
)
def test_create_sqlite_engine_rejects_sqlite_uri_filenames(database: str) -> None:
    with pytest.raises(
        ValueError,
        match="SQLite URI-style database filenames are not supported",
    ):
        create_sqlite_engine(SqliteEngineConfig(database))


def test_session_scope_uses_short_lived_connections(tmp_path: Path) -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(tmp_path / "state.db"))
    opened: list[object] = []
    closed: list[object] = []
    event.listen(engine, "connect", lambda dbapi_connection, _: opened.append(dbapi_connection))
    event.listen(engine.pool, "close", lambda dbapi_connection, _: closed.append(dbapi_connection))
    session_factory = create_session_factory(engine)

    try:
        assert isinstance(engine.pool, NullPool)
        with session_scope(session_factory) as session:
            session.execute(text("SELECT 1"))
        with session_scope(session_factory) as session:
            session.execute(text("SELECT 1"))
    finally:
        engine.dispose()

    assert len(opened) == 2
    assert len(closed) == 2
    assert all(
        any(closed_connection is opened_connection for closed_connection in closed) for opened_connection in opened
    )


def test_session_scope_rolls_back_on_exception(tmp_path: Path) -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(tmp_path / "state.db"))
    session_factory = create_session_factory(engine)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE records (name TEXT PRIMARY KEY)")

        with pytest.raises(RuntimeError, match="boom"), session_scope(session_factory) as session:
            session.execute(text("INSERT INTO records (name) VALUES ('alpha')"))
            raise RuntimeError("boom")

        with engine.connect() as connection:
            rows = connection.exec_driver_sql("SELECT name FROM records").fetchall()
    finally:
        engine.dispose()

    assert rows == []


def test_session_scope_enforces_foreign_keys(tmp_path: Path) -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(tmp_path / "state.db"))
    session_factory = create_session_factory(engine)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE parents (id INTEGER PRIMARY KEY)")
            connection.exec_driver_sql(
                """
                CREATE TABLE children (
                    id INTEGER PRIMARY KEY,
                    parent_id INTEGER NOT NULL REFERENCES parents(id)
                )
                """
            )

        with pytest.raises(IntegrityError), session_scope(session_factory) as session:
            session.execute(text("INSERT INTO children (id, parent_id) VALUES (1, 999)"))
    finally:
        engine.dispose()


def test_immediate_session_scope_serializes_writers(tmp_path: Path) -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(tmp_path / "state.db", busy_timeout_ms=3_000))
    session_factory = create_session_factory(engine)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE records (name TEXT PRIMARY KEY)")

        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        failures: list[BaseException] = []

        def first_writer() -> None:
            try:
                with immediate_session_scope(session_factory) as session:
                    session.execute(text("INSERT INTO records (name) VALUES ('first')"))
                    first_entered.set()
                    release_first.wait(timeout=5)
            except BaseException as exc:
                failures.append(exc)

        def second_writer() -> None:
            try:
                with immediate_session_scope(session_factory) as session:
                    second_entered.set()
                    session.execute(text("INSERT INTO records (name) VALUES ('second')"))
            except BaseException as exc:
                failures.append(exc)

        first = threading.Thread(target=first_writer)
        second = threading.Thread(target=second_writer)
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        time.sleep(0.2)
        assert not second_entered.is_set()
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert failures == []
        with engine.connect() as connection:
            names = connection.exec_driver_sql("SELECT name FROM records ORDER BY name").scalars().all()
    finally:
        engine.dispose()

    assert names == ["first", "second"]


def test_immediate_session_scope_rolls_back_on_integrity_error(tmp_path: Path) -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(tmp_path / "state.db"))
    session_factory = create_session_factory(engine)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE records (name TEXT PRIMARY KEY)")

        with pytest.raises(IntegrityError), immediate_session_scope(session_factory) as session:
            session.execute(text("INSERT INTO records (name) VALUES ('alpha')"))
            session.execute(text("INSERT INTO records (name) VALUES ('alpha')"))

        with engine.connect() as connection:
            rows = connection.exec_driver_sql("SELECT name FROM records").fetchall()
    finally:
        engine.dispose()

    assert rows == []


def test_create_sqlite_engine_allows_plain_memory_database() -> None:
    engine = create_sqlite_engine(SqliteEngineConfig(":memory:"))
    try:
        with engine.connect() as connection:
            journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
            busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()

        assert str(journal_mode).lower() == "memory"
        assert busy_timeout == 5_000
    finally:
        engine.dispose()
