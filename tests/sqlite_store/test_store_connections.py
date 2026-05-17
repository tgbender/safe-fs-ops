from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from safe_fs_ops.filesystem_ops import UnsafePathError
from safe_fs_ops.sqlite_store import SqliteStore

pytestmark = pytest.mark.safe_fs_ops


def test_sqlite_store_opens_short_lived_connections_per_transaction(tmp_path: Path) -> None:
    opened: list[sqlite3.Connection] = []
    closed: list[int] = []

    class TrackingConnection(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

        def close(self) -> None:
            closed.append(id(self))
            super().close()

    def factory(path: Path | str) -> sqlite3.Connection:
        return TrackingConnection(str(path))

    store = SqliteStore(tmp_path / "state.db", connection_factory=factory)

    with store.transaction() as connection:
        connection.execute("CREATE TABLE records (value TEXT)")
    with store.transaction() as connection:
        connection.execute("INSERT INTO records (value) VALUES ('a')")

    opened_ids = [id(connection) for connection in opened]
    assert len(opened_ids) == 2
    assert len(set(opened_ids)) == 2
    assert closed == opened_ids


def test_sqlite_store_read_connection_does_not_begin_immediate(tmp_path: Path) -> None:
    statements: list[str] = []

    class TrackingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=(), /):  # type: ignore[no-untyped-def]
            statements.append(str(sql))
            return super().execute(sql, parameters)

    def factory(path: Path | str) -> sqlite3.Connection:
        return TrackingConnection(str(path))

    store = SqliteStore(tmp_path / "state.db", connection_factory=factory)

    with store.read_connection() as connection:
        connection.execute("SELECT 1")

    assert "BEGIN IMMEDIATE" not in statements


def test_sqlite_store_rejects_memory_path() -> None:
    with pytest.raises(ValueError, match="short-lived connections"):
        SqliteStore(":memory:")


def test_sqlite_store_rejects_symlink_state_parent_before_opening_db(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    state_parent = tmp_path / "state"
    try:
        state_parent.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(UnsafePathError, match="redirects"):
        SqliteStore(state_parent / "state.db")

    assert list(outside.iterdir()) == []


def test_sqlite_store_does_not_share_connections_between_threads(tmp_path: Path) -> None:
    seen: list[tuple[int, sqlite3.Connection]] = []
    seen_lock = threading.Lock()

    class TrackingConnection(sqlite3.Connection):
        pass

    def factory(path: Path | str) -> sqlite3.Connection:
        return TrackingConnection(str(path))

    store = SqliteStore(tmp_path / "state.db", connection_factory=factory)
    store.initialize(["CREATE TABLE IF NOT EXISTS records (thread_id INTEGER)"])

    def write_record() -> None:
        with store.transaction() as connection:
            thread_id = threading.get_ident()
            connection.execute("INSERT INTO records (thread_id) VALUES (?)", (thread_id,))
            with seen_lock:
                seen.append((thread_id, connection))

    threads = [threading.Thread(target=write_record) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(seen) == 6
    assert len({id(connection) for _, connection in seen}) == 6
