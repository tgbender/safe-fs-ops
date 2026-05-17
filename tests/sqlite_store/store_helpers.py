from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from safe_fs_ops.sqlite_store import (
    SchemaDefinition,
    SchemaMigration,
    SchemaStatus,
    SchemaValidationError,
    SqliteStore,
)

pytestmark = pytest.mark.safe_fs_ops


def _expect_table_columns(table_name: str, expected_columns: tuple[str, ...]):
    def _validate(connection: sqlite3.Connection, version: int) -> None:
        del version
        table_row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (table_name,),
        ).fetchone()
        if table_row is None:
            raise SchemaValidationError(f"missing expected table {table_name!r}")
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        actual_columns = tuple(str(row["name"]) for row in rows)
        if actual_columns != expected_columns:
            raise SchemaValidationError(
                f"table {table_name!r} columns {actual_columns!r} did not match expected {expected_columns!r}"
            )

    return _validate


def _fail_after_migration(connection: sqlite3.Connection, marker: str) -> None:
    del connection
    raise RuntimeError(f"migration {marker} failed")


def _initialize_schema_worker(
    state_path: str,
    started: Any,
    release: Any,
    results: Any,
    *,
    connection_timeout: float,
) -> None:
    try:
        applied = False

        def hold(connection: sqlite3.Connection) -> None:
            nonlocal applied
            applied = True
            started.set()
            release.wait(timeout=5)
            connection.execute("SELECT 1")

        store = SqliteStore(
            Path(state_path),
            connection_factory=lambda path: sqlite3.connect(path, timeout=connection_timeout),
        )
        definition = SchemaDefinition(
            identity="journal",
            migrations=(
                SchemaMigration(
                    version=1,
                    statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
                    apply=hold,
                ),
            ),
        )
        status = store.initialize_schema(definition)
        results.put({"version": status.version, "applied": applied})
    except BaseException as exc:
        results.put({"error": f"{type(exc).__name__}: {exc}"})


def _collect_results(results: Any, count: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for _ in range(count):
        try:
            collected.append(results.get(timeout=10))
        except Empty:
            break
    return collected


def test_sqlite_store_initialize_schema_serializes_concurrent_threads(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    started = threading.Event()
    release = threading.Event()
    results: list[SchemaStatus] = []
    errors: list[BaseException] = []
    callback_runs = 0
    results_lock = threading.Lock()

    def run_initializer() -> None:
        try:

            def hold(connection: sqlite3.Connection) -> None:
                nonlocal callback_runs
                with results_lock:
                    callback_runs += 1
                started.set()
                release.wait(timeout=5)
                connection.execute("SELECT 1")

            store = SqliteStore(
                state_path,
                connection_factory=lambda path: sqlite3.connect(path, timeout=30.0),
            )
            status = store.initialize_schema(
                SchemaDefinition(
                    identity="journal",
                    migrations=(
                        SchemaMigration(
                            version=1,
                            statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
                            apply=hold,
                        ),
                    ),
                )
            )
            with results_lock:
                results.append(status)
        except BaseException as exc:
            with results_lock:
                errors.append(exc)

    first = threading.Thread(target=run_initializer)
    second = threading.Thread(target=run_initializer)
    first.start()
    assert started.wait(timeout=5)
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert errors == []
    assert len(results) == 2
    assert {status.version for status in results} == {1}
    assert callback_runs == 1
    with SqliteStore(
        state_path,
        connection_factory=lambda path: sqlite3.connect(path, timeout=30.0),
    ).read_connection() as connection:
        metadata_row = connection.execute(
            """
            SELECT schema_identity, schema_version
            FROM safe_fs_ops_schema_metadata
            """
        ).fetchone()
    assert metadata_row is not None
    assert str(metadata_row["schema_identity"]) == "journal"
    assert int(metadata_row["schema_version"]) == 1


@pytest.mark.stress_lock
def test_sqlite_store_initialize_schema_serializes_concurrent_processes(tmp_path: Path) -> None:
    state_path = tmp_path / "state.db"
    ctx = multiprocessing.get_context("spawn")
    started = ctx.Event()
    release = ctx.Event()
    results = ctx.Queue()
    process_count = 2
    processes = [
        ctx.Process(
            target=_initialize_schema_worker,
            args=(str(state_path), started, release, results),
            kwargs={"connection_timeout": 30.0},
        )
        for _ in range(process_count)
    ]

    processes[0].start()
    assert started.wait(timeout=5)
    processes[1].start()
    time.sleep(0.1)
    release.set()
    collected = _collect_results(results, process_count)
    for process in processes:
        process.join(timeout=10)

    assert len(collected) == process_count
    assert [item for item in collected if "error" in item] == []
    assert [item["applied"] for item in collected].count(True) == 1
    assert [item["version"] for item in collected] == [1, 1]
    assert all(process.exitcode == 0 for process in processes)


def test_sqlite_store_adopt_existing_schema_preserves_schema_validation_error(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize(("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",))

    def validate(_: sqlite3.Connection, __: int) -> None:
        raise SchemaValidationError("legacy schema rejected")

    with pytest.raises(SchemaValidationError, match="legacy schema rejected"):
        store.adopt_existing_schema(
            SchemaDefinition(
                identity="claims",
                migrations=(
                    SchemaMigration(
                        version=1,
                        statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
                    ),
                ),
                validate=validate,
            ),
            version=1,
        )

    with store.read_connection() as connection:
        metadata_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'safe_fs_ops_schema_metadata'
            """
        ).fetchone()
    assert metadata_table is None


__all__ = [name for name in globals() if not name.startswith("__")]
