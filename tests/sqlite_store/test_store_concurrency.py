from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from store_helpers import _collect_results, _initialize_schema_worker

from safe_fs_ops.sqlite_store import SchemaDefinition, SchemaMigration, SchemaStatus, SqliteStore

pytestmark = pytest.mark.safe_fs_ops


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
