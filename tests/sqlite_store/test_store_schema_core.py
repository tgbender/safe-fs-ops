from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from store_helpers import _expect_table_columns

from safe_fs_ops.sqlite_store import (
    SchemaDefinition,
    SchemaIdentityMismatchError,
    SchemaMigration,
    SchemaVersionError,
    SqliteStore,
)

pytestmark = pytest.mark.safe_fs_ops


def test_sqlite_store_initialize_schema_records_initial_version(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    definition = SchemaDefinition(
        identity="leases",
        migrations=(
            SchemaMigration(
                version=1,
                statements=("CREATE TABLE records (value TEXT NOT NULL)",),
            ),
        ),
    )

    status = store.initialize_schema(definition)

    assert status.identity == "leases"
    assert status.version == 1
    with store.read_connection() as connection:
        row = connection.execute(
            """
            SELECT schema_identity, schema_version
            FROM safe_fs_ops_schema_metadata
            """
        ).fetchone()
    assert row is not None
    assert str(row["schema_identity"]) == "leases"
    assert int(row["schema_version"]) == 1


def test_sqlite_store_initialize_schema_applies_migrations_in_order_only_once(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    applied: list[str] = []

    def append_marker(connection: sqlite3.Connection, marker: str) -> None:
        applied.append(marker)
        connection.execute("INSERT INTO migration_audit (marker) VALUES (?)", (marker,))

    definition = SchemaDefinition(
        identity="journal",
        migrations=(
            SchemaMigration(
                version=1,
                statements=(
                    "CREATE TABLE records (value TEXT NOT NULL)",
                    "CREATE TABLE migration_audit (marker TEXT NOT NULL)",
                ),
                apply=lambda connection: append_marker(connection, "v1"),
            ),
            SchemaMigration(
                version=2,
                statements=("ALTER TABLE records ADD COLUMN note TEXT NOT NULL DEFAULT ''",),
                apply=lambda connection: append_marker(connection, "v2"),
            ),
            SchemaMigration(version=3, apply=lambda connection: append_marker(connection, "v3")),
        ),
    )

    first = store.initialize_schema(definition)
    second = store.initialize_schema(definition)

    assert first.version == 3
    assert second.version == 3
    assert applied == ["v1", "v2", "v3"]
    with store.read_connection() as connection:
        audit_rows = connection.execute("SELECT marker FROM migration_audit ORDER BY rowid").fetchall()
    assert [str(row["marker"]) for row in audit_rows] == ["v1", "v2", "v3"]


def test_sqlite_store_schema_definition_rejects_version_gaps() -> None:
    with pytest.raises(ValueError, match="start at version 1 and increase by 1"):
        SchemaDefinition(
            identity="claims",
            migrations=(
                SchemaMigration(version=1, statements=("CREATE TABLE claims (name TEXT PRIMARY KEY)",)),
                SchemaMigration(
                    version=3, statements=("ALTER TABLE claims ADD COLUMN owner TEXT NOT NULL DEFAULT ''",)
                ),
            ),
        )


def test_sqlite_store_initialize_schema_rejects_identity_mismatch(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize_schema(
        SchemaDefinition(
            identity="leases",
            migrations=(SchemaMigration(version=1, statements=("CREATE TABLE leases (name TEXT PRIMARY KEY)",)),),
        )
    )

    with pytest.raises(SchemaIdentityMismatchError, match="contains schema 'leases', not 'claims'"):
        store.initialize_schema(
            SchemaDefinition(
                identity="claims",
                migrations=(SchemaMigration(version=1, statements=("CREATE TABLE claims (name TEXT PRIMARY KEY)",)),),
            )
        )


def test_sqlite_store_initialize_schema_rejects_newer_existing_version(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize_schema(
        SchemaDefinition(
            identity="leases",
            migrations=(
                SchemaMigration(version=1, statements=("CREATE TABLE leases (name TEXT PRIMARY KEY)",)),
                SchemaMigration(
                    version=2, statements=("ALTER TABLE leases ADD COLUMN owner TEXT NOT NULL DEFAULT ''",)
                ),
            ),
        )
    )

    with pytest.raises(SchemaVersionError, match="newer than supported version 1"):
        store.initialize_schema(
            SchemaDefinition(
                identity="leases",
                migrations=(SchemaMigration(version=1, statements=("CREATE TABLE leases (name TEXT PRIMARY KEY)",)),),
            )
        )


def test_sqlite_store_initialize_schema_migrates_forward_preserving_data(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = SqliteStore(path)
    v1 = SchemaDefinition(
        identity="claims",
        migrations=(
            SchemaMigration(
                version=1,
                statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
            ),
        ),
    )
    store.initialize_schema(v1)
    with store.transaction() as connection:
        connection.execute("INSERT INTO records (name, value) VALUES (?, ?)", ("alpha", "one"))

    upgraded = SqliteStore(path)
    v2 = SchemaDefinition(
        identity="claims",
        migrations=(
            SchemaMigration(
                version=1,
                statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
            ),
            SchemaMigration(
                version=2,
                statements=("ALTER TABLE records ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",),
            ),
        ),
    )

    status = upgraded.initialize_schema(v2)

    assert status.version == 2
    with upgraded.read_connection() as connection:
        row = connection.execute("SELECT name, value, status FROM records").fetchone()
    assert row is not None
    assert str(row["name"]) == "alpha"
    assert str(row["value"]) == "one"
    assert str(row["status"]) == "active"


def test_sqlite_store_adopts_legacy_schema_and_upgrades_preserving_data(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store = SqliteStore(path)
    store.initialize(("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",))
    with store.transaction() as connection:
        connection.execute("INSERT INTO records (name, value) VALUES (?, ?)", ("alpha", "one"))

    definition = SchemaDefinition(
        identity="claims",
        migrations=(
            SchemaMigration(
                version=1,
                statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
            ),
            SchemaMigration(
                version=2,
                statements=("ALTER TABLE records ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",),
            ),
        ),
        validate=_expect_table_columns("records", ("name", "value")),
    )

    adopted = store.adopt_existing_schema(definition, version=1)
    assert adopted.identity == "claims"
    assert adopted.version == 1

    upgraded = SqliteStore(path)
    status = upgraded.initialize_schema(definition)

    assert status.identity == "claims"
    assert status.version == 2
    with upgraded.read_connection() as connection:
        row = connection.execute("SELECT name, value, status FROM records").fetchone()
    assert row is not None
    assert str(row["name"]) == "alpha"
    assert str(row["value"]) == "one"
    assert str(row["status"]) == "active"


def test_sqlite_store_adopt_existing_schema_rejects_invalid_version(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    definition = SchemaDefinition(
        identity="claims",
        migrations=(
            SchemaMigration(
                version=1,
                statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
            ),
            SchemaMigration(
                version=2,
                statements=("ALTER TABLE records ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",),
            ),
        ),
        validate=_expect_table_columns("records", ("name", "value")),
    )

    with pytest.raises(SchemaVersionError, match="must be at least 1"):
        store.adopt_existing_schema(definition, version=0)


def test_sqlite_store_initialize_with_schema_strings_remains_compatible(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")

    store.initialize(
        (
            "CREATE TABLE IF NOT EXISTS leases (name TEXT PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS claims (name TEXT PRIMARY KEY, lease_name TEXT NOT NULL)",
        )
    )

    with store.transaction() as connection:
        connection.execute("INSERT INTO leases (name) VALUES ('lease-1')")
        connection.execute("INSERT INTO claims (name, lease_name) VALUES ('claim-1', 'lease-1')")
    with store.read_connection() as connection:
        row = connection.execute(
            """
            SELECT claims.name AS claim_name, claims.lease_name AS lease_name
            FROM claims
            """
        ).fetchone()
    assert row is not None
    assert str(row["claim_name"]) == "claim-1"
    assert str(row["lease_name"]) == "lease-1"
