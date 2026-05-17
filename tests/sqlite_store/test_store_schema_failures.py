from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from store_helpers import _expect_table_columns, _fail_after_migration

from safe_fs_ops.sqlite_store import (
    SchemaDefinition,
    SchemaIdentityMismatchError,
    SchemaMigration,
    SchemaValidationError,
    SchemaVersionError,
    SqliteStore,
)

pytestmark = pytest.mark.safe_fs_ops


def test_sqlite_store_adopt_existing_schema_rejects_incorrect_existing_metadata(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize_schema(
        SchemaDefinition(
            identity="claims",
            migrations=(SchemaMigration(version=1, statements=("CREATE TABLE claims (name TEXT PRIMARY KEY)",)),),
        )
    )

    with pytest.raises(SchemaVersionError, match="already recorded at version 1, not requested version 2"):
        store.adopt_existing_schema(
            SchemaDefinition(
                identity="claims",
                migrations=(
                    SchemaMigration(version=1, statements=("CREATE TABLE claims (name TEXT PRIMARY KEY)",)),
                    SchemaMigration(
                        version=2,
                        statements=("ALTER TABLE claims ADD COLUMN owner TEXT NOT NULL DEFAULT ''",),
                    ),
                ),
                validate=_expect_table_columns("claims", ("name",)),
            ),
            version=2,
        )

    with pytest.raises(SchemaIdentityMismatchError, match="contains schema 'claims', not 'leases'"):
        store.adopt_existing_schema(
            SchemaDefinition(
                identity="leases",
                migrations=(SchemaMigration(version=1, statements=("CREATE TABLE leases (name TEXT PRIMARY KEY)",)),),
                validate=_expect_table_columns("leases", ("name",)),
            ),
            version=1,
        )


def test_sqlite_store_adopt_existing_schema_requires_validation_callback(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize(("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",))

    with pytest.raises(SchemaValidationError, match="requires a validation callback"):
        store.adopt_existing_schema(
            SchemaDefinition(
                identity="claims",
                migrations=(
                    SchemaMigration(
                        version=1,
                        statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
                    ),
                ),
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


def test_sqlite_store_adopt_existing_schema_rejects_mismatched_legacy_schema_without_stamping(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize(("CREATE TABLE unexpected (value TEXT NOT NULL)",))

    with pytest.raises(SchemaValidationError, match="missing expected table 'records'"):
        store.adopt_existing_schema(
            SchemaDefinition(
                identity="claims",
                migrations=(
                    SchemaMigration(
                        version=1,
                        statements=("CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",),
                    ),
                ),
                validate=_expect_table_columns("records", ("name", "value")),
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


def test_sqlite_store_initialize_schema_rolls_back_failed_migration(tmp_path: Path) -> None:
    store = SqliteStore(tmp_path / "state.db")
    definition = SchemaDefinition(
        identity="journal",
        migrations=(
            SchemaMigration(
                version=1,
                statements=(
                    "CREATE TABLE records (value TEXT NOT NULL)",
                    "INSERT INTO records (value) VALUES ('seed')",
                ),
            ),
            SchemaMigration(version=2, statements=("INSERT INTO missing_table (value) VALUES ('boom')",)),
        ),
    )

    with pytest.raises(sqlite3.OperationalError, match="missing_table"):
        store.initialize_schema(definition)

    with store.read_connection() as connection:
        metadata_exists = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'safe_fs_ops_schema_metadata'
            """
        ).fetchone()
        records_exists = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'records'
            """
        ).fetchone()
    assert metadata_exists is None
    assert records_exists is None


def test_sqlite_store_initialize_schema_rolls_back_failed_upgrade_and_keeps_metadata_version(
    tmp_path: Path,
) -> None:
    store = SqliteStore(tmp_path / "state.db")
    store.initialize_schema(
        SchemaDefinition(
            identity="journal",
            migrations=(
                SchemaMigration(
                    version=1,
                    statements=(
                        "CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
                        "INSERT INTO records (name, value) VALUES ('alpha', 'one')",
                    ),
                ),
            ),
        )
    )

    failing_definition = SchemaDefinition(
        identity="journal",
        migrations=(
            SchemaMigration(
                version=1,
                statements=(
                    "CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
                    "INSERT INTO records (name, value) VALUES ('alpha', 'one')",
                ),
            ),
            SchemaMigration(
                version=2,
                statements=(
                    "ALTER TABLE records ADD COLUMN note TEXT NOT NULL DEFAULT ''",
                    "CREATE TABLE migration_audit (marker TEXT NOT NULL)",
                ),
                apply=lambda connection: _fail_after_migration(connection, "v2"),
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="migration v2 failed"):
        store.initialize_schema(failing_definition)

    with store.read_connection() as connection:
        metadata_row = connection.execute(
            """
            SELECT schema_version
            FROM safe_fs_ops_schema_metadata
            """
        ).fetchone()
        columns = connection.execute("PRAGMA table_info(records)").fetchall()
        row = connection.execute("SELECT name, value FROM records").fetchone()
        audit_table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'migration_audit'
            """
        ).fetchone()
    assert metadata_row is not None
    assert int(metadata_row["schema_version"]) == 1
    assert [str(column["name"]) for column in columns] == ["name", "value"]
    assert row is not None
    assert str(row["name"]) == "alpha"
    assert str(row["value"]) == "one"
    assert audit_table is None

    recovered = store.initialize_schema(
        SchemaDefinition(
            identity="journal",
            migrations=(
                SchemaMigration(
                    version=1,
                    statements=(
                        "CREATE TABLE records (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
                        "INSERT INTO records (name, value) VALUES ('alpha', 'one')",
                    ),
                ),
                SchemaMigration(
                    version=2,
                    statements=(
                        "ALTER TABLE records ADD COLUMN note TEXT NOT NULL DEFAULT ''",
                        "CREATE TABLE migration_audit (marker TEXT NOT NULL)",
                    ),
                    apply=lambda connection: connection.execute("INSERT INTO migration_audit (marker) VALUES ('v2')"),
                ),
            ),
        )
    )

    assert recovered.version == 2
    with store.read_connection() as connection:
        metadata_row = connection.execute(
            """
            SELECT schema_version
            FROM safe_fs_ops_schema_metadata
            """
        ).fetchone()
        columns = connection.execute("PRAGMA table_info(records)").fetchall()
        audit_rows = connection.execute("SELECT marker FROM migration_audit").fetchall()
    assert metadata_row is not None
    assert int(metadata_row["schema_version"]) == 2
    assert [str(column["name"]) for column in columns] == ["name", "value", "note"]
    assert [str(row["marker"]) for row in audit_rows] == ["v2"]


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
