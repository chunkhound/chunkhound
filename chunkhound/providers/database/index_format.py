"""Nonmutating format inspection and writable-only legacy format stamps."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from chunkhound.core.index_format import (
    LEGACY_INDEX_FORMAT_VERSION,
    IndexFormatError,
    require_supported,
)
from chunkhound.providers.database.duckdb.schema_constants import (
    _SCHEMA_VERSION_TABLE_COLUMNS,
)

_INSPECTION_FAILED = (
    "Cannot safely inspect index; format compatibility is unknown, "
    "so automatic recovery is blocked. Some healthy WALs require "
    "writes to replay and cannot be inspected read-only. Preserve "
    "the database and any WAL/recovery files. Restore a known-good "
    "complete index or recover a copy using compatible DuckDB/VSS "
    "tooling, then retry. Reopening writable does not bypass this check."
)


def _stored_version(values: list[Any]) -> int:
    if not values:
        return LEGACY_INDEX_FORMAT_VERSION
    if any(type(value) is not int or value < 1 for value in values):
        raise IndexFormatError("Invalid stored index format")
    return cast(int, max(values))


@contextmanager
def read_only_duckdb(path: Path) -> Iterator[Any]:
    """Attach the original as ``inspected`` with the approved WAL safety policy.

    No provider migrations, automatic extension installation, temporary-copy
    recovery, or spill files. This is not lifetime/race protection.
    """
    import duckdb

    # Establish extension policy before opening a file that may replay WAL.
    try:
        with duckdb.connect(
            ":memory:",
            config={
                "autoinstall_known_extensions": "false",
                "autoload_known_extensions": "false",
                "temp_directory": "",
            },
        ) as conn:
            if Path(str(path) + ".wal").exists():
                # Arbitrary WAL may require VSS. Never continue replay after
                # a failed local LOAD, even if this WAL might be ordinary.
                try:
                    conn.execute("LOAD vss")
                    conn.execute("SET hnsw_enable_experimental_persistence = true")
                except duckdb.Error as exc:
                    raise IndexFormatError(
                        "Cannot safely inspect index: WAL is present but local "
                        "DuckDB VSS support could not be loaded or configured; "
                        "format compatibility is unknown, so automatic recovery "
                        "is blocked. Preserve the database and any WAL/recovery "
                        "files. Make a compatible VSS extension available using "
                        "an explicit setup operation, then retry. Reopening "
                        "writable does not bypass this check."
                    ) from exc
            escaped = str(path).replace("'", "''")
            conn.execute(f"ATTACH '{escaped}' AS inspected (READ_ONLY)")
            yield conn
    except duckdb.Error as exc:
        # A failed inspection is not evidence of corruption or a known
        # unsupported format. Some healthy HNSW WALs need writable replay.
        raise IndexFormatError(_INSPECTION_FAILED) from exc


def owned_duckdb_catalog(conn: Any, path: Path) -> str:
    """Bind a live handle to its owned file, never its mutable USE namespace."""
    catalogs = conn.execute(
        "SELECT database_name, path FROM system.main.duckdb_databases() "
        "WHERE NOT internal AND type='duckdb'"
    ).fetchall()
    matches = [
        name
        for name, filename in catalogs
        if filename and Path(filename).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise IndexFormatError(
            "Cannot safely reconnect: owned database catalog identity is ambiguous. "
            "Preserve live handles and pending work before retrying."
        )
    return cast(str, matches[0])


def require_stored_duckdb_table(conn: Any, catalog: str, table: str) -> None:
    """Refuse generated columns in owned contract tables before binding values.

    DuckDB's is_generated field is unset, and column_default conflates defaults
    with generated expressions. Inspect its canonical DDL instead. Remove quoted
    identifiers/literals (including doubled quotes) before checking DDL keywords;
    ordinary defaults containing those words remain supported. Requiring the
    whole contract table to be stored also covers empty tables without reading
    rows or evaluating publisher expressions.
    """
    row = conn.execute(
        "SELECT sql FROM system.main.duckdb_tables() "
        "WHERE database_name=? AND schema_name='main' "
        "AND translate(table_name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')=?",
        [catalog, table],
    ).fetchone()
    if row is None or not isinstance(row[0], str):
        raise IndexFormatError("Owned metadata must be a physical table")
    unquoted = re.sub(r"\"(?:[^\"]|\"\")*\"|'(?:[^']|'')*'", " ", row[0])
    if re.search(r"\bGENERATED\s+ALWAYS\s+AS\b", unquoted, flags=re.ASCII):
        raise IndexFormatError(
            "Owned metadata requires stored columns, not generated columns"
        )


def read_duckdb_format(conn: Any, *, catalog: str) -> int:
    """Read exactly one catalog's physical metadata, without evaluating views."""
    # Both kind and value lookup use the same fully qualified object. Three-part
    # qualification bypasses TEMP shadowing; catalog names are SQL identifiers.
    table = '"' + catalog.replace('"', '""') + '".main.schema_version'
    present = conn.execute(
        "SELECT table_type FROM system.information_schema.tables "
        "WHERE table_catalog=? AND table_schema='main' "
        "AND translate(table_name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')='schema_version'",
        [catalog],
    ).fetchone()
    if present is None:
        return LEGACY_INDEX_FORMAT_VERSION
    if present[0] != "BASE TABLE":
        raise IndexFormatError(
            "Invalid stored index format: schema_version must be a physical table"
        )
    require_stored_duckdb_table(conn, catalog, "schema_version")
    return _stored_version(
        [row[0] for row in conn.execute(f"SELECT version FROM {table}").fetchall()]
    )


def read_format(path: Path, backend: str) -> int:
    """Inspect existing metadata without creating paths or migrating the index."""
    if backend == "duckdb":
        with read_only_duckdb(path) as conn:
            return read_duckdb_format(conn, catalog="inspected")
    if backend == "lancedb":
        import lance  # type: ignore[import-untyped]  # Lance does not ship typing metadata.

        meta = path / "schema_version.lance"
        if not meta.exists():
            return LEGACY_INDEX_FORMAT_VERSION
        values = (
            lance.dataset(str(meta))
            .to_table(columns=["version"])["version"]
            .to_pylist()
        )
        return _stored_version(values)
    raise ValueError("Unsupported database backend")


def preflight_duckdb(path: Path | str, *, live_connection: Any | None = None) -> None:
    """Check every candidate before recovery, using an owned live handle if present.

    A partial reconnect must not attach the live file a second time or close its
    remaining handle before inspection. This is not lifetime/race protection.
    """
    if str(path) == ":memory:":
        return
    live = Path(path)
    if live_connection is not None:
        import duckdb

        try:
            catalog = owned_duckdb_catalog(live_connection, live)
            require_supported(read_duckdb_format(live_connection, catalog=catalog))
        except duckdb.Error as exc:
            raise IndexFormatError(_INSPECTION_FAILED) from exc
    # Both recovery implementations can choose a different file before opening live.
    for suffix in ("", ".compact_backup", ".compact_new", ".old", ".compact"):
        if not suffix and live_connection is not None:
            continue
        candidate = Path(str(live) + suffix)
        if candidate.is_file() and candidate.stat().st_size:
            require_supported(read_format(candidate, "duckdb"))


def stamp_duckdb(conn: Any, *, catalog: str) -> None:
    """Record format 1 only after checking existing physical, stored metadata."""
    require_supported(read_duckdb_format(conn, catalog=catalog))
    table = '"' + catalog.replace('"', '""') + '".main.schema_version'
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {table} ({_SCHEMA_VERSION_TABLE_COLUMNS})"
    )
    conn.execute(
        f"INSERT INTO {table}(version, description) "
        "SELECT ?, 'ChunkHound index format' "
        f"WHERE NOT EXISTS (SELECT 1 FROM {table})",
        [LEGACY_INDEX_FORMAT_VERSION],
    )


def stamp_lancedb(conn: Any) -> None:
    """Record format 1 without replacing existing Lance metadata."""
    # PyArrow does not ship typing metadata.
    import pyarrow as pa  # type: ignore[import-untyped]

    schema = pa.schema([("version", pa.int64()), ("description", pa.string())])
    rows = pa.Table.from_pylist(
        [
            {
                "version": LEGACY_INDEX_FORMAT_VERSION,
                "description": "ChunkHound index format",
            }
        ],
        schema=schema,
    )
    if "schema_version" not in conn.table_names():
        conn.create_table("schema_version", data=rows)
    elif conn.open_table("schema_version").count_rows() == 0:
        conn.open_table("schema_version").add(rows)
