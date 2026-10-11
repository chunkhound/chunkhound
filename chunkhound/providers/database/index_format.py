"""Nonmutating format inspection and writable-only legacy format stamps."""

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


def read_duckdb_format(conn: Any, *, inspected: bool = False) -> int:
    """Read physical metadata on an existing connection, without evaluating views."""
    catalog = "'inspected'" if inspected else "current_database()"
    table = "inspected.main.schema_version" if inspected else "main.schema_version"
    # DuckDB preserves spelling but resolves identifiers case-insensitively
    # using ASCII folding, not Unicode lower/casefold or publisher collation.
    present = conn.execute(
        "SELECT table_type FROM information_schema.tables "
        f"WHERE table_catalog={catalog} AND table_schema='main' "
        "AND translate(table_name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')='schema_version'"
    ).fetchone()
    if present is None:
        return LEGACY_INDEX_FORMAT_VERSION
    if present[0] != "BASE TABLE":
        raise IndexFormatError(
            "Invalid stored index format: schema_version must be a physical table"
        )
    return _stored_version(
        [row[0] for row in conn.execute(f"SELECT version FROM {table}").fetchall()]
    )


def read_format(path: Path, backend: str) -> int:
    """Inspect existing metadata without creating paths or migrating the index."""
    if backend == "duckdb":
        with read_only_duckdb(path) as conn:
            return read_duckdb_format(conn, inspected=True)
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
            require_supported(read_duckdb_format(live_connection))
        except duckdb.Error as exc:
            raise IndexFormatError(_INSPECTION_FAILED) from exc
    # Both recovery implementations can choose a different file before opening live.
    for suffix in ("", ".compact_backup", ".compact_new", ".old", ".compact"):
        if not suffix and live_connection is not None:
            continue
        candidate = Path(str(live) + suffix)
        if candidate.is_file() and candidate.stat().st_size:
            require_supported(read_format(candidate, "duckdb"))


def stamp_duckdb(conn: Any) -> None:
    """Record format 1 for writable new/legacy databases using canonical DDL."""
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS schema_version ({_SCHEMA_VERSION_TABLE_COLUMNS})"
    )
    conn.execute(
        "INSERT INTO schema_version(version, description) "
        "SELECT ?, 'ChunkHound index format' "
        "WHERE NOT EXISTS (SELECT 1 FROM schema_version)",
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
