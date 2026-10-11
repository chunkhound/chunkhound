"""Nonmutating format inspection and writable-only legacy format stamps."""

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


def _stored_version(values: list[Any]) -> int:
    if not values:
        return LEGACY_INDEX_FORMAT_VERSION
    if any(type(value) is not int or value < 1 for value in values):
        raise IndexFormatError("Invalid stored index format")
    return cast(int, max(values))


def read_format(path: Path, backend: str) -> int:
    """Inspect existing metadata without creating paths or migrating the index."""
    if backend == "duckdb":
        import duckdb

        # Establish extension policy before opening a file that may replay WAL.
        # In-memory inspection must not install extensions or spill to disk.
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
                present = conn.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_catalog='inspected' AND table_schema='main' "
                    "AND table_name='schema_version'"
                ).fetchone()
                return _stored_version(
                    [
                        row[0]
                        for row in conn.execute(
                            "SELECT version FROM inspected.main.schema_version"
                        ).fetchall()
                    ]
                    if present
                    else []
                )
        except duckdb.Error as exc:
            # A failed inspection is not evidence of corruption or a known
            # unsupported format. Some healthy HNSW WALs need writable replay.
            raise IndexFormatError(
                "Cannot safely inspect index; format compatibility is unknown, "
                "so automatic recovery is blocked. Some healthy WALs require "
                "writes to replay and cannot be inspected read-only. Preserve "
                "the database and any WAL/recovery files. Restore a known-good "
                "complete index or recover a copy using compatible DuckDB/VSS "
                "tooling, then retry. Reopening writable does not bypass this check."
            ) from exc
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


def preflight_duckdb(path: Path | str) -> None:
    """Check all candidates before recovery; this is not lifetime/race protection."""
    if str(path) == ":memory:":
        return
    live = Path(path)
    # Both recovery implementations can choose a different file before opening live.
    for suffix in ("", ".compact_backup", ".compact_new", ".old", ".compact"):
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
