"""Read-only source readability and embedding inventory for staged databases.

This does not validate every stored page, HNSW search, or active-model sidecars.
It never opens providers, runs migrations, or chooses an active model.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chunkhound.core.embedding_model_drift import ModelGroup
from chunkhound.providers.database.index_format import read_format, read_only_duckdb


@dataclass(frozen=True)
class Inventory:
    """Persisted format and all stored embedding identities, without selection."""

    format_version: int
    groups: list[ModelGroup]


def _time(value: object) -> float:
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value) if isinstance(value, int | float) else float("-inf")


def inspect_database(path: Path, backend: str) -> Inventory:
    """Read actual backend storage without modifying the database or sidecars."""
    if not path.exists():
        raise ValueError("Database is absent")
    version = read_format(path, backend)
    rows = []
    if backend == "duckdb":
        # A direct file reopen would bypass the extension/WAL safety boundary.
        with read_only_duckdb(path) as conn:
            # Fold only ASCII, matching DuckDB identifier resolution. Check
            # object kinds before binding any publisher-defined source/vector SQL.
            objects = dict(
                conn.execute(
                    "SELECT translate(table_name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                    "'abcdefghijklmnopqrstuvwxyz'), table_type "
                    "FROM information_schema.tables "
                    "WHERE table_catalog='inspected' AND table_schema='main'"
                ).fetchall()
            )
            if not {"files", "chunks"}.issubset(objects):
                raise ValueError("Database lacks required tables")
            names = {
                name
                for name in objects
                if name in {"files", "chunks"}
                or re.fullmatch(r"embeddings_([0-9]+)", name)
            }
            if any(objects[name] != "BASE TABLE" for name in names):
                raise ValueError(
                    "Stored source and vector objects must be physical tables"
                )
            conn.execute(
                "SELECT id, path, size, modified_time, content_hash, language "
                "FROM inspected.main.files LIMIT 1"
            ).fetchall()
            conn.execute(
                "SELECT id, file_id, chunk_type, symbol, code, start_line, end_line, "
                "start_byte, end_byte, language, metadata "
                "FROM inspected.main.chunks LIMIT 1"
            ).fetchall()
            for name in sorted(names):
                match = re.fullmatch(r"embeddings_([0-9]+)", name)
                if not match:
                    continue
                dims = int(match.group(1))
                table = f'inspected.main."{name}"'
                # Retrieval joins vectors to chunks and requires real float arrays.
                columns = conn.execute(
                    f"SELECT chunk_id, embedding FROM {table} LIMIT 0"
                ).description
                if str(columns[1][1]) != f"FLOAT[{dims}]":
                    raise ValueError("Stored embedding type is not a float vector")
                malformed = conn.execute(
                    f"SELECT count(*) FROM {table} WHERE dims != ? OR dims IS NULL "
                    "OR embedding IS NULL OR array_length(embedding) != ?",
                    [dims, dims],
                ).fetchone()[0]
                if malformed:
                    raise ValueError("Stored vector dimensions are inconsistent")
                rows.extend(
                    conn.execute(
                        # BLOB keys discard publisher-supplied string collations;
                        # decode only after exact SQL-side aggregation.
                        "SELECT decode(encode(provider)), decode(encode(model)), "
                        f"dims, count(*), max(created_at) FROM {table} "
                        "GROUP BY encode(provider), encode(model), dims"
                    ).fetchall()
                )
    elif backend == "lancedb":
        # These third-party packages do not ship typing metadata.
        import lance  # type: ignore[import-untyped]
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.compute as pc  # type: ignore[import-untyped]

        files = lance.dataset(str(path / "files.lance"))
        files.head(
            1,
            columns=["id", "path", "size", "modified_time", "content_hash", "language"],
        )
        dataset = lance.dataset(str(path / "chunks.lance"))
        dataset.head(
            1,
            columns=[
                "id",
                "file_id",
                "content",
                "start_line",
                "end_line",
                "chunk_type",
                "language",
                "name",
                "metadata",
            ],
        )
        embedding_type = dataset.schema.field("embedding").type
        if not (
            pa.types.is_fixed_size_list(embedding_type)
            or pa.types.is_list(embedding_type)
            or pa.types.is_large_list(embedding_type)
        ) or not pa.types.is_floating(embedding_type.value_type):
            raise ValueError("Stored embedding type is not a float vector")
        fixed = getattr(embedding_type, "list_size", None)
        columns = ["provider", "model", "created_time"]
        if fixed is None:
            columns.append("embedding")
        groups: dict[tuple[str, str, int], tuple[int, float]] = {}
        for batch in dataset.scanner(
            columns=columns, filter="embedding IS NOT NULL", batch_size=8192
        ).to_batches():
            values = batch.to_pydict()
            lengths = (
                [fixed] * batch.num_rows
                if fixed is not None
                else pc.list_value_length(batch.column("embedding")).to_pylist()
            )
            for provider, model, dims, latest in zip(
                values["provider"], values["model"], lengths, values["created_time"]
            ):
                key = (provider, model, dims)
                count, newest = groups.get(key, (0, float("-inf")))
                groups[key] = (count + 1, max(newest, _time(latest)))
        rows = [
            (provider, model, dims, count, newest)
            for (provider, model, dims), (count, newest) in groups.items()
        ]
    else:
        raise ValueError("Unsupported database backend")
    if any(not p or not m or not d or d < 1 for p, m, d, count, latest in rows):
        raise ValueError("Stored embedding identity is incomplete")
    return Inventory(
        version,
        [
            ModelGroup(p, m, int(d), int(count), _time(latest))
            for p, m, d, count, latest in rows
        ],
    )
