"""Convert a ChunkHound LanceDB index into a DuckDB index.

Produces a DuckDB database that can be opened with ``provider=duckdb`` as if the
corpus had been indexed into DuckDB originally (files, chunks, embeddings + HNSW).

This is a one-shot materialization utility for the post-Lance-write "read backend"
path — not part of the cold-index hot path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chunkhound.providers.database.duckdb.schema_constants import (
    _CHUNKS_TABLE_COLUMNS,
    _FILES_TABLE_COLUMNS,
    _SCHEMA_VERSION_TABLE_COLUMNS,
    _create_embedding_table_sql,
    _embedding_chunk_id_index_name,
    _embedding_hnsw_index_name,
    _embedding_provider_model_index_name,
    _embedding_table_name,
    _embedding_unique_index_name,
)
from chunkhound.providers.database.lancedb_provider import _has_valid_embedding


@dataclass
class ConversionStats:
    """Summary of a Lance → DuckDB conversion run."""

    files: int = 0
    chunks: int = 0
    embeddings: int = 0
    embedding_dims: list[int] = field(default_factory=list)
    source_lance: str = ""
    dest_duckdb: str = ""
    skipped_invalid_embeddings: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "embedding_dims": list(self.embedding_dims),
            "source_lance": self.source_lance,
            "dest_duckdb": self.dest_duckdb,
            "skipped_invalid_embeddings": self.skipped_invalid_embeddings,
        }


def resolve_lancedb_dir(path: Path) -> Path:
    """Resolve a user path to the LanceDB directory (``*.lancedb``).

    Accepts:
    - the ``.lancedb`` directory itself
    - a database config directory that contains ``lancedb.lancedb``
    - a project root with ``.chunkhound/lancedb.lancedb``
    """
    path = path.expanduser().resolve()
    if path.is_dir() and path.name.endswith(".lancedb"):
        return path
    candidate = path / "lancedb.lancedb"
    if candidate.is_dir():
        return candidate
    candidate = path / ".chunkhound" / "lancedb.lancedb"
    if candidate.is_dir():
        return candidate
    # Older layouts
    for child in path.iterdir() if path.is_dir() else []:
        if child.is_dir() and child.name.endswith(".lancedb"):
            return child
    raise FileNotFoundError(
        f"No LanceDB directory found under {path}. "
        "Expected a path ending in .lancedb or a config dir with lancedb.lancedb"
    )


def resolve_duckdb_file(path: Path) -> Path:
    """Resolve destination to a DuckDB file path (``chunks.db`` by default)."""
    path = path.expanduser().resolve()
    if path.suffix.lower() in {".db", ".duckdb"}:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    return path / "chunks.db"


def _unix_to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def _path_name_ext(path: str) -> tuple[str, str]:
    p = Path(path.replace("\\", "/"))
    return p.name or path, p.suffix or ""


def _dedupe_rows_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        iid = int(rid)
        if iid in seen:
            continue
        seen.add(iid)
        out.append(row)
    return out


def _read_lance_tables(lance_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import lancedb

    db = lancedb.connect(str(lance_dir))
    try:
        files_tbl = db.open_table("files")
        chunks_tbl = db.open_table("chunks")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open LanceDB tables in {lance_dir}: {e}"
        ) from e

    def _table_rows(table: Any) -> list[dict[str, Any]]:
        # Full-table export — never silently cap (search().limit is not OK here).
        if hasattr(table, "to_arrow"):
            try:
                return table.to_arrow().to_pylist()
            except Exception:
                pass
        if hasattr(table, "to_pandas"):
            try:
                return table.to_pandas().to_dict(orient="records")
            except Exception:
                pass
        # Last resort: scanner if present
        if hasattr(table, "scanner"):
            try:
                return table.scanner().to_table().to_pylist()
            except Exception:
                pass
        raise RuntimeError(
            "Cannot full-scan Lance table (no to_arrow/to_pandas/scanner)"
        )

    files = _dedupe_rows_by_id(_table_rows(files_tbl))
    chunks = _dedupe_rows_by_id(_table_rows(chunks_tbl))
    return files, chunks


def _open_duck_for_write(dest_file: Path) -> Any:
    import duckdb

    # Match DuckDB provider: import numpy before threaded DuckDB use.
    try:
        import numpy  # noqa: F401
        import numpy.core.multiarray  # noqa: F401
    except ImportError:
        pass

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(dest_file))
    try:
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    except Exception:
        # HNSW optional if VSS unavailable; search still works without it (slower).
        pass
    return conn


def _create_duck_schema(conn: Any) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS schema_version ({_SCHEMA_VERSION_TABLE_COLUMNS})"
    )
    conn.execute("CREATE SEQUENCE IF NOT EXISTS files_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS chunks_id_seq")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq")
    conn.execute(f"CREATE TABLE IF NOT EXISTS files ({_FILES_TABLE_COLUMNS})")
    conn.execute(f"CREATE TABLE IF NOT EXISTS chunks ({_CHUNKS_TABLE_COLUMNS})")
    ver = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    if not ver or ver[0] is None:
        conn.execute(
            "INSERT INTO schema_version (version, description) "
            "VALUES (1, 'Imported from LanceDB')"
        )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)")


def _reseed_sequence(conn: Any, name: str, max_id: int) -> None:
    """Advance sequence so nextval returns > max_id (DuckDB cannot DROP in-use seq)."""
    if max_id < 1:
        return
    cur = conn.execute(
        "SELECT last_value FROM duckdb_sequences() "
        f"WHERE sequence_name = '{name}'"
    ).fetchone()
    current = int(cur[0]) if cur and cur[0] is not None else 0
    if current == 0:
        try:
            current = int(conn.execute(f"SELECT nextval('{name}')").fetchone()[0])
        except Exception:
            current = 0
    adv = int(max_id) - current
    if adv > 0:
        conn.execute(f"SELECT nextval('{name}') FROM range({adv})")


def _ensure_embedding_table(conn: Any, dims: int) -> str:
    table = _embedding_table_name(dims)
    exists = conn.execute(
        "SELECT 1 FROM information_schema.tables "
        f"WHERE table_schema = 'main' AND table_name = '{table}'"
    ).fetchone()
    if not exists:
        conn.execute(_create_embedding_table_sql(dims))
    return table


def _create_embedding_indexes(conn: Any, dims: int) -> None:
    table = _embedding_table_name(dims)
    try:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_embedding_hnsw_index_name(dims)} "
            f"ON {table} USING HNSW (embedding) WITH (metric = 'cosine')"
        )
    except Exception:
        pass
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_embedding_chunk_id_index_name(dims)} "
        f"ON {table}(chunk_id)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_embedding_provider_model_index_name(dims)} "
        f"ON {table}(provider, model)"
    )
    try:
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {_embedding_unique_index_name(dims)} "
            f"ON {table}(chunk_id, provider, model)"
        )
    except Exception:
        pass


def convert_lancedb_to_duckdb(
    source: Path,
    dest: Path,
    *,
    overwrite: bool = False,
    batch_size: int = 2000,
) -> ConversionStats:
    """Copy LanceDB files/chunks/embeddings into a DuckDB ``chunks.db``.

    Args:
        source: Lance dir, config dir, or project root containing LanceDB.
        dest: DuckDB file, or directory (writes ``chunks.db`` inside).
        overwrite: If True, replace an existing destination file.
        batch_size: Rows per INSERT batch.

    Returns:
        ConversionStats with counts and paths.
    """
    lance_dir = resolve_lancedb_dir(source)
    dest_file = resolve_duckdb_file(dest)
    stats = ConversionStats(
        source_lance=str(lance_dir),
        dest_duckdb=str(dest_file),
    )

    if dest_file.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest_file}. Pass overwrite=True."
            )
        dest_file.unlink()
        wal = dest_file.with_suffix(dest_file.suffix + ".wal")
        if wal.exists():
            wal.unlink()

    files_rows, chunks_rows = _read_lance_tables(lance_dir)
    conn = _open_duck_for_write(dest_file)
    try:
        _create_duck_schema(conn)

        # Lance uses 64-bit content-hash ids; DuckDB INTEGER PKs are 32-bit.
        # Remap to dense sequential ids so the result matches a native Duck index.
        file_id_map: dict[int, int] = {}
        file_batch: list[tuple[Any, ...]] = []
        for row in files_rows:
            old_fid = int(row["id"])
            new_fid = len(file_id_map) + 1
            file_id_map[old_fid] = new_fid
            path = str(row.get("path") or "")
            name, ext = _path_name_ext(path)
            file_batch.append(
                (
                    new_fid,
                    path,
                    name,
                    ext,
                    int(row["size"]) if row.get("size") is not None else None,
                    _unix_to_datetime(row.get("modified_time")),
                    str(row.get("content_hash") or "") or None,
                    str(row.get("language") or "") or None,
                    row.get("skip_reason"),
                )
            )
        for i in range(0, len(file_batch), batch_size):
            part = file_batch[i : i + batch_size]
            conn.executemany(
                """
                INSERT INTO files (
                    id, path, name, extension, size, modified_time,
                    content_hash, language, skip_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                part,
            )
        stats.files = len(file_batch)
        if file_batch:
            _reseed_sequence(conn, "files_id_seq", len(file_batch))

        # Chunks + collect embeddings
        emb_by_dims: dict[int, list[tuple[Any, ...]]] = {}
        chunk_batch: list[tuple[Any, ...]] = []
        chunk_id_map: dict[int, int] = {}
        for row in chunks_rows:
            old_cid = int(row["id"])
            new_cid = len(chunk_id_map) + 1
            chunk_id_map[old_cid] = new_cid
            old_fid = int(row["file_id"]) if row.get("file_id") is not None else None
            new_fid = file_id_map.get(old_fid) if old_fid is not None else None
            meta = row.get("metadata")
            if meta is not None and not isinstance(meta, str):
                try:
                    meta = json.dumps(meta)
                except Exception:
                    meta = None
            chunk_batch.append(
                (
                    new_cid,
                    new_fid,
                    str(row.get("chunk_type") or "unknown"),
                    str(row.get("name") or "") or None,
                    str(row.get("content") or ""),
                    int(row["start_line"]) if row.get("start_line") is not None else None,
                    int(row["end_line"]) if row.get("end_line") is not None else None,
                    None,  # start_byte
                    None,  # end_byte
                    str(row.get("language") or "") or None,
                    meta if meta else None,
                )
            )
            emb = row.get("embedding")
            if _has_valid_embedding(emb):
                vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                dims = len(vec)
                provider = str(row.get("provider") or "unknown")
                model = str(row.get("model") or "unknown")
                emb_by_dims.setdefault(dims, []).append(
                    (new_cid, provider, model, vec, dims)
                )
            elif emb is not None:
                stats.skipped_invalid_embeddings += 1

        for i in range(0, len(chunk_batch), batch_size):
            part = chunk_batch[i : i + batch_size]
            conn.executemany(
                """
                INSERT INTO chunks (
                    id, file_id, chunk_type, symbol, code,
                    start_line, end_line, start_byte, end_byte,
                    language, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                part,
            )
        stats.chunks = len(chunk_batch)
        if chunk_batch:
            _reseed_sequence(conn, "chunks_id_seq", len(chunk_batch))

        # Embeddings (separate table per dims)
        emb_id = 0
        for dims, rows in sorted(emb_by_dims.items()):
            table = _ensure_embedding_table(conn, dims)
            for i in range(0, len(rows), batch_size):
                part = rows[i : i + batch_size]
                payload = []
                for chunk_id, provider, model, vec, d in part:
                    emb_id += 1
                    payload.append((emb_id, chunk_id, provider, model, vec, d))
                conn.executemany(
                    f"""
                    INSERT INTO {table} (
                        id, chunk_id, provider, model, embedding, dims
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
            stats.embeddings += len(rows)
            stats.embedding_dims.append(dims)
            _create_embedding_indexes(conn, dims)

        if emb_id > 0:
            _reseed_sequence(conn, "embeddings_id_seq", emb_id)

        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    return stats


def activate_duckdb_in_config(
    project_dir: Path,
    *,
    database_path: str | Path = ".chunkhound",
    config_name: str = ".chunkhound.json",
    require_db_exists: bool = False,
) -> Path:
    """Point project config at DuckDB so subsequent runs use the converted index.

    Creates or updates ``.chunkhound.json`` with::

        {"database": {"provider": "duckdb", "path": "<database_path>"}}

    Merges into existing JSON; **refuses** to rewrite if the file exists but is
    not valid JSON (avoids wiping embedding/indexing settings).

    Returns path to the written config file.
    """
    project_dir = project_dir.expanduser().resolve()
    config_path = project_dir / config_name
    data: dict[str, Any] = {}
    if config_path.is_file():
        raw = config_path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Cannot activate: {config_path} is not valid JSON ({e}). "
                "Fix or remove the file, then retry."
            ) from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"Cannot activate: {config_path} root must be a JSON object"
            )
        data = parsed
    db = data.get("database")
    if not isinstance(db, dict):
        db = {}
    db["provider"] = "duckdb"
    db["path"] = str(database_path).replace("\\", "/")
    data["database"] = db

    if require_db_exists:
        # Resolve like DatabaseConfig: dir → chunks.db, or explicit .db file
        p = Path(db["path"])
        if not p.is_absolute():
            p = project_dir / p
        if p.suffix.lower() in {".db", ".duckdb"}:
            duck_file = p
        else:
            duck_file = p / "chunks.db"
        if not duck_file.is_file():
            raise FileNotFoundError(
                f"Cannot activate: DuckDB file not found at {duck_file}. "
                "Run conversion first (without --activate-only)."
            )

    config_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path


def config_path_for_dest(
    project_dir: Path, dest_duckdb_file: Path
) -> str:
    """Choose database.path for config so get_db_path() opens dest_duckdb_file.

    - Explicit ``*.db`` / ``*.duckdb`` file → path is that file.
    - Otherwise → parent directory (product opens ``parent/chunks.db``).
    """
    dest_duckdb_file = dest_duckdb_file.expanduser().resolve()
    project_dir = project_dir.expanduser().resolve()
    if dest_duckdb_file.suffix.lower() in {".db", ".duckdb"}:
        target = dest_duckdb_file
    else:
        target = dest_duckdb_file.parent
    try:
        return target.relative_to(project_dir).as_posix()
    except ValueError:
        return str(target).replace("\\", "/")


def convert_and_activate(
    project_dir: Path,
    *,
    source: Path | None = None,
    dest: Path | None = None,
    overwrite: bool = False,
    activate: bool = True,
    database_path: str | Path | None = None,
) -> ConversionStats:
    """Convert project LanceDB → DuckDB and optionally switch config to DuckDB.

    Default layout (ChunkHound conventions)::

        source:  <project>/.chunkhound/lancedb.lancedb
        dest:    <project>/.chunkhound/chunks.db
        config:  database.provider=duckdb, database.path=.chunkhound
    """
    project_dir = project_dir.expanduser().resolve()
    src = source or (project_dir / ".chunkhound")
    dst = dest or (project_dir / ".chunkhound")
    stats = convert_lancedb_to_duckdb(src, dst, overwrite=overwrite)
    if activate:
        if database_path is not None:
            cfg_path_val = str(database_path).replace("\\", "/")
        else:
            cfg_path_val = config_path_for_dest(
                project_dir, Path(stats.dest_duckdb)
            )
        activate_duckdb_in_config(project_dir, database_path=cfg_path_val)
    return stats
