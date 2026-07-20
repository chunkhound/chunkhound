"""Convert a ChunkHound LanceDB index into a DuckDB index.

Produces a DuckDB database that can be opened with ``provider=duckdb`` as if the
corpus had been indexed into DuckDB originally (files, chunks, embeddings + HNSW).

This is a one-shot materialization utility for the post-Lance-write "read backend"
path — not part of the cold-index hot path.

**Memory:** Lance tables are scanned in Arrow record batches (streaming). DuckDB
inserts are batched. Embedding vectors are not accumulated for the full corpus;
HNSW indexes are built only after all vectors for a dimension are loaded.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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

logger = logging.getLogger(__name__)

# Default Lance scan / Duck insert batch size (rows). Keep modest so peak RSS
# stays ~O(batch × dims × sizeof(float)) not O(corpus).
_DEFAULT_BATCH_SIZE = 2_000

ProgressFn = Callable[[str], None]


def _default_progress(msg: str) -> None:
    """Print progress to stderr (flush) so stdout stays free for --json."""
    print(msg, file=sys.stderr, flush=True)
    logger.debug("%s", msg)


def _format_bytes(n: int) -> str:
    """Human-readable byte size."""
    size = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{n} B"


def _dir_size_bytes(path: Path) -> int:
    """Sum file sizes under a directory (best-effort)."""
    total = 0
    try:
        if not path.is_dir():
            return path.stat().st_size if path.is_file() else 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except OSError:
        return total
    return total


def _duck_on_disk_bytes(dest_file: Path) -> int:
    """DuckDB file + WAL size if present."""
    total = 0
    for p in (dest_file, dest_file.with_suffix(dest_file.suffix + ".wal")):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _pct(current: int, total: int | None) -> str:
    if total is None or total <= 0:
        return "?"
    return f"{min(100.0, 100.0 * current / total):.1f}%"


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
    compacted: bool = False
    stream_batch_size: int = _DEFAULT_BATCH_SIZE

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "embedding_dims": list(self.embedding_dims),
            "source_lance": self.source_lance,
            "dest_duckdb": self.dest_duckdb,
            "skipped_invalid_embeddings": self.skipped_invalid_embeddings,
            "compacted": self.compacted,
            "stream_batch_size": self.stream_batch_size,
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


def _table_row_count(table: Any) -> int | None:
    """Best-effort row count for safety checks (None if unavailable)."""
    for attr in ("count_rows", "count"):
        if hasattr(table, attr):
            try:
                return int(getattr(table, attr)())
            except Exception:
                pass
    return None


def _iter_lance_table_batches(
    table: Any,
    *,
    batch_size: int,
    allow_full_scan: bool = False,
) -> Iterator[list[dict[str, Any]]]:
    """Yield row dicts from a Lance table without full-table materialization.

    Prefers ``table.to_lance().to_batches(batch_size=...)`` (Lance 0.34+).
    Full-table materialization is refused for large tables unless
    ``allow_full_scan`` is True (avoids silent OOM on multi-million-row indexes).
    """
    stream_errors: list[str] = []

    # Preferred: native Lance dataset streaming
    if hasattr(table, "to_lance"):
        try:
            dataset = table.to_lance()
            if hasattr(dataset, "to_batches"):
                for batch in dataset.to_batches(batch_size=batch_size):
                    yield batch.to_pylist()
                return
            stream_errors.append("to_lance() has no to_batches")
        except Exception as e:
            stream_errors.append(f"to_lance/to_batches: {e}")

    # Scanner API if present (must stream — never to_table() full load)
    if hasattr(table, "scanner"):
        try:
            scanner = table.scanner(batch_size=batch_size)
            if hasattr(scanner, "to_batches"):
                for batch in scanner.to_batches():
                    yield batch.to_pylist()
                return
            stream_errors.append("scanner has no to_batches")
        except Exception as e:
            stream_errors.append(f"scanner: {e}")

    n = _table_row_count(table)
    detail = "; ".join(stream_errors) if stream_errors else "no stream API"
    if not allow_full_scan and (n is None or n > batch_size):
        raise RuntimeError(
            "Streaming Lance scan failed; refusing full-table materialization "
            f"(rows={n}, batch_size={batch_size}). {detail}. "
            "Pass allow_full_scan=True only for small tables."
        )

    logger.warning(
        "Streaming Lance scan unavailable (%s); full scan (rows=%s)", detail, n
    )
    if hasattr(table, "to_arrow"):
        yield table.to_arrow().to_pylist()
        return
    if hasattr(table, "to_pandas"):
        yield table.to_pandas().to_dict(orient="records")
        return
    raise RuntimeError(
        "Cannot scan Lance table (no to_lance/to_batches, scanner, to_arrow, or to_pandas)"
    )


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
    # Secondary indexes (files/chunks) are created only after bulk load
    # — see convert_lancedb_to_duckdb — so inserts stay index-free.


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
    """Create HNSW + supporting indexes after bulk load for one dims table."""
    table = _embedding_table_name(dims)
    try:
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {_embedding_hnsw_index_name(dims)} "
            f"ON {table} USING HNSW (embedding) WITH (metric = 'cosine')"
        )
    except Exception as e:
        logger.warning("HNSW index create failed for dims=%s: %s", dims, e)
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


def _flush_embedding_batch(
    conn: Any,
    dims: int,
    emb_rows: list[tuple[Any, ...]],
    *,
    emb_id_start: int,
) -> int:
    """Insert embedding rows for one dims table. Returns next emb_id."""
    if not emb_rows:
        return emb_id_start
    table = _ensure_embedding_table(conn, dims)
    payload: list[tuple[Any, ...]] = []
    emb_id = emb_id_start
    for chunk_id, provider, model, vec, d in emb_rows:
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
    return emb_id


def _maybe_compact_converted_db(
    dest_file: Path,
    *,
    mode: Literal["auto", "always", "never"],
    base_directory: Path | None,
) -> bool:
    """Run product DuckDB compaction if needed (or always/never).

    Uses DuckDBProvider.should_optimize / compact_database so HNSW and
    file-swap behavior match live indexes.
    """
    if mode == "never":
        return False

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.providers.database.duckdb_provider import DuckDBProvider

    dest_file = dest_file.expanduser().resolve()
    # Explicit .db path so get_db_path / connection open the converted file.
    cfg = DatabaseConfig(path=dest_file, provider="duckdb")
    base = (base_directory or dest_file.parent).resolve()
    provider = DuckDBProvider(dest_file, base_directory=base, config=cfg)

    try:
        provider.connect()
        if mode == "always" or provider.should_optimize():
            provider.compact_database()
            return True
        return False
    finally:
        try:
            provider.disconnect(skip_checkpoint=False)
        except Exception:
            pass


def convert_lancedb_to_duckdb(
    source: Path,
    dest: Path,
    *,
    overwrite: bool = False,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    compact: Literal["auto", "always", "never"] = "auto",
    base_directory: Path | None = None,
    allow_full_scan: bool = False,
    progress: ProgressFn | None = _default_progress,
) -> ConversionStats:
    """Copy LanceDB files/chunks/embeddings into a DuckDB ``chunks.db``.

    Streams Lance tables in ``batch_size`` row batches so multi-million-row
    indexes (4M+ embeddings) do not require loading the full corpus into RAM.
    Embedding vectors are inserted per batch; **all** indexes (files/chunks
    b-tree + embedding HNSW) are created only after bulk load.

    Args:
        source: Lance dir, config dir, or project root containing LanceDB.
        dest: DuckDB file, or directory (writes ``chunks.db`` inside).
        overwrite: If True, replace an existing destination file.
        batch_size: Lance scan + Duck INSERT batch size (rows).
        compact: After convert — ``auto`` (product fragmentation check),
            ``always`` (force compact_database), or ``never``.
        base_directory: Project root for DuckDBProvider compaction (optional).
        allow_full_scan: If True, permit full-table materialization when
            streaming APIs fail (unsafe for multi-million-row indexes).
        progress: Optional callback for phase/batch progress (default: print).
            Pass ``None`` to silence progress lines.

    Returns:
        ConversionStats with counts and paths.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    report: ProgressFn = progress if progress is not None else (lambda _m: None)

    lance_dir = resolve_lancedb_dir(source)
    dest_file = resolve_duckdb_file(dest)
    stats = ConversionStats(
        source_lance=str(lance_dir),
        dest_duckdb=str(dest_file),
        stream_batch_size=batch_size,
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

    import lancedb

    db = lancedb.connect(str(lance_dir))
    try:
        files_tbl = db.open_table("files")
        chunks_tbl = db.open_table("chunks")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open LanceDB tables in {lance_dir}: {e}"
        ) from e

    # --- Phase 0: source inventory ---
    lance_bytes = _dir_size_bytes(lance_dir)
    total_files = _table_row_count(files_tbl)
    total_chunks = _table_row_count(chunks_tbl)
    file_batches_est = (
        (total_files + batch_size - 1) // batch_size if total_files else None
    )
    chunk_batches_est = (
        (total_chunks + batch_size - 1) // batch_size if total_chunks else None
    )
    report("=== Phase 0: initialize ===")
    report(f"  source LanceDB:  {lance_dir}")
    report(f"  dest DuckDB:     {dest_file}")
    report(f"  Lance on-disk:   {_format_bytes(lance_bytes)}")
    report(
        f"  Lance files:     {total_files if total_files is not None else '?'}"
        f"  (row count; dups rare)"
    )
    report(
        f"  Lance chunks:    {total_chunks if total_chunks is not None else '?'}"
        f"  (row count; fragment dups possible)"
    )
    report(f"  batch_size:      {batch_size}")
    if file_batches_est is not None:
        report(f"  est. file scan batches:  ~{file_batches_est}")
    if chunk_batches_est is not None:
        report(f"  est. chunk scan batches: ~{chunk_batches_est}")
    report(f"  compact mode:    {compact}")
    report(f"  allow_full_scan: {allow_full_scan}")

    conn = _open_duck_for_write(dest_file)
    try:
        report("=== Phase 1: schema ===")
        _create_duck_schema(conn)
        report("  empty DuckDB schema ready (indexes deferred until after bulk load)")

        # --- Phase 2: files ---
        report("=== Phase 2: files ===")
        file_id_map: dict[int, int] = {}
        seen_file_ids: set[int] = set()
        file_batch: list[tuple[Any, ...]] = []
        file_flush_n = 0
        file_rows_seen = 0

        def _flush_files() -> None:
            nonlocal file_flush_n
            if not file_batch:
                return
            conn.executemany(
                """
                INSERT INTO files (
                    id, path, name, extension, size, modified_time,
                    content_hash, language, skip_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                file_batch,
            )
            stats.files += len(file_batch)
            file_batch.clear()
            file_flush_n += 1
            report(
                f"  [files] batch {file_flush_n}"
                f"  scanned {file_rows_seen}"
                f"/{total_files if total_files is not None else '?'}"
                f" ({_pct(file_rows_seen, total_files)})"
                f"  unique={stats.files}"
                f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
            )

        for rows in _iter_lance_table_batches(
            files_tbl, batch_size=batch_size, allow_full_scan=allow_full_scan
        ):
            for row in rows:
                file_rows_seen += 1
                rid = row.get("id")
                if rid is None:
                    continue
                old_fid = int(rid)
                if old_fid in seen_file_ids:
                    continue
                seen_file_ids.add(old_fid)
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
                if len(file_batch) >= batch_size:
                    _flush_files()
        _flush_files()
        if stats.files:
            _reseed_sequence(conn, "files_id_seq", stats.files)
        report(
            f"  [files] done  scanned={file_rows_seen} unique={stats.files}"
            f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
        )

        # --- Phase 3: chunks + embeddings ---
        report("=== Phase 3: chunks + embeddings ===")
        # old_cid -> (new_cid, has_valid_embedding) for prefer-embedded dedupe
        chunk_state: dict[int, tuple[int, bool]] = {}
        next_cid = 0
        chunk_batch: list[tuple[Any, ...]] = []
        emb_pending: dict[int, list[tuple[Any, ...]]] = {}
        emb_id = 0
        dims_seen: set[int] = set()
        chunk_flush_n = 0
        lance_rows_seen = 0

        def _flush_chunks_and_embeddings() -> None:
            nonlocal emb_id, chunk_flush_n
            if not chunk_batch and not any(emb_pending.values()):
                return
            if chunk_batch:
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        id, file_id, chunk_type, symbol, code,
                        start_line, end_line, start_byte, end_byte,
                        language, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    chunk_batch,
                )
                stats.chunks += len(chunk_batch)
                chunk_batch.clear()
            for dims, erows in list(emb_pending.items()):
                if not erows:
                    continue
                emb_id = _flush_embedding_batch(
                    conn, dims, erows, emb_id_start=emb_id
                )
                stats.embeddings += len(erows)
                dims_seen.add(dims)
                erows.clear()
            emb_pending.clear()
            chunk_flush_n += 1
            report(
                f"  [chunks] batch {chunk_flush_n}"
                f"  scanned {lance_rows_seen}"
                f"/{total_chunks if total_chunks is not None else '?'}"
                f" ({_pct(lance_rows_seen, total_chunks)})"
                f"  unique={stats.chunks}"
                f"  emb={stats.embeddings}"
                f"  dims={sorted(dims_seen) or '-'}"
                f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
            )

        for rows in _iter_lance_table_batches(
            chunks_tbl, batch_size=batch_size, allow_full_scan=allow_full_scan
        ):
            for row in rows:
                lance_rows_seen += 1
                rid = row.get("id")
                if rid is None:
                    continue
                old_cid = int(rid)
                emb = row.get("embedding")
                has_emb = _has_valid_embedding(emb)

                if old_cid in chunk_state:
                    # Prefer-embedded: Lance fragments can surface pre/post-embed
                    # rows for the same id; first-wins would drop the vector.
                    new_cid, prev_has = chunk_state[old_cid]
                    if has_emb and not prev_has:
                        vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                        dims = len(vec)
                        provider = str(row.get("provider") or "unknown")
                        model = str(row.get("model") or "unknown")
                        emb_pending.setdefault(dims, []).append(
                            (new_cid, provider, model, vec, dims)
                        )
                        chunk_state[old_cid] = (new_cid, True)
                    continue

                next_cid += 1
                new_cid = next_cid
                chunk_state[old_cid] = (new_cid, has_emb)
                old_fid = (
                    int(row["file_id"]) if row.get("file_id") is not None else None
                )
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
                        int(row["start_line"])
                        if row.get("start_line") is not None
                        else None,
                        int(row["end_line"])
                        if row.get("end_line") is not None
                        else None,
                        None,  # start_byte
                        None,  # end_byte
                        str(row.get("language") or "") or None,
                        meta if meta else None,
                    )
                )
                if has_emb:
                    vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    dims = len(vec)
                    provider = str(row.get("provider") or "unknown")
                    model = str(row.get("model") or "unknown")
                    emb_pending.setdefault(dims, []).append(
                        (new_cid, provider, model, vec, dims)
                    )
                elif emb is not None:
                    stats.skipped_invalid_embeddings += 1

                if len(chunk_batch) >= batch_size:
                    _flush_chunks_and_embeddings()

        _flush_chunks_and_embeddings()
        if stats.chunks:
            _reseed_sequence(conn, "chunks_id_seq", stats.chunks)
        if emb_id > 0:
            _reseed_sequence(conn, "embeddings_id_seq", emb_id)
        report(
            f"  [chunks] done  unique={stats.chunks}"
            f"  emb={stats.embeddings}"
            f"  skipped_invalid_emb={stats.skipped_invalid_embeddings}"
            f"  lance_rows_scanned={lance_rows_seen}"
            f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
        )

        # --- Phase 4: indexes ---
        report("=== Phase 4: indexes (after bulk load) ===")
        report("  creating files/chunks b-tree indexes …")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol)"
        )
        report(
            f"  files/chunks indexes done"
            f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
        )
        stats.embedding_dims = sorted(dims_seen)
        for dims in stats.embedding_dims:
            report(f"  building embeddings_{dims} indexes (incl. HNSW) …")
            _create_embedding_indexes(conn, dims)
            report(
                f"  embeddings_{dims} indexes done"
                f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
            )

        report("=== Phase 5: checkpoint ===")
        conn.execute("CHECKPOINT")
        report(
            f"  checkpoint done  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
        )
        report(
            f"  load complete: files={stats.files} chunks={stats.chunks} "
            f"embeddings={stats.embeddings} dims={stats.embedding_dims}"
        )
    finally:
        conn.close()

    # Compaction may rename/reopen the file — run after connection closed.
    if compact != "never":
        report(f"=== Phase 6: compact ({compact}) ===")
        try:
            stats.compacted = _maybe_compact_converted_db(
                dest_file,
                mode=compact,
                base_directory=base_directory,
            )
            if stats.compacted:
                report(
                    f"  compacted  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
                )
            else:
                report(
                    f"  compaction skipped (below threshold or not required)"
                    f"  duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
                )
        except Exception as e:
            report(f"  compaction failed/skipped: {e}")
            logger.warning(
                "Post-convert compaction skipped/failed for %s: %s", dest_file, e
            )
    else:
        report("=== Phase 6: compact (never) ===")
        report("  skipped")

    report("=== done ===")
    report(
        f"  source_lance_size={_format_bytes(lance_bytes)}"
        f"  dest_duck={_format_bytes(_duck_on_disk_bytes(dest_file))}"
        f"  files={stats.files} chunks={stats.chunks}"
        f"  embeddings={stats.embeddings} compacted={stats.compacted}"
    )
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


def config_path_for_dest(project_dir: Path, dest_duckdb_file: Path) -> str:
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
    batch_size: int = _DEFAULT_BATCH_SIZE,
    compact: Literal["auto", "always", "never"] = "auto",
    allow_full_scan: bool = False,
    progress: ProgressFn | None = _default_progress,
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
    stats = convert_lancedb_to_duckdb(
        src,
        dst,
        overwrite=overwrite,
        batch_size=batch_size,
        compact=compact,
        base_directory=project_dir,
        allow_full_scan=allow_full_scan,
        progress=progress,
    )
    if activate:
        if database_path is not None:
            cfg_path_val = str(database_path).replace("\\", "/")
        else:
            cfg_path_val = config_path_for_dest(
                project_dir, Path(stats.dest_duckdb)
            )
        activate_duckdb_in_config(project_dir, database_path=cfg_path_val)
    return stats
