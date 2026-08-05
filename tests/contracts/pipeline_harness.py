"""Pipeline contract test harness.

Compares output from the Python indexing pipeline (and eventually the Rust
pipeline) to assert byte-identical chunk output.
"""

from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService


@dataclass
class IndexResult:
    """Normalised result from a single indexing run, suitable for comparison."""

    files_processed: int = 0
    chunks_written: int = 0
    embeddings_generated: int = 0
    # (file_path, chunk_type, symbol, code, start_line, end_line) — sorted
    chunk_tuples: list[tuple[str, str, str, str, int, int]] = field(
        default_factory=list
    )
    # (file_path, chunk_type, symbol, provider, model, dims, embedding[:8]) — sorted
    embedding_tuples: list[tuple[str, str, str, str, str, int, tuple[float, ...]]] = field(
        default_factory=list
    )
    errors: list[str] = field(default_factory=list)
    # Mirrors Python's DiskUsageLimitExceededError contract — set when a
    # mid-run disk-usage check tripped and stopped further writes.
    disk_limit_exceeded: bool = False
    disk_limit_current_mb: float | None = None
    disk_limit_max_mb: float | None = None


def disconnect_registry_db() -> None:
    """Force-disconnect the registry's DuckDB provider to clear per-process cache.

    DuckDB maintains a process-level cache of opened databases. When the Python
    indexing pipeline opens a DuckDB database, subsequent duckdb.connect() calls
    to the same path (even from Rust via py.allow_threads in the same process)
    reuse the cached state, returning stale data.

    Disconnecting the provider and unregistering it forces duckdb to release
    the cached state, so the next connection gets fresh data from disk.
    """
    try:
        from chunkhound.registry import get_registry
        registry = get_registry()
        db = registry.get_provider("database")
        if db is not None and hasattr(db, "disconnect"):
            db.disconnect()
        registry._providers.pop("database", None)
    except Exception:
        pass  # best-effort — test should still pass even if cleanup fails


async def index_with_python(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = True,
    embedding_provider: object = None,
) -> IndexResult:
    """Index *fixture_dir* using the current Python pipeline.

    When *embedding_provider* is given, it is set on the coordinator before
    processing (useful for deterministic mock providers).

    Sets ``CHUNKHOUND_USE_RUST=0`` to explicitly force the Python path
    (the Rust pipeline is on by default; this override is intentional).
    """
    import os

    os.environ["CHUNKHOUND_USE_RUST"] = "0"

    # Build a minimal Config — point the DB at *db_dir* and disable embeddings.
    config = Config(
        target_dir=fixture_dir.resolve(),
        database={
            "provider": "duckdb",
            "path": str(db_dir.resolve()),
        },
        embeddings_disabled=skip_embeddings,
    )

    # Wire the registry with this config (creates providers, parsers, etc.)
    configure_registry(config)

    coordinator = create_indexing_coordinator()

    # Inject mock embedding provider if provided
    if embedding_provider is not None:
        coordinator._embedding_provider = embedding_provider

    service = DirectoryIndexingService(
        indexing_coordinator=coordinator,
        config=config,
    )

    stats = await service.process_directory(
        fixture_dir, no_embeddings=skip_embeddings
    )

    # Collect chunk tuples from the database.
    # IMPORTANT: shut down the coordinator's DB connection before querying.
    # On Windows, DuckDB opens database files in exclusive mode — a second
    # duckdb.connect() would fail while DuckDBProvider holds the file.
    chunk_tuples = _collect_chunk_tuples(coordinator)
    coordinator._db.disconnect()
    embedding_tuples = _collect_embedding_tuples(db_dir)

    return IndexResult(
        files_processed=stats.files_processed,
        chunks_written=stats.chunks_created,
        embeddings_generated=stats.embeddings_generated,
        chunk_tuples=chunk_tuples,
        embedding_tuples=embedding_tuples,
        errors=[str(e) for e in (stats.errors_encountered or [])],
    )


def _collect_chunk_tuples(coordinator) -> list[tuple[str, str, str, str, int, int]]:
    """Query the DB for all chunks and return canonical comparison tuples."""
    db = coordinator._db
    rows = db.execute_query(
        """
        SELECT
            f.path AS file_path,
            c.chunk_type,
            c.symbol,
            c.code,
            c.start_line,
            c.end_line
        FROM chunks c
        JOIN files f ON f.id = c.file_id
        ORDER BY f.path, c.start_line, c.symbol
        """
    )
    tuples: list[tuple[str, str, str, str, int, int]] = []
    for row in rows:
        tuples.append(
            (
                str(row["file_path"]),
                str(row["chunk_type"]),
                str(row["symbol"] or ""),
                str(row["code"] or ""),
                int(row["start_line"] or 0),
                int(row["end_line"] or 0),
            )
        )
    return tuples


def _collect_embedding_tuples(
    db_dir: Path,
) -> list[tuple[str, str, str, str, str, int, tuple[float, ...]]]:
    """Query the DB for all embeddings and return canonical comparison tuples.

    The caller is responsible for ensuring the DB file is not held by
    another connection (e.g., call coordinator._db.disconnect() first).
    """
    import duckdb

    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return []

    conn = duckdb.connect(str(db_file))
    # Find dimensions from any embedding table
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
    ).fetchall()
    tuples: list[tuple[str, str, str, str, str, int, tuple[float, ...]]] = []
    for (table_name,) in tables:
        rows = conn.execute(
            f"""
            SELECT
                f.path AS file_path,
                c.chunk_type,
                c.symbol,
                e.provider,
                e.model,
                e.dims,
                e.embedding
            FROM {table_name} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            ORDER BY f.path, c.start_line, c.symbol
            """
        ).fetchall()
        for row in rows:
            vec = row[6]
            # Store first 8 elements of the vector for comparison
            prefix = tuple(float(v) for v in vec[:8])
            tuples.append(
                (
                    str(row[0]),
                    str(row[1]),
                    str(row[2] or ""),
                    str(row[3]),
                    str(row[4]),
                    int(row[5]),
                    prefix,
                )
            )
    conn.close()
    return tuples


def collect_chunk_tuples_from_duckdb(
    db_dir: Path,
) -> list[tuple[str, str, str, str, int, int]]:
    """Query a DB directory directly for canonical comparison tuples.

    Unlike ``_collect_chunk_tuples``, which reads through a coordinator's live
    connection, this opens ``chunks.db`` directly — used after the Rust
    pipeline writes to the DB, when there is no Python coordinator connection
    to read through.
    """
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    rows = conn.execute(
        """
        SELECT f.path, c.chunk_type, c.symbol, c.code, c.start_line, c.end_line
        FROM chunks c JOIN files f ON f.id = c.file_id
        ORDER BY f.path, c.start_line, c.symbol
        """
    ).fetchall()
    conn.close()
    return [
        (
            str(r[0]),
            str(r[1]),
            str(r[2] or ""),
            str(r[3] or ""),
            int(r[4] or 0),
            int(r[5] or 0),
        )
        for r in rows
    ]


def index_with_rust(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = False,
    incremental: bool = False,
    parse_thread_pool_size: int = 4,
    compaction_threshold: float = 0.60,
    compaction_min_size_mb: int = 10,
    disk_usage_limit_mb: float | None = None,
    progress_callback=None,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline."""
    from tests.contracts.mock_embed import MOCK_MODEL, MOCK_PROVIDER, embed_texts

    from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

    db_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "project_root": str(fixture_dir.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": compaction_threshold,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": compaction_min_size_mb,
        "disk_usage_limit_mb": disk_usage_limit_mb,
        "parse_batch_size": 200,
        "parse_thread_pool_size": parse_thread_pool_size,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "do_cleanup": True,
        "skip_embeddings": skip_embeddings,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": MOCK_PROVIDER,
        "embedding_model": MOCK_MODEL,
    }

    pipeline = IndexingPipeline(config_dict)

    files = sorted(fixture_dir.resolve().glob("*"))
    file_paths = [str(f) for f in files if f.is_file()]

    from chunkhound.pipeline_bridge import parse_batch_callback

    report = pipeline.run(
        files=file_paths,
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=embed_texts if not skip_embeddings else None,
        progress_callback=progress_callback,
        incremental=incremental,
    )

    chunk_tuples = collect_chunk_tuples_from_duckdb(db_dir)
    embedding_tuples = _collect_embedding_tuples(db_dir)

    return IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=chunk_tuples,
        embedding_tuples=embedding_tuples,
        errors=list(report.errors) if report.errors else [],
        disk_limit_exceeded=report.disk_limit_exceeded,
        disk_limit_current_mb=report.disk_limit_current_mb,
        disk_limit_max_mb=report.disk_limit_max_mb,
    )


def assert_identical(result_a: IndexResult, result_b: IndexResult) -> None:
    """Assert two IndexResults are byte-identical.

    Raises ``AssertionError`` with a human-readable diff on mismatch.
    """
    # Top-level counts
    assert result_a.files_processed == result_b.files_processed, (
        f"files_processed mismatch: {result_a.files_processed} != {result_b.files_processed}"
    )
    assert result_a.chunks_written == result_b.chunks_written, (
        f"chunks_written mismatch: {result_a.chunks_written} != {result_b.chunks_written}"
    )
    assert result_a.embeddings_generated == result_b.embeddings_generated, (
        f"embeddings_generated mismatch: {result_a.embeddings_generated} != {result_b.embeddings_generated}"
    )

    # Chunk tuples
    a_only = set(result_a.chunk_tuples) - set(result_b.chunk_tuples)
    b_only = set(result_b.chunk_tuples) - set(result_a.chunk_tuples)

    if a_only or b_only:
        msg_parts = ["Chunk tuple mismatch:"]
        if a_only:
            msg_parts.append(
                f"  Only in A ({len(a_only)}): {sorted(a_only)[:5]}..."
            )
        if b_only:
            msg_parts.append(
                f"  Only in B ({len(b_only)}): {sorted(b_only)[:5]}..."
            )
        raise AssertionError("\n".join(msg_parts))

    # Embedding tuples
    emb_a = set(result_a.embedding_tuples)
    emb_b = set(result_b.embedding_tuples)
    if emb_a or emb_b:
        if emb_a != emb_b:
            raise AssertionError(
                f"Embedding tuple mismatch: A has {len(emb_a)}, B has {len(emb_b)}"
            )