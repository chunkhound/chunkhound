"""Pipeline contract test harness.

Compares output from the Python indexing pipeline (and eventually the Rust
pipeline) to assert byte-identical chunk output.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from chunkhound.core.config.config import Config
from chunkhound.core.utils.path_utils import get_relative_path_safe
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
    Restores the prior value afterward — pytest-xdist workers run many
    test items in one process, so leaving this set process-wide would leak
    into unrelated tests run later in the same worker.
    """
    import os

    _prev_use_rust = os.environ.get("CHUNKHOUND_USE_RUST")
    os.environ["CHUNKHOUND_USE_RUST"] = "0"
    try:
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
    finally:
        if _prev_use_rust is None:
            os.environ.pop("CHUNKHOUND_USE_RUST", None)
        else:
            os.environ["CHUNKHOUND_USE_RUST"] = _prev_use_rust


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
    parse_batch_size: int = 200,
    compaction_threshold: float = 0.60,
    compaction_min_size_mb: int = 10,
    disk_usage_limit_mb: float | None = None,
    progress_callback=None,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline."""
    from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    from tests.contracts.mock_embed import MOCK_MODEL, MOCK_PROVIDER, embed_texts

    db_dir.mkdir(parents=True, exist_ok=True)

    config_dict = default_rust_config(
        db_dir,
        compaction_threshold=compaction_threshold,
        compaction_min_size_mb=compaction_min_size_mb,
        disk_usage_limit_mb=disk_usage_limit_mb,
        parse_batch_size=parse_batch_size,
        parse_thread_pool_size=parse_thread_pool_size,
        skip_embeddings=skip_embeddings,
        embedding_provider=MOCK_PROVIDER,
        embedding_model=MOCK_MODEL,
    )

    pipeline = IndexingPipeline(config_dict)

    resolved_fixture_dir = fixture_dir.resolve()
    files = sorted(resolved_fixture_dir.glob("*"))
    # (absolute_path, relative_key) pairs — relative_key computed via the same
    # production helper Rust now relies on (get_relative_path_safe), so this
    # harness exercises the real symlink-aware path-keying contract rather
    # than a simplified stand-in. Base dir must match what `files` was
    # globbed from (the already-resolved fixture dir), or a symlinked leaf
    # entry's is_symlink() branch (which keeps both sides unresolved as-is)
    # would compare against a differently-spelled base and fail to relativize.
    file_entries = [
        (str(f), get_relative_path_safe(f, resolved_fixture_dir).as_posix())
        for f in files
        if f.is_file()
    ]

    from chunkhound.pipeline_bridge import parse_batch_callback

    report = pipeline.run(
        files=file_entries,
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=embed_texts if not skip_embeddings else None,
        progress_callback=progress_callback,
        incremental=incremental,
    )

    chunk_tuples = collect_chunk_tuples_from_duckdb(db_dir)
    embedding_tuples = _collect_embedding_tuples(db_dir)

    # report.disk_limit is a single Option<(f64, f64)> on the Rust side
    # (current_mb, limit_mb) or None — unpacked here into IndexResult's three
    # convenience fields for easier assertions in tests.
    disk_limit = report.disk_limit
    current_mb, max_mb = disk_limit if disk_limit is not None else (None, None)

    return IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=chunk_tuples,
        embedding_tuples=embedding_tuples,
        errors=list(report.errors) if report.errors else [],
        disk_limit_exceeded=disk_limit is not None,
        disk_limit_current_mb=current_mb,
        disk_limit_max_mb=max_mb,
    )


def assert_chunk_multiset_identical(
    chunks_a: list[tuple],
    chunks_b: list[tuple],
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> None:
    """Assert two chunk-tuple collections are identical, including duplicate counts.

    Counter, not set: a set would silently absorb a duplicated row (e.g. one
    pipeline writing the same chunk twice) since both sides reduce to the same
    set of distinct values. Counter equality requires matching multiplicities too.
    """
    a_counts = Counter(chunks_a)
    b_counts = Counter(chunks_b)

    if a_counts != b_counts:
        a_only = a_counts - b_counts
        b_only = b_counts - a_counts
        msg_parts = ["Chunk tuple mismatch:"]
        if a_only:
            msg_parts.append(
                f"  Only in {label_a} / extra copies ({sum(a_only.values())}): "
                f"{sorted(a_only.elements())[:5]}..."
            )
        if b_only:
            msg_parts.append(
                f"  Only in {label_b} / extra copies ({sum(b_only.values())}): "
                f"{sorted(b_only.elements())[:5]}..."
            )
        raise AssertionError("\n".join(msg_parts))


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

    a_errors = sorted(result_a.errors)
    b_errors = sorted(result_b.errors)
    assert a_errors == b_errors, (
        f"errors mismatch: A has {len(a_errors)} error(s) {a_errors!r}, "
        f"B has {len(b_errors)} error(s) {b_errors!r}"
    )

    assert_chunk_multiset_identical(result_a.chunk_tuples, result_b.chunk_tuples)

    # Embedding tuples — same multiset comparison as chunk tuples above.
    emb_a_counts = Counter(result_a.embedding_tuples)
    emb_b_counts = Counter(result_b.embedding_tuples)
    if emb_a_counts or emb_b_counts:
        if emb_a_counts != emb_b_counts:
            raise AssertionError(
                f"Embedding tuple mismatch: A has {len(result_a.embedding_tuples)} "
                f"entries ({len(emb_a_counts)} unique), B has "
                f"{len(result_b.embedding_tuples)} entries ({len(emb_b_counts)} unique)"
            )


def default_rust_config(
    db_dir: Path,
    **overrides: Any,
) -> dict[str, Any]:
    """Return the standard `IndexingPipeline` config dict.

    Tests that drive `IndexingPipeline` directly (rather than through
    `index_with_rust`, e.g. to inject a failing callback) need this same
    18-key dict. Pass only the fields a given test actually varies via
    `**overrides` instead of repeating the whole shape — a previous
    per-file copy of this dict already drifted (`parse_thread_pool_size`
    differed between two files with no test-specific reason).
    """
    config: dict[str, Any] = {
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_min_size_mb": 10,
        "disk_usage_limit_mb": None,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "do_cleanup": True,
        "skip_embeddings": True,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": "",
        "embedding_model": "",
    }
    config.update(overrides)
    return config


def collect_table_counts(db_dir: Path) -> dict[str, int]:
    """Row counts for files/chunks/any embeddings_* table.

    Returns zeros if `chunks.db` doesn't exist yet (e.g. before the first
    index run). Callers that only care about a subset of these three keys
    can simply ignore the rest.
    """
    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return {"files": 0, "chunks": 0, "embeddings": 0}

    conn = duckdb.connect(str(db_file))
    try:
        files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
        embeddings = 0
        for (table_name,) in tables:
            embeddings += conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()[0]
        return {"files": files, "chunks": chunks, "embeddings": embeddings}
    finally:
        conn.close()


def files_table_paths(db_dir: Path) -> set[str]:
    """All `path` values currently in the `files` table."""
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        rows = conn.execute("SELECT path FROM files").fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def chunk_ids_for_path(db_dir: Path, rel_path: str) -> list[int]:
    """Chunk ids for the file at *rel_path*, ordered by id."""
    conn = duckdb.connect(str(db_dir / "chunks.db"))
    try:
        rows = conn.execute(
            """
            SELECT c.id
            FROM chunks c JOIN files f ON f.id = c.file_id
            WHERE f.path = ?
            ORDER BY c.id
            """,
            [rel_path],
        ).fetchall()
    finally:
        conn.close()
    return [int(r[0]) for r in rows]
