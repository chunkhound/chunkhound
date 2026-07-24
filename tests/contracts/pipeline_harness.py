"""Pipeline contract test harness.

Compares output from the Python indexing pipeline (and eventually the Rust
pipeline) to assert byte-identical chunk output.
"""

from dataclasses import dataclass, field
from pathlib import Path

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

    Sets ``CHUNKHOUND_USE_RUST=0`` so the coordinator always takes the
    Python path, even when the native extension is installed.
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