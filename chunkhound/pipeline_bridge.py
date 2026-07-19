"""Pipeline bridge — thin adapter wrapping existing parser and embedding code.

These callbacks are called from the Rust IndexingPipeline. They adapt the
existing Python parsing and embedding infrastructure to the contract expected
by Rust.
"""

from pathlib import Path

from chunkhound.core.types.common import FileId
from chunkhound.parsers.parser_factory import create_parser_for_language


def parse_file_callback(
    file_path: str,
    detect_embedded_sql: bool = True,
) -> tuple[str, list[dict]]:
    """Adapter: path → (language, chunks).

    Called from the Rust parse thread for each file. Handles:
    - Language detection (40+ mappings in Python)
    - Tree-sitter parsing
    - Config file size threshold
    - Embedded SQL detection
    - Per-file timeout (applied by the caller via multiprocessing)

    Returns:
        (language_value: str, chunks: list[dict])
        Empty string language means unrecognized file (no chunks).
    """
    from chunkhound.core.detection import detect_language
    from chunkhound.core.types.common import Language

    lang = detect_language(Path(file_path))

    if lang is None or lang == Language.UNKNOWN:
        return ("", [])

    # Binary guard — skip files with null bytes
    try:
        with open(file_path, "rb") as fh:
            sample = fh.read(8192)
        if b"\x00" in sample:
            return ("", [])
    except OSError:
        return ("", [])

    # Config file size gate
    if lang.is_structured_config_language:
        try:
            size_kb = Path(file_path).stat().st_size / 1024
            # Default threshold 20 KB
            if size_kb > 20:
                return ("", [])
        except OSError:
            return ("", [])

    parser = create_parser_for_language(
        lang, detect_embedded_sql=detect_embedded_sql
    )
    if parser is None:
        return (lang.value, [])

    file_id = FileId(0)  # Rust assigns the real ID
    chunks = parser.parse_file(Path(file_path), file_id)
    return (lang.value, [c.to_dict() for c in chunks])


def embed_callback(
    texts: list[str],
    provider: str,
    model: str,
) -> list[list[float]]:
    """Adapter: batch embed (called from Rust embed thread).

    All embedding config is baked into the service via global config —
    api_key, base_url, ssl_verify, output_dims, etc.
    """
    from chunkhound.registry import get_registry

    service = get_registry().create_embedding_service()
    # Direct sync embed — the Rust embed thread calls this.
    # Use the provider's embed method directly.
    emb_provider = service._embedding_provider
    if emb_provider is None:
        raise RuntimeError("No embedding provider configured")

    # The embedding service wraps the provider; use it directly
    import asyncio

    async def _embed():
        return await emb_provider.embed(texts)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an event loop — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _embed())
                return future.result()
        return asyncio.run(_embed())
    except RuntimeError:
        return asyncio.run(_embed())


def store_embeddings_callback(
    db_path: str,
    embeddings_data: list[dict],
) -> int:
    """Adapter: store embedding vectors to DuckDB.

    Called from the Rust embed thread after embed_callback returns vectors.
    Uses the same dimension-specific table pattern as Python's
    EmbeddingRepository.insert_embeddings_batch().

    Args:
        db_path: Path to the chunks.db file (already created by Rust pipeline).
        embeddings_data: List of {chunk_id, provider, model, embedding, dims} dicts.

    Returns:
        Number of embeddings stored.
    """
    import duckdb

    if not embeddings_data:
        return 0

    conn = duckdb.connect(str(db_path))

    first = embeddings_data[0]
    dims = first["dims"]
    table_name = f"embeddings_{dims}"

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            chunk_id BIGINT NOT NULL,
            provider VARCHAR NOT NULL,
            model VARCHAR NOT NULL,
            embedding FLOAT[{dims}],
            dims INTEGER NOT NULL,
            PRIMARY KEY (chunk_id, provider, model)
        )
    """)

    rows = [
        (e["chunk_id"], e["provider"], e["model"], e["embedding"], e["dims"])
        for e in embeddings_data
    ]

    conn.executemany(
        f"INSERT OR REPLACE INTO {table_name} VALUES (?, ?, ?, ?, ?)",
        rows,
    )

    conn.close()
    return len(embeddings_data)