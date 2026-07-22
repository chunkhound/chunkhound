"""Pipeline bridge — thin adapter wrapping existing parser and embedding code.

These callbacks are called from the Rust IndexingPipeline. They adapt the
existing Python parsing and embedding infrastructure to the contract expected
by Rust.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chunkhound.core.detection import Language

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


def _parse_one_file(args: tuple[str, bool]) -> tuple[str, list[dict]]:
    """Parse a single file — module-level so ProcessPoolExecutor can pickle it."""
    file_path, detect_embedded_sql = args
    return parse_file_callback(file_path, detect_embedded_sql=detect_embedded_sql)


def parse_batch_callback(
    file_paths: list[str],
    detect_embedded_sql: bool = True,
) -> list[tuple[str, list[dict]]]:
    """Adapter: batch-parse files in parallel (called from Rust parse threads).

    Each file is parsed in its own subprocess via ProcessPoolExecutor for
    true CPU parallelism (tree-sitter holds the GIL, so threads don't help).

    Returns:
        List of (language, chunks) tuples — same order as file_paths.
    """
    from concurrent.futures import ProcessPoolExecutor

    # Use at most N workers — cap to avoid resource exhaustion on large batches
    max_workers = min(len(file_paths), 16)
    args_list = [(p, detect_embedded_sql) for p in file_paths]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_parse_one_file, args_list))


async def run_rust_pipeline(
    files_to_process: list[tuple[Path, str | None]],
    *,
    db_path: Path,
    project_root: Path,
    force_reindex: bool = False,
    skip_embeddings: bool = False,
    config: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run the Rust indexing pipeline and return coordinator-compatible stats.

    Called from IndexingCoordinator.process_directory() Phase 3 when
    ``CHUNKHOUND_USE_RUST=1``.  Offloads the call to
    ``asyncio.to_thread`` so the event loop stays responsive.

    Args:
        files_to_process: List of (path, content_hash) tuples from change detection.
        db_path: Parent directory containing ``chunks.db``.
        project_root: Root directory being indexed.
        force_reindex: Skip incremental diff — re-index every file.
        skip_embeddings: Skip embedding generation (e.g. --no-embeddings).
        config: Coordinator config object (for extracting indexing/embedding settings).
        progress_callback: Optional callable(phase: str, current: int, total: int)
            for streaming progress updates to the coordinator's Rich bars.

    Returns:
        Coordinator-compatible stats dict:
        ``{"total_files": int, "total_chunks": int, "embeddings_generated": int,
           "elapsed_secs": float, "errors": list[dict]}``
    """
    import asyncio
    import os

    from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

    # ── Config mapping ──────────────────────────────────────
    indexing_cfg = getattr(config, "indexing", None) if config else None
    embedding_cfg = getattr(config, "embedding", None) if config else None

    per_file_timeout = float(
        getattr(indexing_cfg, "per_file_timeout_seconds", 3.0) or 3.0
    )
    per_file_timeout_min = int(
        getattr(indexing_cfg, "per_file_timeout_min_size_kb", 128) or 128
    )
    mtime_eps = float(
        getattr(indexing_cfg, "mtime_epsilon_seconds", 0.01) or 0.01
    )
    detect_sql = bool(getattr(indexing_cfg, "detect_embedded_sql", True))
    config_file_threshold = int(
        getattr(indexing_cfg, "config_file_size_threshold_kb", 20) or 20
    )
    db_batch_size = int(getattr(indexing_cfg, "db_batch_size", 100) or 100)

    embedding_provider = str(
        getattr(embedding_cfg, "provider", "") or ""
    )
    embedding_model = str(
        getattr(embedding_cfg, "model", "") or ""
    )

    parse_thread_pool_size = (os.cpu_count() or 4)

    config_dict = {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_path.resolve()),
        "db_batch_size": db_batch_size,
        "compaction_threshold": 0.30,
        "compaction_batch_threshold": 50,
        "compaction_min_size_mb": 50,
        "parse_batch_size": 200,
        "parse_thread_pool_size": parse_thread_pool_size,
        "embed_batch_size": 200,
        "force_reindex": force_reindex,
        "mtime_epsilon_seconds": mtime_eps,
        "skip_cleanup": True,  # coordinator handles cleanup separately
        "skip_embeddings": skip_embeddings,
        "per_file_timeout_secs": per_file_timeout,
        "per_file_timeout_min_size_kb": per_file_timeout_min,
        "detect_embedded_sql": detect_sql,
        "config_file_size_threshold_kb": config_file_threshold,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }

    pipeline = IndexingPipeline(config_dict)

    # Extract file paths from (path, hash) tuples
    file_paths = [str(p.resolve()) for p, _ in files_to_process]

    # Run pipeline in thread pool — pipeline releases the GIL internally
    report = await asyncio.to_thread(
        pipeline.run,
        files=file_paths,
        parse_callback=parse_file_callback,
        embed_callback=embed_callback if not skip_embeddings else None,
        progress_callback=progress_callback,
        incremental=not force_reindex,
        parse_batch_callback=parse_batch_callback,
    )

    # Map PipelineReport → coordinator stats dict
    return {
        "total_files": report.files_processed,
        "total_chunks": report.chunks_written,
        "embeddings_generated": report.embeddings_generated,
        "elapsed_secs": report.elapsed_secs,
        "errors": [
            {"file": None, "error": err}
            for err in (list(report.errors) if report.errors else [])
        ],
    }