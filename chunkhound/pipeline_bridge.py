"""Pipeline bridge — thin adapter wrapping existing parser and embedding code.

These callbacks are called from the Rust IndexingPipeline. They adapt the
existing Python parsing and embedding infrastructure to the contract expected
by Rust.
"""

import atexit
import functools
import os
import threading
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from chunkhound.core.exceptions import DiskUsageLimitExceededError, RustPipelineError
from chunkhound.core.types.common import FileId
from chunkhound.parsers.parser_factory import create_parser_for_language

if TYPE_CHECKING:
    import asyncio

    from chunkhound.interfaces.embedding_provider import EmbeddingProvider

_embed_providers: dict[int, "EmbeddingProvider"] = {}
_embed_loops: dict[int, "asyncio.AbstractEventLoop"] = {}
_embed_cache_lock = threading.Lock()

_parse_pool: ProcessPoolExecutor | None = None
_parse_pool_lock = threading.Lock()


def _default_parse_pool_workers() -> int:
    """Single source of truth for the parse pool's default worker count.

    Used both by _ParsePoolConfig's own default (below) and by
    run_rust_pipeline() when it builds the "parse_thread_pool_size" entry
    Rust reads — keeping both call sites pointed at one formula instead of
    two copies that could silently drift apart.
    """
    return min(os.cpu_count() or 4, 16)


@dataclass(frozen=True)
class _ParsePoolConfig:
    """Plain, picklable mirror of chunkhound_native.ParseCallConfig.

    ProcessPoolExecutor.map() pickles every argument to send it to a worker
    process, and a Rust #[pyclass] instance (what parse_batch_callback()
    actually receives) has no pickle support — so parse_batch_callback()
    copies the fields it needs out of whatever parse_config it's given
    (real or absent) into one of these before it ever reaches _get_parse_pool()
    or _parse_one_file(). Also doubles as the default used when
    parse_batch_callback()/_parse_one_file() are called directly (tests, or
    any caller that bypasses the Rust pipeline) without a parse_config.

    per_file_timeout_secs defaults to 0.0 (disabled) here rather than
    matching Rust's own default, so direct/test callers don't unexpectedly
    start paying subprocess-spawn overhead for large fixture files.
    """

    detect_embedded_sql: bool = True
    per_file_timeout_secs: float = 0.0
    per_file_timeout_min_size_kb: int = 128
    config_file_size_threshold_kb: int = 20
    parse_thread_pool_size: int = field(default_factory=_default_parse_pool_workers)
    index_unknown_files: bool = False

    @classmethod
    def from_any(
        cls, parse_config: Any, *, index_unknown_files: bool = False
    ) -> "_ParsePoolConfig":
        """Build a picklable config from whatever parse_batch_callback()
        received — the real Rust ParseCallConfig, a duck-typed stand-in
        (e.g. in tests), or None."""
        if parse_config is None:
            if not index_unknown_files:
                return _DEFAULT_PARSE_CONFIG
            return cls(index_unknown_files=True)
        return cls(
            detect_embedded_sql=parse_config.detect_embedded_sql,
            per_file_timeout_secs=parse_config.per_file_timeout_secs,
            per_file_timeout_min_size_kb=parse_config.per_file_timeout_min_size_kb,
            config_file_size_threshold_kb=parse_config.config_file_size_threshold_kb,
            parse_thread_pool_size=parse_config.parse_thread_pool_size,
            index_unknown_files=index_unknown_files,
        )


_DEFAULT_PARSE_CONFIG = _ParsePoolConfig()


def _get_parse_pool(max_workers: int) -> ProcessPoolExecutor:
    """Lazily create the module-level parse worker pool, once per process.

    Reused across every parse_batch_callback() call — i.e. across every
    parse batch within one IndexingPipeline.run(), and across every run in
    this process — instead of being spawned and torn down per batch. Sizing
    is fixed at first use, from whichever call happens to win the creation
    race below — later calls' max_workers values are ignored once the pool
    already exists, same as before this took an explicit parameter.

    A lock guards creation because multiple concurrent IndexingPipeline.run()
    calls (e.g. two simultaneous indexing operations in one long-lived MCP
    process) each spawn their own dedicated parse thread, so a first-use race
    across threads is a real scenario here, not a hypothetical one.
    """
    global _parse_pool
    if _parse_pool is None:
        with _parse_pool_lock:
            if _parse_pool is None:
                pool = ProcessPoolExecutor(max_workers=max(1, max_workers))
                atexit.register(pool.shutdown, cancel_futures=True)
                _parse_pool = pool
    return _parse_pool


def parse_file_callback(
    file_path: str,
    detect_embedded_sql: bool = True,
    config_file_size_threshold_kb: int = 20,
    index_unknown_files: bool = False,
) -> tuple[str, list[dict]]:
    """Adapter: path → (language, chunks).

    Called from the Rust parse thread for each file (directly for small/
    typical files, or via _parse_with_timeout()'s child process for large
    ones — see _parse_one_file()). Handles:
    - Language detection (40+ mappings in Python)
    - Tree-sitter parsing
    - Config file size threshold
    - Embedded SQL detection

    Args:
        config_file_size_threshold_kb: Structured config files (JSON, YAML,
            etc.) larger than this are skipped entirely. <= 0 disables the
            gate (matches chunkhound.services.batch_processor's convention).

    Returns:
        (language_value: str, chunks: list[dict])
        Empty string language means unrecognized file (no chunks).
    """
    from chunkhound.core.detection import detect_language
    from chunkhound.core.types.common import Language

    lang = detect_language(Path(file_path))

    if lang is None or lang == Language.UNKNOWN:
        if not index_unknown_files:
            return ("", [])
        lang = Language.TEXT

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
            if (
                config_file_size_threshold_kb > 0
                and size_kb > config_file_size_threshold_kb
            ):
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


def embed_batch_callback(texts: list[str]) -> list[list[float]]:
    """Parallel batch embed (called from Rust rayon threads with GIL held).

    Signature matches what ``embed_batch_parallel`` expects:
    ``callback.call1((texts,)) → List[List[float]]``.

    Used when ``embed_thread_pool_size > 1`` — each rayon thread
    processes one batch at a time, so the provider sees N concurrent
    API requests.
    """
    return _embed_batch(texts)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Shared embed helper — run the async provider.embed() synchronously.

    Uses a provider AND an event loop cached per OS thread (keyed by
    ``threading.get_ident()``), both created once per thread and reused for
    the thread's lifetime.

    NOTE: this used to use ``threading.local()``, which turned out not to
    persist across separate ``Python::with_gil()`` calls from the same
    Rust-native rayon thread — each call got a fresh ``threading.local()``
    namespace (visible as CPython auto-naming the attaching thread
    "Dummy-N" every time), silently defeating the cache and creating a new
    provider + event loop on *every single batch* instead of once per
    thread. Keying an explicit module-level dict by the raw OS thread id
    (which — unlike the ``threading.local()`` namespace — really is stable
    across calls) fixes this; see git history for the diagnostic that
    confirmed only ``embed_thread_pool_size`` distinct thread ids exist but
    hundreds of provider/loop instances were being created.

    The Rust embed thread's rayon pool is built once per pipeline run and
    reused across every streamed batch, so its worker OS threads are
    long-lived — the same thread makes many calls to this function over the
    life of a run. httpx's AsyncClient binds internal locks/transports to
    whichever event loop is running when it's first used; calling
    ``asyncio.run()`` here would create and tear down a *new* loop on every
    call while still reusing the same cached client, and running that
    client under a succession of different loops hangs (the client's locks
    stay attached to the loop they were created under, which is already
    closed). Reusing one loop per thread via ``run_until_complete`` keeps
    the client bound to a single, stable loop for as long as the thread
    lives.
    """
    import asyncio

    tid = threading.get_ident()
    if tid not in _embed_providers:
        from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
        from chunkhound.registry import get_registry

        config = get_registry()._config
        if config is None or config.embedding is None:
            raise RuntimeError("No embedding configuration available")
        provider = EmbeddingProviderFactory.create_provider(config.embedding)
        with _embed_cache_lock:
            _embed_providers.setdefault(tid, provider)

    emb_provider = _embed_providers[tid]

    async def _embed() -> list[list[float]]:
        return await emb_provider.embed(texts)

    # Rayon threads are NOT the main thread and have no event loop of their
    # own — give each thread a persistent loop, created once and reused for
    # every call made from that thread.
    if threading.current_thread() is not threading.main_thread():
        if tid not in _embed_loops:
            loop = asyncio.new_event_loop()
            with _embed_cache_lock:
                _embed_loops.setdefault(tid, loop)
        loop = _embed_loops[tid]
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_embed())

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _embed())
                return future.result()
        return asyncio.run(_embed())
    except RuntimeError:
        return asyncio.run(_embed())


def _parse_file_worker_for_timeout(
    file_path: str,
    detect_embedded_sql: bool,
    config_file_size_threshold_kb: int,
    conn: Any,
    index_unknown_files: bool = False,
) -> None:
    """Child-process entry point for _parse_with_timeout().

    Runs in a dedicated spawned process so the parent can enforce a strict
    wall-clock timeout by terminating this process outright — a hung
    ProcessPoolExecutor task can't be selectively cancelled in place, only
    the process running it can be killed. Mirrors
    chunkhound.services.batch_processor._parse_file_worker's approach,
    adapted to call parse_file_callback() (which already does language
    detection + the binary guard + the config-file size gate) instead of
    parsing directly.
    """
    try:
        lang, chunks = parse_file_callback(
            file_path,
            detect_embedded_sql=detect_embedded_sql,
            config_file_size_threshold_kb=config_file_size_threshold_kb,
            index_unknown_files=index_unknown_files,
        )
        conn.send(("ok", (lang, chunks)))
    except Exception as e:
        try:
            conn.send(("error", str(e)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _parse_with_timeout(
    file_path: str, cfg: "_ParsePoolConfig"
) -> tuple[str, list[dict], str | None]:
    """Parse one file in a dedicated child process with a wall-clock timeout.

    Only used for files at or above cfg.per_file_timeout_min_size_kb (see
    _parse_one_file) — the extra process-spawn cost isn't worth paying for
    every small file, only the large ones that could plausibly hang a
    parser (e.g. pathological minified/generated files).
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_parse_file_worker_for_timeout,
        args=(
            file_path,
            cfg.detect_embedded_sql,
            cfg.config_file_size_threshold_kb,
            child_conn,
            cfg.index_unknown_files,
        ),
        daemon=True,
    )
    p.start()
    try:
        child_conn.close()
    except Exception:
        pass

    try:
        if parent_conn.poll(cfg.per_file_timeout_secs):
            status, payload = parent_conn.recv()
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
            if status == "ok":
                lang, chunks = payload
                return (lang, chunks, None)
            return ("", [], str(payload))
        # Timed out — terminate the child process cleanly.
        p.terminate()
        p.join(timeout=0.5)
        return ("", [], f"parse timed out after {cfg.per_file_timeout_secs}s")
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass


def _parse_one_file(
    args: tuple[str, "_ParsePoolConfig"],
) -> tuple[str, list[dict], str | None]:
    """Parse a single file — module-level so ProcessPoolExecutor can pickle it.

    Catches any exception so one bad file can't abort the whole batch —
    ProcessPoolExecutor.map() would otherwise re-raise it for the whole
    parse_batch_callback() call, aborting the entire pipeline run.

    Files at or above cfg.per_file_timeout_min_size_kb are parsed via
    _parse_with_timeout() instead of the direct fast path, so a hung parse
    on one large file can be killed rather than stalling this pool worker
    (and every batch queued behind it) indefinitely.
    """
    file_path, cfg = args
    try:
        if cfg.per_file_timeout_secs > 0:
            try:
                size_kb = os.path.getsize(file_path) / 1024
            except OSError as e:
                return ("", [], str(e))
            if size_kb >= cfg.per_file_timeout_min_size_kb:
                return _parse_with_timeout(file_path, cfg)

        lang, chunks = parse_file_callback(
            file_path,
            detect_embedded_sql=cfg.detect_embedded_sql,
            config_file_size_threshold_kb=cfg.config_file_size_threshold_kb,
            index_unknown_files=cfg.index_unknown_files,
        )
        return (lang, chunks, None)
    except Exception as e:
        # No file_path prefix here — the Rust caller already knows which
        # path this tuple corresponds to and prepends it when aggregating
        # into PipelineReport.errors.
        return ("", [], str(e))


def parse_batch_callback(
    file_paths: list[str],
    parse_config: Any = None,
    *,
    index_unknown_files: bool = False,
) -> list[tuple[str, list[dict], str | None]]:
    """Adapter: batch-parse files in parallel (called from Rust parse thread).

    Callback contract is batch-shaped, not per-file: Rust's parse thread
    hands over one whole batch per call (one call per parse_batch_size chunk
    of files) and this function fans it out across subprocesses for true CPU
    parallelism (tree-sitter holds the GIL, so Python threads wouldn't help).
    The worker pool (see _get_parse_pool()) is a process-wide singleton —
    created once and reused for every call, both within one
    IndexingPipeline.run() and across multiple runs in this process.

    Args:
        parse_config: The chunkhound_native.ParseCallConfig Rust constructs
            once per IndexingPipeline.run() and passes to every call for
            that run. Direct callers that omit it (tests bypassing Rust) get
            _DEFAULT_PARSE_CONFIG instead.
        index_unknown_files: Passed through to _ParsePoolConfig — Rust's own
            ParseCallConfig has no such field, so run_rust_pipeline() binds
            this from the resolved indexing config via functools.partial
            rather than it flowing through parse_config.

    Returns:
        List of (language, chunks, error) tuples — same order as file_paths.
        `error` is `None` on success, or a message string if that file's
        parse raised or timed out.
    """
    cfg = _ParsePoolConfig.from_any(
        parse_config, index_unknown_files=index_unknown_files
    )
    args_list = [(p, cfg) for p in file_paths]
    pool = _get_parse_pool(cfg.parse_thread_pool_size)
    return list(pool.map(_parse_one_file, args_list))


def _detect_embed_concurrency(embedding_cfg: Any) -> int:
    """Auto-detect embed concurrency from the provider, matching
    ``EmbeddingService.__init__``'s behavior for the legacy Python path —
    without this, the Rust pipeline's embed thread pool silently defaults
    to a single thread (serial embedding) whenever ``max_concurrent_batches``
    isn't explicitly set in config.
    """
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    if embedding_cfg is None:
        return 8

    try:
        provider = EmbeddingProviderFactory.create_provider(embedding_cfg)
    except (ValueError, ImportError):
        return 8

    get_concurrency = getattr(provider, "get_recommended_concurrency", None)
    if get_concurrency is None:
        return 8
    return int(get_concurrency())


_T = TypeVar("_T")
_MISSING = object()


def _cfg_or(obj: Any, attr: str, default: _T, cast: Callable[[Any], _T]) -> _T:
    """Read obj.attr, cast it, falling back to default only if missing or None.

    Unlike a naive ``getattr(obj, attr, default) or default``, this honors
    explicit falsy values (e.g. ``0`` to disable a size gate, ``0.0`` for an
    exact mtime match) instead of silently overriding them — matching how
    the legacy Python path reads these same config fields.
    """
    value = getattr(obj, attr, _MISSING)
    if value is _MISSING or value is None:
        return default
    return cast(value)


async def run_rust_pipeline(
    files_to_process: list[tuple[Path, str | None]],
    *,
    db_path: Path,
    project_root: Path,
    force_reindex: bool = False,
    skip_embeddings: bool = False,
    do_cleanup: bool = True,
    config: Any = None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run the Rust indexing pipeline and return coordinator-compatible stats.

    Called from IndexingCoordinator.process_directory() Phase 3. The Rust
    pipeline is the default path; set ``CHUNKHOUND_USE_RUST=0`` to opt out
    and fall back to the Python path.  Offloads the call to
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

    from chunkhound_native import IndexingPipeline

    # ── Config mapping ──────────────────────────────────────
    indexing_cfg = getattr(config, "indexing", None) if config else None
    embedding_cfg = getattr(config, "embedding", None) if config else None
    database_cfg = getattr(config, "database", None) if config else None

    per_file_timeout = _cfg_or(indexing_cfg, "per_file_timeout_seconds", 0.0, float)
    per_file_timeout_min = _cfg_or(
        indexing_cfg, "per_file_timeout_min_size_kb", 128, int
    )
    mtime_eps = _cfg_or(indexing_cfg, "mtime_epsilon_seconds", 0.01, float)
    detect_sql = bool(getattr(indexing_cfg, "detect_embedded_sql", True))
    # Rust's config_file_size_threshold_kb is a u32 — a negative value (the
    # documented "<=0 disables the gate" convention, see batch_processor.py)
    # would overflow PyO3's u64 extraction and crash pipeline construction,
    # so clamp to 0 (which disables the gate the same way) before crossing.
    config_file_threshold = max(
        0, _cfg_or(indexing_cfg, "config_file_size_threshold_kb", 20, int)
    )
    db_batch_size = _cfg_or(indexing_cfg, "db_batch_size", 100, int)

    # fragmentation_threshold_pct is a percentage (30.0 = 30%); Rust's
    # compaction_threshold expects a ratio (0.30) — same setting the Python
    # indexing path already honors via --fragmentation-threshold-pct.
    fragmentation_pct = _cfg_or(
        database_cfg, "fragmentation_threshold_pct", 30.0, float
    )
    compaction_threshold = fragmentation_pct / 100.0

    embedding_provider = _cfg_or(embedding_cfg, "provider", "", str)
    embedding_model = _cfg_or(embedding_cfg, "model", "", str)

    embed_batch_size = _cfg_or(embedding_cfg, "batch_size", 200, int)
    max_concurrent = _cfg_or(embedding_cfg, "max_concurrent_batches", 0, int)
    if max_concurrent <= 0 and not skip_embeddings:
        max_concurrent = _detect_embed_concurrency(embedding_cfg)
    max_concurrent = max_concurrent or 1
    _parse_concurrent = _cfg_or(indexing_cfg, "max_concurrent", 0, int)
    parse_thread_pool_size = (
        _parse_concurrent if _parse_concurrent > 0 else _default_parse_pool_workers()
    )
    _index_unknown = bool(getattr(indexing_cfg, "index_unknown_files", False))
    disk_usage_limit_mb = getattr(database_cfg, "max_disk_usage_mb", None)

    config_dict = {
        "project_root": str(project_root.resolve()),
        "db_path": str(db_path.resolve()),
        "db_batch_size": db_batch_size,
        "compaction_threshold": compaction_threshold,
        "compaction_min_size_mb": 50,
        "parse_batch_size": 200,
        "parse_thread_pool_size": parse_thread_pool_size,
        "embed_thread_pool_size": max_concurrent,
        "embed_batch_size": embed_batch_size,
        "mtime_epsilon_seconds": mtime_eps,
        "do_cleanup": do_cleanup,
        "skip_embeddings": skip_embeddings,
        "per_file_timeout_secs": per_file_timeout,
        "per_file_timeout_min_size_kb": per_file_timeout_min,
        "detect_embedded_sql": detect_sql,
        "config_file_size_threshold_kb": config_file_threshold,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "disk_usage_limit_mb": disk_usage_limit_mb,
    }

    # Extract file paths from (path, hash) tuples
    file_paths = [str(p.resolve()) for p, _ in files_to_process]

    # Both the pipeline construction and pipeline.run() raise a plain PyO3
    # PyRuntimeError on failure (see src/pipeline/pipeline.rs), indistinguishable
    # from any other exception IndexingCoordinator.process_directory() can hit.
    # Re-raise as RustPipelineError so callers can tell "the native pipeline
    # itself failed" apart from surrounding Python glue-code errors.
    try:
        pipeline = IndexingPipeline(config_dict)

        # Run pipeline in thread pool — pipeline releases the GIL internally
        report = await asyncio.to_thread(
            pipeline.run,
            files=file_paths,
            parse_batch_callback=functools.partial(
                parse_batch_callback, index_unknown_files=_index_unknown
            ),
            embed_batch_callback=embed_batch_callback if not skip_embeddings else None,
            progress_callback=progress_callback,
            incremental=not force_reindex,
        )
    except Exception as e:
        raise RustPipelineError(reason=str(e)) from e

    # Map PipelineReport → coordinator stats dict.
    # `report.errors` entries are Rust-formatted as "{path}: {message}"
    # (see src/pipeline/pipeline.rs's parse_errors accumulator) — split back
    # into (file, error) so callers like IndexingCoordinator can log the
    # actual failing path instead of "Failed to process None: ...".
    def _split_rust_error(err: str) -> dict[str, str | None]:
        file, sep, message = err.partition(": ")
        if not sep:
            return {"file": None, "error": err}
        return {"file": file, "error": message}

    errors: list[dict[str, Any]] = [
        _split_rust_error(err) for err in (list(report.errors) if report.errors else [])
    ]

    # Mid-run disk-usage check (mirrors _check_disk_usage_limit's contract) —
    # reported as structured data on the report (a single Option<(f64, f64)>
    # on the Rust side, so "tripped" and its two numbers can't desync), not a
    # raised exception. Reuses DiskUsageLimitExceededError.to_error_dict() --
    # the same single source of truth IndexingCoordinator._store_parsed_results
    # uses for the Python path (indexing_coordinator.py:1039-1042) -- so
    # IndexingCoordinator.process_directory()'s generic disk-limit scan
    # (indexing_coordinator.py:2175-2183) picks this up with no
    # pipeline-specific handling and no second copy of the message format.
    disk_limit = getattr(report, "disk_limit", None)
    if disk_limit is not None:
        current_mb, limit_mb = disk_limit
        errors.append(
            DiskUsageLimitExceededError(
                current_size_mb=current_mb, limit_mb=limit_mb
            ).to_error_dict()
        )

    return {
        "total_files": report.files_processed,
        "total_chunks": report.chunks_written,
        "embeddings_generated": report.embeddings_generated,
        "elapsed_secs": report.elapsed_secs,
        "files_skipped_unchanged": report.files_skipped,
        "errors": errors,
    }
