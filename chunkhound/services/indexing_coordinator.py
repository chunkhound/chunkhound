"""Indexing coordinator service for ChunkHound - orchestrates indexing workflows.

# FILE_CONTEXT: Central orchestrator for the parse→chunk→embed→store pipeline
# ROLE: Coordinates complex multi-phase workflows with parallel batch processing
# CONCURRENCY: Parsing parallelized across CPU cores, storage remains single-threaded
# PERFORMANCE: Smart chunk diffing preserves existing embeddings (10x speedup)
#
# PERFORMANCE TUNING:
# - File batch processing scales workers based on file count (100/1000 thresholds)
#   to balance parallelism overhead vs throughput
# - Directory discovery uses parallel mode only when ≥4 top-level dirs
# - Worker limits (4/8/16) prevent resource exhaustion on high-core machines
# - See module constants below for tunable parameters
"""

import asyncio
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger
from rich.progress import Progress, TaskID

from chunkhound.core.detection import detect_language
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import FilePath, Language
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.parsers.universal_parser import UniversalParser
from chunkhound.utils.hashing import compute_file_hash

from .base_service import BaseService
from .batch_processor import ParsedFileResult, process_file_batch
from .chunk_cache_service import ChunkCacheService

# File pattern utilities for directory discovery
from chunkhound.utils.file_patterns import (
    load_gitignore_patterns,
    scan_directory_files,
    walk_directory_tree,
    walk_subtree_worker,
)


# CRITICAL FIX: Force spawn multiprocessing start method to prevent fork + asyncio issues
# RATIONALE: Linux defaults to 'fork' which is unsafe with asyncio event loops
# - Forking an active asyncio event loop causes segfaults (background threads/locks copied)
# - 'spawn' starts fresh Python interpreter, avoiding fork-related issues
# - Windows/macOS already use 'spawn' by default
# - Python 3.14 will make 'spawn' the default on all platforms
# - See: https://github.com/chunkhound/chunkhound/pull/47
desired_mp_method = os.getenv("CHUNKHOUND_MP_START_METHOD", "spawn")
current_mp_method = multiprocessing.get_start_method(allow_none=True)
if current_mp_method != desired_mp_method:
    try:
        multiprocessing.set_start_method(desired_mp_method, force=True)
        logger.debug(
            f"Set multiprocessing start method to '{desired_mp_method}' (was {current_mp_method})"
        )
    except RuntimeError:
        # Start method may already be set elsewhere; log and continue
        logger.debug(
            f"Multiprocessing start method remains '{multiprocessing.get_start_method()}'; desired '{desired_mp_method}'"
        )


# Performance tuning constants for parallel operations
# RATIONALE: Balance parallelism overhead vs throughput for different workload sizes

# File parsing batch sizes
SMALL_FILE_COUNT_THRESHOLD = (
    100  # Below this: use minimal workers (overhead not worth it)
)
MEDIUM_FILE_COUNT_THRESHOLD = 1000  # Above this: scale up for enterprise monorepos
MAX_WORKERS_SMALL_BATCH = 4  # Worker cap for small file batches
MAX_WORKERS_MEDIUM_BATCH = 8  # Worker cap for medium file batches (original behavior)
MAX_WORKERS_LARGE_BATCH = (
    16  # Worker cap for large file batches (prevents resource exhaustion)
)

# Fallback CPU count when os.cpu_count() returns None
DEFAULT_CPU_COUNT = 4


def _calculate_worker_count(file_count: int, cpu_count: int) -> int:
    """Calculate optimal worker count based on file count and available CPUs.

    Args:
        file_count: Number of files to process
        cpu_count: Number of available CPU cores

    Returns:
        Optimal number of workers (capped based on workload size)
    """
    if file_count < SMALL_FILE_COUNT_THRESHOLD:
        return min(cpu_count, MAX_WORKERS_SMALL_BATCH, file_count)
    elif file_count < MEDIUM_FILE_COUNT_THRESHOLD:
        return min(cpu_count, MAX_WORKERS_MEDIUM_BATCH, file_count)
    else:
        return min(cpu_count, MAX_WORKERS_LARGE_BATCH, file_count)


class IndexingCoordinator(BaseService):
    """Coordinates file indexing workflows with parsing, chunking, and embeddings.

    # CLASS_CONTEXT: Orchestrates the three-phase indexing process
    # RELATIONSHIP: Uses -> LanguageParser, ChunkCacheService, DatabaseProvider
    # CONCURRENCY_MODEL:
    #   - Parse: CPU-bound, can parallelize across files
    #   - Embed: IO-bound, rate-limited batching
    #   - Store: Serial execution required (DB constraint)
    # TRANSACTION_SAFETY: All DB operations wrapped in transactions
    """

    def __init__(
        self,
        database_provider: DatabaseProvider,
        base_directory: Path,
        embedding_provider: EmbeddingProvider | None = None,
        language_parsers: dict[Language, UniversalParser] | None = None,
        progress: Progress | None = None,
        config: Any | None = None,
    ):
        """Initialize indexing coordinator.

        Args:
            database_provider: Database provider for persistence
            base_directory: Base directory for path normalization (always set)
            embedding_provider: Optional embedding provider for vector generation
            language_parsers: Optional mapping of language to parser implementations
            progress: Optional Rich Progress instance for hierarchical progress display
            config: Optional configuration object with indexing settings
        """
        super().__init__(database_provider)
        self._embedding_provider = embedding_provider
        self.progress = progress
        self._language_parsers = language_parsers or {}
        self.config = config

        # Performance optimization: shared instances
        self._parser_cache: dict[Language, UniversalParser] = {}

        # Chunk cache service for content-based comparison
        self._chunk_cache = ChunkCacheService()

        # Per-run cache for repo-aware ignore engines to avoid repeated tree scans
        # Key: (root, tuple(sources), chignore_file, tuple(cfg_excludes))
        self._ignore_engine_cache: dict[tuple[str, tuple[str, ...], str, tuple[str, ...]], object] = {}

        # Per-run cache for repo root detection to avoid repeated directory walks
        # Key: (root_path, tuple(sorted(cfg_excludes))) -> list[Path]
        self._repo_roots_cache: dict[tuple[str, tuple[str, ...]], list[Path]] = {}

        # SECTION: File_Level_Locking
        # CRITICAL: Prevents race conditions during concurrent file processing
        # PATTERN: Lazy lock creation within event loop context
        # WHY: asyncio.Lock() must be created inside the event loop
        self._file_locks: dict[str, asyncio.Lock] = {}
        self._locks_lock = None  # Will be initialized when first needed

        # Base directory for path normalization (immutable after initialization)
        # Store raw path - will resolve at usage time for consistent symlink handling
        self._base_directory: Path = base_directory

    def _get_relative_path(self, file_path: Path) -> Path:
        """Get relative path with consistent symlink resolution.

        Resolves both file path and base directory at the same time to ensure
        consistent symlink handling, preventing ValueError on Ubuntu CI systems
        where temporary directories often involve symlinks.
        """
        resolved_file = file_path.resolve()
        resolved_base = self._base_directory.resolve()
        return resolved_file.relative_to(resolved_base)

    def add_language_parser(self, language: Language, parser: UniversalParser) -> None:
        """Add or update a language parser.

        Args:
            language: Programming language identifier
            parser: Parser implementation for the language
        """
        self._language_parsers[language] = parser
        # Clear cache for this language
        if language in self._parser_cache:
            del self._parser_cache[language]

    def get_parser_for_language(self, language: Language) -> UniversalParser | None:
        """Get parser for specified language with caching.

        Args:
            language: Programming language identifier

        Returns:
            Parser instance or None if not supported
        """
        if language not in self._parser_cache:
            if language in self._language_parsers:
                parser = self._language_parsers[language]
                # Parser setup() already called during registration - no need to call again
                self._parser_cache[language] = parser
            else:
                return None

        return self._parser_cache[language]

    def detect_file_language(self, file_path: Path) -> Language | None:
        """Detect programming language from file.

        Uses content-based detection for ambiguous extensions (.m files).

        Args:
            file_path: Path to the file

        Returns:
            Language enum value or None if unsupported
        """
        language = detect_language(file_path)
        return language if language != Language.UNKNOWN else None

    # ------------------------------------------------------------------
    # Ignore engine caching helpers (per-run, process-local)
    # ------------------------------------------------------------------
    def _engine_cache_key(
        self, root: Path, sources: list[str], chf: str, cfg: list[str] | tuple[str, ...], backend: str = "python", overlay: bool | None = None
    ) -> tuple[str, tuple[str, ...], str, tuple[str, ...], str, int]:
        return (
            str(root.resolve()),
            tuple(sources),
            chf,
            tuple(cfg),
            backend,
            1 if overlay else 0,
        )

    def _get_or_build_ignore_engine(
        self, root: Path, sources: list[str], chf: str, cfg: list[str] | tuple[str, ...], backend: str = "python", overlay: bool | None = None
    ) -> object:
        key = self._engine_cache_key(root, sources, chf, cfg, backend, overlay)
        eng = self._ignore_engine_cache.get(key)
        if eng is not None:
            return eng
        try:
            from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine as _bre

            eng = _bre(root=root, sources=sources, chignore_file=chf, config_exclude=list(cfg), backend=backend, workspace_root_only_gitignore=overlay)
            self._ignore_engine_cache[key] = eng
            return eng
        except Exception:
            return None

    def _determine_db_batch_size(self, pending_inserts: list[Chunk]) -> int:
        """Compute an insert batch size using env/config or dynamic memory heuristics.

        Priority:
        - Env CHUNKHOUND_DB_BATCH_SIZE when set (>0)
        - Config indexing.db_batch_size when set (>0)
        - Dynamic: use a fraction of available memory with sane limits.
        """
        # 1) Environment override
        try:
            env_bs = int(os.environ.get("CHUNKHOUND_DB_BATCH_SIZE", "0") or "0")
            if env_bs > 0:
                return max(1, min(env_bs, 20000))
        except Exception:
            pass

        # 2) Config override
        try:
            if self.config and getattr(self.config, "indexing", None):
                cfg_bs = int(getattr(self.config.indexing, "db_batch_size", 0) or 0)
                if cfg_bs > 0:
                    return max(1, min(cfg_bs, 20000))
        except Exception:
            pass

        # 3) Dynamic heuristic
        # - Budget 10% of available RAM (min 64MB, max 512MB)
        # - Estimate average bytes per chunk from a sample (code length dominates)
        # - Constrain final batch size to [1000, 20000]
        def _mem_available_bytes() -> int:
            # Try psutil first
            try:
                import psutil  # type: ignore

                return int(getattr(psutil.virtual_memory(), "available", 0)) or 0
            except Exception:
                pass
            # Linux /proc/meminfo
            try:
                with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            parts = line.split()
                            if len(parts) >= 2:
                                return int(parts[1]) * 1024  # kB → B
            except Exception:
                pass
            return 0

        avail = _mem_available_bytes()
        if avail <= 0:
            # Fallback when unknown: adopt a conservative default
            avail = 512 * 1024 * 1024  # 512MB

        # Budget: 10% of available, clamped
        budget = int(avail * 0.10)
        budget = max(64 * 1024 * 1024, min(budget, 512 * 1024 * 1024))  # 64MB..512MB

        # Estimate bytes per chunk from a small sample (code length dominates payload)
        sample = pending_inserts[: min(200, len(pending_inserts))]
        if not sample:
            return 5000
        total_bytes = 0
        for ch in sample:
            code = getattr(ch, "code", "") or ""
            total_bytes += len(code.encode("utf-8", errors="ignore")) + 256  # overhead estimate
        avg = max(512, total_bytes // len(sample))

        # Compute batch size and clamp
        est = max(1, budget // avg)
        est = max(1000, min(int(est), 20000))
        return est

    async def _get_file_lock(self, file_path: Path) -> asyncio.Lock:
        """Get or create a lock for the given file path.

        # PATTERN: Double-checked locking for thread-safe lazy initialization
        # CONSTRAINT: asyncio.Lock() must be created in event loop context
        # EDGE_CASE: First call initializes _locks_lock itself

        Args:
            file_path: Path to the file

        Returns:
            AsyncIO lock for the file
        """
        # Initialize the locks lock if needed (first time, in event loop context)
        if self._locks_lock is None:
            self._locks_lock = asyncio.Lock()

        # Use resolve() instead of absolute() to handle symlinks consistently
        file_key = str(file_path.resolve())

        # Use the locks lock to ensure thread-safe access to the locks dictionary
        async with self._locks_lock:
            if file_key not in self._file_locks:
                # Create the lock within the event loop context
                self._file_locks[file_key] = asyncio.Lock()
            return self._file_locks[file_key]

    def _cleanup_file_lock(self, file_path: Path) -> None:
        """Remove lock for a file that no longer exists.

        Args:
            file_path: Path to the file
        """
        # Use resolve() instead of absolute() to handle symlinks consistently
        file_key = str(file_path.resolve())
        if file_key in self._file_locks:
            del self._file_locks[file_key]
            logger.debug(f"Cleaned up lock for deleted file: {file_key}")

    async def process_file(
        self, file_path: Path, skip_embeddings: bool = False
    ) -> dict[str, Any]:
        """Process a single file through the complete indexing pipeline.

        Uses the same parallel batch processing path as process_directory,
        but with a single-file batch for consistency.

        Args:
            file_path: Path to the file to process
            skip_embeddings: If True, skip embedding generation

        Returns:
            Dictionary with processing results including status, chunks, and embeddings
        """
        # CRITICAL: File-level locking prevents concurrent async processing
        # PATTERN: All processing happens inside the lock
        # PREVENTS: Race conditions in read-modify-write operations
        file_lock = await self._get_file_lock(file_path)
        async with file_lock:
            # Use batch processor with single file for consistency
            parsed_results = await self._process_files_in_batches([(file_path, None)])

            if not parsed_results:
                return {
                    "status": "error",
                    "chunks": 0,
                    "error": "No results from batch processor",
                }

            result = parsed_results[0]

            if result.status == "error":
                logger.warning(
                    f"Parse error for {file_path}: {result.error}"
                )
                return {"status": "error", "chunks": 0, "error": result.error}

            if result.status == "skipped":
                return {"status": "skipped", "reason": result.error, "chunks": 0}

            # Store the single file result
            store_result = await self._store_parsed_results([result], file_task=None)

            # Handle tuple return for single-file case
            if isinstance(store_result, tuple):
                stats, file_id = store_result
            else:
                # Should not happen for single file, but handle gracefully
                stats = store_result
                file_id = None

            # Generate embeddings if needed
            # CRITICAL FIX: Wrap embedding generation in transaction with checkpoint
            # RATIONALE: Embeddings must be checkpointed to be visible to semantic search
            # BUG: Previously inserted into WAL without checkpoint, invisible to queries
            embeddings_generated = 0
            embedding_error = None
            if not skip_embeddings and self._embedding_provider:
                if stats["chunk_ids_needing_embeddings"]:
                    # Generate embeddings
                    # NOTE: Transaction management is handled internally by the database provider
                    # to avoid transaction context issues during concurrent operations
                    try:
                        embeddings_generated = await self._generate_embeddings(
                            stats["chunk_ids_needing_embeddings"],
                            [chunk for r in parsed_results for chunk in r.chunks],
                        )

                        # Verify embeddings were actually generated
                        expected_embeddings = len(stats["chunk_ids_needing_embeddings"])
                        if embeddings_generated < expected_embeddings:
                            embedding_error = (
                                f"Only generated {embeddings_generated}/{expected_embeddings} embeddings. "
                                f"Some chunks may have empty content."
                            )
                            logger.warning(f"[IndexCoord] {embedding_error}")
                    except Exception as e:
                        embedding_error = str(e)
                        logger.error(
                            f"Failed to generate embeddings for {file_path}: {e}"
                        )
                        # Don't fail the entire operation if embeddings fail
                        # File chunks are already committed and searchable via regex

            return_dict = {
                "status": "success" if not stats["errors"] else "error",
                "chunks": stats["total_chunks"],
                "errors": stats["errors"],
                "embeddings_skipped": skip_embeddings,
                "embeddings_generated": embeddings_generated,
                "embedding_error": embedding_error,
            }

            # Include file_id for single-file operations
            if file_id is not None:
                return_dict["file_id"] = file_id

            if return_dict.get("status") == "error":
                logger.warning(
                    f"Store error for {file_path}: {return_dict.get('errors')}"
                )
            return return_dict

    async def _process_files_in_batches(
        self,
        files: list[tuple[Path, str | None]] | list[Path],
        config_file_size_threshold_kb: int = 20,
        parse_task: TaskID | None = None,
        on_batch: Any | None = None,
    ) -> list[ParsedFileResult]:
        """Process files in parallel batches across CPU cores.

        # PARALLELIZATION_STRATEGY:
        #   - File parsing: CPU-bound tree-sitter operations (parallelizable)
        #   - Batch processing: Each worker handles multiple files independently
        #   - Result aggregation: Collected in main thread for serial storage
        # CRITICAL: Only parsing is parallel, database operations remain single-threaded

        Each CPU core receives a batch of files and performs the complete
        read→parse→chunk pipeline independently before returning results.

        Args:
            files: List of file paths to process
            config_file_size_threshold_kb: Skip structured config files (JSON/YAML/TOML) larger than this (KB)

        Returns:
            List of ParsedFileResult objects with parsed chunks and metadata
        """
        if not files:
            return []

        # Calculate optimal worker count based on file count
        cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT
        file_count = len(files)

        # Determine timeout settings early to adjust worker count and batch sizing
        timeout_s_probe = 0.0
        try:
            if self.config and getattr(self.config, "indexing", None):
                timeout_s_probe = float(
                    getattr(self.config.indexing, "per_file_timeout_seconds", 0.0) or 0.0
                )
        except Exception:
            timeout_s_probe = 0.0

        # Inspect explicit concurrency override
        max_concurrent = 0
        try:
            if self.config and getattr(self.config, "indexing", None):
                max_concurrent = int(getattr(self.config.indexing, "max_concurrent", 0) or 0)
        except Exception:
            max_concurrent = 0

        # Default behavior:
        # - If timeouts are enabled and no explicit max_concurrent given,
        #   auto-scale to cpu_count (clamped to a safe upper bound).
        # - Otherwise, use heuristic based on file count with existing caps.
        SAFE_MAX = 32
        if timeout_s_probe > 0 and max_concurrent <= 0:
            num_workers = max(1, min(cpu_count, SAFE_MAX))
        else:
            num_workers = _calculate_worker_count(file_count, cpu_count)
            if max_concurrent > 0:
                num_workers = max(1, min(num_workers, max_concurrent))

        logger.debug(f"Parsing {file_count} files with {num_workers} workers (timeout={timeout_s_probe}s, max_concurrent={max_concurrent or 'auto'})")

        # Fast path for single-file processing: avoid creating a ProcessPoolExecutor
        # This eliminates sporadic BrokenProcessPool errors seen in CI on tiny files
        # and removes overhead when there is nothing to parallelize.
        if file_count == 1:
            # Build the same config dict as the parallel branch uses
            try:
                timeout_s = 0.0
                if self.config and getattr(self.config, "indexing", None):
                    timeout_s = float(
                        getattr(self.config.indexing, "per_file_timeout_seconds", 0.0)
                        or 0.0
                    )
            except Exception:
                timeout_s = 0.0

            try:
                min_timeout_kb = 128
                if self.config and getattr(self.config, "indexing", None):
                    min_timeout_kb = int(
                        getattr(
                            self.config.indexing, "per_file_timeout_min_size_kb", 128
                        )
                    )
            except Exception:
                min_timeout_kb = 128

            config_dict = {
                "config_file_size_threshold_kb": config_file_size_threshold_kb,
                "per_file_timeout_seconds": timeout_s,
                "per_file_timeout_min_size_kb": min_timeout_kb,
                # Cap concurrent timeout children to avoid resource exhaustion
                "max_concurrent_timeouts": min(max(1, num_workers) * 2, 32),
            }

            # Normalize to the batch-processor input format
            norm: list[tuple[Path, str | None]] = []
            for item in files:
                if isinstance(item, tuple):
                    norm.append(item)
                else:
                    norm.append((item, None))

            # Execute synchronously in-process for the single file
            results = process_file_batch(norm, config_dict)

            # Stream directly to storage to keep behavior consistent with the
            # parallel path where batches are stored as they complete.
            if on_batch is not None and results:
                try:
                    if asyncio.iscoroutinefunction(on_batch):
                        await on_batch(results)
                    else:
                        on_batch(results)
                except Exception as e:
                    logger.warning(f"on_batch handler failed: {e}")

            return results

        # Heuristic: increase batch granularity when per-file timeouts are enabled to avoid long silent stalls
        # (long stalls happen when large batches contain many files that each hit the timeout)
        dynamic_factor = 8 if timeout_s_probe > 0 else 4
        target_batches = max(num_workers * dynamic_factor, 1)
        # Compute base size from target_batches then clamp to [MIN_BATCH, file_count]
        # Use smaller minimum when timeouts are enabled to surface progress more frequently
        MIN_BATCH = 16 if timeout_s_probe > 0 else 128
        base = max(1, math.ceil(len(files) / target_batches))
        batch_size = min(len(files), max(MIN_BATCH, base))
        # Additional clamp: target ~60s worst-case batch wall time when timeouts are enabled
        if timeout_s_probe > 0:
            target_secs = 60.0
            per_file_cap = max(4, int(target_secs / max(0.001, timeout_s_probe)))
            batch_size = min(batch_size, per_file_cap)
        # Normalize input to list[tuple[Path, str|None]]
        norm: list[tuple[Path, str | None]] = []
        for item in files:
            if isinstance(item, tuple):
                norm.append(item)
            else:
                norm.append((item, None))
        file_batches = [norm[i : i + batch_size] for i in range(0, len(norm), batch_size)]

        # Process batches in parallel using ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches for concurrent processing
            # Pass config for structured file size filtering (JSON/YAML/TOML)
            # Include optional per-file timeout (seconds) if configured
            timeout_s = 0.0
            try:
                if self.config and getattr(self.config, "indexing", None):
                    timeout_s = float(
                        getattr(self.config.indexing, "per_file_timeout_seconds", 0.0)
                        or 0.0
                    )
            except Exception:
                timeout_s = 0.0

            min_timeout_kb = 128
            try:
                if self.config and getattr(self.config, "indexing", None):
                    # Respect explicit 0 so users can apply timeout to all file sizes.
                    min_timeout_kb = int(
                        getattr(self.config.indexing, "per_file_timeout_min_size_kb", 128)
                    )
            except Exception:
                min_timeout_kb = 128

            config_dict = {
                "config_file_size_threshold_kb": config_file_size_threshold_kb,
                "per_file_timeout_seconds": timeout_s,
                "per_file_timeout_min_size_kb": min_timeout_kb,
                # Cap concurrent timeout children to avoid resource exhaustion
                "max_concurrent_timeouts": min(num_workers * 2, 32),
            }
            futures = [
                loop.run_in_executor(executor, process_file_batch, batch, config_dict)
                for batch in file_batches
            ]

            # Consume results as they complete to stream progress
            all_results: list[ParsedFileResult] = []
            completed_files = 0
            for fut in asyncio.as_completed(futures):
                batch_result = await fut
                all_results.extend(batch_result)
                # Update parse progress
                if parse_task is not None and self.progress:
                    inc = len(batch_result)
                    completed_files += inc
                    self.progress.advance(parse_task, inc)
                    self.progress.update(parse_task, info=f"{completed_files} parsed")
                # Stream results to storage if callback provided
                if on_batch is not None:
                    try:
                        if asyncio.iscoroutinefunction(on_batch):
                            await on_batch(batch_result)
                        else:
                            on_batch(batch_result)
                    except Exception as e:
                        logger.warning(f"on_batch handler failed: {e}")

        return all_results

    async def _store_parsed_results(
        self,
        results: list[ParsedFileResult],
        file_task: TaskID | None = None,
        cumulative_counters: dict[str, int] | None = None,
    ) -> dict[str, Any] | tuple[dict[str, Any], int]:
        """Store all parsed results in database (single-threaded).

        Args:
            results: List of parsed file results from batch processing
            file_task: Optional progress task ID for tracking

        Returns:
            For multiple files: Dictionary with processing statistics
            For single file: Tuple of (statistics dict, file_id)
        """
        stats = {
            "total_files": 0,
            "total_chunks": 0,
            "errors": [],
            "chunk_ids_needing_embeddings": [],
        }

        # Track file_ids for single-file case
        file_ids = []

        # Process each file independently (per-file transaction)
        for result in results:
                # Handle errors
                if result.status == "error":
                    stats["errors"].append(
                        {"file": str(result.file_path), "error": result.error}
                    )
                    if file_task is not None and self.progress:
                        self.progress.advance(file_task, 1)
                        if cumulative_counters is not None:
                            cumulative_counters['errors'] = cumulative_counters.get('errors', 0) + 1
                            stored = cumulative_counters.get('stored', 0)
                            skipped = cumulative_counters.get('skipped', 0)
                            errs = cumulative_counters.get('errors', 0)
                            chunks_so_far = cumulative_counters.get('chunks', 0)
                            self.progress.update(file_task, info=f"stored {stored} | skipped {skipped} | err {errs} | {chunks_so_far} chunks")
                    continue

                # Handle skipped files
                if result.status == "skipped":
                    # Track skip reason in stats for single-file case
                    if "skip_reason" not in stats:
                        stats["skip_reason"] = result.error
                    if file_task is not None and self.progress:
                        self.progress.advance(file_task, 1)
                        if cumulative_counters is not None:
                            cumulative_counters['skipped'] = cumulative_counters.get('skipped', 0) + 1
                            stored = cumulative_counters.get('stored', 0)
                            skipped = cumulative_counters.get('skipped', 0)
                            errs = cumulative_counters.get('errors', 0)
                            chunks_so_far = cumulative_counters.get('chunks', 0)
                            self.progress.update(file_task, info=f"stored {stored} | skipped {skipped} | err {errs} | {chunks_so_far} chunks")
                    continue

                # Detect language for storage
                language = result.language

                # Per-file transaction boundaries
                self._db.begin_transaction()
                try:
                    # Store file metadata
                    file_stat_dict = {
                        "st_size": result.file_size,
                        "st_mtime": result.file_mtime,
                    }

                    # Create mock stat object for _store_file_record
                    class StatResult:
                        def __init__(self, size: int, mtime: float):
                            self.st_size = size
                            self.st_mtime = mtime

                    file_stat = StatResult(result.file_size, result.file_mtime)
                    # Extract content hash if available (from parsing result or precomputed)
                    content_hash = getattr(result, "content_hash", None)
                    file_id = self._store_file_record(result.file_path, file_stat, language, content_hash)

                    # Track file_id for single-file case
                    file_ids.append(file_id)

                    if file_id is None:
                        self._db.rollback_transaction()
                        stats["errors"].append(
                            {
                                "file": str(result.file_path),
                                "error": "Failed to store file record",
                            }
                        )
                        if file_task is not None and self.progress:
                            self.progress.advance(file_task, 1)
                        continue
                    # Check for existing chunks to enable smart diffing
                    relative_path = self._get_relative_path(result.file_path)
                    existing_file = self._db.get_file_by_path(relative_path.as_posix())

                    # Convert result chunks to Chunk models using from_dict()
                    new_chunk_models = [
                        Chunk.from_dict({**chunk_data, "file_id": file_id})
                        for chunk_data in result.chunks
                    ]

                    if existing_file:
                        # Get existing chunks for diffing
                        existing_chunks = self._db.get_chunks_by_file_id(
                            file_id, as_model=True
                        )

                        if existing_chunks:
                            # Smart diff to preserve embeddings
                            chunk_diff = self._chunk_cache.diff_chunks(
                                new_chunk_models, existing_chunks
                            )

                            # Delete modified/removed chunks
                            chunks_to_delete = chunk_diff.deleted + chunk_diff.modified
                            if chunks_to_delete:
                                chunk_ids_to_delete = [
                                    chunk.id
                                    for chunk in chunks_to_delete
                                    if chunk.id is not None
                                ]
                                if chunk_ids_to_delete:
                                    self._db.delete_chunks_batch(chunk_ids_to_delete)

                            # Store new/modified chunks (pass models directly)
                            chunks_to_store = chunk_diff.added + chunk_diff.modified
                            ids = self._db.insert_chunks_batch(chunks_to_store) if chunks_to_store else []
                        else:
                            # No existing chunks - store all as new
                            ids = self._db.insert_chunks_batch(new_chunk_models)
                    else:
                        # New file - store all
                        ids = self._db.insert_chunks_batch(new_chunk_models)

                    stats["chunk_ids_needing_embeddings"].extend(ids)
                    stats["total_chunks"] += len(ids)
                    # Count this file as processed successfully (stored or updated)
                    stats["total_files"] += 1

                    # Commit per-file
                    try:
                        self._db.commit_transaction()
                    except TypeError:
                        try:
                            self._db.commit_transaction(force_checkpoint=True)
                        except Exception:
                            pass

                    # Update progress
                    if file_task is not None and self.progress:
                        self.progress.advance(file_task, 1)
                        if cumulative_counters is not None:
                            cumulative_counters['stored'] = cumulative_counters.get('stored', 0) + 1
                            base = int(cumulative_counters.get('chunks', 0))
                            display_chunks = base + stats["total_chunks"]
                            stored = cumulative_counters.get('stored', 0)
                            skipped = cumulative_counters.get('skipped', 0)
                            errs = cumulative_counters.get('errors', 0)
                            self.progress.update(file_task, info=f"stored {stored} | skipped {skipped} | err {errs} | {display_chunks} chunks")

                except Exception as e:
                    self._db.rollback_transaction()
                    stats["errors"].append({"file": str(result.file_path), "error": str(e)})
                    if file_task is not None and self.progress:
                        self.progress.advance(file_task, 1)
                    continue

        # Update external cumulative counters
        if cumulative_counters is not None:
            cumulative_counters['chunks'] = cumulative_counters.get('chunks', 0) + stats["total_chunks"]
            cumulative_counters['files'] = cumulative_counters.get('files', 0) + stats["total_files"]

        # Return file_id for single-file case
        if len(results) == 1 and file_ids and file_ids[0] is not None:
            return stats, file_ids[0]
        return stats

    async def process_directory(
        self,
        directory: Path,
        patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        config_file_size_threshold_kb: int = 20,
    ) -> dict[str, Any]:
        """Process all supported files in a directory with batch optimization and consistency checks.

        Args:
            directory: Directory path to process
            patterns: Optional file patterns to include
            exclude_patterns: Optional file patterns to exclude
            config_file_size_threshold_kb: Skip structured config files (JSON/YAML/TOML) larger than this (KB)

        Returns:
            Dictionary with processing statistics
        """
        try:
            import time as _t
            _t0 = _t.perf_counter() if getattr(self, "profile_startup", False) else None
            # Phase 1: Discovery - Discover files in directory (now parallelized)
            files = await self._discover_files(directory, patterns, exclude_patterns)
            _t1 = _t.perf_counter() if _t0 is not None else None

            if not files:
                return {"status": "no_files", "files_processed": 0, "total_chunks": 0}

            # Phase 2: Reconciliation - Ensure database consistency by removing orphaned files
            cleaned_files = 0
            try:
                do_cleanup = True
                if self.config and getattr(self.config, "indexing", None) is not None:
                    do_cleanup = bool(getattr(self.config.indexing, "cleanup", True))
                if do_cleanup:
                    _t2 = _t.perf_counter() if _t0 is not None else None
                    cleaned_files = self._cleanup_orphaned_files(
                        directory, files, exclude_patterns
                    )
                    _t3 = _t.perf_counter() if _t0 is not None else None
                else:
                    logger.debug("Skipping orphaned file cleanup (cleanup disabled)")
            except Exception as e:
                logger.warning(f"Cleanup phase skipped due to error: {e}")

            logger.debug(
                f"Directory consistency: {len(files)} files discovered, {cleaned_files} orphaned files cleaned"
            )

            # Phase 2.5: Change detection (skip unchanged files unless force_reindex)
            force_reindex = False
            try:
                if self.config and getattr(self.config, "indexing", None):
                    force_reindex = bool(getattr(self.config.indexing, "force_reindex", False))
            except Exception:
                force_reindex = False

            files_to_process: list[Path] = files
            skipped_unchanged = 0
            if not force_reindex:
                _t4 = _t.perf_counter() if _t0 is not None else None
                change_task: TaskID | None = None
                if self.progress:
                    change_task = self.progress.add_task(
                        "  └─ Scanning changes", total=len(files), speed="", info=""
                    )

                debug_skip = bool(os.getenv("CHUNKHOUND_DEBUG_SKIP"))
                reasons = {"not_found": 0, "size": 0, "mtime": 0, "ok": 0, "error": 0}
                # Configurable tolerances
                mtime_eps = 0.01
                try:
                    if self.config and getattr(self.config, "indexing", None):
                        mtime_eps = float(getattr(self.config.indexing, "mtime_epsilon_seconds", 0.01) or 0.01)
                except Exception:
                    mtime_eps = 0.01

                files_to_process = []
                # Batch-fetch DB metadata once to avoid per-file lookups
                db_meta_map: dict[str, tuple[int | None, float | None, str | None]] = {}
                try:
                    # Prefer content_hash if available but tolerate schema variance
                    try:
                        rows = self._db.execute_query(
                            "SELECT path, size, modified_time, content_hash FROM files",
                            [],
                        )
                    except Exception:
                        rows = self._db.execute_query(
                            "SELECT path, size, modified_time FROM files",
                            [],
                        )
                    for r in rows or []:
                        p = r.get("path") if isinstance(r, dict) else None
                        if not p:
                            continue
                        sz = r.get("size") if isinstance(r, dict) else None
                        mt = r.get("modified_time") if isinstance(r, dict) else None
                        try:
                            mtv = float(mt.timestamp()) if hasattr(mt, "timestamp") else float(mt)
                        except Exception:
                            mtv = None
                        ch = r.get("content_hash") if isinstance(r, dict) else None
                        db_meta_map[str(p)] = (int(sz) if sz is not None else None, mtv, ch)
                except Exception:
                    db_meta_map = {}
                precomputed_hashes: dict[str, str] = {}
                for f in files:
                    try:
                        rel = self._get_relative_path(f).as_posix()
                        db_tuple = db_meta_map.get(rel)
                        if db_tuple is None:
                            # Fallback for providers without execute_query() (e.g., fake or limited providers)
                            try:
                                rec = self._db.get_file_by_path(rel, as_model=False)
                                if rec:
                                    sz = rec.get("size") if isinstance(rec, dict) else None
                                    mt = rec.get("modified_time") if isinstance(rec, dict) else None
                                    try:
                                        mtv = float(mt.timestamp()) if hasattr(mt, "timestamp") else float(mt)
                                    except Exception:
                                        mtv = None
                                    ch = rec.get("content_hash") if isinstance(rec, dict) else None
                                    db_tuple = (int(sz) if sz is not None else None, mtv, ch)
                            except Exception:
                                db_tuple = None
                        st = f.stat()
                        if db_tuple is not None:
                            db_size, stored_mtime, db_hash = db_tuple
                            db_size = int(db_size) if db_size is not None else -1
                            same_size = db_size == int(st.st_size)
                            smt = float(stored_mtime) if stored_mtime is not None else -1.0
                            same_mtime = abs(smt - float(st.st_mtime)) <= mtime_eps
                            if same_size and same_mtime:
                                # Fast skip - trust filesystem metadata (mtime+size match)
                                skipped_unchanged += 1
                                reasons["ok"] += 1
                            else:
                                # Size or mtime changed - verify if content actually changed via checksum
                                cur_hash = self._compute_hash_with_fallback(f)

                                if db_hash and cur_hash and db_hash == cur_hash:
                                    # False positive - metadata changed but content didn't
                                    skipped_unchanged += 1
                                    reasons["ok"] += 1
                                else:
                                    # Content actually changed (or hash unavailable) - reindex
                                    precomputed_hashes[str(f.resolve())] = cur_hash
                                    files_to_process.append((f, cur_hash))
                                    if not same_size:
                                        reasons["size"] += 1
                                    elif not same_mtime:
                                        reasons["mtime"] += 1
                                    if cur_hash is None:
                                        reasons["error"] += 1
                        else:
                            # New file not in DB - compute hash for skip optimization on next run
                            cur_hash = self._compute_hash_with_fallback(f)
                            precomputed_hashes[str(f.resolve())] = cur_hash
                            files_to_process.append((f, cur_hash))
                            if cur_hash is None:
                                reasons["error"] += 1
                            else:
                                reasons["not_found"] += 1
                    except Exception:
                        files_to_process.append((f, None))
                        reasons["error"] += 1
                    finally:
                        if change_task is not None and self.progress:
                            self.progress.advance(change_task, 1)

                if change_task is not None and self.progress:
                    task = self.progress.tasks[change_task]
                    if task.total:
                        self.progress.update(change_task, completed=task.total)
                _t5 = _t.perf_counter() if _t0 is not None else None
                if debug_skip:
                    logger.warning(
                        f"Skip-check summary: ok={reasons['ok']} not_found={reasons['not_found']} "
                        f"size_mismatch={reasons['size']} mtime_mismatch={reasons['mtime']} error={reasons['error']} "
                        f"mtime_eps={mtime_eps} files_to_process={len(files_to_process)} total={len(files)}"
                    )

            # Phase 3: Parse + Store
            parse_task: TaskID | None = None
            store_task: TaskID | None = None
            if self.progress:
                parse_task = self.progress.add_task(
                    "  └─ Parsing files", total=len(files_to_process), speed="", info=""
                )
                store_task = self.progress.add_task(
                    "  └─ Handling files", total=len(files_to_process), speed="", info=""
                )

            # Aggregators for streamed storage
            agg_total_files = 0
            agg_total_chunks = 0
            agg_errors: list[dict[str, Any]] = []
            agg_skipped = 0
            agg_skipped_timeout: list[str] = []

            store_progress_counters = {"chunks": 0, "files": 0, "stored": 0, "skipped": 0, "errors": 0}

            async def _on_batch_store(batch: list[ParsedFileResult]) -> None:
                nonlocal agg_total_files, agg_total_chunks, agg_errors, agg_skipped, agg_skipped_timeout
                # Update skip counters from parse results
                for r in batch:
                    if r.status == "skipped":
                        agg_skipped += 1
                        if (r.error or "").lower() == "timeout":
                            agg_skipped_timeout.append(str(r.file_path))

                # Store this batch immediately
                stats_part = await self._store_parsed_results(
                    batch, store_task, cumulative_counters=store_progress_counters
                )
                if isinstance(stats_part, tuple):
                    stats_part = stats_part[0]
                agg_total_files += stats_part.get("total_files", 0)
                agg_total_chunks += stats_part.get("total_chunks", 0)
                agg_errors.extend(stats_part.get("errors", []))

            # Parse files (streaming progress as batches complete and store concurrently)
            # Pass files_to_process directly - preserves hash for each file
            parsed_results = await self._process_files_in_batches(
                files_to_process, config_file_size_threshold_kb, parse_task, on_batch=_on_batch_store
            )

            # Mark parse task complete
            if parse_task is not None and self.progress:
                task = self.progress.tasks[parse_task]
                if task.total:
                    self.progress.update(parse_task, completed=task.total)

            # Record startup profile if enabled (before heavy parse+store dominates totals)
            if _t0 is not None:
                try:
                    self._startup_profile = {
                        "discovery_ms": round(((_t1 - _t0) if (_t1 and _t0) else 0.0) * 1000.0, 3),
                        "cleanup_ms": round(((_t3 - _t2) if (locals().get("_t3") and locals().get("_t2")) else 0.0) * 1000.0, 3),
                        "change_scan_ms": round(((_t5 - _t4) if (locals().get("_t5") and locals().get("_t4")) else 0.0) * 1000.0, 3),
                        "files_discovered": len(files),
                        "orphaned_cleaned": cleaned_files,
                        "files_after_change_scan": len(files_to_process),
                        "parallel_used": bool(getattr(self, "_profile_parallel_used", False)),
                    }
                except Exception:
                    pass

            # At this point, all parsed results have been stored via _on_batch_store
            stats = {
                "total_files": agg_total_files,
                "total_chunks": agg_total_chunks,
                "errors": agg_errors,
            }

            # Track skipped files (including timeouts)
            skipped_total = agg_skipped
            skipped_due_to_timeout = agg_skipped_timeout
            # Split parse-time skips into timeout vs other (filtered/unsupported/config)
            skipped_filtered = max(0, skipped_total - len(skipped_due_to_timeout))

            total_files = stats["total_files"]
            total_chunks = stats["total_chunks"]

            # Log any errors
            for error in stats["errors"]:
                logger.warning(f"Failed to process {error['file']}: {error['error']}")

            # Complete the file processing progress bar
            if store_task is not None and self.progress:
                task = self.progress.tasks[store_task]
                if task.total:
                    self.progress.update(store_task, completed=task.total)

            # Note: Embedding generation is handled separately via generate_missing_embeddings()
            # to provide a unified progress experience

            # Optimize tables after bulk operations (provider-specific)
            if total_chunks > 0 and hasattr(self._db, "optimize_tables"):
                logger.debug("Optimizing database tables after bulk operations...")
                self._db.optimize_tables()

            return {
                "status": "success",
                "files_processed": total_files,
                "total_chunks": total_chunks,
                "skipped": skipped_total + skipped_unchanged,
                "skipped_due_to_timeout": skipped_due_to_timeout,
                "skipped_unchanged": skipped_unchanged,
                "skipped_filtered": skipped_filtered,
            }

        except Exception as e:
            import traceback

            logger.error(f"Failed to process directory {directory}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "error": str(e)}

    def _extract_file_id(self, file_record: dict[str, Any] | File) -> int | None:
        """Safely extract file ID from either dict or File model."""
        if isinstance(file_record, File):
            return file_record.id
        elif isinstance(file_record, dict) and "id" in file_record:
            return file_record["id"]
        else:
            return None

    def _compute_hash_with_fallback(self, file_path: Path) -> str | None:
        """Compute file hash with consistent error handling.

        Args:
            file_path: Path to the file to hash

        Returns:
            Hash string on success, None on error
        """
        try:
            return compute_file_hash(file_path)
        except (OSError, IOError) as e:
            rel_path = self._get_relative_path(file_path).as_posix()
            logger.warning(f"Failed to compute hash for {rel_path}: {e}")
            return None

    def _store_file_record(
        self, file_path: Path, file_stat: Any, language: Language, content_hash: str | None = None
    ) -> int:
        """Store or update file record in database.

        Args:
            file_path: Path to the file
            file_stat: File stat object with st_size and st_mtime
            language: Programming language of the file
            content_hash: Optional content hash for change detection

        Returns:
            File ID
        """
        # Check if file already exists
        # Use consistent symlink-safe path resolution
        relative_path = self._get_relative_path(file_path)
        existing_file = self._db.get_file_by_path(relative_path.as_posix())

        if existing_file:
            # Update existing file with new metadata
            if isinstance(existing_file, dict) and "id" in existing_file:
                file_id = existing_file["id"]
                self._db.update_file(
                    file_id, size_bytes=file_stat.st_size, mtime=file_stat.st_mtime,
                    content_hash=content_hash
                )
                return file_id

        # Create new File model instance with relative path
        # Use consistent symlink-safe path resolution
        relative_path = self._get_relative_path(file_path)
        file_model = File(
            path=FilePath(relative_path.as_posix()),
            size_bytes=file_stat.st_size,
            mtime=file_stat.st_mtime,
            language=language,
            content_hash=content_hash,
        )
        return self._db.insert_file(file_model)

    def _store_chunks(
        self, file_id: int, chunk_models: list[Chunk], language: Language
    ) -> list[int]:
        """Store chunks in database and return chunk IDs.

        Args:
            file_id: File ID for the chunks
            chunk_models: List of Chunk model instances to store
            language: Language (for compatibility, already set in models)

        Returns:
            List of chunk IDs from database insertion
        """
        if not chunk_models:
            return []

        # Use batch insertion for optimal performance
        chunk_ids = self._db.insert_chunks_batch(chunk_models)

        # Log batch operation
        logger.debug(f"Batch inserted {len(chunk_ids)} chunks for file_id {file_id}")

        return chunk_ids

    async def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with file, chunk, and embedding counts
        """
        return self._db.get_stats()

    async def remove_file(self, file_path: str) -> int:
        """Remove a file and all its chunks from the database.

        Args:
            file_path: Path to the file to remove

        Returns:
            Number of chunks removed
        """
        try:
            # Convert path to relative format for database lookup
            file_path_obj = Path(file_path)
            if file_path_obj.is_absolute():
                base_dir = self._base_directory
                relative_path = file_path_obj.relative_to(base_dir).as_posix()
            else:
                relative_path = file_path_obj.as_posix()

            # Get file record to get chunk count before deletion
            file_record = self._db.get_file_by_path(relative_path)
            if not file_record:
                return 0

            # Get file ID
            file_id = self._extract_file_id(file_record)
            if file_id is None:
                return 0

            # Count chunks before deletion
            chunks = self._db.get_chunks_by_file_id(file_id)
            chunk_count = len(chunks) if chunks else 0

            # Delete the file completely (this will also delete chunks and embeddings)
            success = self._db.delete_file_completely(relative_path)

            # Clean up the file lock since the file no longer exists
            if success:
                self._cleanup_file_lock(Path(file_path))

            return chunk_count if success else 0

        except Exception as e:
            logger.error(f"Failed to remove file {file_path}: {e}")
            return 0

    async def generate_missing_embeddings(
        self,
        exclude_patterns: list[str] | None = None,
        thin_check: bool = False,
    ) -> dict[str, Any]:
        """Generate embeddings for chunks that don't have them.

        Args:
            exclude_patterns: Optional file patterns to exclude from embedding generation

        Returns:
            Dictionary with generation results
        """
        if not self._embedding_provider:
            return {
                "status": "error",
                "error": "No embedding provider configured",
                "generated": 0,
            }

        try:
            # Optional fast-path: only if explicitly requested
            if thin_check and self._embedding_provider:
                try:
                    provider = getattr(self._embedding_provider, "name", None)
                    model = getattr(self._embedding_provider, "model", None)
                    if provider and model:
                        # Discover embedding tables
                        tables: list[str] = []
                        try:
                            rows = self._db.execute_query(
                                """
                                SELECT table_name AS name
                                FROM information_schema.tables
                                WHERE table_name LIKE 'embeddings_%'
                                """,
                                [],
                            )
                            for r in rows or []:
                                name = r.get("name") if isinstance(r, dict) else None
                                if name:
                                    tables.append(str(name))
                        except Exception:
                            tables = []

                        # If we cannot enumerate tables, fallback to normal path
                        if tables:
                            clauses = []
                            for _ in tables:
                                clauses.append(
                                    """
                                    NOT EXISTS (
                                        SELECT 1 FROM {table} e
                                        WHERE e.chunk_id = c.id AND e.provider = ? AND e.model = ?
                                    )
                                    """
                                )
                            # Build query by injecting table names safely
                            not_exists_sql = " AND ".join(
                                clause.format(table=t) for clause in clauses
                            )
                            probe_sql = f"""
                                SELECT 1
                                FROM chunks c
                                WHERE {not_exists_sql}
                                LIMIT 1
                            """
                            params = []
                            # Repeat provider/model params for each table
                            for _ in tables:
                                params.extend([provider, model])
                            probe = self._db.execute_query(probe_sql, params)
                            if not probe:
                                # No missing embeddings: short-circuit
                                return {"status": "complete", "generated": 0}
                        else:
                            # No embedding tables present; if there are no chunks, report complete
                            try:
                                any_chunk = self._db.execute_query(
                                    "SELECT 1 FROM chunks LIMIT 1", []
                                )
                                if not any_chunk:
                                    return {"status": "complete", "generated": 0}
                            except Exception:
                                pass
                except Exception:
                    # On any probe error, fall through to normal generation
                    pass

            # Use EmbeddingService for embedding generation
            from .embedding_service import EmbeddingService

            # Get optimization frequency from config or use default
            optimization_batch_frequency = 1000
            if hasattr(self._db, "_config") and self._db._config:
                optimization_batch_frequency = getattr(
                    self._db._config.embedding, "optimization_batch_frequency", 1000
                )

            embedding_service = EmbeddingService(
                database_provider=self._db,
                embedding_provider=self._embedding_provider,
                optimization_batch_frequency=optimization_batch_frequency,
                progress=self.progress,
            )

            return await embedding_service.generate_missing_embeddings(
                exclude_patterns=exclude_patterns
            )

        except Exception as e:
            # Debug log to trace if this is the mystery error source
            import os
            from datetime import datetime

            debug_file = os.getenv("CHUNKHOUND_DEBUG_FILE", "/tmp/chunkhound_debug.log")
            timestamp = datetime.now().isoformat()
            try:
                with open(debug_file, "a") as f:
                    f.write(
                        f"[{timestamp}] [COORDINATOR-MISSING] Failed to generate missing embeddings: {e}\n"
                    )
                    f.flush()
            except Exception:
                pass

            logger.error(
                f"[IndexCoord-Missing] Failed to generate missing embeddings: {e}"
            )
            return {"status": "error", "error": str(e), "generated": 0}

    async def _generate_embeddings(
        self, chunk_ids: list[int], chunks: list[dict[str, Any]], connection=None
    ) -> int:
        """Generate embeddings for chunks."""
        if not self._embedding_provider:
            return 0

        # VALIDATION: Ensure chunk IDs and chunks are aligned
        if len(chunk_ids) != len(chunks):
            error_msg = (
                f"Data mismatch in embedding generation: "
                f"{len(chunk_ids)} chunk IDs but {len(chunks)} chunks. "
                f"This indicates a bug in chunk processing."
            )
            logger.error(f"[IndexCoord] {error_msg}")
            raise ValueError(error_msg)

        try:
            # Filter out chunks with empty text content before embedding
            valid_chunk_data = []
            empty_count = 0
            for chunk_id, chunk in zip(chunk_ids, chunks):
                from chunkhound.utils.normalization import normalize_content

                text = normalize_content(chunk.get("code", ""))
                if text:  # Only include chunks with actual content
                    valid_chunk_data.append((chunk_id, chunk, text))
                else:
                    empty_count += 1

            # Log metrics for empty chunks
            if empty_count > 0:
                logger.debug(
                    f"Filtered {empty_count} empty text chunks before embedding generation"
                )

            if not valid_chunk_data:
                logger.debug(
                    "No valid chunks with text content for embedding generation"
                )
                return 0

            # Extract data for embedding generation
            valid_chunk_ids = [chunk_id for chunk_id, _, _ in valid_chunk_data]
            texts = [text for _, _, text in valid_chunk_data]

            # Generate embeddings (progress tracking handled by missing embeddings phase)
            embedding_results = await self._embedding_provider.embed(texts)

            # Store embeddings in database
            embeddings_data = []
            for chunk_id, vector in zip(valid_chunk_ids, embedding_results):
                embeddings_data.append(
                    {
                        "chunk_id": chunk_id,
                        "provider": self._embedding_provider.name,
                        "model": self._embedding_provider.model,
                        "dims": len(vector),
                        "embedding": vector,
                    }
                )

            # CRITICAL FIX: Ensure clean transaction state before database insertion
            # In concurrent scenarios, the executor thread may have an aborted transaction
            # from a previous operation. Try insertion, and if we get a transaction error,
            # clean up and retry once.
            try:
                result = self._db.insert_embeddings_batch(
                    embeddings_data, connection=connection
                )
                return result
            except Exception as e:
                if "transaction is aborted" in str(e).lower():
                    logger.warning(
                        f"[IndexCoord] Transaction aborted during embedding insertion, "
                        f"attempting recovery and retry"
                    )
                    # Try to clean up the aborted transaction
                    try:
                        self._db.rollback_transaction()
                    except Exception:
                        pass  # Ignore errors during cleanup

                    # Retry the insertion with a fresh transaction
                    result = self._db.insert_embeddings_batch(
                        embeddings_data, connection=connection
                    )
                    logger.info(
                        f"[IndexCoord] Successfully inserted {result} embeddings after "
                        f"transaction recovery"
                    )
                    return result
                else:
                    # Not a transaction error, re-raise
                    raise

        except Exception as e:
            # Log chunk details for debugging oversized chunks
            text_sizes = [len(text) for text in texts] if "texts" in locals() else []
            max_chars = max(text_sizes) if text_sizes else 0
            logger.error(
                f"[IndexCoord] Failed to generate embeddings (chunks: {len(text_sizes)}, max_chars: {max_chars}): {e}"
            )
            return 0

    async def _generate_embeddings_batch(
        self, file_chunks: list[tuple[int, dict[str, Any]]]
    ) -> int:
        """Generate embeddings for chunks in optimized batches."""
        if not self._embedding_provider or not file_chunks:
            return 0

        # Extract chunk IDs and text content
        chunk_ids = [chunk_id for chunk_id, _ in file_chunks]
        chunks = [chunk_data for _, chunk_data in file_chunks]

        return await self._generate_embeddings(chunk_ids, chunks)

    async def _discover_files_parallel(
        self,
        directory: Path,
        patterns: list[str],
        exclude_patterns: list[str],
        use_inode_ordering: bool = False,
    ) -> list[Path] | None:
        """Parallel directory discovery using multi-core traversal.

        ARCHITECTURE: Partitions directory tree at top level and processes subtrees
        in parallel using ProcessPoolExecutor. Workers are isolated processes to avoid
        GIL contention and enable true parallelism.

        Args:
            directory: Resolved directory path to search
            patterns: File patterns to include (validated non-empty)
            exclude_patterns: Patterns to exclude
            use_inode_ordering: Sort directories by inode

        Returns:
            List of discovered file paths on successful parallel discovery,
            or None to signal fallback to sequential mode is needed

        Raises:
            Logs warnings and returns None on errors
        """
        # Get top-level directories (first level subdirectories)
        # RACE CONDITION SAFETY: Handle directories deleted/modified during iteration
        top_level_items = []
        # Use effective config excludes (includes defaults even when sentinel is set)
        effective_excludes = list(
            (self.config.indexing.get_effective_config_excludes() if self.config and getattr(self.config, "indexing", None) else [])
        )
        # Also add dynamic DB path exclusion when DB lives under the directory
        try:
            dbp = getattr(self._db, "db_path", None)
            if dbp:
                dbp_res = Path(dbp).resolve()
                dir_res = directory.resolve()
                try:
                    rel = dbp_res.relative_to(dir_res)
                    if dbp_res.is_dir():
                        rp = rel.as_posix()
                        effective_excludes.extend([rp, f"{rp}/**"])  # safe duplicates
                    else:
                        effective_excludes.append(rel.as_posix())
                except Exception:
                    pass
        except Exception:
            pass
        try:
            for item in directory.iterdir():
                try:
                    # Check if item is still a directory (could change during iteration)
                    if not item.is_dir():
                        continue

                    # Check if this directory should be excluded
                    rel_path = item.relative_to(directory)
                    should_skip = False
                    # Prefer effective_excludes for early pruning; fall back to provided list
                    prune_patterns = effective_excludes or exclude_patterns
                    for pattern in prune_patterns:
                        if pattern.startswith("**/") and pattern.endswith("/**"):
                            target_dir = pattern[3:-3]
                            if target_dir in rel_path.parts:
                                should_skip = True
                                break
                        elif fnmatch(str(rel_path), pattern) or fnmatch(
                            item.name, pattern
                        ):
                            should_skip = True
                            break
                    if not should_skip:
                        top_level_items.append(item)
                except (FileNotFoundError, NotADirectoryError, ValueError):
                    # Item deleted, changed type, or can't be made relative - skip it
                    continue
        except (PermissionError, OSError) as e:
            logger.warning(f"Error accessing directory {directory}: {e}")
            return None

        # Check if parallel mode is beneficial
        # Use config value if available, otherwise use default of 4
        min_dirs_threshold = (
            self.config.indexing.min_dirs_for_parallel if self.config else 4
        )
        if len(top_level_items) < min_dirs_threshold:
            logger.info(
                f"Using sequential discovery: {len(top_level_items)} top-level dirs "
                f"< {min_dirs_threshold} threshold (parallel overhead not worthwhile)"
            )
            return None

        # CRITICAL: Pre-load root .gitignore before spawning workers
        # Workers need parent patterns to correctly apply gitignore inheritance
        parent_gitignores: dict[Path, list[str]] = {}
        parent_gitignores[directory] = load_gitignore_patterns(directory, directory)

        # Pre-compute repo roots once in the main process to avoid re-scanning
        # the entire tree in each worker when building repo-aware engines.
        precomputed_roots = []
        try:
            from chunkhound.utils.ignore_engine import detect_repo_roots  # type: ignore
            precomputed_roots = detect_repo_roots(directory, effective_excludes)
        except Exception:
            precomputed_roots = []

        # Determine number of workers for directory discovery
        # Scale based on number of subtrees and available cores
        # Use config value if available, otherwise use default of 16
        max_workers = self.config.indexing.max_discovery_workers if self.config else 16
        num_workers = min(
            os.cpu_count() or DEFAULT_CPU_COUNT, len(top_level_items), max_workers
        )

        logger.info(
            f"Using parallel discovery: {len(top_level_items)} top-level dirs, "
            f"{num_workers} workers (max: {max_workers})"
        )

        # Process subtrees in parallel
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for subtree in top_level_items:
                # Filter precomputed repo roots to those under this subtree
                roots_for_subtree = []
                try:
                    sres = subtree.resolve()
                    for rr in precomputed_roots:
                        try:
                            if rr.resolve().is_relative_to(sres):
                                roots_for_subtree.append(rr)
                        except Exception:
                            # Fallback for Python versions without is_relative_to or resolution issues
                            try:
                                rr.resolve().relative_to(sres)
                                roots_for_subtree.append(rr)
                            except Exception:
                                pass
                except Exception:
                    roots_for_subtree = precomputed_roots

                fut = loop.run_in_executor(
                    executor,
                    walk_subtree_worker,
                    subtree,
                    directory,
                    patterns,
                    exclude_patterns,
                    parent_gitignores,
                    use_inode_ordering,
                    (
                        {
                            "mode": "repo_aware",
                            "root": directory,
                            "sources": (self.config.indexing.resolve_ignore_sources() if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ["config"]),
                            "chf": (getattr(self.config.indexing, "chignore_file", ".chignore") if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ".chignore"),  # deprecated; ignored
                            "cfg": list(effective_excludes),
                            "roots": roots_for_subtree,
                        }
                    ),
                )
                futures.append(fut)

            # Wait for all subtrees to complete
            subtree_results = await asyncio.gather(*futures)

        # Aggregate and log worker errors
        all_errors = []
        for files, errors in subtree_results:
            all_errors.extend(errors)
        if all_errors:
            logger.error(
                f"Parallel discovery encountered {len(all_errors)} worker errors:"
            )
            for error in all_errors[:5]:  # Log first 5 errors
                logger.error(f"  - {error}")
            if len(all_errors) > 5:
                logger.error(f"  ... and {len(all_errors) - 5} more errors")

        # Scan files in the root directory itself (not in subdirs) using helper
        root_gitignore_patterns = parent_gitignores.get(directory, [])
        # Build or reuse local repo-aware engine for the root directory scan
        local_engine = self._get_or_build_ignore_engine(
            root=directory,
            sources=(self.config.indexing.resolve_ignore_sources() if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ["config"]),
            chf=(getattr(self.config.indexing, "chignore_file", ".chignore") if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ".chignore"),  # deprecated; ignored
            cfg=list(effective_excludes),
            backend=(getattr(self.config.indexing, "gitignore_backend", "python") if getattr(self, "config", None) and getattr(self.config, "indexing", None) else "python"),
        )

        root_files = scan_directory_files(
            directory,
            patterns,
            exclude_patterns,
            None,
            local_engine,
        )

        # Merge sorted worker results efficiently using heap-based merge
        # Workers already sort their results, so we merge k sorted lists
        import heapq

        # Collect sorted file lists from workers
        sorted_worker_results = []
        for files, errors in subtree_results:
            if files:  # Only include non-empty results
                sorted_worker_results.append(sorted(files))

        # Add root files as a sorted list
        if root_files:
            sorted_worker_results.append(sorted(root_files))

        # Merge sorted results: O(n log k) where k is number of workers
        # More efficient than concatenating and sorting: O(n log n)
        if sorted_worker_results:
            all_files = list(heapq.merge(*sorted_worker_results, key=str))
        else:
            all_files = []

        logger.info(f"Parallel discovery complete: found {len(all_files)} files")
        return all_files

    async def _discover_files(
        self,
        directory: Path,
        patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        parallel_discovery: bool | None = None,
        use_inode_ordering: bool = False,
    ) -> list[Path]:
        """Discover files in directory matching patterns with efficient exclude filtering.

        PERFORMANCE: Automatically selects parallel vs sequential discovery based on:
        - Config setting (parallel_discovery)
        - Directory structure size (min_dirs_for_parallel threshold)
        - Falls back to sequential on any parallel errors

        Args:
            directory: Directory to search
            patterns: File patterns to include (REQUIRED - must be provided by configuration layer)
            exclude_patterns: File patterns to exclude (optional - will load from config if None)
            parallel_discovery: Enable parallel directory traversal (default: from config)
                - Activates when >= min_dirs_for_parallel top-level directories exist
                - Scales workers based on number of subdirectories (max: max_discovery_workers)
                - Falls back to sequential for small directory structures
            use_inode_ordering: Sort directories by inode for improved disk locality (default: False)
                - Beneficial on rotational drives (HDDs) to reduce seek time
                - Minimal benefit on SSDs
                - Slight overhead from stat() calls per directory

        Raises:
            ValueError: If patterns is None/empty (configuration layer error)
        """

        # Validate inputs - fail fast on configuration errors
        if not patterns:
            raise ValueError(
                "patterns parameter is required for directory discovery. "
                "Configuration layer must provide file patterns."
            )

        # Default exclude patterns if not provided
        if not exclude_patterns:
            exclude_patterns = []

        # Prepare IgnoreEngine parameters (defer heavy engine build unless sequential path is taken)
        engine_args = None
        ignore_engine_obj = None
        if getattr(self, "config", None) is not None and getattr(self.config, "indexing", None) is not None:
            # Resolve ignore sources/config with backward-compatible fallbacks
            _idx = getattr(self, "config", None)
            _idx = getattr(_idx, "indexing", None)
            if _idx is not None and callable(getattr(_idx, "resolve_ignore_sources", None)):
                sources = _idx.resolve_ignore_sources()
            else:
                # Default to gitignore-only semantics when unspecified
                sources = ["gitignore"]
            chf = getattr(_idx, "chignore_file", ".chignore") if _idx is not None else ".chignore"
            if _idx is not None and callable(getattr(_idx, "get_effective_config_excludes", None)):
                cfg_excludes = _idx.get_effective_config_excludes()
            else:
                from chunkhound.core.config.indexing_config import IndexingConfig as _Idx
                cfg_excludes = _Idx._default_excludes()
            # Dynamically exclude the database path when it lives under the target directory
            try:
                dbp = getattr(self._db, "db_path", None)
                if dbp:
                    dbp_res = Path(dbp).resolve()
                    dir_res = directory.resolve()
                    try:
                        rel = dbp_res.relative_to(dir_res)
                        # If DB path is a directory, exclude the whole subtree; if file, exclude the file
                        if dbp_res.is_dir():
                            rp = rel.as_posix()
                            cfg_excludes.extend([rp, f"{rp}/**"])  # idempotent later by validator
                            exclude_patterns.extend([rp, f"{rp}/**"])  # local excludes too
                        else:
                            cfg_excludes.append(rel.as_posix())
                            exclude_patterns.append(rel.as_posix())
                    except Exception:
                        pass
            except Exception:
                pass
            backend = getattr(_idx, "gitignore_backend", "python") if _idx is not None else "python"
            engine_args = {
                "mode": "repo_aware",
                "root": directory.resolve(),
                "sources": sources,
                "chf": chf,
                "cfg": list(cfg_excludes),
                "backend": backend,
                "workspace_nonrepo_overlay": bool(getattr(_idx, "workspace_gitignore_nonrepo", False)) if _idx is not None else False,
            }
            # Provide precomputed repo roots to parallel workers so they can
            # avoid re-detecting per process
            try:
                roots = self._get_or_detect_repo_roots(directory.resolve(), list(cfg_excludes))
                if roots:
                    engine_args["roots"] = roots
            except Exception:
                pass

        # Use default (enabled) if not explicitly specified
        if parallel_discovery is None:
            parallel_discovery = True  # Default: parallel discovery enabled

        # Resolve directory once for consistent path handling
        directory = directory.resolve()

        # Normalize include patterns for consistent matching
        try:
            from chunkhound.utils.file_patterns import normalize_include_patterns as _norm
            patterns = _norm(list(patterns)) if patterns else patterns
        except Exception:
            pass

        # If configured, try Git-backed discovery first (fast path)
        try:
            _disc_backend = (
                getattr(self.config.indexing, "discovery_backend", "auto")
                if getattr(self, "config", None) and getattr(self.config, "indexing", None)
                else "auto"
            )
        except Exception:
            _disc_backend = "auto"

        # Resolve 'auto' to a concrete backend using a fast heuristic
        def _decide_backend() -> tuple[str, list[str]]:
            reasons: list[str] = []
            try:
                eff = (self.config.indexing.get_effective_config_excludes() if getattr(self, "config", None) and getattr(self.config, "indexing", None) else [])
                repo_roots = self._get_or_detect_repo_roots(directory, eff)
            except Exception:
                repo_roots = []
            if not repo_roots:
                reasons.append("no_repos")
                return "python", reasons
            # Count non-repo top-level entries (dirs/files) that are not excluded
            non_repo_items = 0
            try:
                for item in directory.iterdir():
                    try:
                        if any((item.resolve().is_relative_to(rr.resolve()) for rr in repo_roots)):
                            continue
                    except Exception:
                        try:
                            _ = [item.resolve().relative_to(rr.resolve()) for rr in repo_roots]
                            # if any succeeded, it's in a repo
                            inside = False
                            for rr in repo_roots:
                                try:
                                    item.resolve().relative_to(rr.resolve())
                                    inside = True
                                    break
                                except Exception:
                                    pass
                            if inside:
                                continue
                        except Exception:
                            pass
                    non_repo_items += 1
                    if non_repo_items > 1:
                        break
            except Exception:
                pass
            if non_repo_items == 0:
                reasons.append("all_repos")
                return "git_only", reasons
            reasons.append("mixed")
            return "git", reasons

        if _disc_backend == "auto":
            _resolved, _reasons = _decide_backend()
        else:
            _resolved, _reasons = _disc_backend, ["explicit"]

        # Expose resolved backend for diagnostics/profiling
        try:
            setattr(self, "_resolved_discovery_backend", _resolved)
            setattr(self, "_resolved_discovery_reasons", _reasons)
        except Exception:
            pass

        use_git_backend = _resolved in ("git", "git_only")
        git_only_mode = _resolved == "git_only"

        if use_git_backend:
            files_git = self._discover_files_via_git(
                directory, patterns, exclude_patterns, fallback_to_python=(not git_only_mode)
            )
            # If Git enumeration succeeded and produced files, return them.
            # If it returned an empty list in git_only mode (e.g., fake repos with only
            # a .git directory but no initialized repo), fall back to Python traversal
            # to honor repo-boundary ignore semantics used in tests, but only when repos
            # were actually detected. When truly no repos exist, git_only should yield
            # an empty result by design.
            if files_git is not None:
                repo_detected = bool(getattr(self, "_git_repo_roots_detected", False))
                if files_git or (not git_only_mode) or (git_only_mode and not repo_detected):
                    try:
                        setattr(self, "_profile_parallel_used", False)
                    except Exception:
                        pass
                    return sorted(files_git)

        # Try parallel discovery if enabled
        if parallel_discovery:
            try:
                discovered_files = await self._discover_files_parallel(
                    directory, patterns, exclude_patterns, use_inode_ordering
                )
                # Check if parallel succeeded (returns files) or signaled fallback (returns None)
                if discovered_files is not None:
                    # Parallel discovery returns pre-sorted results (via heapq.merge)
                    try:
                        # mark for startup profile that parallel was used
                        setattr(self, "_profile_parallel_used", True)
                    except Exception:
                        pass
                    return discovered_files
                # Otherwise fall through to sequential (None signal)
            except Exception as e:
                # Preserve full error context for debugging large repo issues
                import traceback

                error_traceback = traceback.format_exc()
                logger.warning(
                    f"Parallel discovery failed for {directory}, falling back to sequential:\n"
                    f"  Error: {type(e).__name__}: {e}\n"
                    f"  Traceback (last 3 frames):\n"
                    f"{''.join(traceback.format_tb(e.__traceback__)[-3:])}"
                )
                logger.debug(f"Full traceback:\n{error_traceback}")
                # Fall through to sequential

        # Sequential discovery (fallback or explicitly requested)
        # Build the ignore engine only now (avoid heavy pre-build when parallel succeeds)
        if engine_args is not None and ignore_engine_obj is None:
            try:
                ignore_engine_obj = self._get_or_build_ignore_engine(
                    root=engine_args["root"],
                    sources=engine_args["sources"],
                    chf=engine_args["chf"],
                    cfg=engine_args["cfg"],
                    backend=engine_args.get("backend", "python"),
                    overlay=engine_args.get("workspace_nonrepo_overlay", False),
                )
            except Exception:
                ignore_engine_obj = None

        discovered_files = self._walk_directory_with_excludes(
            directory, patterns, exclude_patterns, use_inode_ordering, ignore_engine_obj, engine_args
        )
        try:
            setattr(self, "_profile_parallel_used", False)
        except Exception:
            pass
        return sorted(discovered_files)

    def _discover_files_via_git(
        self,
        directory: Path,
        patterns: list[str] | None,
        exclude_patterns: list[str] | None,
        fallback_to_python: bool = True,
    ) -> list[Path] | None:
        """Enumerate files using `git ls-files` per repo when available.

        Falls back (returns None) when Git is missing, errors occur, or no repos are found.
        Always applies ChunkHound include patterns and config/default excludes on top
        of Git results. Non-repo portions of the directory are scanned using the
        Python walker while pruning repo subtrees.
        """
        from fnmatch import fnmatch as _fnmatch
        try:
            # Quick probe: ensure git exists
            import shutil as _sh

            if _sh.which("git") is None:
                return None
        except Exception:
            return None

        try:
            from chunkhound.utils.git_discovery import list_repo_files_via_git as _git_list
            from chunkhound.utils.file_patterns import (
                walk_directory_tree as _walk,
                load_gitignore_patterns as _load_gi,
            )
        except Exception:
            return None

        patterns = list(patterns or [])
        exclude_patterns_local = list(exclude_patterns or [])

        # Effective config excludes (includes defaults even with sentinel)
        try:
            _idx = getattr(self, "config", None)
            _idx = getattr(_idx, "indexing", None)
            if _idx is not None and callable(getattr(_idx, "get_effective_config_excludes", None)):
                effective_excludes = list(_idx.get_effective_config_excludes())
            else:
                from chunkhound.core.config.indexing_config import IndexingConfig as _Idx
                effective_excludes = _Idx._default_excludes()
        except Exception:
            from chunkhound.core.config.indexing_config import IndexingConfig as _Idx
            effective_excludes = _Idx._default_excludes()
        # Also exclude dynamic DB path when it lives under directory
        try:
            dbp = getattr(self._db, "db_path", None)
            if dbp:
                dbp_res = Path(dbp).resolve()
                dir_res = directory.resolve()
                try:
                    rel = dbp_res.relative_to(dir_res)
                    if dbp_res.is_dir():
                        rp = rel.as_posix()
                        effective_excludes.extend([rp, f"{rp}/**"])  # safe duplicates
                        exclude_patterns_local.extend([rp, f"{rp}/**"])  # local excludes too
                    else:
                        effective_excludes.append(rel.as_posix())
                        exclude_patterns_local.append(rel.as_posix())
                except Exception:
                    pass
        except Exception:
            pass

        # Detect repo roots under directory (pruned by effective_excludes) with cache reuse
        try:
            repo_roots = self._get_or_detect_repo_roots(directory, effective_excludes)
        except Exception:
            repo_roots = []
        # Expose whether any repos were detected for caller decisions
        try:
            setattr(self, "_git_repo_roots_detected", bool(repo_roots))
        except Exception:
            pass

        # If no repos detected
        if not repo_roots:
            return None if fallback_to_python else []

        results: list[Path] = []
        # Instrumentation totals across repos
        tot_rows_tracked = 0
        tot_rows_others = 0
        tot_pathspecs = 0
        tot_capped = False

        # For each repo, list files via Git, restricted to the subtree being indexed
        for rr in repo_roots:
            try:
                # Determine the subtree to list for this repo root
                # If the requested directory is an ancestor of the repo root, list the whole repo (start at rr)
                # If the repo root is an ancestor of the requested directory, limit to that subdir
                # Otherwise (disjoint), skip (shouldn't occur with our detection)
                start_for_repo = rr
                try:
                    directory.resolve().relative_to(rr.resolve())
                    # directory is inside rr
                    start_for_repo = directory
                except Exception:
                    try:
                        rr.resolve().relative_to(directory.resolve())
                        # rr is inside directory → start at rr
                        start_for_repo = rr
                    except Exception:
                        # Disjoint; skip
                        continue

                repo_files, stats = _git_list(
                    rr,
                    start_dir=start_for_repo,
                    include_patterns=patterns,
                    config_excludes=effective_excludes,
                    filter_root=directory,
                )
                tot_rows_tracked += int(stats.get("git_rows_tracked", 0))
                tot_rows_others += int(stats.get("git_rows_others", 0))
                tot_pathspecs += int(stats.get("git_pathspecs", 0))
                if bool(stats.get("git_pathspecs_capped")):
                    tot_capped = True
                results.extend(repo_files)
            except Exception:
                # If any repo fails, skip it (we'll still scan non-repo areas)
                continue

        # Scan non-repo areas by pruning repo subtrees during walk
        if fallback_to_python:
            try:
                # Build a fast set of immediate children to prune, but do a general prune inside walker too
                parent_gitignores: dict[Path, list[str]] = {directory: _load_gi(directory, directory)}
                # Build a repo-aware engine so we can control whether the workspace (non-repo)
                # side honors the CH root .gitignore (root-only) or ignores it entirely.
                try:
                    wr_only = bool(getattr(self.config.indexing, "workspace_gitignore_nonrepo", False)) if getattr(self, "config", None) and getattr(self.config, "indexing", None) else False
                except Exception:
                    wr_only = False
                # Simple overlay prefixes (directory-only) parsed from root .gitignore (best-effort)
                overlay_prefixes: list[str] = []
                if wr_only:
                    try:
                        gi = directory / ".gitignore"
                        if gi.exists():
                            for raw in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
                                if not raw or raw.lstrip().startswith("#"):
                                    continue
                                ln = raw.strip()
                                if ln.endswith("/"):
                                    ln = ln[:-1]
                                    if ln.startswith("/"):
                                        ln = ln[1:]
                                    overlay_prefixes.append(ln.strip("/"))
                    except Exception:
                        overlay_prefixes = []
                # Reuse same sources selection; engine respects config overlay flag
                local_engine = self._get_or_build_ignore_engine(
                    root=directory,
                    sources=(self.config.indexing.resolve_ignore_sources() if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ["config"]),
                    chf=(getattr(self.config.indexing, "chignore_file", ".chignore") if getattr(self, "config", None) and getattr(self.config, "indexing", None) else ".chignore"),
                    cfg=list(effective_excludes),
                    backend=(getattr(self.config.indexing, "gitignore_backend", "python") if getattr(self, "config", None) and getattr(self.config, "indexing", None) else "python"),
                    overlay=wr_only,
                )
                non_repo_files, _ = _walk(
                    directory,
                    directory,
                    patterns,
                    exclude_patterns_local,
                    parent_gitignores,
                    False,
                    ignore_engine=local_engine,
                )

                # Remove any files that are inside detected repo roots to avoid duplicates
                pruned_non_repo: list[Path] = []
                for fp in non_repo_files:
                    try:
                        inside_repo = False
                        for rr in repo_roots:
                            try:
                                fp.resolve().relative_to(rr.resolve())
                                inside_repo = True
                                break
                            except Exception:
                                continue
                        if not inside_repo:
                            # Overlay prefix shortcut (best-effort) for non-repo files
                            try:
                                rel = fp.resolve().relative_to(directory.resolve()).as_posix()
                            except Exception:
                                rel = fp.name
                            if overlay_prefixes:
                                skip = False
                                for pref in overlay_prefixes:
                                    if rel == pref or rel.startswith(pref + "/"):
                                        skip = True
                                        break
                                if skip:
                                    continue
                            # Apply workspace overlay engine to non-repo files
                            try:
                                if local_engine and getattr(local_engine, "matches", None) and local_engine.matches(fp, is_dir=False):  # type: ignore[attr-defined]
                                    continue
                            except Exception:
                                pass
                            pruned_non_repo.append(fp)
                    except Exception:
                        pruned_non_repo.append(fp)
                results.extend(pruned_non_repo)
            except Exception:
                # Ignore non-repo scan errors; we still return repo files
                pass

        # Dedup and return
        seen = set()
        uniq: list[Path] = []
        for p in results:
            s = str(p)
            if s not in seen:
                seen.add(s)
                uniq.append(p)
        # Expose instrumentation for simulate/run
        try:
            setattr(self, "_git_rows_tracked", int(tot_rows_tracked))
            setattr(self, "_git_rows_others", int(tot_rows_others))
            setattr(self, "_git_rows_total", int(tot_rows_tracked + tot_rows_others))
            setattr(self, "_git_pathspecs", int(tot_pathspecs))
            if tot_capped:
                setattr(self, "_git_pathspecs_capped", True)
        except Exception:
            pass
        return uniq

    # --------------------------- Repo-roots caching ---------------------------
    def _repo_roots_cache_key(self, root: Path, cfg_excludes: list[str] | tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
        try:
            base = str(root.resolve())
        except Exception:
            base = str(root)
        try:
            items = tuple(sorted([str(x) for x in (list(cfg_excludes) if not isinstance(cfg_excludes, tuple) else list(cfg_excludes))]))
        except Exception:
            items = tuple()
        return (base, items)

    def _get_or_detect_repo_roots(self, root: Path, cfg_excludes: list[str] | tuple[str, ...]) -> list[Path]:
        key = self._repo_roots_cache_key(root, cfg_excludes)
        cached = self._repo_roots_cache.get(key)
        if cached is not None:
            return cached
        try:
            from chunkhound.utils.ignore_engine import detect_repo_roots as _detect
            roots = _detect(root, cfg_excludes)  # type: ignore[arg-type]
        except Exception:
            roots = []
        self._repo_roots_cache[key] = roots
        return roots

    def _walk_directory_with_excludes(
        self,
        directory: Path,
        patterns: list[str],
        exclude_patterns: list[str],
        use_inode_ordering: bool = False,
        ignore_engine_obj: object | None = None,
        ignore_engine_args: tuple | None = None,
    ) -> list[Path]:
        """Optimized directory walker using os.walk() with optional inode ordering.

        PERFORMANCE OPTIMIZATIONS:
        - Uses os.walk() with scandir (3-50x faster than manual recursion)
        - Compiled regex patterns (cached, 2-3x faster than fnmatch)
        - Optional inode ordering (reduces disk seeks on large filesystems)
        - Early directory pruning (skips excluded subtrees entirely)

        Args:
            directory: Root directory to walk
            patterns: File patterns to include
            exclude_patterns: Patterns to exclude (applied to both files and directories)
            use_inode_ordering: Sort directories by inode to reduce disk seeks (default: False)

        Returns:
            List of file paths that match include patterns and don't match exclude patterns
        """
        # Resolve directory path once at the beginning for consistent comparison
        directory = directory.resolve()

        # Pre-load root gitignore (consistent with parallel mode)
        parent_gitignores: dict[Path, list[str]] = {}
        if ignore_engine_obj is None:
            parent_gitignores[directory] = load_gitignore_patterns(directory, directory)
        else:
            parent_gitignores[directory] = []

        # Use shared directory traversal logic
        files, _ = walk_directory_tree(
            directory,
            directory,
            patterns,
            exclude_patterns,
            parent_gitignores,
            use_inode_ordering,
            ignore_engine=ignore_engine_obj,
        )

        # Fallback: if repo-aware engine is present and no files were found, perform
        # a conservative scan that filters files by engine and include patterns.
        if not files and getattr(ignore_engine_obj, "matches", None):
            try:
                import os as _os
                from chunkhound.utils.file_patterns import should_include_file as _inc
                from chunkhound.utils.file_patterns import compile_pattern as _cp
                from fnmatch import translate as _translate
                from pathlib import Path as _Path

                # Pre-compile include patterns for a minimal filter
                pat_cache = {}
                _ = [_cp(p, pat_cache) for p in (patterns or [])]

                collected: list[Path] = []
                for dp, dn, fn in _os.walk(directory, topdown=True):
                    cur = _Path(dp)
                    # Prune dirs by engine
                    pruned = []
                    for d in list(dn):
                        if ignore_engine_obj.matches(cur / d, is_dir=True):  # type: ignore[attr-defined]
                            pruned.append(d)
                    for d in pruned:
                        dn.remove(d)
                    # Files
                    for name in fn:
                        fp = cur / name
                        if ignore_engine_obj.matches(fp, is_dir=False):  # type: ignore[attr-defined]
                            continue
                        # Include filter
                        if _inc(fp, directory, patterns or [], pat_cache):
                            collected.append(fp)
                files = collected
            except Exception:
                pass

        return files

    def _cleanup_orphaned_files(
        self,
        directory: Path,
        current_files: list[Path],
        exclude_patterns: list[str] | None = None,
    ) -> int:
        """Remove database entries for files that no longer exist in the directory.

        Args:
            directory: Directory being processed
            current_files: List of files currently in the directory
            exclude_patterns: Optional list of exclude patterns to check against

        Returns:
            Number of orphaned files cleaned up
        """
        try:
            # Create set of relative paths for fast lookup
            base_dir = self._base_directory
            current_file_paths = {
                file_path.relative_to(base_dir).as_posix()
                for file_path in current_files
            }

            # Get all files in database (stored as relative paths)
            query = """
                SELECT id, path
                FROM files
            """
            db_files = self._db.execute_query(query, [])

            # Find orphaned files (in DB but not on disk or excluded by patterns)
            orphaned_files = []
            if not exclude_patterns:
                # Prefer the coordinator's current config; fall back to defaults
                try:
                    cfg = self.config if getattr(self, "config", None) else None
                    if cfg is None:
                        from chunkhound.core.config.config import Config as _Cfg
                        cfg = _Cfg()
                    patterns_to_check = cfg.indexing.get_effective_config_excludes()
                except Exception:
                    # Final fallback to static defaults
                    from chunkhound.core.config.indexing_config import IndexingConfig as _Idx
                    patterns_to_check = _Idx._default_excludes()
            else:
                patterns_to_check = exclude_patterns

            for db_file in db_files:
                file_path = db_file["path"]

                # Check if file should be excluded based on current patterns
                should_exclude = False

                # File path is already relative (stored as relative with forward slashes)
                rel_path = Path(file_path)

                for exclude_pattern in patterns_to_check:
                    # Check relative path pattern
                    if fnmatch(str(rel_path), exclude_pattern):
                        should_exclude = True
                        break

                # Mark for removal if not in current files or should be excluded
                if file_path not in current_file_paths or should_exclude:
                    orphaned_files.append(file_path)

            # Remove orphaned files with progress tracking
            orphaned_count = 0
            if orphaned_files:
                cleanup_task: TaskID | None = None
                if self.progress:
                    cleanup_task = self.progress.add_task(
                        "  └─ Cleaning orphaned files",
                        total=len(orphaned_files),
                        speed="",
                        info="",
                    )

                for file_path in orphaned_files:
                    if self._db.delete_file_completely(file_path):
                        orphaned_count += 1
                        # Clean up the file lock for orphaned file
                        self._cleanup_file_lock(Path(file_path))

                    if cleanup_task is not None and self.progress:
                        self.progress.advance(cleanup_task, 1)

                # Complete the cleanup progress bar
                if cleanup_task is not None and self.progress:
                    task = self.progress.tasks[cleanup_task]
                    if task.total:
                        self.progress.update(cleanup_task, completed=task.total)

                logger.info(f"Cleaned up {orphaned_count} orphaned files from database")

            return orphaned_count

        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned files: {e}")
            return 0
