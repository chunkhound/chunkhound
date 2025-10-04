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
import os
from concurrent.futures import ProcessPoolExecutor
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from loguru import logger
from rich.progress import Progress, TaskID

from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import FilePath, Language
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.parsers.universal_parser import UniversalParser

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


# Performance tuning constants for parallel operations
# RATIONALE: Balance parallelism overhead vs throughput for different workload sizes

# File parsing batch sizes
SMALL_FILE_COUNT_THRESHOLD = 100  # Below this: use minimal workers (overhead not worth it)
MEDIUM_FILE_COUNT_THRESHOLD = 1000  # Above this: scale up for enterprise monorepos
MAX_WORKERS_SMALL_BATCH = 4  # Worker cap for small file batches
MAX_WORKERS_MEDIUM_BATCH = 8  # Worker cap for medium file batches (original behavior)
MAX_WORKERS_LARGE_BATCH = 16  # Worker cap for large file batches (prevents resource exhaustion)

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
        """Detect programming language from file extension.

        Args:
            file_path: Path to the file

        Returns:
            Language enum value or None if unsupported
        """
        language = Language.from_file_extension(file_path)
        return language if language != Language.UNKNOWN else None

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
            parsed_results = await self._process_files_in_batches([file_path])

            if not parsed_results:
                return {"status": "error", "chunks": 0, "error": "No results from batch processor"}

            result = parsed_results[0]

            if result.status == "error":
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
            if not skip_embeddings and self._embedding_provider:
                if stats["chunk_ids_needing_embeddings"]:
                    await self._generate_embeddings(
                        stats["chunk_ids_needing_embeddings"],
                        [chunk for r in parsed_results for chunk in r.chunks]
                    )

            return_dict = {
                "status": "success" if not stats["errors"] else "error",
                "chunks": stats["total_chunks"],
                "errors": stats["errors"],
                "embeddings_skipped": skip_embeddings,
            }

            # Include file_id for single-file operations
            if file_id is not None:
                return_dict["file_id"] = file_id

            return return_dict

    async def _process_files_in_batches(
        self, files: list[Path], config_file_size_threshold_kb: int = 20
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
        num_workers = _calculate_worker_count(file_count, cpu_count)

        logger.debug(f"Parsing {file_count} files with {num_workers} workers")

        # Split files into batches for parallel processing
        batch_size = math.ceil(len(files) / num_workers)
        file_batches = [
            files[i : i + batch_size] for i in range(0, len(files), batch_size)
        ]

        # Process batches in parallel using ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batches for concurrent processing
            # Pass config for structured file size filtering (JSON/YAML/TOML)
            config_dict = {"config_file_size_threshold_kb": config_file_size_threshold_kb}
            futures = [
                loop.run_in_executor(
                    executor, process_file_batch, batch, config_dict
                )
                for batch in file_batches
            ]

            # Wait for all batches to complete
            batch_results = await asyncio.gather(*futures)

        # Flatten results from all batches
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)

        return all_results

    async def _store_parsed_results(
        self, results: list[ParsedFileResult], file_task: TaskID | None = None
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

        for result in results:
            # Handle errors
            if result.status == "error":
                stats["errors"].append(
                    {"file": str(result.file_path), "error": result.error}
                )
                if file_task is not None and self.progress:
                    self.progress.advance(file_task, 1)
                continue

            # Handle skipped files
            if result.status == "skipped":
                # Track skip reason in stats for single-file case
                if "skip_reason" not in stats:
                    stats["skip_reason"] = result.error
                if file_task is not None and self.progress:
                    self.progress.advance(file_task, 1)
                continue

            # Detect language for storage
            language = result.language

            # Store file record with transaction
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
                file_id = self._store_file_record(result.file_path, file_stat, language)

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

                if existing_file:
                    # Get existing chunks for diffing
                    existing_chunks = self._db.get_chunks_by_file_id(file_id, as_model=True)

                    # Convert result chunks to Chunk models using from_dict()
                    new_chunk_models = [
                        Chunk.from_dict({**chunk_data, "file_id": file_id})
                        for chunk_data in result.chunks
                    ]

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
                            for chunk_id in chunk_ids_to_delete:
                                self._db.delete_chunk(chunk_id)

                        # Store new/modified chunks (pass models directly)
                        chunks_to_store = chunk_diff.added + chunk_diff.modified

                        if chunks_to_store:
                            chunk_ids_new = self._store_chunks(
                                file_id, chunks_to_store, language
                            )
                        else:
                            chunk_ids_new = []

                        # Track chunks needing embeddings (new + modified)
                        stats["chunk_ids_needing_embeddings"].extend(chunk_ids_new)

                        stats["total_chunks"] += len(result.chunks)
                    else:
                        # No existing chunks - store all as new (pass models directly)
                        chunk_ids = self._store_chunks(file_id, new_chunk_models, language)
                        stats["chunk_ids_needing_embeddings"].extend(chunk_ids)
                        stats["total_chunks"] += len(chunk_ids)
                else:
                    # New file - convert dicts to models, then store
                    chunk_models = [
                        Chunk.from_dict({**chunk_data, "file_id": file_id})
                        for chunk_data in result.chunks
                    ]
                    chunk_ids = self._store_chunks(file_id, chunk_models, language)
                    stats["chunk_ids_needing_embeddings"].extend(chunk_ids)
                    stats["total_chunks"] += len(chunk_ids)

                self._db.commit_transaction()
                stats["total_files"] += 1

                # Update progress
                if file_task is not None and self.progress:
                    self.progress.advance(file_task, 1)
                    self.progress.update(file_task, info=f"{stats['total_chunks']} chunks")

            except Exception as e:
                self._db.rollback_transaction()
                stats["errors"].append({"file": str(result.file_path), "error": str(e)})
                if file_task is not None and self.progress:
                    self.progress.advance(file_task, 1)

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
            # Phase 1: Discovery - Discover files in directory (now parallelized)
            files = await self._discover_files(directory, patterns, exclude_patterns)

            if not files:
                return {"status": "no_files", "files_processed": 0, "total_chunks": 0}

            # Phase 2: Reconciliation - Ensure database consistency by removing orphaned files
            cleaned_files = self._cleanup_orphaned_files(
                directory, files, exclude_patterns
            )

            logger.debug(
                f"Directory consistency: {len(files)} files discovered, {cleaned_files} orphaned files cleaned"
            )

            # Phase 3: Update - Process files in parallel batches
            # Create progress task for file processing
            file_task: TaskID | None = None
            if self.progress:
                file_task = self.progress.add_task(
                    "  └─ Processing files", total=len(files), speed="", info=""
                )

            # Parse files in parallel batches across CPU cores
            parsed_results = await self._process_files_in_batches(files, config_file_size_threshold_kb)

            # Store results in database (single-threaded for safety)
            stats = await self._store_parsed_results(parsed_results, file_task)

            total_files = stats["total_files"]
            total_chunks = stats["total_chunks"]

            # Log any errors
            for error in stats["errors"]:
                logger.warning(f"Failed to process {error['file']}: {error['error']}")

            # Complete the file processing progress bar
            if file_task is not None and self.progress:
                task = self.progress.tasks[file_task]
                if task.total:
                    self.progress.update(file_task, completed=task.total)

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

    def _store_file_record(
        self, file_path: Path, file_stat: Any, language: Language
    ) -> int:
        """Store or update file record in database."""
        # Check if file already exists
        # Use consistent symlink-safe path resolution
        relative_path = self._get_relative_path(file_path)
        existing_file = self._db.get_file_by_path(relative_path.as_posix())

        if existing_file:
            # Update existing file with new metadata
            if isinstance(existing_file, dict) and "id" in existing_file:
                file_id = existing_file["id"]
                self._db.update_file(
                    file_id, size_bytes=file_stat.st_size, mtime=file_stat.st_mtime
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
        self, exclude_patterns: list[str] | None = None
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

            # Database storage - use provided connection for transaction context
            result = self._db.insert_embeddings_batch(
                embeddings_data, connection=connection
            )

            return result

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
        try:
            for item in directory.iterdir():
                try:
                    # Check if item is still a directory (could change during iteration)
                    if not item.is_dir():
                        continue

                    # Check if this directory should be excluded
                    rel_path = item.relative_to(directory)
                    should_skip = False
                    for pattern in exclude_patterns:
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

        # Determine number of workers for directory discovery
        # Scale based on number of subtrees and available cores
        # Use config value if available, otherwise use default of 16
        max_workers = (
            self.config.indexing.max_discovery_workers if self.config else 16
        )
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
            futures = [
                loop.run_in_executor(
                    executor,
                    walk_subtree_worker,
                    subtree,
                    directory,
                    patterns,
                    exclude_patterns,
                    parent_gitignores,
                    use_inode_ordering,
                )
                for subtree in top_level_items
            ]

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
        root_files = scan_directory_files(
            directory, patterns, exclude_patterns, root_gitignore_patterns or None
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

        # Use default (enabled) if not explicitly specified
        if parallel_discovery is None:
            parallel_discovery = True  # Default: parallel discovery enabled

        # Resolve directory once for consistent path handling
        directory = directory.resolve()

        # Try parallel discovery if enabled
        if parallel_discovery:
            try:
                discovered_files = await self._discover_files_parallel(
                    directory, patterns, exclude_patterns, use_inode_ordering
                )
                # Check if parallel succeeded (returns files) or signaled fallback (returns None)
                if discovered_files is not None:
                    # Parallel discovery returns pre-sorted results (via heapq.merge)
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
        discovered_files = self._walk_directory_with_excludes(
            directory, patterns, exclude_patterns, use_inode_ordering
        )
        return sorted(discovered_files)

    def _walk_directory_with_excludes(
        self, directory: Path, patterns: list[str], exclude_patterns: list[str], use_inode_ordering: bool = False
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
        parent_gitignores[directory] = load_gitignore_patterns(directory, directory)

        # Use shared directory traversal logic
        files, _ = walk_directory_tree(
            directory,
            directory,
            patterns,
            exclude_patterns,
            parent_gitignores,
            use_inode_ordering,
        )

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
                from chunkhound.core.config.config import Config

                config = Config.from_environment()
                patterns_to_check = config.indexing.get_default_exclude_patterns()
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
