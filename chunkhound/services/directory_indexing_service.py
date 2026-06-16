"""Directory indexing service - extracted from CLI indexer for shared use."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.diagnostics.batch_metrics import BatchMetricsCollector
from chunkhound.services.indexing_coordinator import run_batch_compaction_boundary
from chunkhound.utils.file_patterns import normalize_include_patterns


@dataclass
class IndexingStats:
    """Statistics from directory processing."""

    files_processed: int = 0
    files_skipped: int = 0
    files_errors: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    processing_time: float = 0.0
    cleanup_deleted_files: int = 0
    cleanup_deleted_chunks: int = 0
    errors_encountered: list[str] = field(default_factory=list)
    skipped_due_to_timeout: list[str] = field(default_factory=list)
    skipped_unchanged: int = 0
    skipped_filtered: int = 0
    db_compactions: int = 0


class DirectoryIndexingService:
    """
    Complete directory indexing pipeline extracted from CLI indexer.
    Handles file discovery, processing, and embedding generation.
    """

    def __init__(
        self,
        indexing_coordinator: Any,
        config: Any,
        progress_callback: Callable[[str], None] | None = None,
        progress: Any = None,
        metrics_collector: BatchMetricsCollector | None = None,
    ):
        """Initialize directory indexing service.

        Args:
            indexing_coordinator: Indexing coordinator service
            config: Configuration object
            progress_callback: Optional callback for progress messages
            progress: Optional Rich Progress instance for hierarchical progress display
            metrics_collector: Optional metrics collector for batch diagnostics
        """
        self.indexing_coordinator = indexing_coordinator
        self.config = config
        self.progress_callback = progress_callback or (lambda msg: None)
        self.progress = progress
        self._metrics_collector = metrics_collector

        # Pass progress to coordinator if it supports it
        if hasattr(self.indexing_coordinator, "progress"):
            self.indexing_coordinator.progress = progress

    async def process_directory(
        self, target_path: Path, no_embeddings: bool = False
    ) -> IndexingStats:
        """
        Main processing pipeline - extracted from run.py.

        Args:
            target_path: Directory to process
            no_embeddings: Skip embedding generation

        Returns:
            IndexingStats with processing results
        """
        start_time = time.time()
        stats = IndexingStats()

        try:
            # === Drop HNSW indexes before bulk indexing ===
            # Eliminates per-batch index maintenance overhead during
            # large embedding inserts and compaction steps.
            await self._drop_hnsw_indexes()

            # File pattern resolution (extracted from run.py:61-65)
            include_patterns, exclude_patterns = self._resolve_file_patterns()

            # Directory processing (extracted from run.py:80-82, 253-284)
            self.progress_callback("Discovering files...")
            process_result = await self._process_directory_files(
                target_path, include_patterns, exclude_patterns
            )

            # Update stats from processing result
            self._update_stats_from_process_result(stats, process_result)

            # Only compact when actual work was done — avoids a costly
            # no-op compaction on the MCP initial scan path where all
            # files are already indexed and unchanged.
            if stats.files_processed > 0 or stats.chunks_created > 0:
                await self._run_batch_compaction(stats)

            # Embedding generation (extracted from run.py:85-88, 287-312)
            if not no_embeddings:
                self.progress_callback("Checking for missing embeddings...")
                embed_result = await self._generate_missing_embeddings(exclude_patterns)
                stats.embeddings_generated = embed_result.get("generated", 0)

            # Second compaction boundary: needed when embeddings were
            # generated, or when files were processed (but the first
            # compaction may have been skipped).
            if (
                stats.files_processed > 0
                or stats.chunks_created > 0
                or stats.embeddings_generated > 0
            ):
                await self._run_batch_compaction(stats)

            # === Rebuild HNSW indexes as final step ===
            # Must be last: compaction produces a clean DB file, and HNSW
            # is built once on the clean DB instead of being rewritten on
            # every checkpoint.
            await self._ensure_hnsw_indexes()

            stats.processing_time = time.time() - start_time

        except Exception as e:
            stats.errors_encountered.append(str(e))
            raise

        return stats

    async def _run_batch_compaction(self, stats: IndexingStats) -> None:
        """Run one mandatory batch-compaction boundary."""
        await run_batch_compaction_boundary(self.indexing_coordinator, stats)

    async def _drop_hnsw_indexes(self) -> None:
        """Drop HNSW indexes before bulk indexing."""
        db = getattr(self.indexing_coordinator, "_db", None)
        if db is not None and hasattr(db, "drop_all_hnsw_indexes"):
            db.drop_all_hnsw_indexes()

    async def _ensure_hnsw_indexes(self) -> None:
        """Rebuild HNSW indexes after bulk indexing completes."""
        db = getattr(self.indexing_coordinator, "_db", None)
        if db is not None and hasattr(db, "ensure_all_hnsw_indexes"):
            db.ensure_all_hnsw_indexes()

    def _resolve_file_patterns(self) -> tuple[list[str], list[str]]:
        """Extracted from run.py:152-175 - file pattern resolution logic."""
        # Use patterns from config. CLI overrides are already applied during
        # config creation.
        include_patterns = list(self.config.indexing.include)
        exclude_patterns = list(self.config.indexing.exclude)

        return include_patterns, exclude_patterns

    async def _process_directory_files(
        self,
        target_path: Path,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ) -> dict[str, Any]:
        """Extracted from run.py:237-284 - directory processing logic."""
        # Normalize patterns using shared utility (prevents double-prefixing)
        processed_patterns: list[str] = normalize_include_patterns(include_patterns)

        # Process directory using indexing coordinator with config threshold
        result = await self.indexing_coordinator.process_directory(
            target_path,
            patterns=processed_patterns,
            exclude_patterns=exclude_patterns,
            config_file_size_threshold_kb=self.config.indexing.config_file_size_threshold_kb,
        )

        if result["status"] not in ["complete", "success", "no_files"]:
            raise RuntimeError(f"Directory processing failed: {result}")

        return result

    async def _generate_missing_embeddings(
        self, exclude_patterns: list[str]
    ) -> dict[str, Any]:
        """Extracted from run.py:287-312 - embedding generation workflow."""
        embed_result = await self.indexing_coordinator.generate_missing_embeddings(
            exclude_patterns=exclude_patterns,
            metrics_collector=self._metrics_collector,
        )

        if embed_result["status"] not in ["success", "up_to_date", "complete"]:
            logger.warning(f"Embedding generation failed: {embed_result}")

        return embed_result

    def _update_stats_from_process_result(
        self, stats: IndexingStats, result: dict[str, Any]
    ) -> None:
        """Update stats from processing result."""
        stats.files_processed = result.get(
            "files_processed", result.get("processed", 0)
        )
        stats.files_skipped = result.get("skipped", 0)
        stats.files_errors = result.get("errors", 0)
        stats.chunks_created = result.get("total_chunks", 0)
        stats.skipped_due_to_timeout = result.get("skipped_due_to_timeout", [])
        stats.skipped_unchanged = result.get("skipped_unchanged", 0)
        stats.skipped_filtered = result.get("skipped_filtered", 0)

        # Cleanup statistics
        cleanup = result.get("cleanup", {})
        stats.cleanup_deleted_files = cleanup.get("deleted_files", 0)
        stats.cleanup_deleted_chunks = cleanup.get("deleted_chunks", 0)
