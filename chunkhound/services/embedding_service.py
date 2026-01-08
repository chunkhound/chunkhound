"""Embedding service for ChunkHound - manages embedding generation and caching."""

import asyncio
import time
from typing import Any

from loguru import logger
from rich.progress import Progress, TaskID

from chunkhound.core.exceptions import (
    EmbeddingErrorClassification,
    EmbeddingErrorClassifier,
)
from chunkhound.core.types.common import ChunkId
from chunkhound.core.utils import estimate_tokens
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider

from .base_service import BaseService


class EmbeddingBatchProcessor:
    """Handles the complex batch processing loop for generating missing embeddings."""

    def __init__(
        self,
        service: "EmbeddingService",
        error_handler: "EmbeddingErrorHandler",
        progress_manager: "EmbeddingProgressManager",
    ):
        """Initialize the batch processor.

        Args:
            service: The embedding service instance
            error_handler: Error handler for tracking failures
            progress_manager: Progress manager for UI updates
        """
        self.service = service
        self.error_handler = error_handler
        self.progress_manager = progress_manager

    async def process_all_batches(
        self,
        target_provider: str,
        target_model: str,
        initial_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        target_batch_time: float,
        slow_threshold: float,
        fast_threshold: float,
        embed_task: TaskID | None,
    ) -> dict[str, Any]:
        """Process all batches of chunks without embeddings.

        Args:
            target_provider: Provider name for embeddings
            target_model: Model name for embeddings
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_batch_time: Target time per batch
            slow_threshold: Threshold for slow batches
            fast_threshold: Threshold for fast batches
            embed_task: Progress task ID

        Returns:
            Dictionary with aggregated batch processing results
        """
        import time as time_module

        current_batch_size = initial_batch_size
        total_generated = 0
        total_attempted = 0
        total_failed = 0
        total_permanent_failures = 0
        batch_count = 0

        while True:
            batch_count += 1

            # Measure time for chunk ID retrieval
            start_time = time_module.time()

            try:
                logger.debug(f"Retrieving chunks batch: provider={target_provider}, model={target_model}, limit={current_batch_size}")
                # Get next batch of chunks without embeddings
                chunks_data = self.service._db.get_chunks_without_embeddings_paginated(
                    target_provider, target_model, limit=current_batch_size
                )
                logger.debug(f"Retrieved {len(chunks_data) if chunks_data else 0} chunks")
            except Exception as e:
                # If batch fails, reduce batch size and retry once
                if current_batch_size > min_batch_size:
                    old_batch_size = current_batch_size
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    logger.warning(f"Batch retrieval failed (size={old_batch_size}), reducing to {current_batch_size}: {e}")
                    continue
                else:
                    logger.error(f"Batch retrieval failed permanently after reducing batch size: {e}")
                    raise  # Re-raise if already at minimum

            retrieval_time = time_module.time() - start_time

            if not chunks_data:  # temp remove for testing: or len(chunks_data) < current_batch_size:
                # No more chunks to process
                logger.info("No more chunks to process, exiting batch loop")
                break

            logger.debug(f"Batch {batch_count}: size={len(chunks_data)}, retrieval_time={retrieval_time:.2f}s")

            # Adjust batch size based on retrieval time (only if dynamic sizing enabled)
            if target_batch_time != float('inf'):
                if retrieval_time > slow_threshold and current_batch_size > min_batch_size:
                    # Too slow, reduce batch size
                    old_batch_size = current_batch_size
                    current_batch_size = max(min_batch_size, current_batch_size // 2)
                    logger.info(f"Batch too slow ({retrieval_time:.2f}s > {slow_threshold}s), reducing batch size: {old_batch_size} -> {current_batch_size}")
                elif retrieval_time < fast_threshold and current_batch_size < max_batch_size:
                    # Very fast, can increase batch size more aggressively
                    old_batch_size = current_batch_size
                    current_batch_size = min(max_batch_size, int(current_batch_size * 2.0))
                    logger.debug(f"Batch very fast ({retrieval_time:.2f}s < {fast_threshold}s), increasing batch size: {old_batch_size} -> {current_batch_size}")
                elif retrieval_time < target_batch_time and current_batch_size < max_batch_size:
                    # Reasonably fast, small increase
                    old_batch_size = current_batch_size
                    current_batch_size = min(max_batch_size, int(current_batch_size * 1.5))
                    logger.debug(f"Batch reasonably fast ({retrieval_time:.2f}s), small increase: {old_batch_size} -> {current_batch_size}")

            # Extract chunk IDs and texts from retrieved chunks
            chunk_id_list = [chunk["id"] for chunk in chunks_data]
            chunk_texts = [chunk["code"] for chunk in chunks_data]
            logger.debug(f"Extracted content for {len(chunks_data)} chunks")

            total_attempted += len(chunks_data)

            # Update progress total with attempted chunks
            self.progress_manager.increment_total(embed_task, len(chunks_data))

            # Generate embeddings for this batch
            logger.debug(f"Calling generate_embeddings_for_chunks with {len(chunk_id_list)} chunks, provider={self.service._embedding_provider}")
            batch_result = await self.service.generate_embeddings_for_chunks(
                chunk_id_list, chunk_texts, show_progress=True, embed_task=embed_task, chunks_data=chunks_data
            )
            logger.debug(f"generate_embeddings_for_chunks returned {batch_result}")

            # Track transient failures for flow control
            self.error_handler.track_batch_failures(batch_result)
            if self.error_handler.should_abort_generation():
                break

            total_generated += batch_result["total_generated"]
            total_failed += batch_result["failed_chunks"]
            total_permanent_failures += batch_result["permanent_failures"]

            # Update progress info with current statistics
            info = f"success: {total_generated}, failed: {total_failed}, permanent: {total_permanent_failures}"
            self.progress_manager.update_info(embed_task, info)

            # Enhanced progress logging with error statistics
            error_breakdown = []
            for error_type, count in batch_result.get("error_stats", {}).items():
                error_breakdown.append(f"{error_type}: {count}")
            error_breakdown_str = "; ".join(error_breakdown) if error_breakdown else "none"

            # Batch-level error reporting with specific format
            total_chunks_in_batch = len(chunks_data)
            successful_in_batch = batch_result['total_generated']
            failed_in_batch = batch_result['failed_chunks'] + batch_result['permanent_failures']

            error_breakdown_formatted = []
            for i, (error_type, count) in enumerate(batch_result.get("error_stats", {}).items(), 1):
                error_breakdown_formatted.append(f"{i}. {error_type}: {count}")
            error_breakdown_formatted_str = ". ".join(error_breakdown_formatted) if error_breakdown_formatted else "none"

            logger.info(f"embedding batch processed {total_chunks_in_batch}/{total_chunks_in_batch} chunks. success: {successful_in_batch} failed {failed_in_batch} failure reason breakdown: {error_breakdown_formatted_str}")

            logger.info(f"Embedding batch {batch_count} processed {len(chunks_data)} chunks: success: {batch_result['total_generated']}, failed: {batch_result['failed_chunks']}, permanent: {batch_result['permanent_failures']}, error breakdown: {error_breakdown_str}")

            logger.info(f"Total progress: attempted={total_attempted}, generated={total_generated}, failed={total_failed}, permanent={total_permanent_failures}, current_batch_size={current_batch_size}")

            # Check for fragmentation optimization after each page
            if self.service._db.should_optimize_fragments(operation="during-embedding-page"):
                logger.info("Running fragmentation optimization during embedding generation...")
                try:
                    self.service._db.optimize_tables()
                    logger.info("Fragmentation optimization completed")
                except Exception as opt_error:
                    logger.warning(f"Fragmentation optimization failed: {opt_error}")

        return {
            "total_generated": total_generated,
            "total_attempted": total_attempted,
            "total_failed": total_failed,
            "total_permanent_failures": total_permanent_failures,
        }


class EmbeddingErrorHandler:
    """Handles transient failure tracking for embedding generation."""

    def __init__(
        self,
        max_consecutive_failures: int = 5,
        transient_error_window_seconds: int = 300,
    ):
        """Initialize the error handler.

        Args:
            max_consecutive_failures: Maximum consecutive transient failures before aborting
            transient_error_window_seconds: Time window in seconds for tracking consecutive transient failures
        """
        self._max_consecutive_failures = max_consecutive_failures
        self._transient_error_window_seconds = transient_error_window_seconds
        self._transient_failure_timestamps: list[float] = []

    def track_batch_failures(self, batch_result: dict[str, Any]) -> None:
        """Track transient failures from a batch result.

        Args:
            batch_result: Result dictionary from embedding generation batch
        """
        transient_failures_in_batch = batch_result.get("failed_chunks", 0) - batch_result.get("permanent_failures", 0)
        if transient_failures_in_batch > 0:
            current_time = time.time()
            self._transient_failure_timestamps.append(current_time)
            # Remove timestamps outside the window
            self._transient_failure_timestamps = [
                t for t in self._transient_failure_timestamps
                if current_time - t <= self._transient_error_window_seconds
            ]
        else:
            # Successful batch, reset failure tracking
            self._transient_failure_timestamps.clear()

    def should_abort_generation(self) -> bool:
        """Check if generation should abort due to excessive transient failures.

        Returns:
            True if should abort, False otherwise
        """
        consecutive = len(self._transient_failure_timestamps)
        if consecutive >= self._max_consecutive_failures:
            logger.error(
                f"Aborting embedding generation due to {consecutive} consecutive transient failures "
                f"within {self._transient_error_window_seconds}s window"
            )
            return True
        elif consecutive >= self._max_consecutive_failures * 0.8:
            logger.warning(
                f"Approaching transient failure threshold: {consecutive}/{self._max_consecutive_failures} "
                f"consecutive failures within {self._transient_error_window_seconds}s"
            )
        return False


class EmbeddingProgressManager:
    """Manages progress tracking for embedding generation operations."""

    def __init__(self, progress: Progress | None):
        """Initialize the progress manager.

        Args:
            progress: Optional Rich Progress instance for hierarchical progress display
        """
        self.progress = progress

    def create_task(self, description: str = "Generating embeddings") -> TaskID | None:
        """Create a new progress task.

        Args:
            description: Description for the progress task

        Returns:
            TaskID if progress is enabled, None otherwise
        """
        if not self.progress:
            return None

        task = self.progress.add_task(description, total=0, info="")
        # Store initial completed count on the task for speed calculation
        self.progress.tasks[task].initial_completed = 0
        return task

    def update_total(self, task: TaskID | None, total: int) -> None:
        """Update the total count for a progress task.

        Args:
            task: Progress task ID
            total: New total count
        """
        if task and self.progress:
            self.progress.update(task, total=total)

    def increment_total(self, task: TaskID | None, increment: int) -> None:
        """Increment the total count for a progress task.

        Args:
            task: Progress task ID
            increment: Amount to increment the total by
        """
        if task and self.progress:
            current_total = self.progress.tasks[task].total
            self.progress.update(task, total=current_total + increment)

    def update_info(self, task: TaskID | None, info: str) -> None:
        """Update the info string for a progress task.

        Args:
            task: Progress task ID
            info: Info string to display
        """
        if task and self.progress:
            self.progress.update(task, info=info)

    def advance(self, task: TaskID | None, advance: int) -> None:
        """Advance the progress by a given amount.

        Args:
            task: Progress task ID
            advance: Number of steps to advance
        """
        if task and self.progress:
            self.progress.advance(task, advance)

    def update_speed(self, task: TaskID | None) -> None:
        """Calculate and update the speed display for a progress task.

        Args:
            task: Progress task ID
        """
        if task and self.progress:
            task_obj = self.progress.tasks[task]
            initial_completed = getattr(task_obj, 'initial_completed', 0)
            newly_processed = task_obj.completed - initial_completed
            if task_obj.elapsed > 0:
                speed = newly_processed / task_obj.elapsed
                self.progress.update(task, speed=f"{speed:.1f} chunks/s")


class EmbeddingService(BaseService):
    """Service for managing embedding generation, caching, and optimization."""

    def __init__(
        self,
        database_provider: DatabaseProvider,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_batch_size: int = 1000,
        db_batch_size: int = 2000,
        max_concurrent_batches: int | None = None,
        optimization_batch_frequency: int = 100,
        progress: Progress | None = None,
        missing_embeddings_initial_batch_size: int = 1000,
        missing_embeddings_min_batch_size: int = 50,
        missing_embeddings_max_batch_size: int = 30000,
        missing_embeddings_target_batch_time: float = 15.0,
        missing_embeddings_slow_threshold: float = 25.0,
        missing_embeddings_fast_threshold: float = 5.0,
        error_sample_limit: int = 5,
        max_consecutive_transient_failures: int = 5,
        transient_error_window_seconds: int = 300,
    ):
        """Initialize embedding service.

        Args:
            database_provider: Database provider for persistence
            embedding_provider: Embedding provider for vector generation
            embedding_batch_size: Number of texts per embedding API request
            db_batch_size: Number of records per database transaction
            max_concurrent_batches: Maximum concurrent batches (None = auto-detect from provider)
            optimization_batch_frequency: Optimize DB every N batches (provider-aware)
            progress: Optional Rich Progress instance for hierarchical progress display
            missing_embeddings_initial_batch_size: Initial batch size for missing embeddings generation
            missing_embeddings_min_batch_size: Minimum batch size for missing embeddings generation
            missing_embeddings_max_batch_size: Maximum batch size for missing embeddings generation
            missing_embeddings_target_batch_time: Target time per batch in seconds for dynamic sizing
            missing_embeddings_slow_threshold: Threshold above which batch is considered slow (seconds)
            missing_embeddings_fast_threshold: Threshold below which batch is considered fast (seconds)
            error_sample_limit: Maximum number of error samples to collect per error type (default: 5)
            max_consecutive_transient_failures: Maximum consecutive transient failures before aborting (default: 5)
            transient_error_window_seconds: Time window in seconds for tracking consecutive transient failures (default: 300)
        """
        super().__init__(database_provider)
        self._embedding_provider = embedding_provider
        self._embedding_batch_size = embedding_batch_size
        self._db_batch_size = db_batch_size

        # Auto-detect optimal concurrency from provider if not explicitly set
        if max_concurrent_batches is None:
            if embedding_provider and hasattr(
                embedding_provider, "get_recommended_concurrency"
            ):
                self._max_concurrent_batches = (
                    embedding_provider.get_recommended_concurrency()
                )
                logger.info(
                    f"Auto-detected concurrency: {self._max_concurrent_batches} "
                    f"concurrent batches for {embedding_provider.name}"
                )
            else:
                self._max_concurrent_batches = 8  # Safe default
                if embedding_provider:
                    logger.warning(
                        f"Provider {embedding_provider.name} does not implement "
                        f"get_recommended_concurrency(), using default: {self._max_concurrent_batches}"
                    )
                else:
                    logger.debug(
                        f"No embedding provider, using default concurrency: {self._max_concurrent_batches}"
                    )
        else:
            self._max_concurrent_batches = max_concurrent_batches
            if embedding_provider:
                logger.info(
                    f"Using explicit concurrency: {self._max_concurrent_batches} "
                    f"concurrent batches (overrides provider recommendation)"
                )

        self._optimization_batch_frequency = optimization_batch_frequency
        self._progress_manager = EmbeddingProgressManager(progress)

        # Dynamic batch size configuration
        self._missing_embeddings_initial_batch_size = missing_embeddings_initial_batch_size
        self._missing_embeddings_min_batch_size = missing_embeddings_min_batch_size
        self._missing_embeddings_max_batch_size = missing_embeddings_max_batch_size
        self._missing_embeddings_target_batch_time = missing_embeddings_target_batch_time
        self._missing_embeddings_slow_threshold = missing_embeddings_slow_threshold
        self._missing_embeddings_fast_threshold = missing_embeddings_fast_threshold
        self._error_sample_limit = error_sample_limit
        self._error_handler = EmbeddingErrorHandler(
            max_consecutive_failures=max_consecutive_transient_failures,
            transient_error_window_seconds=transient_error_window_seconds,
        )

        self._batch_processor = EmbeddingBatchProcessor(
            service=self,
            error_handler=self._error_handler,
            progress_manager=self._progress_manager,
        )

    def set_embedding_provider(self, provider: EmbeddingProvider) -> None:
        """Set or update the embedding provider.

        Args:
            provider: New embedding provider implementation
        """
        self._embedding_provider = provider

    async def generate_embeddings_for_chunks(
        self,
        chunk_ids: list[ChunkId],
        chunk_texts: list[str],
        show_progress: bool = True,
        embed_task: TaskID | None = None,
        chunks_data: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings for a list of chunks.

        Args:
            chunk_ids: List of chunk IDs to generate embeddings for
            chunk_texts: Corresponding text content for each chunk
            show_progress: Whether to show progress bar (default True)

        Returns:
            Dictionary with detailed generation statistics including successful, failed, and permanent failure counts
        """
        if not self._embedding_provider:
            logger.warning(f"No embedding provider configured in generate_embeddings_for_chunks (provider={self._embedding_provider})")
            return {"total_generated": 0, "failed_chunks": 0, "permanent_failures": 0}

        if len(chunk_ids) != len(chunk_texts):
            raise ValueError("chunk_ids and chunk_texts must have the same length")

        try:
            logger.debug(f"Generating embeddings for {len(chunk_ids)} chunks")

            # Filter out chunks that already have embeddings
            filtered_chunks = await self._filter_existing_embeddings(
                chunk_ids, chunk_texts
            )

            if not filtered_chunks:
                logger.debug("All chunks already have embeddings, returning 0")
                return {"total_generated": 0, "failed_chunks": 0, "permanent_failures": 0}

            # Generate embeddings in batches
            result = await self._generate_embeddings_in_batches(
                filtered_chunks, show_progress, self._progress_manager, embed_task, chunks_data
            )

            logger.debug(
                f"Successfully generated {result['total_generated']} embeddings "
                f"(processed: {result['total_processed']}, "
                f"failed: {result['failed_chunks']}, "
                f"permanent: {result['permanent_failures']})"
            )
            return result

        except Exception as e:
            # Log chunk details for debugging oversized chunks
            chunk_sizes = [len(text) for text in chunk_texts]
            max_size = max(chunk_sizes) if chunk_sizes else 0
            total_chars = sum(chunk_sizes)
            logger.error(f"Failed to generate embeddings (chunks: {len(chunk_sizes)}, total_chars: {total_chars}, max_chars: {max_size}): {e}")

            # Special handling for LanceDB resources exhausted error - fail fast
            if "Resources exhausted" in str(e):
                raise e

            return {"total_generated": 0, "failed_chunks": 0, "permanent_failures": 0, "error": str(e)}

    async def generate_missing_embeddings(
        self,
        provider_name: str | None = None,
        model_name: str | None = None,
        exclude_patterns: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Generate embeddings for all chunks that don't have them yet.

        Uses pagination to handle large codebases without memory issues.

        Args:
            provider_name: Optional specific provider to generate for
            model_name: Optional specific model to generate for
            exclude_patterns: Optional file patterns to exclude from embedding generation
            batch_size: Optional batch size for processing chunks (default: 10000)

        Returns:
            Dictionary with generation statistics
        """
        try:
            if not self._embedding_provider:
                logger.warning("No embedding provider configured, returning error")
                return {
                    "status": "error",
                    "error": "No embedding provider configured",
                    "generated": 0,
                }

            # Use provided provider/model or fall back to configured defaults
            target_provider = provider_name or self._embedding_provider.name
            target_model = model_name or self._embedding_provider.model

            # Invalidate embeddings that don't match the current provider/model combination
            logger.info(f"Invalidating embeddings that don't match {target_provider}/{target_model}")
            self._db.invalidate_embeddings_by_provider_model(target_provider, target_model)

            # Call optimize_tables after invalidation to reclaim space
            logger.info("Optimizing database after embedding invalidation...")
            self._db.optimize_tables()

            # Dynamic batch size configuration
            if batch_size is None:
                # Use configured values for dynamic batch sizing
                initial_batch_size = self._missing_embeddings_initial_batch_size
                min_batch_size = self._missing_embeddings_min_batch_size
                max_batch_size = self._missing_embeddings_max_batch_size
                target_batch_time = self._missing_embeddings_target_batch_time
                slow_threshold = self._missing_embeddings_slow_threshold
                fast_threshold = self._missing_embeddings_fast_threshold
            else:
                # Use fixed batch size if explicitly provided
                initial_batch_size = batch_size
                min_batch_size = batch_size
                max_batch_size = batch_size
                target_batch_time = float('inf')  # Disable dynamic sizing
                slow_threshold = float('inf')
                fast_threshold = 0.0

            logger.info(f"Starting missing embeddings generation (initial_batch_size={initial_batch_size})")

            # Get initial stats for progress tracking
            initial_stats = self._db.get_stats()
            total_chunks = initial_stats.get('chunks', 0)
            current_embeddings = initial_stats.get('embeddings', 0)

            # Create progress task for embedding generation
            embed_task = self._progress_manager.create_task("Generating embeddings")

            # Process all batches using the dedicated processor
            batch_results = await self._batch_processor.process_all_batches(
                target_provider=target_provider,
                target_model=target_model,
                initial_batch_size=initial_batch_size,
                min_batch_size=min_batch_size,
                max_batch_size=max_batch_size,
                target_batch_time=target_batch_time,
                slow_threshold=slow_threshold,
                fast_threshold=fast_threshold,
                embed_task=embed_task,
            )

            total_generated = batch_results["total_generated"]
            total_attempted = batch_results["total_attempted"]
            total_failed = batch_results["total_failed"]
            total_permanent_failures = batch_results["total_permanent_failures"]

            # Optimize if fragmentation high after embedding generation
            optimize_tables = getattr(self._db, "optimize_tables", None)
            if total_generated > 0 and optimize_tables and self._db.should_optimize_fragments(operation="post-embedding"):
                logger.debug("Optimizing database after embedding generation...")
                optimize_tables()

            logger.info(f"Completed missing embeddings generation: generated={total_generated}, attempted={total_attempted}, failed={total_failed}, permanent_failures={total_permanent_failures}")
            return {
                "status": "success",
                "generated": total_generated,
                "attempted": total_attempted,
                "failed": total_failed,
                "permanent_failures": total_permanent_failures,
                "provider": target_provider,
                "model": target_model,
            }

        except Exception as e:
            logger.error(f"Failed to generate missing embeddings: {e}")
            return {"status": "error", "error": str(e), "generated": 0}

    async def regenerate_embeddings(
        self, file_path: str | None = None, chunk_ids: list[ChunkId] | None = None
    ) -> dict[str, Any]:
        """Regenerate embeddings for specific files or chunks.

        Args:
            file_path: Optional file path to regenerate embeddings for
            chunk_ids: Optional specific chunk IDs to regenerate

        Returns:
            Dictionary with regeneration statistics
        """
        try:
            if not self._embedding_provider:
                return {
                    "status": "error",
                    "error": "No embedding provider configured",
                    "regenerated": 0,
                }

            # Determine which chunks to regenerate
            if chunk_ids:
                chunks_to_regenerate = self._get_chunks_by_ids(chunk_ids)
            elif file_path:
                chunks_to_regenerate = self._get_chunks_by_file_path(file_path)
            else:
                return {
                    "status": "error",
                    "error": "Must specify either file_path or chunk_ids",
                    "regenerated": 0,
                }

            if not chunks_to_regenerate:
                return {
                    "status": "complete",
                    "regenerated": 0,
                    "message": "No chunks found",
                }

            logger.info(
                f"Regenerating embeddings for {len(chunks_to_regenerate)} chunks"
            )

            # Delete existing embeddings
            provider_name = self._embedding_provider.name
            model_name = self._embedding_provider.model

            chunk_ids_to_regenerate = [chunk["id"] for chunk in chunks_to_regenerate]
            self._delete_embeddings_for_chunks(
                chunk_ids_to_regenerate, provider_name, model_name
            )

            # Generate new embeddings
            chunk_texts = [chunk["code"] for chunk in chunks_to_regenerate]
            result = await self.generate_embeddings_for_chunks(
                chunk_ids_to_regenerate, chunk_texts
            )
            regenerated_count = result["total_generated"]

            return {
                "status": "success",
                "regenerated": regenerated_count,
                "total_chunks": len(chunks_to_regenerate),
                "provider": provider_name,
                "model": model_name,
            }

        except Exception as e:
            logger.error(f"Failed to regenerate embeddings: {e}")
            return {"status": "error", "error": str(e), "regenerated": 0}

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get statistics about embeddings in the database.

        Returns:
            Dictionary with embedding statistics by provider and model
        """
        try:
            # Get all embedding tables
            embedding_tables = self._get_all_embedding_tables()

            if not embedding_tables:
                return {
                    "total_embeddings": 0,
                    "total_unique_chunks": 0,
                    "providers": [],
                    "configured_provider": self._embedding_provider.name
                    if self._embedding_provider
                    else None,
                    "configured_model": self._embedding_provider.model
                    if self._embedding_provider
                    else None,
                }

            # Query each table and aggregate results
            all_results = []
            all_chunks: set[tuple[str, str, Any]] = set()

            for table_name in embedding_tables:
                query = f"""
                    SELECT
                        provider,
                        model,
                        dims,
                        COUNT(*) as count,
                        COUNT(DISTINCT chunk_id) as unique_chunks
                    FROM {table_name}
                    GROUP BY provider, model, dims
                    ORDER BY provider, model, dims
                """

                table_results = self._db.execute_query(query)
                all_results.extend(table_results)

                # Get chunk IDs for total unique calculation
                chunk_query = f"SELECT provider, model, chunk_id FROM {table_name}"
                chunk_results = self._db.execute_query(chunk_query)
                all_chunks.update(
                    (row["provider"], row["model"], row["chunk_id"])
                    for row in chunk_results
                )

            # Calculate totals
            total_embeddings = sum(row["count"] for row in all_results)
            total_unique_chunks = len(all_chunks)

            return {
                "total_embeddings": total_embeddings,
                "total_unique_chunks": total_unique_chunks,
                "providers": all_results,
                "configured_provider": self._embedding_provider.name
                if self._embedding_provider
                else None,
                "configured_model": self._embedding_provider.model
                if self._embedding_provider
                else None,
            }

        except Exception as e:
            logger.error(f"Failed to get embedding stats: {e}")
            return {"error": str(e)}

    async def _filter_existing_embeddings(
        self, chunk_ids: list[ChunkId], chunk_texts: list[str]
    ) -> list[tuple[ChunkId, str]]:
        """Filter out chunks that already have embeddings.

        Args:
            chunk_ids: List of chunk IDs
            chunk_texts: Corresponding chunk texts

        Returns:
            List of (chunk_id, text) tuples for chunks without embeddings
        """
        if not self._embedding_provider:
            logger.debug("_filter_existing_embeddings: no embedding provider, returning empty list")
            return []

        provider_name = self._embedding_provider.name
        model_name = self._embedding_provider.model

        # Get existing embeddings from database
        try:
            # Determine table name based on embedding dimensions
            # We need to check what dimensions this provider/model uses
            if hasattr(self._embedding_provider, "get_dimensions"):
                dims = self._embedding_provider.get_dimensions()
            else:
                # Default to 1536 for most embedding models (OpenAI, etc.)
                dims = 1536

            table_name = f"embeddings_{dims}"

            existing_chunk_ids = self._db.get_existing_embeddings(
                chunk_ids=[int(cid) for cid in chunk_ids],
                provider=provider_name,
                model=model_name,
            )
        except Exception as e:
            logger.error(f"Failed to get existing embeddings: {e}")
            existing_chunk_ids = set()

        # Filter out chunks that already have embeddings or would be empty after normalization
        filtered_chunks = []
        skipped_empty = 0
        for chunk_id, text in zip(chunk_ids, chunk_texts):
            if chunk_id not in existing_chunk_ids:
                # Text is already normalized by IndexingCoordinator
                if not text.strip():
                    skipped_empty += 1
                    logger.debug(f"Skipping chunk {chunk_id}: empty content")
                    continue
                filtered_chunks.append((chunk_id, text))

        if skipped_empty > 0:
            logger.info(
                f"Skipped {skipped_empty} chunks with empty content after normalization"
            )

        logger.debug(
            f"Filtered {len(filtered_chunks)} chunks (out of {len(chunk_ids)}) need embeddings"
        )
        return filtered_chunks

    async def _generate_embeddings_in_batches(
        self, chunk_data: list[tuple[ChunkId, str]], show_progress: bool = True, progress_manager: EmbeddingProgressManager | None = None, task: TaskID | None = None, chunks_data: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Generate embeddings for chunks in optimized batches with granular failure handling.

        Args:
            chunk_data: List of (chunk_id, text) tuples
            show_progress: Whether to show progress updates
            embed_task: Optional progress task ID
            chunks_data: Optional chunk metadata for database operations

        Returns:
            Dictionary with detailed results including success/failure statistics. All processed chunks receive status updates in the database.
        """
        import time
        from dataclasses import dataclass, field

        @dataclass
        class BatchResult:
            """Result of processing a single batch."""
            batch_num: int
            successful_chunks: list[tuple[ChunkId, list[float]]] = field(default_factory=list)
            failed_chunks: list[tuple[ChunkId, str, EmbeddingErrorClassification]] = field(default_factory=list)
            retry_queue: list[tuple[ChunkId, str]] = field(default_factory=list)
            error_stats: dict[str, int] = field(default_factory=dict)
            error_samples: dict[str, list[str]] = field(default_factory=dict)

        if not chunk_data:
            logger.debug("_generate_embeddings_in_batches: no chunk data provided")
            return {
                "total_generated": 0,
                "total_processed": 0,
                "error_stats": {},
                "retry_attempts": 0,
                "permanent_failures": 0,
            }

        # Initialize error classifier and counters
        error_classifier = EmbeddingErrorClassifier()

        # Create token-aware batches
        batches = self._create_token_aware_batches(chunk_data)
        logger.debug(f"Processing {len(batches)} token-aware batches")

        # Track overall statistics
        total_generated = 0
        total_processed = 0
        retry_attempts = 0
        permanent_failures = 0
        all_error_stats = {}
        all_error_samples = {}

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        async def process_single_batch(
            batch: list[tuple[ChunkId, str]], batch_num: int
        ) -> BatchResult:
            """Process a single batch attempt."""
            async with semaphore:
                result = BatchResult(batch_num=batch_num)

                try:
                    # Extract chunk IDs and texts
                    chunk_ids = [chunk_id for chunk_id, _ in batch]
                    texts = [text for _, text in batch]

                    # Generate embeddings
                    if not self._embedding_provider:
                        # Mark all as failed
                        for chunk_id, text in batch:
                            result.failed_chunks.append((
                                chunk_id, text, EmbeddingErrorClassification.PERMANENT
                            ))
                        return result

                    embedding_results = await self._embedding_provider.embed(texts)

                    if len(embedding_results) != len(chunk_ids):
                        logger.warning(
                            f"Batch {batch_num}: Expected {len(chunk_ids)} embeddings, got {len(embedding_results)}"
                        )
                        # Mark all as failed for this attempt
                        for chunk_id, text in batch:
                            result.failed_chunks.append((
                                chunk_id, text, EmbeddingErrorClassification.TRANSIENT
                            ))
                        return result

                    # All embeddings succeeded - add to successful chunks
                    for chunk_id, vector in zip(chunk_ids, embedding_results):
                        result.successful_chunks.append((chunk_id, vector))

                    logger.debug(f"Batch {batch_num} completed successfully with {len(result.successful_chunks)} embeddings")
                    return result

                except Exception as e:
                    # Classify the error
                    classification = error_classifier.classify_exception(
                        e,
                        provider=self._embedding_provider.name if self._embedding_provider else None,
                        model=self._embedding_provider.model if self._embedding_provider else None,
                        context={"batch_num": batch_num, "batch_size": len(batch)}
                    )

                    # Update error statistics
                    error_type = classification.value
                    result.error_stats[error_type] = result.error_stats.get(error_type, 0) + 1

                    # Collect error samples (up to limit per error type)
                    if error_type not in result.error_samples:
                        result.error_samples[error_type] = []
                    if len(result.error_samples[error_type]) < self._error_sample_limit:
                        result.error_samples[error_type].append(str(e))

                    if classification == EmbeddingErrorClassification.BATCH_RECOVERABLE:
                        # Split batch and retry parts
                        if len(batch) > 1:
                            logger.warning(
                                f"Batch {batch_num} recoverable error, splitting batch: {e}"
                            )
                            mid = len(batch) // 2
                            # Process first half
                            sub_result1 = await process_single_batch(
                                batch[:mid], f"{batch_num}a"
                            )
                            # Process second half
                            sub_result2 = await process_single_batch(
                                batch[mid:], f"{batch_num}b"
                            )

                            # Merge results
                            result.successful_chunks.extend(sub_result1.successful_chunks)
                            result.successful_chunks.extend(sub_result2.successful_chunks)
                            result.failed_chunks.extend(sub_result1.failed_chunks)
                            result.failed_chunks.extend(sub_result2.failed_chunks)
                            result.retry_queue.extend(sub_result1.retry_queue)
                            result.retry_queue.extend(sub_result2.retry_queue)

                            # Merge error stats
                            for key, value in sub_result1.error_stats.items():
                                result.error_stats[key] = result.error_stats.get(key, 0) + value
                            for key, value in sub_result2.error_stats.items():
                                result.error_stats[key] = result.error_stats.get(key, 0) + value
                        else:
                            # Can't split further
                            for chunk_id, text in batch:
                                result.failed_chunks.append((chunk_id, text, classification))
                    else:
                        # For transient or permanent errors, mark all as failed for this attempt
                        for chunk_id, text in batch:
                            result.failed_chunks.append((chunk_id, text, classification))

                    return result

        async def process_batch_with_retries(
            batch: list[tuple[ChunkId, str]], batch_num: int, max_retries: int = 3
        ) -> BatchResult:
            """Process a batch with intelligent retry logic."""
            result = BatchResult(batch_num=batch_num)
            current_batch = batch
            attempt = 0

            while attempt < max_retries and current_batch:
                attempt += 1
                if attempt > 1:
                    nonlocal retry_attempts
                    retry_attempts += 1
                    logger.debug(f"Batch {batch_num} retry attempt {attempt}/{max_retries}")
                    # Exponential backoff between retries (reduced for testing)
                    await asyncio.sleep(min(2 ** attempt, 5))  # Cap at 5 seconds for faster test completion

                # Process the current batch
                batch_result = await process_single_batch(current_batch, batch_num)

                # Check results
                if batch_result.successful_chunks:
                    # Some chunks succeeded - add them to result
                    result.successful_chunks.extend(batch_result.successful_chunks)

                # Handle failed chunks based on their classification
                transient_failures = []
                permanent_failures = []
                recoverable_failures = []

                for chunk_id, text, classification in batch_result.failed_chunks:
                    if classification == EmbeddingErrorClassification.TRANSIENT:
                        transient_failures.append((chunk_id, text))
                    elif classification == EmbeddingErrorClassification.PERMANENT:
                        permanent_failures.append((chunk_id, text))
                    elif classification == EmbeddingErrorClassification.BATCH_RECOVERABLE:
                        recoverable_failures.append((chunk_id, text))

                # Permanent failures are final
                for chunk_id, text in permanent_failures:
                    result.failed_chunks.append((chunk_id, text, EmbeddingErrorClassification.PERMANENT))

                # For transient failures, retry if attempts remain
                if transient_failures and attempt <= max_retries:
                    current_batch = transient_failures
                    continue
                else:
                    # Max retries reached or no transient failures
                    for chunk_id, text in transient_failures:
                        result.failed_chunks.append((chunk_id, text, EmbeddingErrorClassification.PERMANENT))

                # Recoverable failures should have been handled by splitting in process_single_batch
                # If they still failed after splitting, treat as permanent failures
                for chunk_id, text in recoverable_failures:
                    result.failed_chunks.append((chunk_id, text, EmbeddingErrorClassification.PERMANENT))

                # Merge error stats
                for error_type, count in batch_result.error_stats.items():
                    result.error_stats[error_type] = result.error_stats.get(error_type, 0) + count

                # Merge error samples
                for error_type, samples in batch_result.error_samples.items():
                    if error_type not in result.error_samples:
                        result.error_samples[error_type] = []
                    result.error_samples[error_type].extend(samples[:self._error_sample_limit - len(result.error_samples[error_type])])

                # Only break when max retries reached
                if attempt > max_retries:
                    break

                # Prepare for next retry attempt
                current_batch = transient_failures

            # Deduplicate results to prevent duplicate inserts/updates
            # This can happen when chunks succeed in multiple retry attempts
            def _classification_priority(classification):
                """Get priority for classification deduplication (higher = more severe)."""
                priorities = {
                    EmbeddingErrorClassification.PERMANENT: 3,
                    EmbeddingErrorClassification.TRANSIENT: 2,
                    EmbeddingErrorClassification.BATCH_RECOVERABLE: 1,
                }
                return priorities.get(classification, 0)

            # Deduplicate successful chunks (keep last vector)
            successful_by_id = {}
            for chunk_id, vector in result.successful_chunks:
                successful_by_id[chunk_id] = vector
            result.successful_chunks = [(cid, vec) for cid, vec in successful_by_id.items()]

            # Deduplicate failed chunks (keep highest priority classification)
            failed_by_id = {}
            for chunk_id, text, classification in result.failed_chunks:
                if chunk_id not in failed_by_id or _classification_priority(classification) > _classification_priority(failed_by_id[chunk_id][1]):
                    failed_by_id[chunk_id] = (text, classification)
            result.failed_chunks = [(cid, text, cls) for cid, (text, cls) in failed_by_id.items()]

            return result

        # Process batches with optional progress tracking
        import threading
        update_lock = threading.Lock()

        async def process_batch_with_progress(
            batch: list[tuple[ChunkId, str]], batch_num: int
        ) -> BatchResult:
            result = await process_batch_with_retries(batch, batch_num)

            # Thread-safe progress update
            if show_progress and progress_manager:
                with update_lock:
                    # Only count successful chunks for progress
                    successful_count = len(result.successful_chunks)
                    progress_manager.advance(task, successful_count)
                    progress_manager.update_speed(task)

            return result

        # Process all batches concurrently
        tasks = [
            process_batch_with_progress(batch, i)
            for i, batch in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect and process results
        all_embeddings_data = []
        successful_chunks_total = 0
        failed_chunks_total = 0

        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"Batch {i + 1} failed with exception: {result}")
                continue

            batch_result = result

            # Add successful embeddings to database batch
            for chunk_id, vector in batch_result.successful_chunks:
                all_embeddings_data.append({
                    "chunk_id": chunk_id,
                    "provider": self._embedding_provider.name if self._embedding_provider else "unknown",
                    "model": self._embedding_provider.model if self._embedding_provider else "unknown",
                    "dims": len(vector),
                    "embedding": vector,
                    "status": "success",
                })



            # Merge error statistics
            for error_type, count in batch_result.error_stats.items():
                all_error_stats[error_type] = all_error_stats.get(error_type, 0) + count

            # Merge error samples
            for error_type, samples in batch_result.error_samples.items():
                if error_type not in all_error_samples:
                    all_error_samples[error_type] = []
                all_error_samples[error_type].extend(samples[:self._error_sample_limit - len(all_error_samples[error_type])])

        # Update chunks_data with failed statuses
        chunk_id_to_status = {}
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                continue
            for chunk_id, text, classification in batch_result.failed_chunks:
                status = "permanent_failure" if classification == EmbeddingErrorClassification.PERMANENT else "failed"
                chunk_id_to_status[chunk_id] = status
        if chunks_data:
            for chunk in chunks_data:
                if chunk["id"] in chunk_id_to_status:
                    chunk["embedding_status"] = chunk_id_to_status[chunk["id"]]

        logger.debug(f"Prepared status updates for {len(chunk_id_to_status)} failed chunks")

        # Derive counters from batch results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                continue
            successful_chunks_total += len(batch_result.successful_chunks)
            for chunk_id, text, classification in batch_result.failed_chunks:
                if classification == EmbeddingErrorClassification.PERMANENT:
                    permanent_failures += 1
                failed_chunks_total += 1

        # Insert successful embeddings in one batch
        if all_embeddings_data or failed_chunks_total > 0:
            logger.debug(f"Inserting {len(all_embeddings_data)} embeddings and updating status for {failed_chunks_total} failed chunks in one batch")

            relevant_chunks_data = chunks_data or []

            try:
                inserted_count = self._db.insert_embeddings_batch(
                    all_embeddings_data, relevant_chunks_data
                )
                # Handle mock objects in tests
                if hasattr(inserted_count, '__int__'):
                    total_generated = int(inserted_count)
                else:
                    total_generated = len(all_embeddings_data)  # Fallback for mocks
                logger.debug(f"Successfully inserted {total_generated} embeddings")
            except Exception as e:
                logger.error(f"Failed to insert embeddings batch: {e}")
                total_generated = 0

            # Run optimization after bulk insert
            try:
                start_time = time.time()
                self._db.optimize_tables()
                elapsed = time.time() - start_time
                logger.info(f"Post-batch optimization completed in {elapsed:.2f}s")
            except Exception as e:
                logger.warning(f"Post-batch optimization failed: {e}")

        total_processed = successful_chunks_total + failed_chunks_total

        logger.debug(
            f"_generate_embeddings_in_batches: completed, generated={total_generated}, "
            f"processed={total_processed} (all with status updates), failed={failed_chunks_total}, "
            f"permanent_failures={permanent_failures}, retries={retry_attempts}"
        )

        return {
            "total_generated": total_generated,
            "total_processed": total_processed,
            "successful_chunks": successful_chunks_total,
            "failed_chunks": failed_chunks_total,
            "permanent_failures": permanent_failures,
            "retry_attempts": retry_attempts,
            "error_stats": all_error_stats,
            "error_samples": all_error_samples,
        }

    def _create_token_aware_batches(
        self, chunk_data: list[tuple[ChunkId, str]]
    ) -> list[list[tuple[ChunkId, str]]]:
        """Create batches that respect provider token limits using provider-agnostic logic.

        Args:
            chunk_data: List of (chunk_id, text) tuples

        Returns:
            List of optimized batches that respect provider token limits
        """
        if not chunk_data:
            return []

        # Maximum chunks per batch - critical tuning for DB write performance
        #
        # Performance tradeoff:
        # - Larger batches = fewer serialized DB writes (DB is single-threaded bottleneck)
        # - Smaller batches = more concurrent embedding requests (parallelizable)
        #
        # Value of 300 chosen empirically:
        # - With 40 concurrent batches: 12,000 chunks in flight (good saturation)
        # - With 8 concurrent batches: 2,400 chunks in flight (still acceptable)
        # - DB writes complete in ~50-100ms, avoiding idle time between batch completions
        # - Not so large that a single slow batch blocks progress significantly
        #
        # Can be tuned per-provider if profiling shows different optimal values
        MAX_CHUNKS_PER_BATCH = 300

        if not self._embedding_provider:
            # No provider - use simple batching
            batch_size = 20  # Conservative default
            batches = []
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i : i + batch_size]
                batches.append(batch)
            return batches

        # Get provider's token and document limits
        max_tokens = self._embedding_provider.get_max_tokens_per_batch()
        max_documents = self._embedding_provider.get_max_documents_per_batch()
        # Apply safety margin (20% for conservative batching)
        safe_limit = int(max_tokens * 0.80)

        # Provider-agnostic token-aware batching
        batches = []
        current_batch: list[tuple[ChunkId, str]] = []
        current_tokens = 0

        for chunk_id, text in chunk_data:
            # Use accurate provider-specific token estimation
            if self._embedding_provider:
                text_tokens = estimate_tokens(
                    text, self._embedding_provider.name, self._embedding_provider.model
                )
            else:
                # Fallback for no provider (conservative default)
                text_tokens = max(1, int(len(text) / 3.5))

            # Check if adding this chunk would exceed token, document, or chunk limit
            if (
                (current_tokens + text_tokens > safe_limit and current_batch)
                or len(current_batch) >= max_documents
                or len(current_batch) >= MAX_CHUNKS_PER_BATCH
            ):
                # Start new batch
                batches.append(current_batch)
                current_batch = [(chunk_id, text)]
                current_tokens = text_tokens
            else:
                current_batch.append((chunk_id, text))
                current_tokens += text_tokens

        # Add remaining batch if not empty
        if current_batch:
            batches.append(current_batch)

        # Calculate effective concurrency
        effective_concurrency = min(len(batches), self._max_concurrent_batches)

        logger.info(
            f"Created {len(batches)} batches for {len(chunk_data)} chunks "
            f"(concurrency limit: {self._max_concurrent_batches}, "
            f"effective concurrency: {effective_concurrency}, "
            f"max_chunks_per_batch: {MAX_CHUNKS_PER_BATCH})"
        )

        logger.debug(
            f"Batch constraints: max_tokens={max_tokens}, max_documents={max_documents}, "
            f"safe_limit={safe_limit}, max_chunks={MAX_CHUNKS_PER_BATCH}"
        )
        return batches


    def _get_chunks_by_ids(self, chunk_ids: list[ChunkId]) -> list[dict[str, Any]]:
        """Get chunk data for specific chunk IDs."""
        if not chunk_ids:
            logger.debug("_get_chunks_by_ids: no chunk_ids provided, returning empty list")
            return []

        # Use batch query for better performance
        chunks = self._db.get_chunks_by_ids(chunk_ids)

        # Normalize field names for compatibility between providers
        filtered_chunks = []
        for chunk in chunks:
            filtered_chunk = {
                "id": chunk["id"],
                "code": chunk.get(
                    "content", chunk.get("code", "")
                ),  # LanceDB uses 'content'
                "symbol": chunk.get(
                    "name", chunk.get("symbol", "")
                ),  # LanceDB uses 'name'
                "path": chunk.get("file_path", ""),
            }
            filtered_chunks.append(filtered_chunk)

        logger.debug(f"_get_chunks_by_ids: found {len(filtered_chunks)} chunks out of {len(chunk_ids)} requested")
        return filtered_chunks

    def _get_chunks_by_file_path(self, file_path: str) -> list[dict[str, Any]]:
        """Get all chunks for a specific file path."""
        query = """
            SELECT c.id, c.code, c.symbol, f.path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE f.path = ?
            ORDER BY c.id
        """

        return self._db.execute_query(query, [file_path])

    def _delete_embeddings_for_chunks(
        self, chunk_ids: list[ChunkId], provider: str, model: str
    ) -> None:
        """Delete existing embeddings for specific chunks and provider/model."""
        if not chunk_ids:
            return

        # Get all embedding tables and delete from each
        embedding_tables = self._get_all_embedding_tables()

        if not embedding_tables:
            logger.debug("No embedding tables found, nothing to delete")
            return

        placeholders = ",".join("?" for _ in chunk_ids)
        deleted_count = 0

        for table_name in embedding_tables:
            query = f"""
                DELETE FROM {table_name}
                WHERE chunk_id IN ({placeholders})
                AND provider = ?
                AND model = ?
            """

            params = chunk_ids + [provider, model]
            try:
                # Execute the deletion for this table
                self._db.execute_query(query, params)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete from {table_name}: {e}")

        logger.debug(
            f"Deleted existing embeddings for {len(chunk_ids)} chunks from {deleted_count} tables"
        )

