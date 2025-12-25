"""Embedding service for ChunkHound - manages embedding generation and caching."""

import asyncio
from typing import Any

from loguru import logger
from rich.progress import Progress, TaskID

from chunkhound.core.types.common import ChunkId
from chunkhound.core.utils import estimate_tokens
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider

from .base_service import BaseService


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
        self.progress = progress

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
    ) -> int:
        """Generate embeddings for a list of chunks.

        Args:
            chunk_ids: List of chunk IDs to generate embeddings for
            chunk_texts: Corresponding text content for each chunk
            show_progress: Whether to show progress bar (default True)

        Returns:
            Number of embeddings successfully generated
        """
        if not self._embedding_provider:
            logger.warning(f"No embedding provider configured in generate_embeddings_for_chunks (provider={self._embedding_provider})")
            return 0

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
                return 0

            # Generate embeddings in batches
            total_generated = await self._generate_embeddings_in_batches(
                filtered_chunks, show_progress, embed_task
            )

            logger.debug(f"Successfully generated {total_generated} embeddings")
            return total_generated

        except Exception as e:
            # Log chunk details for debugging oversized chunks
            chunk_sizes = [len(text) for text in chunk_texts]
            max_size = max(chunk_sizes) if chunk_sizes else 0
            total_chars = sum(chunk_sizes)
            logger.error(f"Failed to generate embeddings (chunks: {len(chunk_sizes)}, total_chars: {total_chars}, max_chars: {max_size}): {e}")

            # Special handling for LanceDB resources exhausted error - fail fast
            if "Resources exhausted" in str(e):
                raise e

            return 0

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

            # Optimize database before starting if fragmentation is high
            if self._db.should_optimize_fragments(operation="pre-embedding"):
                logger.info("Optimizing database before embedding generation to prevent fragmentation issues...")
                self._db.optimize_tables()

            # Use provided provider/model or fall back to configured defaults
            target_provider = provider_name or self._embedding_provider.name
            target_model = model_name or self._embedding_provider.model

            # Dynamic batch size configuration
            if batch_size is None:
                # Start with conservative batch size for large databases
                initial_batch_size = 100
                min_batch_size = 50
                max_batch_size = 10000
                target_batch_time = 15.0  # Target 15 seconds per batch
                slow_threshold = 30.0     # Consider >30s as too slow
                fast_threshold = 5.0      # Consider <5s as very fast
            else:
                # Use fixed batch size if explicitly provided
                initial_batch_size = batch_size
                min_batch_size = batch_size
                max_batch_size = batch_size
                target_batch_time = float('inf')  # Disable dynamic sizing
                slow_threshold = float('inf')
                fast_threshold = 0.0

            current_batch_size = initial_batch_size
            logger.info(f"Starting missing embeddings generation (initial_batch_size={current_batch_size})")

            # Get initial stats for progress tracking
            initial_stats = self._db.get_stats()
            total_chunks = initial_stats.get('chunks', 0)

            # Create progress task for embedding generation
            embed_task: TaskID | None = None
            if self.progress:
                embed_task = self.progress.add_task(
                    "Generating embeddings", total=total_chunks, speed="", info=""
                )

            total_generated = 0
            total_processed = 0
            offset = 0
            batch_count = 0

            while True:
                batch_count += 1
                import time as time_module

                # Measure time for chunk ID retrieval
                start_time = time_module.time()

                try:
                    logger.debug(f"Retrieving chunks batch: provider={target_provider}, model={target_model}, limit={current_batch_size}, offset={offset}")
                    # Get next batch of chunks without embeddings
                    chunks_data = self._db.get_chunks_without_embeddings_paginated(
                        target_provider, target_model, limit=current_batch_size, offset=offset
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

                if not chunks_data:
                    # No more chunks to process
                    logger.info("No more chunks to process, exiting batch loop")
                    break

                logger.debug(f"Batch {batch_count}: offset={offset}, size={len(chunks_data)}, retrieval_time={retrieval_time:.2f}s")

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

                # Generate embeddings for this batch
                logger.debug(f"Calling generate_embeddings_for_chunks with {len(chunk_id_list)} chunks, provider={self._embedding_provider}")
                batch_generated = await self.generate_embeddings_for_chunks(
                    chunk_id_list, chunk_texts, show_progress=True, embed_task=embed_task
                )
                logger.debug(f"generate_embeddings_for_chunks returned {batch_generated}")

                total_generated += batch_generated
                total_processed += len(chunks_data)
                offset += len(chunks_data)  # Use actual batch size, not current_batch_size

                logger.info(f"Batch {batch_count} completed: generated={batch_generated}, total_processed={total_processed}, total_generated={total_generated}, current_batch_size={current_batch_size}")

                # Check for fragmentation optimization after each page
                if self._db.should_optimize_fragments(operation="during-embedding-page"):
                    logger.info("Running fragmentation optimization during embedding generation...")
                    try:
                        self._db.optimize_tables()
                        logger.info("Fragmentation optimization completed")
                    except Exception as opt_error:
                        logger.warning(f"Fragmentation optimization failed: {opt_error}")

            # Optimize if fragmentation high after embedding generation
            optimize_tables = getattr(self._db, "optimize_tables", None)
            if total_generated > 0 and optimize_tables and self._db.should_optimize_fragments(operation="post-embedding"):
                logger.debug("Optimizing database after embedding generation...")
                optimize_tables()

            logger.info(f"Completed missing embeddings generation: generated={total_generated}, processed={total_processed}")
            return {
                "status": "success",
                "generated": total_generated,
                "total_chunks": total_processed,
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
            regenerated_count = await self.generate_embeddings_for_chunks(
                chunk_ids_to_regenerate, chunk_texts
            )

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
        self, chunk_data: list[tuple[ChunkId, str]], show_progress: bool = True, embed_task: TaskID | None = None
    ) -> int:
        """Generate embeddings for chunks in optimized batches.

        Args:
            chunk_data: List of (chunk_id, text) tuples

        Returns:
            Number of embeddings successfully generated
        """
        if not chunk_data:
            logger.debug("_generate_embeddings_in_batches: no chunk data provided, returning 0")
            return 0

        # Create token-aware batches immediately (fast operation)
        batches = self._create_token_aware_batches(chunk_data)

        avg_batch_size = (
            sum(len(batch) for batch in batches) / len(batches) if batches else 0
        )
        logger.debug(
            f"Processing {len(batches)} token-aware batches (avg {avg_batch_size:.1f} chunks each)"
        )

        # Track batch count for periodic optimization
        completed_batch_count = 0
        optimize_tables = getattr(self._db, "optimize_tables", None)
        should_optimize = (
            optimize_tables is not None
            and self._optimization_batch_frequency > 0
        )

        # Process batches with concurrency control
        semaphore = asyncio.Semaphore(self._max_concurrent_batches)

        async def process_batch(
            batch: list[tuple[ChunkId, str]], batch_num: int, retry_depth: int = 0
        ) -> list[dict[str, Any]]:
            """Process a single batch of embeddings."""
            async with semaphore:
                try:
                    logger.debug(
                        f"Processing batch {batch_num + 1}/{len(batches)} with {len(batch)} chunks"
                    )

                    # Extract chunk IDs and texts
                    chunk_ids = [chunk_id for chunk_id, _ in batch]
                    texts = [text for _, text in batch]

                    # Generate embeddings
                    if not self._embedding_provider:
                        return []
                    embedding_results = await self._embedding_provider.embed(texts)

                    if len(embedding_results) != len(chunk_ids):
                        logger.warning(
                            f"Batch {batch_num}: Expected {len(chunk_ids)} embeddings, got {len(embedding_results)}"
                        )
                        return []

                    # Prepare embedding data for database
                    embeddings_data = []
                    for chunk_id, vector in zip(chunk_ids, embedding_results):
                        embeddings_data.append(
                            {
                                "chunk_id": chunk_id,
                                "provider": self._embedding_provider.name
                                if self._embedding_provider
                                else "unknown",
                                "model": self._embedding_provider.model
                                if self._embedding_provider
                                else "unknown",
                                "dims": len(vector),
                                "embedding": vector,
                            }
                        )

                    logger.debug(
                        f"Batch {batch_num + 1} completed: {len(embeddings_data)} embeddings generated"
                    )

                    return embeddings_data

                except Exception as e:
                    # Check if this is a token limit error that can be retried
                    error_message = str(e).lower()
                    is_token_limit_error = (
                        "max allowed tokens" in error_message
                        or "token limit" in error_message
                        or "tokens per batch" in error_message
                    )

                    if is_token_limit_error and len(batch) > 1 and retry_depth < 3:
                        # Split batch in half and retry both parts
                        logger.warning(
                            f"Token limit exceeded for batch {batch_num + 1}, splitting and retrying "
                            f"(depth {retry_depth + 1}/3)"
                        )
                        mid = len(batch) // 2
                        batch1 = batch[:mid]
                        batch2 = batch[mid:]

                        # Recursively process both halves
                        result1 = await process_batch(
                            batch1, batch_num, retry_depth + 1
                        )
                        result2 = await process_batch(
                            batch2, batch_num, retry_depth + 1
                        )
                        return result1 + result2

                    # Log batch details for non-retryable errors or max retries exceeded
                    batch_sizes = [len(text) for _, text in batch]
                    max_size = max(batch_sizes) if batch_sizes else 0
                    logger.error(f"Batch {batch_num + 1} failed (chunks: {len(batch)}, max_chars: {max_size}): {e}")

                    return []


        # Process batches with optional progress tracking
        import threading

        update_lock = threading.Lock()
        processed_count = 0

        async def process_batch_with_optional_progress(
            batch: list[tuple[ChunkId, str]], batch_num: int
        ) -> list[dict[str, Any]]:
            nonlocal processed_count
            result = await process_batch(batch, batch_num)

            # Thread-safe progress update if progress tracking is enabled
            if show_progress:
                with update_lock:
                    processed_count += len(batch)
                    if embed_task and self.progress:
                        self.progress.advance(embed_task, len(batch))

                        # Calculate and display speed
                        task_obj = self.progress.tasks[embed_task]
                        if task_obj.elapsed and task_obj.elapsed > 0:
                            speed = processed_count / task_obj.elapsed
                            self.progress.update(
                                embed_task, speed=f"{speed:.1f} chunks/s"
                            )

            return result

        # Create tasks (always process batches, with optional progress tracking)
        tasks = [
            process_batch_with_optional_progress(batch, i)
            for i, batch in enumerate(batches)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all embeddings data and handle errors
        all_embeddings_data = []
        successful_batches = 0
        for i, result in enumerate(results):
            if isinstance(result, list):
                all_embeddings_data.extend(result)
                if result:  # Non-empty list means successful
                    successful_batches += 1
            else:
                # Find the failed batch and extract chunk details
                failed_batch = batches[i] if i < len(batches) else []
                batch_sizes = [len(text) for _, text in failed_batch]
                max_chars = max(batch_sizes) if batch_sizes else 0
                total_chars = sum(batch_sizes)
                # Also keep the standard logging
                logger.error(f"Batch {i + 1} failed (chunks: {len(batch_sizes)}, total_chars: {total_chars:,}, max_chars: {max_chars:,}): {result}")

        # Insert all embeddings in one big batch
        total_generated = 0
        if all_embeddings_data:
            logger.debug(f"Inserting {len(all_embeddings_data)} embeddings in one batch")
            total_generated = self._db.insert_embeddings_batch(
                all_embeddings_data, self._db_batch_size
            )
            logger.debug(f"Successfully inserted {total_generated} embeddings")

        # Update completed batch count and run optimization if needed
        if should_optimize and successful_batches > 0:
            completed_batch_count += successful_batches

            # Check if we've reached the optimization threshold
            batches_since_last_optimize = (
                completed_batch_count % self._optimization_batch_frequency
            )
            if (
                batches_since_last_optimize < successful_batches
                or completed_batch_count == self._optimization_batch_frequency
            ):
                logger.debug(
                    f"Running periodic DB optimization after {completed_batch_count} total batches"
                )
                try:
                    optimize_tables()
                    logger.debug("Periodic optimization completed")
                except Exception as e:
                    logger.warning(f"Periodic optimization failed: {e}")

        logger.debug(f"_generate_embeddings_in_batches: completed, total_generated={total_generated}")
        return total_generated

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

