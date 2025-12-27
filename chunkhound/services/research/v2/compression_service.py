"""Compression Service for hierarchical map-reduce compression.

This module implements the compression logic for the v2 coverage-first algorithm,
which uses recursive hierarchical map-reduce to compress content to fit token budgets.

Algorithm Foundation:
    The compression algorithm implements hierarchical map-reduce based on:
    - ToM (Tree-oriented MapReduce for Long-Context Reasoning), EMNLP 2025
      https://aclanthology.org/2025.emnlp-main.899.pdf
    - RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
      https://arxiv.org/abs/2401.18059

    Unlike flat iterative approaches, this uses recursive compression where
    each cluster is compressed to fit its budget BEFORE merging with siblings.
    This guarantees no single LLM call receives more than MAX_SINGLE_CALL_TOKENS.

Key Features:
- Equal budget split: each cluster gets target_tokens/num_clusters
- Recursive compression: large clusters are recursively compressed BEFORE
  moving to the next level (unlike flat iterative approaches)
- Bounded context: NO single LLM call receives more than MAX_SINGLE_CALL_TOKENS
- Stagnation detection: Falls back to force compression if no progress
"""

import asyncio
from typing import Any

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services.clustering_service import ClusterGroup, ClusteringService
from chunkhound.services.prompts import CITATION_REQUIREMENTS
from chunkhound.services.research.shared.models import SINGLE_PASS_TIMEOUT_SECONDS


class CompressionService:
    """Service for hierarchical map-reduce compression of code content.

    Implements recursive compression algorithm that:
    1. Clusters content into semantically related groups
    2. Compresses each cluster to its budget (recursively if needed)
    3. Merges cluster summaries until budget is met
    4. Force compresses oversized single items via truncation
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: Any,
        config: Any,
        progress: Any | None = None,
    ):
        """Initialize compression service.

        Args:
            llm_manager: LLM manager for compression providers
            embedding_manager: Embedding manager for clustering
            config: ResearchConfig with compression parameters
            progress: Optional progress callback for emitting events
        """
        self._llm_manager = llm_manager
        self._embedding_manager = embedding_manager
        self._config = config
        self._progress = progress

    async def _emit_event(
        self,
        event_type: str,
        message: str,
        **metadata: Any,
    ) -> None:
        """Emit a progress event (compatible with TreeProgressDisplay)."""
        if not self._progress:
            return
        try:
            await self._progress.emit_event(
                event_type=event_type,
                message=message,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to emit progress event '{event_type}': {e}")

    async def compress_to_budget(
        self,
        root_query: str,
        gap_queries: list[str],
        content_dict: dict[str, str],
        target_tokens: int,
        file_imports: dict[str, list[str]],
        depth: int = 0,
        prev_tokens: int | None = None,
    ) -> dict[str, str]:
        """Recursively compress content until it fits target budget.

        Uses hierarchical map-reduce based on:
        - ToM (Tree-oriented MapReduce for Long-Context Reasoning), EMNLP 2025
          https://aclanthology.org/2025.emnlp-main.899.pdf
        - RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
          https://arxiv.org/abs/2401.18059

        Key algorithm properties:
        - Equal budget split: each cluster gets target_tokens/num_clusters
        - Recursive compression: large clusters are recursively compressed BEFORE
          moving to the next level (unlike flat iterative approaches)
        - Bounded context: NO single LLM call receives more than MAX_SINGLE_CALL_TOKENS
        - Stagnation detection: Falls back to force compression if no progress

        Args:
            root_query: Original research query for context
            gap_queries: Gap queries for compound context
            content_dict: File path -> content mapping to compress
            target_tokens: Target budget for final output
            file_imports: Pre-extracted imports per file (for injection without re-parsing)
            depth: Current recursion depth (for safety limit)
            prev_tokens: Token count from previous level (for stagnation detection)

        Returns:
            Compressed content dict (typically single "summary" key when converged)

        Raises:
            RuntimeError: If recursion exceeds MAX_DEPTH levels or stagnation detected
        """
        # Compression constants (enforced relationship)
        max_depth = self._config.compression_max_depth
        final_synthesis_threshold = self._config.final_synthesis_threshold
        # Ensures 2 clusters always fit
        cluster_budget_fixed = final_synthesis_threshold // 2

        llm = self._llm_manager.get_utility_provider()
        current_tokens = sum(llm.estimate_tokens(c) for c in content_dict.values())

        logger.info(
            f"Compression depth {depth}: {len(content_dict)} items, "
            f"{current_tokens:,} tokens (target: {target_tokens:,})"
        )

        # Emit compression start event
        await self._emit_event(
            "compression_start",
            f"Starting compression: {current_tokens:,} tokens",
            depth=depth,
            tokens=current_tokens,
        )

        # Stagnation detection: no meaningful reduction from previous level
        if prev_tokens is not None and current_tokens >= prev_tokens * 0.9:
            raise RuntimeError(
                f"Compression stagnated at depth {depth}: "
                f"{prev_tokens:,} -> {current_tokens:,} tokens (< 10% reduction). "
                f"This indicates a bug in LLM compression or clustering."
            )

        if depth > max_depth:
            raise RuntimeError(
                f"Compression exceeded max depth {max_depth} with {current_tokens:,} tokens "
                f"remaining (target: {target_tokens:,}). "
                f"This indicates a bug in clustering reduction logic."
            )

        # Base case: single result within budget
        if len(content_dict) == 1 and current_tokens <= target_tokens:
            logger.info(
                f"Converged at depth {depth}: single item, "
                f"{current_tokens:,} <= {target_tokens:,} tokens"
            )
            await self._emit_event(
                "compression_complete",
                f"Compression complete: {current_tokens:,} tokens",
                tokens=current_tokens,
            )
            return content_dict

        # Handle oversized single item - can't cluster further, force compress
        if len(content_dict) == 1 and current_tokens > final_synthesis_threshold:
            logger.info(
                f"Single oversized item at depth {depth}: {current_tokens:,} tokens > "
                f"{final_synthesis_threshold:,}, force compressing to {target_tokens:,}"
            )
            key = list(content_dict.keys())[0]
            content = content_dict[key]

            compressed_single = await self._force_compress_single(
                root_query,
                key,
                content,
                target_tokens,
                final_synthesis_threshold,
            )
            return {"summary": compressed_single}

        # If fits in final synthesis threshold, merge directly (regardless of cluster count)
        if current_tokens <= final_synthesis_threshold:
            logger.info(
                f"Merging {len(content_dict)} items at depth {depth} "
                f"({current_tokens:,} tokens fits in final synthesis call)"
            )
            merged = await self._merge_summaries(
                root_query, content_dict, target_tokens
            )
            return {"summary": merged}

        # Cluster content: HDBSCAN at depth=0, k-means otherwise
        clusters = await self._cluster_content(content_dict, cluster_budget_fixed, depth)

        # Fixed budget per cluster (half of final_synthesis_threshold)
        cluster_budget = cluster_budget_fixed

        logger.info(
            f"Clustered into {len(clusters)} groups at depth {depth}, "
            f"cluster_budget={cluster_budget:,} tokens each"
        )

        # Emit clustering event
        await self._emit_event(
            "compression_cluster",
            f"Clustered into {len(clusters)} groups",
            depth=depth,
            clusters=len(clusters),
        )

        # Compress clusters in parallel with concurrency limit
        # Use synthesis provider's recommended concurrency to avoid rate limiting
        synthesis_provider = self._llm_manager.get_synthesis_provider()
        max_concurrency = synthesis_provider.get_synthesis_concurrency()
        semaphore = asyncio.Semaphore(max_concurrency)

        async def compress_with_limit(
            i: int, cluster: ClusterGroup
        ) -> tuple[str, str]:
            async with semaphore:
                return await self._compress_single_cluster(
                    i,
                    cluster,
                    depth,
                    root_query,
                    gap_queries,
                    cluster_budget,
                    file_imports,
                    final_synthesis_threshold,
                )

        tasks = [compress_with_limit(i, cluster) for i, cluster in enumerate(clusters)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            raise RuntimeError(f"Cluster compression failed: {errors[0]}")
        compressed_results = dict(results)  # type: ignore[arg-type]

        # Check if we need another level of compression
        merged_tokens = sum(llm.estimate_tokens(c) for c in compressed_results.values())

        logger.info(
            f"Depth {depth} compression complete: {len(compressed_results)} summaries, "
            f"{merged_tokens:,} tokens (target: {target_tokens:,})"
        )

        if merged_tokens > target_tokens or len(compressed_results) > 1:
            # Emit iteration event before recursion
            await self._emit_event(
                "compression_iteration",
                f"Depth {depth}: {merged_tokens:,} tokens remaining",
                depth=depth,
                tokens=merged_tokens,
            )
            # Recurse on compressed results to merge them
            return await self.compress_to_budget(
                root_query,
                gap_queries,
                compressed_results,
                target_tokens,
                file_imports,
                depth + 1,
                prev_tokens=current_tokens,
            )

        return compressed_results

    async def _cluster_content(
        self,
        content_dict: dict[str, str],
        cluster_budget: int,
        depth: int = 0,
    ) -> list[ClusterGroup]:
        """Cluster content dict using HDBSCAN (depth=0) or k-means (depth>0).

        First pass (depth=0): HDBSCAN discovers natural semantic clusters
        Subsequent passes (depth>0): k-means with budget-based target_k

        Args:
            content_dict: File path -> content mapping
            cluster_budget: Target tokens per cluster
            depth: Current recursion depth (0 for first pass)

        Returns:
            List of ClusterGroup objects
        """
        embedding_provider = self._embedding_manager.get_provider()
        synthesis_provider = self._llm_manager.get_synthesis_provider()

        clustering_service = ClusteringService(
            embedding_provider=embedding_provider,
            llm_provider=synthesis_provider,
        )

        if depth == 0:
            # HDBSCAN for natural semantic discovery (first pass)
            cluster_groups, cluster_metadata = (
                await clustering_service.cluster_files_hdbscan(
                    content_dict,
                    min_cluster_size=2,
                )
            )

            logger.info(
                f"HDBSCAN: {cluster_metadata['num_native_clusters']} native clusters, "
                f"{cluster_metadata['num_outliers']} outliers reassigned, "
                f"{cluster_metadata['num_clusters']} final clusters"
            )
        else:
            # k-means for subsequent passes (structure already known)
            total_tokens = sum(
                synthesis_provider.estimate_tokens(c) for c in content_dict.values()
            )

            # Smart k selection: aim for clusters that fit budget
            # At least 2 to ensure reduction, at most half input count
            target_k = max(2, total_tokens // cluster_budget)
            target_k = min(target_k, len(content_dict) // 2)

            # Clamp to valid range
            target_k = max(1, min(target_k, len(content_dict)))

            cluster_groups, cluster_metadata = await clustering_service.cluster_files(
                content_dict, n_clusters=target_k
            )

            logger.debug(
                f"K-means: {cluster_metadata['num_clusters']} clusters, "
                f"avg {cluster_metadata['avg_tokens_per_cluster']:,} tokens/cluster"
            )

        return cluster_groups

    async def _compress_single_cluster(
        self,
        i: int,
        cluster: ClusterGroup,
        depth: int,
        root_query: str,
        gap_queries: list[str],
        cluster_budget: int,
        file_imports: dict[str, list[str]],
        final_synthesis_threshold: int,
    ) -> tuple[str, str]:
        """Compress a single cluster (helper for parallel processing).

        Args:
            i: Cluster index
            cluster: ClusterGroup to compress
            depth: Current recursion depth
            root_query: Original research query
            gap_queries: Gap queries for compound context
            cluster_budget: Token budget per cluster
            file_imports: Pre-extracted imports per file
            final_synthesis_threshold: Threshold for single-pass compression

        Returns:
            Tuple of (key, compressed_content)
        """
        llm = self._llm_manager.get_utility_provider()
        cluster_tokens = sum(
            llm.estimate_tokens(c) for c in cluster.files_content.values()
        )

        summary_key = f"cluster_{depth}_{i}"

        if cluster_tokens > final_synthesis_threshold:
            # RECURSIVE: this cluster is too large for final synthesis
            logger.info(
                f"Cluster {i} at depth {depth} has {cluster_tokens:,} tokens > "
                f"{final_synthesis_threshold:,}, recursing..."
            )
            compressed = await self.compress_to_budget(
                root_query,
                gap_queries,
                cluster.files_content,
                cluster_budget,
                file_imports,
                depth + 1,
                prev_tokens=None,  # No stagnation check for cluster subdivision
            )
            # Extract the single summary from recursive result
            return (summary_key, list(compressed.values())[0])
        else:
            # Safe to compress in single call
            summary = await self._compress_cluster(
                root_query,
                cluster.files_content,
                cluster_budget,
                file_imports,
            )
            return (summary_key, summary)

    async def _compress_cluster(
        self,
        root_query: str,
        cluster_content: dict[str, str],
        target_tokens: int,
        file_imports: dict[str, list[str]],
    ) -> str:
        """Compress a cluster of files using LLM (helper for compression loop).

        Args:
            root_query: Original research query
            cluster_content: File path -> content mapping for this cluster
            target_tokens: Target output size in tokens
            file_imports: Pre-extracted imports per file (lookup, no re-parsing)

        Returns:
            Compressed summary of cluster content
        """
        llm = self._llm_manager.get_synthesis_provider()

        input_tokens = sum(llm.estimate_tokens(c) for c in cluster_content.values())
        needs_compression = input_tokens > target_tokens

        if needs_compression:
            target_ratio = target_tokens / max(input_tokens, 1)
            logger.debug(
                f"Compressing cluster: {len(cluster_content)} files, "
                f"{input_tokens:,} -> {target_tokens:,} tokens ({target_ratio:.0%})"
            )
        else:
            logger.debug(
                f"Summarizing cluster (under budget): {len(cluster_content)} files, "
                f"{input_tokens:,} tokens, max {target_tokens:,}"
            )

        # Use root query only for compression (gaps are injected in final synthesis)
        query_context = f"QUERY: {root_query}"

        # Inject imports header for each file (using pre-extracted file_imports)
        # Cluster summaries (e.g., "cluster_0_0") won't be in file_imports, so
        # they're naturally skipped - imports were already included when first compressed
        enriched_content = dict(cluster_content)
        for file_path, content in cluster_content.items():
            imports = file_imports.get(file_path)  # Simple lookup, no re-parsing
            if imports:
                imports_header = "# Imports:\n" + "\n".join(imports) + "\n\n"
                enriched_content[file_path] = imports_header + content

        # Build code context
        code_sections = []
        for file_path, content in enriched_content.items():
            code_sections.append(f"### {file_path}\n{'=' * 80}\n{content}\n{'=' * 80}")

        code_context = "\n\n".join(code_sections)

        # Conditional prompts: compression guidance only when over budget
        if needs_compression:
            system = f"""You are compressing code files for a research synthesis.

Compress to MAXIMUM {target_tokens:,} tokens (input: {input_tokens:,} tokens).
Be concise—include only what's essential for the query.

Focus on:
- Key architectural patterns and relationships
- Essential implementation details
- Remove redundant examples and boilerplate

{CITATION_REQUIREMENTS}

Maintain citation references to files."""
            prompt_suffix = "Provide a concise summary."
        else:
            # Under budget: summarize concisely
            system = f"""You are summarizing code files for a research synthesis.

Be concise. Maximum {target_tokens:,} tokens.

Focus on:
- Key architectural patterns and relationships
- Essential implementation details

{CITATION_REQUIREMENTS}

Maintain citation references to files."""
            prompt_suffix = "Provide a concise summary."

        prompt = f"""{query_context}

Summarize the following code files:

{code_context}

{prompt_suffix}"""

        # No artificial floor - let LLM choose natural length
        if needs_compression:
            effective_max = target_tokens
        else:
            # Under budget: cap at input to prevent expansion
            effective_max = min(target_tokens, input_tokens)

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=effective_max,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )

        tokens = llm.estimate_tokens(response.content)
        logger.debug(f"Cluster compression complete: {tokens:,} tokens")

        return response.content

    async def _merge_summaries(
        self,
        root_query: str,
        summaries: dict[str, str],
        target_tokens: int,
    ) -> str:
        """Merge multiple summaries into one when they fit in context.

        Used when total tokens fit the budget but we have multiple items to combine.
        Similar to _compress_cluster but optimized for already-compressed summaries.

        Args:
            root_query: Original research query
            summaries: Key -> summary content mapping
            target_tokens: Target output size in tokens

        Returns:
            Single merged summary string
        """
        llm = self._llm_manager.get_synthesis_provider()

        input_tokens = sum(llm.estimate_tokens(c) for c in summaries.values())
        needs_compression = input_tokens > target_tokens

        if needs_compression:
            target_ratio = target_tokens / max(input_tokens, 1)
            logger.debug(
                f"Merging {len(summaries)} summaries with compression: "
                f"{input_tokens:,} -> {target_tokens:,} tokens ({target_ratio:.0%})"
            )
        else:
            logger.debug(
                f"Merging {len(summaries)} summaries (under budget): "
                f"{input_tokens:,} tokens, max {target_tokens:,}"
            )

        # Use root query only for merging (gaps are injected in final synthesis)
        query_context = f"QUERY: {root_query}"

        # Build summaries context
        summary_sections = []
        for key, content in summaries.items():
            summary_sections.append(f"### {key}\n{'=' * 40}\n{content}\n{'=' * 40}")

        summaries_context = "\n\n".join(summary_sections)

        # Conditional prompts: compression guidance only when over budget
        if needs_compression:
            system = (
                f"You are integrating multiple code analysis summaries.\n\n"
                f"MAXIMUM {target_tokens:,} tokens (input: {input_tokens:,} tokens).\n"
                f"Be concise—combine essentials only.\n\n"
                f"Your task:\n"
                f"1. Combine insights coherently\n"
                f"2. Eliminate redundancy\n"
                f"3. PRESERVE citation references [N]\n\n"
                f"{CITATION_REQUIREMENTS}"
            )
            prompt_suffix = "Provide a concise integrated summary."
        else:
            # Under budget: let LLM choose natural length
            system = (
                f"You are integrating multiple code analysis summaries.\n\n"
                f"Be concise. Maximum {target_tokens:,} tokens.\n\n"
                f"Your task:\n"
                f"1. Combine insights coherently\n"
                f"2. Eliminate redundancy\n"
                f"3. PRESERVE citation references [N]\n\n"
                f"{CITATION_REQUIREMENTS}"
            )
            prompt_suffix = "Provide a concise integrated summary."

        prompt = f"""{query_context}

Integrate the following summaries:

{summaries_context}

{prompt_suffix}"""

        # No artificial floor - let LLM choose natural length
        if needs_compression:
            effective_max = target_tokens
        else:
            # Under budget: cap at input to prevent expansion
            effective_max = min(target_tokens, input_tokens)

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=effective_max,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )

        tokens = llm.estimate_tokens(response.content)
        logger.debug(f"Merged {len(summaries)} summaries into {tokens:,} tokens")

        return response.content

    async def _force_compress_single(
        self,
        root_query: str,
        key: str,
        content: str,
        target_tokens: int,
        max_context_tokens: int,
    ) -> str:
        """Force compress an oversized single item via truncation + LLM summarization.

        Used when a single item (file or previous summary) exceeds the LLM context
        window and cannot be further clustered. Truncates content to fit context,
        then asks LLM to summarize.

        Args:
            root_query: Original research query
            key: Identifier for the content (file path or summary key)
            content: The oversized content to compress
            target_tokens: Target output size in tokens
            max_context_tokens: Maximum tokens that fit in LLM context

        Returns:
            Compressed summary of the content
        """
        llm = self._llm_manager.get_synthesis_provider()
        input_tokens = llm.estimate_tokens(content)

        logger.info(
            f"Force compressing '{key}': {input_tokens:,} tokens -> "
            f"target {target_tokens:,} tokens"
        )

        # Truncate content to fit LLM context, preserving beginning and end
        # Reserve tokens for system prompt (~500) and response (~target_tokens)
        available_for_content = max_context_tokens - 500 - min(target_tokens, 5000)
        chars_per_token = len(content) / max(input_tokens, 1)
        max_chars = int(available_for_content * chars_per_token)

        if len(content) > max_chars:
            # Keep 70% from beginning, 30% from end for context continuity
            head_chars = int(max_chars * 0.7)
            tail_chars = max_chars - head_chars
            truncated_content = (
                content[:head_chars]
                + "\n\n[... CONTENT TRUNCATED FOR LENGTH ...]\n\n"
                + content[-tail_chars:]
            )
            logger.debug(
                f"Truncated content from {len(content):,} to {len(truncated_content):,} chars"
            )
        else:
            truncated_content = content

        # Use root query only (gaps are injected in final synthesis)
        query_context = f"QUERY: {root_query}"

        system = f"""Compress a large code file for research synthesis.

MAXIMUM {target_tokens:,} tokens. Be concise.

Focus on:
- Key architectural patterns
- Essential implementation details
- Preserve citation references

{CITATION_REQUIREMENTS}"""

        prompt = f"""{query_context}

Summarize '{key}':

{"=" * 80}
{truncated_content}
{"=" * 80}

Provide a concise summary."""

        # Force compression always targets reduction (input > context limit)
        # No artificial floor - use target_tokens as ceiling
        effective_max = target_tokens

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=effective_max,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )

        tokens = llm.estimate_tokens(response.content)
        logger.info(f"Force compression complete: {tokens:,} tokens")

        return response.content
