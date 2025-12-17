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
                gap_queries,
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
                root_query, gap_queries, content_dict, target_tokens
            )
            return {"summary": merged}

        # Cluster content using existing ClusteringService
        clusters = await self._cluster_content(content_dict, cluster_budget_fixed)

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

        compressed_results: dict[str, str] = {}
        for i, cluster in enumerate(clusters):
            cluster_tokens = sum(
                llm.estimate_tokens(c) for c in cluster.files_content.values()
            )

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
                summary_key = f"cluster_{depth}_{i}"
                compressed_results[summary_key] = list(compressed.values())[0]
            else:
                # Safe to compress in single call
                summary = await self._compress_cluster(
                    root_query,
                    gap_queries,
                    cluster.files_content,
                    cluster_budget,
                    file_imports,
                )
                compressed_results[f"cluster_{depth}_{i}"] = summary

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
    ) -> list[ClusterGroup]:
        """Cluster content dict using ClusteringService.

        Args:
            content_dict: File path -> content mapping
            cluster_budget: Fixed cluster budget (37,500 tokens)

        Returns:
            List of ClusterGroup objects
        """
        # Must match compress_to_budget
        final_synthesis_threshold = self._config.final_synthesis_threshold

        embedding_provider = self._embedding_manager.get_provider()
        synthesis_provider = self._llm_manager.get_synthesis_provider()

        # Use fixed cluster budget directly
        max_tokens_per_cluster = cluster_budget

        clustering_service = ClusteringService(
            embedding_provider=embedding_provider,
            llm_provider=synthesis_provider,
            max_tokens_per_cluster=max_tokens_per_cluster,
            min_cluster_size=self._config.min_cluster_size,
        )

        cluster_groups, cluster_metadata = await clustering_service.cluster_files(
            content_dict
        )

        logger.debug(
            f"HDBSCAN: {cluster_metadata['num_clusters']} clusters, "
            f"avg {cluster_metadata['avg_tokens_per_cluster']:,} tokens/cluster"
        )

        # Stagnation detection: HDBSCAN produced same or more clusters than input,
        # OR returned a single oversized cluster that can't fit in final synthesis
        input_count = len(content_dict)
        output_count = len(cluster_groups)

        # Check if single cluster is too large for final synthesis
        single_cluster_too_large = (
            output_count == 1
            and input_count > 1
            and cluster_groups[0].total_tokens > final_synthesis_threshold
        )

        if (
            output_count >= input_count and input_count > 1
        ) or single_cluster_too_large:
            # Fallback to k-means with guaranteed reduction
            # Ensure k < input_count: min(input_count - 1, ...) guarantees reduction
            target_k = max(1, min(input_count - 1, input_count // 2))
            reason = (
                f"single cluster too large ({cluster_groups[0].total_tokens:,} tokens)"
                if single_cluster_too_large
                else f"{input_count} -> {output_count} clusters"
            )
            logger.info(
                f"HDBSCAN stagnated ({reason}), "
                f"falling back to k-means with k={target_k}"
            )

            (
                cluster_groups,
                cluster_metadata,
            ) = await clustering_service.cluster_files_kmeans(
                content_dict, n_clusters=target_k
            )

            logger.debug(
                f"K-means: {cluster_metadata['num_clusters']} clusters, "
                f"avg {cluster_metadata['avg_tokens_per_cluster']:,} tokens/cluster"
            )

            # Sanity check: k-means must reduce cluster count
            if len(cluster_groups) >= input_count:
                raise RuntimeError(
                    f"K-means failed to reduce cluster count: "
                    f"{input_count} -> {len(cluster_groups)} (target_k={target_k}). "
                    f"This should never happen with k < input_count."
                )

        return cluster_groups

    async def _compress_cluster(
        self,
        root_query: str,
        gap_queries: list[str],
        cluster_content: dict[str, str],
        target_tokens: int,
        file_imports: dict[str, list[str]],
    ) -> str:
        """Compress a cluster of files using LLM (helper for compression loop).

        Includes ROOT + gap queries in prompt to maintain query focus.

        Args:
            root_query: Original research query
            gap_queries: Gap queries for compound context
            cluster_content: File path -> content mapping for this cluster
            target_tokens: Target output size in tokens
            file_imports: Pre-extracted imports per file (lookup, no re-parsing)

        Returns:
            Compressed summary of cluster content
        """
        llm = self._llm_manager.get_synthesis_provider()

        # Calculate compression ratio for prompt guidance
        input_tokens = sum(llm.estimate_tokens(c) for c in cluster_content.values())
        target_ratio = target_tokens / max(input_tokens, 1)

        logger.debug(
            f"Compressing cluster: {len(cluster_content)} files, "
            f"{input_tokens:,} -> {target_tokens:,} tokens ({target_ratio:.0%})"
        )

        # Build compound query context
        compound_context = f"PRIMARY QUERY: {root_query}"
        if gap_queries:
            compound_context += "\n\nRELATED GAPS IDENTIFIED:\n"
            for gap in gap_queries:
                compound_context += f"- {gap}\n"

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

        # Compress cluster with explicit budget guidance
        system = f"""You are compressing code files for a research synthesis.

CRITICAL: You must compress to approximately {target_tokens:,} tokens.
Input is {input_tokens:,} tokens - compress to {target_ratio:.0%} of original size.

Focus on:
- Key architectural patterns and relationships
- Essential implementation details for the query
- Remove redundant examples and boilerplate

{CITATION_REQUIREMENTS}

Maintain citation references to files."""

        prompt = f"""{compound_context}

Analyze the following code files and provide a concise summary focusing on the query:

{code_context}

Provide a compressed summary of approximately {target_tokens:,} tokens."""

        # Bounded max_completion_tokens: 20k-50k range for quality
        min_llm_tokens = 20000
        max_llm_tokens = 50000
        effective_max = max(min_llm_tokens, min(target_tokens * 2, max_llm_tokens))

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
        gap_queries: list[str],
        summaries: dict[str, str],
        target_tokens: int,
    ) -> str:
        """Merge multiple summaries into one when they fit in context.

        Used when total tokens fit the budget but we have multiple items to combine.
        Similar to _compress_cluster but optimized for already-compressed summaries.

        Args:
            root_query: Original research query
            gap_queries: Gap queries for compound context
            summaries: Key -> summary content mapping
            target_tokens: Target output size in tokens

        Returns:
            Single merged summary string
        """
        llm = self._llm_manager.get_synthesis_provider()

        # Calculate compression ratio for prompt guidance
        input_tokens = sum(llm.estimate_tokens(c) for c in summaries.values())
        target_ratio = target_tokens / max(input_tokens, 1)

        logger.debug(
            f"Merging {len(summaries)} summaries: "
            f"{input_tokens:,} -> {target_tokens:,} tokens ({target_ratio:.0%})"
        )

        # Build compound query context
        compound_context = f"PRIMARY QUERY: {root_query}"
        if gap_queries:
            compound_context += "\n\nRELATED GAPS IDENTIFIED:\n"
            for gap in gap_queries:
                compound_context += f"- {gap}\n"

        # Build summaries context
        summary_sections = []
        for key, content in summaries.items():
            summary_sections.append(f"### {key}\n{'=' * 40}\n{content}\n{'=' * 40}")

        summaries_context = "\n\n".join(summary_sections)

        system = (
            f"You are integrating multiple code analysis summaries into a "
            f"unified response.\n\n"
            f"CRITICAL: Compress to approximately {target_tokens:,} tokens.\n"
            f"Input is {input_tokens:,} tokens - compress to {target_ratio:.0%}.\n\n"
            f"Your task:\n"
            f"1. Combine insights from all summaries coherently\n"
            f"2. Eliminate redundancy while preserving important details\n"
            f"3. Maintain focus on the original query\n"
            f"4. PRESERVE ALL citation references [N] from the summaries\n\n"
            f"{CITATION_REQUIREMENTS}\n\n"
            f"Be thorough but concise - produce a well-integrated summary."
        )

        prompt = f"""{compound_context}

Integrate the following analysis summaries into a single, coherent response:

{summaries_context}

Provide an integrated summary of approximately {target_tokens:,} tokens."""

        # Bounded max_completion_tokens: 20k-50k range for quality
        min_llm_tokens = 20000
        max_llm_tokens = 50000
        effective_max = max(min_llm_tokens, min(target_tokens * 2, max_llm_tokens))

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
        gap_queries: list[str],
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
            gap_queries: Gap queries for compound context
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

        # Build compound query context
        compound_context = f"PRIMARY QUERY: {root_query}"
        if gap_queries:
            compound_context += "\n\nRELATED GAPS IDENTIFIED:\n"
            for gap in gap_queries:
                compound_context += f"- {gap}\n"

        target_ratio = target_tokens / max(input_tokens, 1)
        system = f"""You are compressing a large code file/summary for research synthesis.

CRITICAL: The content was truncated to fit context. Compress to approximately {target_tokens:,} tokens.
Original was {input_tokens:,} tokens - compress to {target_ratio:.0%} of original size.

Focus on:
- Key architectural patterns and relationships
- Essential implementation details for the query
- Preserve important code references and citations

{CITATION_REQUIREMENTS}"""

        prompt = f"""{compound_context}

Compress and summarize the following content from '{key}':

{"=" * 80}
{truncated_content}
{"=" * 80}

Provide a compressed summary of approximately {target_tokens:,} tokens."""

        # Use bounded max_completion_tokens
        min_llm_tokens = 20000
        max_llm_tokens = 50000
        effective_max = max(min_llm_tokens, min(target_tokens * 2, max_llm_tokens))

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=effective_max,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )

        tokens = llm.estimate_tokens(response.content)
        logger.info(f"Force compression complete: {tokens:,} tokens")

        return response.content
