"""Synthesis Engine for Deep Research Service.

This module contains the synthesis logic for combining search results into
comprehensive answers. It implements both single-pass and map-reduce synthesis
strategies with dynamic budget management based on repository size.

Architecture:
    - Single-pass synthesis: For small to medium result sets that fit in context
    - Map-reduce synthesis: For large result sets requiring clustering
    - Budget management: Dynamic token allocation based on repository size
    - Citation system: Numbered file references with validation

The synthesis engine uses:
    - LLM providers for answer generation
    - Embedding providers for file reranking
    - Clustering service for map-reduce grouping
    - Citation management for source tracking
"""

import asyncio
import time
from typing import Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.clustering_service import ClusterGroup, ClusteringService
from chunkhound.services.research.models import (
    CLUSTER_OUTPUT_TOKEN_BUDGET,
    MAX_CHUNKS_PER_FILE_REPR,
    MAX_TOKENS_PER_CLUSTER,
    MAX_TOKENS_PER_FILE_REPR,
    SINGLE_PASS_TIMEOUT_SECONDS,
    _CITATION_PATTERN,
)


class SynthesisEngine:
    """Engine for synthesizing research results into comprehensive answers.

    The SynthesisEngine coordinates the synthesis phase of deep research,
    managing token budgets, file reranking, and LLM calls for answer generation.

    Key responsibilities:
        1. Budget management: Allocate tokens across files and chunks
        2. File prioritization: Rerank files by relevance to avoid diversity collapse
        3. Single-pass synthesis: Generate answers in one LLM call
        4. Map-reduce synthesis: Cluster and combine for large result sets
        5. Citation management: Build and validate numbered file references

    Token Budget Architecture:
        - Input tokens: Dynamic (30k-150k) based on repository size
        - Output tokens: Fixed (30k) for reasoning model compatibility
        - Overhead: ~5k for prompts and formatting
        - Footer: Appended outside token budget (~100-500 tokens)
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        database_services: DatabaseServices,
        parent_service: Any,
    ):
        """Initialize synthesis engine.

        Args:
            llm_manager: LLM manager for synthesis providers
            database_services: Database services for stats and context
            parent_service: Parent DeepResearchService for accessing citation_manager,
                          quality_validator, file_reader, and _emit_event
        """
        self._llm_manager = llm_manager
        self._db_services = database_services
        self._parent = parent_service

    async def _manage_token_budget_for_synthesis(
        self,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        root_query: str,
        synthesis_budgets: dict[str, int],
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
        """Manage token budget to fit within synthesis budget limit.

        Prioritizes files using reranking when available to ensure diverse,
        relevant file selection. Falls back to accumulated chunk scores if
        reranking fails. This prevents file diversity collapse where deep
        exploration causes score accumulation in few files.

        Args:
            chunks: All chunks from BFS traversal
            files: All file contents from BFS traversal
            root_query: Original research query (for reranking files)
            synthesis_budgets: Dynamic budgets based on repository size

        Returns:
            Tuple of (prioritized_chunks, budgeted_files, budget_info)
        """
        llm = self._llm_manager.get_utility_provider()

        # Use output token budget from dynamic calculation
        output_tokens = synthesis_budgets["output_tokens"]

        # Get available tokens for code content from synthesis budgets
        available_tokens = synthesis_budgets["input_tokens"]

        logger.info(
            f"Managing token budget: {available_tokens:,} tokens available for code content"
        )

        # Sort chunks by score from multi-hop semantic search (highest first)
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)

        # Build file-to-chunks mapping
        file_to_chunks: dict[str, list[dict[str, Any]]] = {}
        for chunk in sorted_chunks:
            file_path = chunk.get("file_path", "")
            if file_path:
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)

        # Create file representative documents for reranking
        # This prevents file diversity collapse where files explored at multiple
        # BFS depths accumulate exponentially higher scores
        file_paths = []
        file_documents = []

        for file_path, file_chunks in file_to_chunks.items():
            # Sort chunks by score and take top N chunks (configured by constant)
            sorted_file_chunks = sorted(
                file_chunks, key=lambda c: c.get("score", 0.0), reverse=True
            )
            top_chunks = sorted_file_chunks[:MAX_CHUNKS_PER_FILE_REPR]

            # Build representative document
            repr_parts = []
            for chunk in top_chunks:
                start_line = chunk.get("start_line", 1)
                end_line = chunk.get("end_line", 1)
                content = chunk.get("content", "")
                repr_parts.append(f"Lines {start_line}-{end_line}:\n{content}")

            document = f"{file_path}\n\n" + "\n\n".join(repr_parts)

            # Truncate to token limit
            if llm.estimate_tokens(document) > MAX_TOKENS_PER_FILE_REPR:
                chars_to_include = MAX_TOKENS_PER_FILE_REPR * 4
                document = document[:chars_to_include]

            file_paths.append(file_path)
            file_documents.append(document)

        # Rerank files by relevance to root query
        # This ensures file priority is based on relevance to original query,
        # not accumulated scores from multi-level BFS exploration
        await self._parent._emit_event(
            "synthesis_rerank", f"Reranking {len(file_documents)} files by relevance"
        )
        embedding_provider = self._parent._embedding_manager.get_provider()

        # Check provider batch limits (providers handle batch splitting automatically)
        max_batch = embedding_provider.get_max_rerank_batch_size()
        if len(file_documents) > max_batch:
            logger.info(
                f"Reranking {len(file_documents)} files will be automatically split into "
                f"batches of {max_batch} (provider: {embedding_provider.name})"
            )

        # Initialize file priorities dict (populated by either reranking or fallback)
        file_priorities: dict[str, float] = {}

        # Try reranking with fallback to chunk score accumulation
        try:
            rerank_results = await embedding_provider.rerank(
                query=root_query, documents=file_documents, top_k=None
            )
        except Exception as e:
            logger.warning(
                f"File reranking failed ({type(e).__name__}: {e}), "
                f"falling back to accumulated chunk scores"
            )
            logger.debug(f"Reranking failure traceback: {e}", exc_info=True)

            # Fallback: Use accumulated chunk scores (original behavior before reranking)
            for file_path, file_chunks in file_to_chunks.items():
                file_priorities[file_path] = sum(
                    c.get("score", 0.0) for c in file_chunks
                )

            logger.info(f"Using chunk score fallback for {len(file_priorities)} files")
        else:
            # Build priority map from rerank scores. Some providers may return
            # result indexes that do not align with the documents array (for
            # example, due to internal truncation). In that case we skip the
            # out-of-range entries instead of raising IndexError so that
            # deep-research callers (including MCP code_research) degrade
            # gracefully instead of failing the entire run.
            for result in rerank_results:
                idx = result.index
                if 0 <= idx < len(file_paths):
                    file_path = file_paths[idx]
                    file_priorities[file_path] = result.score
                else:
                    logger.warning(
                        "Rerank result index %s out of range for %s files; "
                        "skipping this entry",
                        idx,
                        len(file_paths),
                    )

            logger.info(
                f"Reranked {len(file_priorities)} files for synthesis budget allocation"
            )

            if not file_priorities:
                logger.warning(
                    "All rerank results were out of range; falling back to chunk scores"
                )
                for file_path, file_chunks in file_to_chunks.items():
                    file_priorities[file_path] = sum(
                        c.get("score", 0.0) for c in file_chunks
                    )

        # Sort files by priority score (highest first)
        # Works for both reranking and fallback paths
        sorted_files = sorted(file_priorities.items(), key=lambda x: x[1], reverse=True)

        # Build budgeted file contents
        budgeted_files: dict[str, str] = {}
        total_tokens = 0
        files_included_fully = 0
        files_included_partial = 0
        files_excluded = 0

        for file_path, priority in sorted_files:
            if file_path not in files:
                continue

            content = files[file_path]
            content_tokens = llm.estimate_tokens(content)

            if total_tokens + content_tokens <= available_tokens:
                # Include full file
                budgeted_files[file_path] = content
                total_tokens += content_tokens
                files_included_fully += 1
            else:
                # Check if we can include a snippet
                remaining_tokens = available_tokens - total_tokens

                if remaining_tokens > 1000:  # Only include if meaningful
                    # Include top chunks from this file as snippets
                    file_chunks = file_to_chunks[file_path]
                    snippet_parts = []

                    for chunk in file_chunks[:5]:  # Top 5 chunks max
                        start_line = chunk.get("start_line", 1)
                        end_line = chunk.get("end_line", 1)
                        chunk_content = chunk.get("content", "")

                        snippet_parts.append(
                            f"# Lines {start_line}-{end_line}\n{chunk_content}"
                        )

                    snippet = "\n\n".join(snippet_parts)
                    snippet_tokens = llm.estimate_tokens(snippet)

                    if snippet_tokens <= remaining_tokens:
                        budgeted_files[file_path] = snippet
                        total_tokens += snippet_tokens
                        files_included_partial += 1
                    else:
                        # Truncate snippet to fit
                        chars_to_include = remaining_tokens * 4  # ~4 chars per token
                        budgeted_files[file_path] = snippet[:chars_to_include]
                        total_tokens = available_tokens
                        files_included_partial += 1
                        break  # Budget exhausted
                else:
                    files_excluded += 1
                    break  # Budget exhausted

        budget_info = {
            "available_tokens": available_tokens,
            "used_tokens": total_tokens,
            "utilization": f"{(total_tokens / available_tokens) * 100:.1f}%",
            "files_included_fully": files_included_fully,
            "files_included_partial": files_included_partial,
            "files_excluded": files_excluded,
            "total_files": len(sorted_files),
        }

        logger.info(
            f"Token budget managed: {total_tokens:,}/{available_tokens:,} tokens used ({budget_info['utilization']}), "
            f"{files_included_fully} full files, {files_included_partial} partial, {files_excluded} excluded"
        )

        return sorted_chunks, budgeted_files, budget_info

    async def _single_pass_synthesis(
        self,
        root_query: str,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        context: Any,
        synthesis_budgets: dict[str, int],
    ) -> str:
        """Perform single-pass synthesis with all aggregated data.

        Uses modern LLM large context windows to synthesize answer from complete
        data in one pass, avoiding information loss from progressive compression.

        Token Budget:
            - The max_output_tokens limit applies only to the LLM-generated content
            - A sources footer is appended AFTER synthesis (outside the token budget)
            - Total output = LLM content + sources footer (~100-500 tokens)
            - Footer size scales with number of files/chunks analyzed

        Args:
            root_query: Original research query
            chunks: All chunks from BFS traversal (will be filtered to match budgeted files)
            files: Budgeted file contents (subset within token limits)
            context: Research context
            synthesis_budgets: Dynamic budgets based on repository size

        Returns:
            Synthesized answer from single LLM call with appended sources footer
        """
        # Use output token budget from dynamic calculation
        max_output_tokens = synthesis_budgets["output_tokens"]

        # Filter chunks to only include those from budgeted files
        # This ensures consistency between reference map, citations, and footer
        original_chunk_count = len(chunks)
        budgeted_chunks = self._parent._citation_manager.filter_chunks_to_files(chunks, files)

        logger.info(
            f"Starting single-pass synthesis with {len(files)} files, "
            f"{len(budgeted_chunks)} chunks (filtered from {original_chunk_count} total, "
            f"output_limit={max_output_tokens:,})"
        )

        llm = self._llm_manager.get_synthesis_provider()

        # SAFETY NET: Final validation before synthesis
        # This should never happen due to earlier validations, but catch it just in case
        if not files:
            logger.error(
                f"Synthesis called with empty files dict despite {original_chunk_count} chunks. "
                "This indicates a bug in aggregation or budget management."
            )
            raise RuntimeError(
                f"Cannot synthesize answer: no code context available. "
                f"Found {original_chunk_count} chunks but received 0 files for synthesis. "
                f"This is a bug - earlier validation should have caught this. "
                f"Check aggregation and budget management logs."
            )

        # Build code context sections
        code_sections = []

        # Group chunks by file for better presentation
        chunks_by_file: dict[str, list[dict[str, Any]]] = {}
        for chunk in budgeted_chunks:
            file_path = chunk.get("file_path", "unknown")
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)

        # Build sections from files (already budgeted)
        for file_path, content in files.items():
            # If we have chunks for this file, build content with individual line markers
            if file_path in chunks_by_file:
                file_chunks = chunks_by_file[file_path]
                # Sort chunks by start_line for logical ordering
                sorted_chunks = sorted(
                    file_chunks, key=lambda c: c.get("start_line", 0)
                )

                # Build content with line markers for each chunk
                chunk_sections = []
                for chunk in sorted_chunks:
                    start_line = chunk.get("start_line", "?")
                    end_line = chunk.get("end_line", "?")
                    chunk_code = chunk.get("content", "")

                    # Add line marker before chunk code
                    chunk_sections.append(
                        f"# Lines {start_line}-{end_line}\n{chunk_code}"
                    )

                file_content = "\n\n".join(chunk_sections)
            else:
                # No chunks for this file, use full content from budget
                file_content = content

            code_sections.append(
                f"### {file_path}\n{'=' * 80}\n{file_content}\n{'=' * 80}"
            )

        code_context = "\n\n".join(code_sections)

        # Build file reference map for numbered citations
        file_reference_map = self._parent._build_file_reference_map(budgeted_chunks, files)
        reference_table = self._parent._format_reference_table(file_reference_map)

        # Default: curated but still detailed analysis.
        output_guidance = (
            f"**Target Output:** Provide a thorough and detailed analysis of approximately "
            f"{max_output_tokens:,} tokens (includes reasoning). Focus on all relevant "
            f"architectural layers, patterns, and implementation details with technical accuracy."
        )

        # Build comprehensive synthesis prompt (adapted from Code Expert methodology)
        system = prompts.SYNTHESIS_SYSTEM_BUILDER(output_guidance)

        prompt = prompts.SYNTHESIS_USER.format(
            root_query=root_query,
            reference_table=reference_table,
            code_context=code_context,
        )

        logger.info(
            f"Calling LLM for single-pass synthesis "
            f"(max_completion_tokens={max_output_tokens:,}, "
            f"timeout={SINGLE_PASS_TIMEOUT_SECONDS}s)"
        )

        await self._parent._emit_event(
            "llm_synthesis",
            f"Calling {llm.name}:{llm.model} for synthesis",
            max_completion_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=max_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,
        )
        duration_s = time.perf_counter() - started
        await self._parent._emit_event(
            "llm_synthesis_complete",
            f"Synthesis LLM call completed in {duration_s:.1f}s",
            duration=duration_s,
        )

        answer = response.content

        # Validate synthesis response
        answer_length = len(answer.strip()) if answer else 0
        logger.info(
            f"LLM synthesis response: length={answer_length}, "
            f"finish_reason={response.finish_reason}"
        )

        # Minimum threshold for valid synthesis (excluding footer)
        MIN_SYNTHESIS_LENGTH = 100  # Chars, not tokens

        if answer_length < MIN_SYNTHESIS_LENGTH:
            logger.error(
                f"Synthesis returned suspiciously short answer: {answer_length} chars "
                f"(minimum: {MIN_SYNTHESIS_LENGTH}, finish_reason={response.finish_reason})"
            )
            raise RuntimeError(
                f"LLM synthesis failed: generated only {answer_length} characters "
                f"(minimum: {MIN_SYNTHESIS_LENGTH}). finish_reason={response.finish_reason}. "
                "This indicates an LLM error, content filter, or model refusal."
            )

        # Append sources footer with file and chunk information
        try:
            footer = self._parent._build_sources_footer(budgeted_chunks, files, file_reference_map)
            if footer:
                answer = f"{answer}\n\n{footer}"
        except Exception as e:
            logger.warning(
                f"Failed to generate sources footer: {e}. Continuing without footer."
            )

        logger.info(
            f"Single-pass synthesis complete: {llm.estimate_tokens(answer):,} tokens generated"
        )

        return answer

    async def _cluster_sources_for_synthesis(
        self,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        synthesis_budgets: dict[str, int],
    ) -> tuple[list[ClusterGroup], dict[str, int]]:
        """Cluster files into token-bounded groups for map-reduce synthesis.

        Args:
            chunks: Prioritized chunks from BFS traversal
            files: Budgeted file contents
            synthesis_budgets: Dynamic budgets based on repository size

        Returns:
            Tuple of (cluster_groups, metadata)
        """
        logger.info(f"Clustering {len(files)} files for map-reduce synthesis")

        # Initialize clustering service
        embedding_provider = self._parent._embedding_manager.get_provider()
        llm_provider = self._llm_manager.get_synthesis_provider()
        clustering_service = ClusteringService(
            embedding_provider=embedding_provider,  # type: ignore[arg-type]
            llm_provider=llm_provider,
            max_tokens_per_cluster=MAX_TOKENS_PER_CLUSTER,
        )

        # Cluster the files
        cluster_groups, metadata = await clustering_service.cluster_files(files)

        logger.info(
            f"Clustered into {metadata['num_clusters']} groups, "
            f"avg {metadata['avg_tokens_per_cluster']:,} tokens/cluster"
        )

        return cluster_groups, metadata

    async def _map_synthesis_on_cluster(
        self,
        cluster: ClusterGroup,
        root_query: str,
        chunks: list[dict[str, Any]],
        synthesis_budgets: dict[str, int],
    ) -> dict[str, Any]:
        """Synthesize partial answer for one cluster of files.

        Args:
            cluster: Cluster group with files to synthesize
            root_query: Original research query
            chunks: All chunks (will be filtered to cluster files)
            synthesis_budgets: Dynamic budgets based on repository size

        Returns:
            Dictionary with:
                - cluster_id: int
                - summary: str (synthesized content for this cluster)
                - sources: list[dict] (files and chunks used)
        """
        # Filter chunks to only those in this cluster's files
        # This ensures consistency between reference map, citations, and cluster content
        original_chunk_count = len(chunks)
        cluster_chunks = self._parent._citation_manager.filter_chunks_to_files(chunks, cluster.files_content)

        logger.debug(
            f"Synthesizing cluster {cluster.cluster_id} "
            f"({len(cluster.file_paths)} files, {len(cluster_chunks)} chunks filtered from {original_chunk_count}, "
            f"{cluster.total_tokens:,} tokens)"
        )

        llm = self._llm_manager.get_synthesis_provider()

        # Build code context for this cluster (same logic as single-pass)
        code_sections = []
        chunks_by_file: dict[str, list[dict[str, Any]]] = {}
        for chunk in cluster_chunks:
            file_path = chunk.get("file_path", "unknown")
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)

        for file_path, content in cluster.files_content.items():
            if file_path in chunks_by_file:
                file_chunks = chunks_by_file[file_path]
                sorted_chunks = sorted(
                    file_chunks, key=lambda c: c.get("start_line", 0)
                )
                chunk_sections = []
                for chunk in sorted_chunks:
                    start_line = chunk.get("start_line", "?")
                    end_line = chunk.get("end_line", "?")
                    chunk_code = chunk.get("content", "")
                    chunk_sections.append(
                        f"# Lines {start_line}-{end_line}\n{chunk_code}"
                    )
                file_content = "\n\n".join(chunk_sections)
            else:
                file_content = content

            code_sections.append(
                f"### {file_path}\n{'=' * 80}\n{file_content}\n{'=' * 80}"
            )

        code_context = "\n\n".join(code_sections)

        # Build file reference map for numbered citations (cluster-specific)
        cluster_files = cluster.files_content
        file_reference_map = self._parent._build_file_reference_map(cluster_chunks, cluster_files)
        reference_table = self._parent._format_reference_table(file_reference_map)

        # Build cluster-specific synthesis prompt
        # Use smaller output budget per cluster (will be combined in reduce step)
        cluster_output_tokens = min(
            CLUSTER_OUTPUT_TOKEN_BUDGET, synthesis_budgets["output_tokens"] // 2
        )

        system = f"""You are analyzing a subset of code files as part of a larger codebase analysis.

Focus on:
1. Key architectural patterns and components in these files
2. Important implementation details and relationships
3. How these files contribute to answering the query

{prompts.CITATION_REQUIREMENTS}

Be thorough but concise - your analysis will be combined with other clusters.
Target output: ~{cluster_output_tokens:,} tokens (includes reasoning)."""

        prompt = f"""Query: {root_query}

{reference_table}

Analyze the following code files and provide insights relevant to the query above:

{code_context}

Provide a comprehensive analysis focusing on the query."""

        logger.debug(
            f"Calling LLM for cluster {cluster.cluster_id} synthesis "
            f"(max_completion_tokens={cluster_output_tokens:,}, "
            f"timeout={SINGLE_PASS_TIMEOUT_SECONDS}s)"
        )

        await self._parent._emit_event(
            "llm_synthesis",
            f"Calling {llm.name}:{llm.model} for cluster {cluster.cluster_id} synthesis",
            max_completion_tokens=cluster_output_tokens,
        )
        started = time.perf_counter()
        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=cluster_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,
        )
        duration_s = time.perf_counter() - started
        await self._parent._emit_event(
            "llm_synthesis_complete",
            f"Cluster {cluster.cluster_id} synthesis completed in {duration_s:.1f}s",
            duration=duration_s,
        )

        # Build sources list for this cluster
        sources = []
        for chunk in cluster_chunks:
            sources.append(
                {
                    "file_path": chunk.get("file_path"),
                    "start_line": chunk.get("start_line"),
                    "end_line": chunk.get("end_line"),
                }
            )

        logger.debug(
            f"Cluster {cluster.cluster_id} synthesis complete: "
            f"{llm.estimate_tokens(response.content):,} tokens generated"
        )

        return {
            "cluster_id": cluster.cluster_id,
            "summary": response.content,
            "sources": sources,
            "file_paths": cluster.file_paths,
            "file_reference_map": file_reference_map,
        }

    async def _reduce_synthesis(
        self,
        root_query: str,
        cluster_results: list[dict[str, Any]],
        all_chunks: list[dict[str, Any]],
        all_files: dict[str, str],
        synthesis_budgets: dict[str, int],
    ) -> str:
        """Combine cluster summaries into final answer.

        Args:
            root_query: Original research query
            cluster_results: Results from map step (cluster summaries)
            all_chunks: All chunks from clusters (will be filtered to match synthesized files)
            all_files: All files that were synthesized across clusters
            synthesis_budgets: Dynamic budgets based on repository size

        Returns:
            Final synthesized answer with sources footer
        """
        # Filter chunks to only include those from synthesized files
        # This ensures consistency between reference map, citations, and footer
        original_chunk_count = len(all_chunks)
        budgeted_chunks = self._parent._citation_manager.filter_chunks_to_files(all_chunks, all_files)

        logger.info(
            f"Reducing {len(cluster_results)} cluster summaries into final answer "
            f"({len(budgeted_chunks)} chunks filtered from {original_chunk_count})"
        )

        llm = self._llm_manager.get_synthesis_provider()
        max_output_tokens = synthesis_budgets["output_tokens"]

        # Build global file reference map for all clusters
        file_reference_map = self._parent._build_file_reference_map(budgeted_chunks, all_files)
        reference_table = self._parent._format_reference_table(file_reference_map)

        # Remap cluster-local citations to global reference numbers
        logger.info("Remapping cluster-local citations to global references")
        for result in cluster_results:
            cluster_file_map = result["file_reference_map"]
            original_summary = result["summary"]
            remapped_summary = self._parent._remap_cluster_citations(
                original_summary, cluster_file_map, file_reference_map
            )
            result["summary"] = remapped_summary

        # Combine all cluster summaries (now with global references)
        cluster_summaries = []
        for i, result in enumerate(cluster_results, 1):
            summary = result["summary"]
            file_paths = result["file_paths"]
            files = ", ".join(file_paths[:5])  # Show first 5 files
            if len(file_paths) > 5:
                remaining = len(file_paths) - 5
                files += f", ... (+{remaining} more)"

            cluster_summaries.append(
                f"## Cluster {i} Analysis\n**Files**: {files}\n\n{summary}"
            )

        combined_summaries = "\n\n" + "=" * 80 + "\n\n".join(cluster_summaries)

        # Build reduce prompt
        system = f"""You are synthesizing multiple partial analyses into a comprehensive final answer.

Your task:
1. Integrate insights from all cluster analyses
2. Eliminate redundancy and contradictions
3. Organize information coherently
4. Maintain focus on the original query
5. PRESERVE ALL reference number citations [N] from cluster analyses
   - Citation numbers have already been remapped to global references
   - Do NOT generate new citations (you don't have access to code)
   - DO preserve existing [N] citations when combining insights
   - Maintain citation density throughout the integrated answer

Target output: ~{max_output_tokens:,} tokens (includes reasoning)."""

        prompt = f"""Query: {root_query}

{reference_table}

You have been provided with analyses of different code clusters.
Synthesize these into a comprehensive, well-organized answer to the query.

NOTE: All citation numbers [N] in the cluster analyses have been remapped to match the global Source References table above. Simply preserve these citations as you integrate the analyses.

{combined_summaries}

Provide a complete, integrated analysis that addresses the original query."""

        logger.debug(
            f"Calling LLM for reduce synthesis "
            f"(max_completion_tokens={max_output_tokens:,})"
        )

        await self._parent._emit_event(
            "llm_synthesis",
            f"Calling {llm.name}:{llm.model} for reduce synthesis",
            max_completion_tokens=max_output_tokens,
        )
        started = time.perf_counter()
        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=max_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,  # type: ignore[call-arg]
        )
        duration_s = time.perf_counter() - started
        await self._parent._emit_event(
            "llm_synthesis_complete",
            f"Reduce synthesis completed in {duration_s:.1f}s",
            duration=duration_s,
        )

        answer = response.content

        # Validate minimum length
        MIN_SYNTHESIS_LENGTH = 100
        answer_length = len(answer.strip()) if answer else 0

        if answer_length < MIN_SYNTHESIS_LENGTH:
            logger.error(
                f"Reduce synthesis returned suspiciously short answer: {answer_length} chars"
            )
            raise RuntimeError(
                f"LLM reduce synthesis failed: generated only {answer_length} characters "
                f"(minimum: {MIN_SYNTHESIS_LENGTH}). finish_reason={response.finish_reason}."
            )

        # Validate citation references are valid
        invalid_citations = self._parent._validate_citation_references(answer, file_reference_map)
        if invalid_citations:
            logger.warning(
                f"Found {len(invalid_citations)} invalid citation references after reduce: "
                f"{invalid_citations[:10]}"
                + (f" ... and {len(invalid_citations) - 10} more" if len(invalid_citations) > 10 else "")
            )

        # Append sources footer (aggregate all sources from all clusters)
        try:
            footer = self._parent._build_sources_footer(budgeted_chunks, all_files, file_reference_map)
            if footer:
                answer = f"{answer}\n\n{footer}"
        except Exception as e:
            logger.warning(
                f"Failed to generate sources footer: {e}. Continuing without footer."
            )

        logger.info(
            f"Reduce synthesis complete: {llm.estimate_tokens(answer):,} tokens generated"
        )

        return answer
