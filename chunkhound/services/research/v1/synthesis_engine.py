"""Synthesis Engine for Deep Research Service.

This module contains the synthesis logic for combining search results into
comprehensive answers. It implements both single-pass and map-reduce synthesis
strategies.

Architecture:
    - Single-pass synthesis: For small to medium result sets that fit in context
    - Map-reduce synthesis: For large result sets requiring clustering
    - Elbow detection: Relevance filtering via score distribution analysis
    - Citation system: Numbered file references with validation

The synthesis engine uses:
    - LLM providers for answer generation
    - Elbow detection for relevance filtering
    - Citation management for source tracking
"""

from typing import Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research.shared.evidence_ledger import (
    CONSTANTS_INSTRUCTION_FULL,
    CONSTANTS_INSTRUCTION_SHORT,
)
from chunkhound.services.research.shared.models import (
    SINGLE_PASS_TIMEOUT_SECONDS,
    TARGET_OUTPUT_TOKENS,
    build_output_guidance,
)

# Minimum characters for a valid synthesis answer (not tokens)
MIN_SYNTHESIS_LENGTH = 100


class SynthesisEngine:
    """Engine for synthesizing research results into comprehensive answers.

    The SynthesisEngine coordinates the synthesis phase of deep research,
    managing file selection and LLM calls for answer generation.

    Key responsibilities:
        1. File selection: Include all files from elbow-filtered chunks
        2. Single-pass synthesis: Generate answers in one LLM call
        3. Map-reduce synthesis: Cluster and combine for large result sets
        4. Citation management: Build and validate numbered file references

    Architecture:
        - Input selection: Elbow detection filters irrelevant chunks (no token cap)
        - Output tokens: Fixed budget for LLM call limits
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
        """Pass through pre-filtered chunks and files for synthesis.

        Filtering (elbow detection, reranking) is now done in the exploration
        strategy before data reaches synthesis. This method simply calculates
        token counts for logging/metadata purposes.

        Args:
            chunks: Pre-filtered chunks from exploration strategy
            files: Pre-filtered file contents from exploration strategy
            root_query: Original research query (unused, kept for API compat)
            synthesis_budgets: Dynamic budgets (unused, kept for API compat)

        Returns:
            Tuple of (chunks, files, selection_info)
        """
        llm = self._llm_manager.get_utility_provider()

        # Calculate total tokens for metadata
        total_tokens = sum(llm.estimate_tokens(content) for content in files.values())

        selection_info = {
            "files_selected": len(files),
            "total_tokens": total_tokens,
            "chunks_count": len(chunks),
        }

        logger.info(
            f"Synthesis input: {len(files)} files, {len(chunks)} chunks, "
            f"{total_tokens:,} tokens"
        )

        return chunks, files, selection_info

    async def _single_pass_synthesis(
        self,
        root_query: str,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        context: Any,
        synthesis_budgets: dict[str, int],
        constants_context: str = "",
        facts_context: str = "",
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
            constants_context: Constants ledger context for LLM prompts
            facts_context: Facts ledger context for LLM prompts

        Returns:
            Synthesized answer from single LLM call with appended sources footer
        """
        # Use output token budget from dynamic calculation
        max_output_tokens = synthesis_budgets["output_tokens"]

        # Filter chunks to only include those from budgeted files
        # This ensures consistency between reference map, citations, and footer
        original_chunk_count = len(chunks)
        budgeted_chunks = self._parent._citation_manager.filter_chunks_to_files(
            chunks, files
        )

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
        file_reference_map = self._parent._citation_manager.build_file_reference_map(
            budgeted_chunks, files
        )
        reference_table = self._parent._citation_manager.format_reference_table(
            file_reference_map
        )

        # Build constants context section
        constants_section = ""
        if constants_context:
            constants_section = f"\n\n{constants_context}\n\n{CONSTANTS_INSTRUCTION_FULL}"

        # Build facts context section
        facts_section = ""
        if facts_context:
            facts_section = f"\n\n{facts_context}"

        # Build output guidance using shared utility (targets ~15k for concise output)
        output_guidance = build_output_guidance(TARGET_OUTPUT_TOKENS)

        # Build comprehensive synthesis prompt (adapted from Code Expert methodology)
        system = prompts.SYNTHESIS_SYSTEM_BUILDER(output_guidance)

        # Combine root_query with constants and facts context
        query_with_context = root_query + constants_section + facts_section

        prompt = prompts.SYNTHESIS_USER.format(
            root_query=query_with_context,
            reference_table=reference_table,
            code_context=code_context,
        )

        logger.info(
            f"Calling LLM for single-pass synthesis "
            f"(max_completion_tokens={max_output_tokens:,}, "
            f"timeout={SINGLE_PASS_TIMEOUT_SECONDS}s)"
        )

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=max_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,
        )

        answer = response.content

        # Validate synthesis response
        answer_length = len(answer.strip()) if answer else 0
        logger.info(
            f"LLM synthesis response: length={answer_length}, "
            f"finish_reason={response.finish_reason}"
        )

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
            footer = self._parent._citation_manager.build_sources_footer(
                budgeted_chunks, files, file_reference_map
            )
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

    async def _map_synthesis_on_cluster(
        self,
        cluster: ClusterGroup,
        root_query: str,
        chunks: list[dict[str, Any]],
        synthesis_budgets: dict[str, int],
        total_input_tokens: int,
        constants_context: str = "",
        facts_context: str = "",
    ) -> dict[str, Any]:
        """Synthesize partial answer for one cluster of files.

        Args:
            cluster: Cluster group with files to synthesize
            root_query: Original research query
            chunks: All chunks (will be filtered to cluster files)
            synthesis_budgets: Dynamic budgets based on repository size
            total_input_tokens: Sum of all cluster tokens (for proportional budget allocation)
            constants_context: Constants ledger context for LLM prompts
            facts_context: Facts ledger context for LLM prompts

        Returns:
            Dictionary with:
                - cluster_id: int
                - summary: str (synthesized content for this cluster)
                - sources: list[dict] (files and chunks used)
        """
        # Filter chunks to only those in this cluster's files
        # This ensures consistency between reference map, citations, and cluster content
        original_chunk_count = len(chunks)
        cluster_chunks = self._parent._citation_manager.filter_chunks_to_files(
            chunks, cluster.files_content
        )

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
        file_reference_map = self._parent._citation_manager.build_file_reference_map(
            cluster_chunks, cluster_files
        )
        reference_table = self._parent._citation_manager.format_reference_table(
            file_reference_map
        )

        # Build cluster-specific synthesis prompt
        # Proportional output budget: each cluster gets output proportional to its input share
        # This ensures larger clusters (more code) get more output tokens
        cluster_proportion = (
            cluster.total_tokens / total_input_tokens if total_input_tokens > 0 else 1.0
        )
        cluster_output_tokens = max(5000, int(total_input_tokens * cluster_proportion))
        # Cap cluster target to half of TARGET_OUTPUT_TOKENS (clusters combine in reduce)
        cluster_target = min(cluster_output_tokens, TARGET_OUTPUT_TOKENS // 2)

        # Build constants section if available
        constants_section = ""
        if constants_context:
            constants_section = f"\n{constants_context}\n\n{CONSTANTS_INSTRUCTION_SHORT}"

        # Build facts section if available
        facts_section = ""
        if facts_context:
            facts_section = f"\n{facts_context}"

        system = f"""You are analyzing a subset of code files as part of a larger codebase analysis.

Focus on:
1. Key architectural patterns and components in these files
2. Important implementation details and relationships
3. How these files contribute to answering the query

{prompts.CITATION_REQUIREMENTS}

Be thorough but concise - your analysis will be combined with other clusters.
{build_output_guidance(cluster_target)}"""

        prompt = f"""Query: {root_query}
{constants_section}{facts_section}
{reference_table}

Analyze the following code files and provide insights relevant to the query above:

{code_context}

Provide a comprehensive analysis focusing on the query."""

        logger.debug(
            f"Calling LLM for cluster {cluster.cluster_id} synthesis "
            f"(max_completion_tokens={cluster_output_tokens:,}, "
            f"timeout={SINGLE_PASS_TIMEOUT_SECONDS}s)"
        )

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=cluster_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,
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
        constants_context: str = "",
        facts_context: str = "",
    ) -> str:
        """Combine cluster summaries into final answer.

        The reduce step receives variable-sized input from cluster map phases.
        With proportional cluster outputs, input can range from 50k to 150k+ tokens
        depending on cluster sizes. The LLM is informed of this context to help
        it understand the scope of integration work.

        Token Budget Strategy:
            - Input: Variable (no cap) - whatever the clusters produced
            - Output: Fixed 30k tokens - sufficient for comprehensive final answer
            - The LLM is told both values to help it prioritize content

        Args:
            root_query: Original research query
            cluster_results: Results from map step (cluster summaries)
            all_chunks: All chunks from clusters (will be filtered to match synthesized files)
            all_files: All files that were synthesized across clusters
            synthesis_budgets: Dynamic budgets based on repository size
            constants_context: Constants ledger context for LLM prompts
            facts_context: Facts ledger context for LLM prompts

        Returns:
            Final synthesized answer with sources footer
        """
        # Filter chunks to only include those from synthesized files
        # This ensures consistency between reference map, citations, and footer
        original_chunk_count = len(all_chunks)
        budgeted_chunks = self._parent._citation_manager.filter_chunks_to_files(
            all_chunks, all_files
        )

        llm = self._llm_manager.get_synthesis_provider()
        max_output_tokens = synthesis_budgets["output_tokens"]

        # Calculate actual input size from cluster summaries
        # This helps inform the LLM about the scope of integration work
        total_input_tokens = sum(
            llm.estimate_tokens(result["summary"]) for result in cluster_results
        )

        logger.info(
            f"Reducing {len(cluster_results)} cluster summaries into final answer "
            f"(input: {total_input_tokens:,} tokens, output budget: {max_output_tokens:,} tokens, "
            f"{len(budgeted_chunks)} chunks filtered from {original_chunk_count})"
        )

        # Build global file reference map for all clusters
        file_reference_map = self._parent._citation_manager.build_file_reference_map(
            budgeted_chunks, all_files
        )
        reference_table = self._parent._citation_manager.format_reference_table(
            file_reference_map
        )

        # Remap cluster-local citations to global reference numbers
        logger.info("Remapping cluster-local citations to global references")
        for result in cluster_results:
            cluster_file_map = result["file_reference_map"]
            original_summary = result["summary"]
            remapped_summary = self._parent._citation_manager.remap_cluster_citations(
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

        combined_summaries = "\n\n" + "=" * 80 + "\n\n" + "\n\n".join(cluster_summaries)

        # Build constants section if available
        constants_section = ""
        if constants_context:
            constants_section = f"\n\n{constants_context}\n\n{CONSTANTS_INSTRUCTION_FULL}"

        # Build facts section if available
        facts_section = ""
        if facts_context:
            facts_section = f"\n\n{facts_context}"

        # Build reduce prompt with input size context
        # The LLM needs to know how much content it's integrating to prioritize effectively
        system = f"""You are integrating {total_input_tokens:,} tokens of cluster analyses into a final answer.

Context:
- You are synthesizing {len(cluster_results)} cluster analyses totaling ~{total_input_tokens:,} tokens
- Prioritize the most important insights when consolidating

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

{build_output_guidance(TARGET_OUTPUT_TOKENS)}"""

        prompt = f"""Query: {root_query}
{constants_section}{facts_section}
{reference_table}

You have been provided with analyses of different code clusters.
Synthesize these into a comprehensive, well-organized answer to the query.

NOTE: All citation numbers [N] in the cluster analyses have been remapped to match the global Source References table above. Simply preserve these citations as you integrate the analyses.

{combined_summaries}

Provide a complete, integrated analysis that addresses the original query."""

        logger.debug(
            f"Calling LLM for reduce synthesis "
            f"(input: {total_input_tokens:,} tokens, max_completion_tokens={max_output_tokens:,})"
        )

        response = await llm.complete(
            prompt,
            system=system,
            max_completion_tokens=max_output_tokens,
            timeout=SINGLE_PASS_TIMEOUT_SECONDS,
        )

        answer = response.content

        # Validate minimum length
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
        invalid_citations = self._parent._citation_manager.validate_citation_references(
            answer, file_reference_map
        )
        if invalid_citations:
            logger.warning(
                f"Found {len(invalid_citations)} invalid citation references after reduce: "
                f"{invalid_citations[:10]}"
                + (
                    f" ... and {len(invalid_citations) - 10} more"
                    if len(invalid_citations) > 10
                    else ""
                )
            )

        # Append sources footer (aggregate all sources from all clusters)
        try:
            footer = self._parent._citation_manager.build_sources_footer(
                budgeted_chunks, all_files, file_reference_map
            )
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
