"""Pluggable Research Service for ChunkHound.

This service orchestrates deep code research using a pluggable exploration strategy.
The exploration strategy (BFS, wide coverage, etc.) is injected via constructor,
enabling different algorithms to be swapped without changing the synthesis pipeline.

The service coordinates:
1. Initial search (unified semantic + symbol-based search)
2. Exploration (delegated to injected ExplorationStrategy)
3. Evidence extraction (constants, facts from clusters)
4. MAP-REDUCE synthesis (parallel cluster synthesis + final reduction)
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research.shared.chunk_dedup import get_chunk_id
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.evidence_ledger import (
    EvidenceLedger,
    extract_facts_with_clustering,
)
from chunkhound.services.research.shared.exploration import ExplorationStrategy
from chunkhound.services.research.shared.models import (
    ENABLE_SMART_BOUNDARIES,
    EXTRA_CONTEXT_TOKENS,
    MAX_BOUNDARY_EXPANSION_LINES,
    MAX_FILE_CONTENT_TOKENS,
    NUM_LLM_EXPANDED_QUERIES,
    OUTPUT_TOKENS_WITH_REASONING,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_TOKENS,
    TOKEN_BUDGET_PER_FILE,
    ResearchContext,
)
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.v1.quality_validator import QualityValidator
from chunkhound.services.research.v1.synthesis_engine import SynthesisEngine

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay
    from chunkhound.core.config.research_config import ResearchConfig


class PluggableResearchService:
    """Service for performing deep research with pluggable exploration strategies.

    This service orchestrates the research pipeline while delegating exploration
    to an injected ExplorationStrategy. The synthesis logic (MAP-REDUCE clustering
    and final answer generation) remains in this service.
    """

    def __init__(
        self,
        database_services: DatabaseServices,
        embedding_manager: EmbeddingManager,
        llm_manager: LLMManager,
        exploration_strategy: ExplorationStrategy,
        tool_name: str = "code_research",
        progress: "TreeProgressDisplay | None" = None,
        path_filter: str | None = None,
        config: "ResearchConfig | None" = None,
    ):
        """Initialize pluggable research service.

        Args:
            database_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
            llm_manager: LLM manager for generating follow-ups and synthesis
            exploration_strategy: Exploration strategy for chunk discovery (required).
                Uses strategy.explore() after initial search to expand coverage.
            tool_name: Name of the MCP tool (used in followup suggestions)
            progress: Optional TreeProgressDisplay instance for terminal UI (None for MCP)
            path_filter: Optional path filter to limit research scope
            config: Optional ResearchConfig for query expansion settings.
                If None, falls back to hardcoded constants.
        """
        self._db_services = database_services
        self._embedding_manager = embedding_manager
        self._llm_manager = llm_manager
        self._tool_name = tool_name
        self._node_counter = 0
        self.progress = progress  # Store progress instance for event emission
        self._progress_lock: asyncio.Lock = asyncio.Lock()
        self._synthesis_engine = SynthesisEngine(llm_manager, database_services, self)
        self._citation_manager = CitationManager()
        self._quality_validator = QualityValidator(llm_manager)
        self._path_filter = path_filter
        self._config = config
        self._unified_search_helper = UnifiedSearch(
            db_services=database_services,
            embedding_manager=embedding_manager,
            config=config,
        )

        # Store exploration strategy (required - pluggable algorithm for chunk discovery)
        self._exploration_strategy = exploration_strategy

    @property
    def _query_expansion_enabled(self) -> bool:
        """Get query expansion enabled setting from config or fallback to constant."""
        if self._config is not None:
            return self._config.query_expansion_enabled
        return QUERY_EXPANSION_ENABLED

    @property
    def _num_expanded_queries(self) -> int:
        """Get number of expanded queries from config or fallback to constant."""
        if self._config is not None:
            return self._config.num_expanded_queries
        return NUM_LLM_EXPANDED_QUERIES

    async def _emit_event(
        self,
        event_type: str,
        message: str,
        node_id: int | None = None,
        depth: int | None = None,
        **metadata: Any,
    ) -> None:
        """Emit a progress event with lock protection.

        Args:
            event_type: Event type identifier
            message: Human-readable event description
            node_id: Optional BFS node ID
            depth: Optional BFS depth level
            **metadata: Additional event data (chunks, files, tokens, etc.)
        """
        if not self.progress:
            return
        async with self._progress_lock:
            await self.progress.emit_event(
                event_type=event_type,
                message=message,
                node_id=node_id,
                depth=depth,
                metadata=metadata,
            )

    async def deep_research(self, query: str) -> dict[str, Any]:
        """Perform deep research on a query.

        Uses fixed BFS depth (max_depth=1) with dynamic synthesis budgets that scale
        based on repository size. Empirical evidence shows shallow exploration with
        comprehensive synthesis outperforms deep BFS traversal.

        Args:
            query: Research question to investigate

        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Starting deep research for query: '{query}'")

        # Emit main start event
        await self._emit_event("main_start", f"Starting deep research: {query[:60]}...")

        # Fixed max depth (empirically proven optimal)
        max_depth = 1
        logger.info(f"Using max_depth={max_depth} (fixed)")

        # Calculate synthesis budgets (output-only, input determined by elbow detection)
        synthesis_budgets = self._calculate_synthesis_budgets()
        logger.info(
            f"Synthesis output budget: {synthesis_budgets['output_tokens']:,} tokens"
        )

        # Emit configuration info
        await self._emit_event(
            "main_info",
            f"Max depth: {max_depth}, output budget: {synthesis_budgets['output_tokens'] // 1000}k tokens",
        )

        # Phase 1: Initial search
        context = ResearchContext(root_query=query)
        await self._emit_event(
            "depth_start",
            "Phase 1: Initial search",
            depth=0,
            nodes=1,
            max_depth=1,
        )

        initial_chunks = await self._unified_search(query, context, node_id=0, depth=0)

        logger.info(f"Initial search found {len(initial_chunks)} chunks")

        # Build evidence ledger for constants context (used in exploration)
        initial_evidence = EvidenceLedger.from_chunks(initial_chunks)
        constants_context = initial_evidence.get_constants_prompt_context()
        if initial_evidence.constants_count > 0:
            await self._emit_event(
                "evidence_ledger",
                f"Initial evidence: {initial_evidence.constants_count} constants",
                evidence_table=initial_evidence.format_progress_table(),
                constants_count=initial_evidence.constants_count,
                facts_count=initial_evidence.facts_count,
            )

        # Phase 2: Exploration via strategy
        await self._emit_event(
            "depth_start",
            f"Phase 2: Exploration ({self._exploration_strategy.name})",
            depth=1,
            nodes=1,
            max_depth=1,
        )

        # Use default threshold (0.0) since v1 doesn't use elbow detection for exploration cutoff
        phase1_threshold = 0.0

        (
            expanded_chunks,
            exploration_stats,
            file_contents,
        ) = await self._exploration_strategy.explore(
            root_query=query,
            initial_chunks=initial_chunks,
            phase1_threshold=phase1_threshold,
            path_filter=self._path_filter,
            constants_context=constants_context,
        )

        logger.info(
            f"Exploration complete: {exploration_stats.get('chunks_total', len(expanded_chunks))} total chunks, "
            f"{exploration_stats.get('nodes_explored', 0)} nodes explored, "
            f"{exploration_stats.get('files_read', len(file_contents))} files read"
        )

        # Aggregate chunks into synthesis format
        await self._emit_event(
            "synthesis_start", "Aggregating findings from exploration"
        )

        aggregated = self._aggregate_all_findings(expanded_chunks, file_contents)

        # Build evidence ledger from all aggregated chunks
        evidence_ledger = EvidenceLedger.from_chunks(aggregated.get("chunks", []))
        constants_context = evidence_ledger.get_constants_prompt_context()
        if evidence_ledger.constants_count > 0:
            await self._emit_event(
                "evidence_ledger",
                f"Evidence: {evidence_ledger.constants_count} constants",
                evidence_table=evidence_ledger.format_progress_table(),
                constants_count=evidence_ledger.constants_count,
                facts_count=evidence_ledger.facts_count,
            )

        # Early return: no context found (avoid scary synthesis error when empty)
        if not aggregated.get("chunks") and not aggregated.get("files"):
            logger.info(
                "No chunks or files aggregated; skipping synthesis and returning guidance"
            )
            await self._emit_event(
                "synthesis_skip",
                "No code context found; skipping synthesis",
                depth=0,
            )
            friendly = (
                f"No relevant code context found for: '{query}'.\n\n"
                "Try a more code-specific question. Helpful patterns:\n"
                "- Name files or modules (e.g., 'services/deep_research_service.py')\n"
                "- Mention classes/functions (e.g., 'DeepResearchService._single_pass_synthesis')\n"
                "- Include keywords that appear in code (constants, config keys)\n"
            )
            return {
                "answer": friendly,
                "metadata": {
                    "depth_reached": 0,
                    "nodes_explored": aggregated.get("stats", {}).get("total_nodes", 1),
                    "chunks_analyzed": 0,
                    "files_analyzed": 0,
                    "skipped_synthesis": True,
                },
            }

        # Pass pre-filtered chunks to synthesis (elbow detection done in exploration strategies)
        (
            prioritized_chunks,
            budgeted_files,
            selection_info,
        ) = await self._synthesis_engine._manage_token_budget_for_synthesis(
            aggregated["chunks"], aggregated["files"], query, synthesis_budgets
        )

        # Emit synthesizing event
        await self._emit_event(
            "synthesis_start",
            f"Synthesizing final answer ({selection_info['files_selected']} files, "
            f"{selection_info['total_tokens']:,} tokens, "
            f"{selection_info['chunks_count']} chunks)",
            chunks=len(prioritized_chunks),
            files=len(budgeted_files),
            input_tokens_used=selection_info["total_tokens"],
        )

        # Cluster files and extract facts in one pass (k-means with ~50k tokens/cluster)
        await self._emit_event(
            "fact_extraction",
            f"Clustering and extracting facts from {len(budgeted_files)} files",
            files=len(budgeted_files),
        )
        extraction_result = await extract_facts_with_clustering(
            files=budgeted_files,
            root_query=query,
            llm_provider=self._llm_manager.get_utility_provider(),
            embedding_provider=self._embedding_manager.get_provider(),
        )
        cluster_groups = extraction_result.cluster_groups
        cluster_metadata = extraction_result.cluster_metadata
        evidence_ledger = evidence_ledger.merge(extraction_result.evidence_ledger)

        # Update evidence ledger event with facts
        if evidence_ledger.facts_count > 0 or evidence_ledger.constants_count > 0:
            await self._emit_event(
                "evidence_ledger",
                f"Evidence: {evidence_ledger.constants_count} constants, {evidence_ledger.facts_count} facts",
                evidence_table=evidence_ledger.format_progress_table(),
                constants_count=evidence_ledger.constants_count,
                facts_count=evidence_ledger.facts_count,
            )

        # If only 1 cluster, use single-pass (no benefit from map-reduce)
        if cluster_metadata["num_clusters"] == 1:
            logger.info("Single cluster detected - using single-pass synthesis")
            facts_context = evidence_ledger.get_facts_reduce_prompt_context()
            answer = await self._synthesis_engine._single_pass_synthesis(
                root_query=query,
                chunks=prioritized_chunks,
                files=budgeted_files,
                context=context,
                synthesis_budgets=synthesis_budgets,
                constants_context=constants_context,
                facts_context=facts_context,
            )
        else:
            # Map-reduce synthesis with parallel execution
            logger.info(
                f"Multiple clusters detected - using map-reduce synthesis with "
                f"{cluster_metadata['num_clusters']} clusters"
            )

            # Get provider concurrency limit
            synthesis_provider = self._llm_manager.get_synthesis_provider()
            max_concurrency = synthesis_provider.get_synthesis_concurrency()
            logger.info(f"Using concurrency limit: {max_concurrency}")

            # Map step: Synthesize each cluster in parallel
            await self._emit_event(
                "synthesis_map",
                f"Synthesizing {cluster_metadata['num_clusters']} clusters in parallel "
                f"(concurrency={max_concurrency})",
            )

            semaphore = asyncio.Semaphore(max_concurrency)

            # Calculate total input tokens across all clusters for proportional budget allocation
            total_input_tokens = sum(cluster.total_tokens for cluster in cluster_groups)

            async def map_with_semaphore(cluster: ClusterGroup) -> dict[str, Any]:
                async with semaphore:
                    # Get cluster-specific facts context
                    cluster_files = set(cluster.file_paths)
                    cluster_facts_context = (
                        evidence_ledger.get_facts_map_prompt_context(cluster_files)
                    )
                    return await self._synthesis_engine._map_synthesis_on_cluster(
                        cluster,
                        query,
                        prioritized_chunks,
                        synthesis_budgets,
                        total_input_tokens,
                        constants_context=constants_context,
                        facts_context=cluster_facts_context,
                    )

            map_tasks = [map_with_semaphore(cluster) for cluster in cluster_groups]
            cluster_results = await asyncio.gather(*map_tasks)

            logger.info(
                f"Map step complete: {len(cluster_results)} cluster summaries generated"
            )

            # Reduce step: Combine cluster summaries
            await self._emit_event(
                "synthesis_reduce",
                f"Combining {len(cluster_results)} cluster summaries into final answer",
            )

            # Get global facts context for reduce phase
            reduce_facts_context = evidence_ledger.get_facts_reduce_prompt_context()

            answer = await self._synthesis_engine._reduce_synthesis(
                query,
                cluster_results,
                prioritized_chunks,
                budgeted_files,
                synthesis_budgets,
                constants_context=constants_context,
                facts_context=reduce_facts_context,
            )

        # Emit validating event
        await self._emit_event("synthesis_validate", "Validating output quality")

        # Validate output quality (conciseness, actionability)
        llm = self._llm_manager.get_utility_provider()
        target_tokens = llm.estimate_tokens(answer)
        answer, quality_warnings = self._quality_validator.validate_output_quality(
            answer, target_tokens
        )
        if quality_warnings:
            logger.warning("Quality issues detected:\n" + "\n".join(quality_warnings))

        # Validate citations in answer
        answer = self._quality_validator.validate_citations(answer, expanded_chunks)

        # Calculate metadata
        metadata = {
            "depth_reached": exploration_stats.get("depth_reached", 1),
            "nodes_explored": exploration_stats.get("nodes_explored", 1),
            "chunks_analyzed": len(expanded_chunks),
            "aggregation_stats": aggregated["stats"],
            "selection_info": selection_info,
        }

        logger.info(f"Deep research completed: {metadata}")

        # Emit completion event
        await self._emit_event(
            "main_complete",
            "Deep research complete",
            depth_reached=metadata["depth_reached"],
            nodes_explored=metadata["nodes_explored"],
            chunks_analyzed=metadata["chunks_analyzed"],
        )

        return {
            "answer": answer,
            "metadata": metadata,
        }

    def _build_search_query(self, query: str, context: ResearchContext) -> str:
        """Build search query combining input with BFS context.

        Evidence-based design (per research on embedding model position bias):
        - Current query FIRST (embedding models weight beginning 15-50% more heavily)
        - Minimal parent context (last 1-2 ancestors for disambiguation)
        - Clear separator to distinguish query from context
        - Root query implicitly preserved through ancestor chain

        Args:
            query: Current query
            context: Research context with ancestors

        Returns:
            Combined search query optimized for semantic search
        """
        if not context.ancestors:
            # Root node: just the query itself
            return query

        # For child nodes: prioritize current query, add minimal parent context
        # Take last 1-2 ancestors (not more to avoid redundancy)
        parent_context = (
            context.ancestors[-2:]
            if len(context.ancestors) >= 2
            else context.ancestors[-1:]
        )
        context_str = " → ".join(parent_context)

        # Current query FIRST (position bias optimization), then context
        return f"{query} | Context: {context_str}"

    async def _expand_query_with_llm(
        self, query: str, context: ResearchContext
    ) -> list[str]:
        """Expand query into multiple diverse semantic search queries.

        Uses LLM to generate different perspectives on the same question,
        improving recall by casting a wider semantic net.

        Args:
            query: Current query to expand
            context: Research context with root query and ancestors

        Returns:
            List of expanded queries (defaults to [query] if expansion fails)
        """
        llm = self._llm_manager.get_utility_provider()

        # Define JSON schema for structured output
        num_queries = self._num_expanded_queries
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Array of exactly {num_queries} expanded search queries (semantically complete sentences)",
                }
            },
            "required": ["queries"],
            "additionalProperties": False,
        }

        # Simplified system prompt per GPT-5-Nano best practices
        system = prompts.QUERY_EXPANSION_SYSTEM

        # Build context string
        context_str = ""
        if context.ancestors:
            ancestor_path = " → ".join(context.ancestors[-2:])
            context_str = f"\nPrior: {ancestor_path}"

        # Optimized prompt for semantic diversity
        prompt = prompts.QUERY_EXPANSION_USER.format(
            query=query,
            context_root_query=context.root_query,
            context_str=context_str,
            num_queries=num_queries,
        )

        logger.debug(
            f"Query expansion budget: {QUERY_EXPANSION_TOKENS:,} tokens (model: {llm.model})"
        )

        try:
            result = await llm.complete_structured(
                prompt=prompt,
                json_schema=schema,
                system=system,
                max_completion_tokens=QUERY_EXPANSION_TOKENS,
            )

            expanded = result.get("queries", [])

            # Validation: expect exactly num_queries from LLM
            if not expanded or len(expanded) < num_queries:
                logger.warning(
                    f"LLM returned {len(expanded) if expanded else 0} queries, expected {num_queries}, using original query only"
                )
                return [query]

            # Filter empty strings
            expanded = [q.strip() for q in expanded if q and q.strip()]

            # PREPEND ORIGINAL QUERY (new logic)
            # Original query goes first for position bias in embedding models
            final_queries = [query] + expanded[:num_queries]

            logger.debug(
                f"Expanded query into {len(final_queries)} variations: {final_queries}"
            )
            return final_queries

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query only")
            return [query]

    async def _unified_search(
        self,
        query: str,
        context: ResearchContext,
        node_id: int | None = None,
        depth: int | None = None,
    ) -> list[dict[str, Any]]:
        """Perform unified semantic + symbol-based regex search (Steps 2-7).

        Delegates to shared UnifiedSearch after handling v1-specific query expansion.

        Args:
            query: Search query
            context: Research context with root query and ancestors
            node_id: Optional BFS node ID for event emission
            depth: Optional BFS depth for event emission

        Returns:
            List of unified chunks
        """
        # Step 1: Query expansion (v1-specific - handled before delegation)
        expanded_queries = None
        if self._query_expansion_enabled:
            await self._emit_event(
                "query_expand", "Expanding query", node_id=node_id, depth=depth
            )
            expanded_queries = await self._expand_query_with_llm(query, context)
            await self._emit_event(
                "query_expand_complete",
                f"Expanded to {len(expanded_queries)} queries",
                node_id=node_id,
                depth=depth,
                queries=len(expanded_queries),
            )

        # Steps 2-7: Delegate to shared UnifiedSearch
        return await self._unified_search_helper.unified_search(
            query=query,
            context=context,
            expanded_queries=expanded_queries,
            rerank_queries=None,  # v1 uses single-query reranking
            emit_event_callback=self._emit_event,
            node_id=node_id,
            depth=depth,
            path_filter=self._path_filter,
        )

    def _expand_to_natural_boundaries(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
        chunk: dict[str, Any],
        file_path: str,
    ) -> tuple[int, int]:
        """Expand chunk boundaries to complete function/class definitions.

        Uses existing chunk metadata (symbol, kind) and language-specific heuristics
        to detect natural code boundaries instead of using fixed 50-line windows.

        Args:
            lines: File content split by lines
            start_line: Original chunk start line (1-indexed)
            end_line: Original chunk end line (1-indexed)
            chunk: Chunk metadata with symbol, kind fields
            file_path: File path for language detection

        Returns:
            Tuple of (expanded_start_line, expanded_end_line) in 1-indexed format,
            or (0, 0) if inputs are invalid.
        """
        # Validate 1-indexed inputs
        if (
            start_line < 1
            or end_line < 1
            or start_line > end_line
            or start_line > len(lines)
            or end_line > len(lines)
        ):
            return (0, 0)

        if not ENABLE_SMART_BOUNDARIES:
            # Fallback to legacy fixed-window behavior
            context_lines = EXTRA_CONTEXT_TOKENS // 20  # ~50 lines
            start_idx = max(1, start_line - context_lines)
            end_idx = min(len(lines), end_line + context_lines)
            return start_idx, end_idx

        # Check if chunk metadata indicates this is already a complete unit
        metadata = chunk.get("metadata", {})
        chunk_kind = metadata.get("kind") or chunk.get("symbol_type", "")

        # If this chunk is marked as a complete function/class/method, use its exact boundaries
        if chunk_kind in ("function", "method", "class", "interface", "struct", "enum"):
            # Chunk is already a complete unit - just add small padding for context
            padding = 3  # A few lines for docstrings/decorators/comments
            start_idx = max(1, start_line - padding)
            end_idx = min(len(lines), end_line + padding)
            logger.debug(
                f"Using complete {chunk_kind} boundaries: {file_path}:{start_idx}-{end_idx}"
            )
            return start_idx, end_idx

        # For non-complete chunks, expand to natural boundaries
        # Detect language from file extension for language-specific logic
        file_path_lower = file_path.lower()
        is_python = file_path_lower.endswith((".py", ".pyw"))
        is_brace_lang = file_path_lower.endswith(
            (
                ".c",
                ".cpp",
                ".cc",
                ".cxx",
                ".h",
                ".hpp",
                ".rs",
                ".go",
                ".java",
                ".js",
                ".ts",
                ".tsx",
                ".jsx",
                ".cs",
                ".swift",
                ".kt",
                ".scala",
            )
        )

        # Convert to 0-indexed for array access
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines) - 1, end_line - 1)

        # Expand backward to find function/class start
        expanded_start = start_idx
        if is_python:
            # Look for def/class keywords at start of line with proper indentation
            for i in range(start_idx - 1, max(0, start_idx - 200), -1):
                line = lines[i].strip()
                if line.startswith(("def ", "class ", "async def ")):
                    expanded_start = i
                    break
                # Stop at empty lines followed by significant dedents (module boundary)
                if not line and i > 0:
                    next_line = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
                    if next_line and not next_line.startswith((" ", "\t")):
                        break
        elif is_brace_lang:
            # Look for opening braces and function signatures
            brace_depth = 0
            for i in range(start_idx, max(0, start_idx - 200), -1):
                line = lines[i]
                # Count braces
                open_braces = line.count("{")
                close_braces = line.count("}")
                brace_depth += close_braces - open_braces

                # Found matching opening brace
                if brace_depth > 0 and "{" in line:
                    # Look backward for function signature
                    for j in range(i, max(0, i - 10), -1):
                        sig_line = lines[j].strip()
                        # Heuristic: function signatures often have (...) or start with keywords
                        if "(" in sig_line and (")" in sig_line or j < i):
                            expanded_start = j
                            break
                    if expanded_start != start_idx:
                        break

        # Expand forward to find function/class end
        expanded_end = end_idx
        if is_python:
            # Find end by detecting dedentation back to original level
            if expanded_start < len(lines):
                start_indent = len(lines[expanded_start]) - len(
                    lines[expanded_start].lstrip()
                )
                for i in range(end_idx + 1, min(len(lines), end_idx + 200)):
                    line = lines[i]
                    if line.strip():  # Non-empty line
                        line_indent = len(line) - len(line.lstrip())
                        # Dedented to same or less indentation = end of block
                        if line_indent <= start_indent:
                            expanded_end = i - 1
                            break
                else:
                    # Reached search limit, use current position
                    expanded_end = min(len(lines) - 1, end_idx + 50)
        elif is_brace_lang:
            # Find matching closing brace
            brace_depth = 0
            for i in range(expanded_start, min(len(lines), end_idx + 200)):
                line = lines[i]
                open_braces = line.count("{")
                close_braces = line.count("}")
                brace_depth += open_braces - close_braces

                # Found matching closing brace
                if brace_depth == 0 and i > expanded_start and "}" in line:
                    expanded_end = i
                    break

        # Safety: Don't expand beyond max limit
        if expanded_end - expanded_start > MAX_BOUNDARY_EXPANSION_LINES:
            logger.debug(
                f"Boundary expansion too large ({expanded_end - expanded_start} lines), "
                f"limiting to {MAX_BOUNDARY_EXPANSION_LINES}"
            )
            expanded_end = expanded_start + MAX_BOUNDARY_EXPANSION_LINES

        # Convert back to 1-indexed
        final_start = expanded_start + 1
        final_end = expanded_end + 1

        logger.debug(
            f"Expanded boundaries: {file_path}:{start_line}-{end_line} -> "
            f"{final_start}-{final_end} ({final_end - final_start} lines)"
        )

        return final_start, final_end

    async def _read_files_with_budget(
        self, chunks: list[dict[str, Any]], max_tokens: int | None = None
    ) -> dict[str, str]:
        """Read files containing chunks within token budget (Step 8).

        Per algorithm: Limit overall data to adaptive budget (or legacy MAX_FILE_CONTENT_TOKENS).

        Args:
            chunks: List of chunks
            max_tokens: Maximum tokens for file contents (uses adaptive budget if provided)

        Returns:
            Dictionary mapping file paths to contents (limited to budget)
        """
        # Group chunks by file
        files_to_chunks: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk.get("file_path") or chunk.get("path", "")
            if file_path:
                if file_path not in files_to_chunks:
                    files_to_chunks[file_path] = []
                files_to_chunks[file_path].append(chunk)

        # Use adaptive budget or fall back to legacy constant
        budget_limit = max_tokens if max_tokens is not None else MAX_FILE_CONTENT_TOKENS

        # Read files with budget (track total tokens per algorithm spec)
        file_contents: dict[str, str] = {}
        total_tokens = 0
        llm = self._llm_manager.get_utility_provider()

        # Get base directory for path resolution
        base_dir = self._db_services.provider.get_base_directory()

        for file_path, file_chunks in files_to_chunks.items():
            # Check if we've hit the overall token limit
            if total_tokens >= budget_limit:
                logger.debug(
                    f"Reached token limit ({budget_limit:,}), stopping file reading"
                )
                break

            try:
                # Resolve path relative to base directory
                if Path(file_path).is_absolute():
                    path = Path(file_path)
                else:
                    path = base_dir / file_path

                if not path.exists():
                    logger.warning(f"File not found (expected at {path}): {file_path}")
                    continue

                # Calculate token budget for this file
                num_chunks = len(file_chunks)
                budget = TOKEN_BUDGET_PER_FILE * num_chunks

                # Read file
                content = path.read_text(encoding="utf-8", errors="ignore")

                # Estimate tokens
                estimated_tokens = llm.estimate_tokens(content)

                if estimated_tokens <= budget:
                    # File fits in budget, check against overall limit
                    if total_tokens + estimated_tokens <= budget_limit:
                        file_contents[file_path] = content
                        total_tokens += estimated_tokens
                    else:
                        # Truncate to fit within overall limit
                        remaining_tokens = budget_limit - total_tokens
                        if remaining_tokens > 500:  # Only include if meaningful
                            chars_to_include = remaining_tokens * 4
                            file_contents[file_path] = content[:chars_to_include]
                            total_tokens = budget_limit
                        break
                else:
                    # File too large, extract chunks with smart boundary detection
                    chunk_contents = []
                    lines = content.split("\n")  # Pre-split for all chunks in this file

                    for chunk in file_chunks:
                        start_line = chunk.get("start_line", 1)
                        end_line = chunk.get("end_line", 1)

                        # Use smart boundary detection to expand to complete functions/classes
                        expanded_start, expanded_end = (
                            self._expand_to_natural_boundaries(
                                lines, start_line, end_line, chunk, file_path
                            )
                        )

                        # Skip chunks with invalid boundary expansion
                        if expanded_start == 0 and expanded_end == 0:
                            logger.warning(
                                f"Skipping chunk with invalid boundaries: "
                                f"{file_path}:{start_line}-{end_line}"
                            )
                            continue

                        # Store expanded range in chunk for later deduplication
                        chunk["expanded_start_line"] = expanded_start
                        chunk["expanded_end_line"] = expanded_end

                        # Extract chunk with smart boundaries (convert 1-indexed to 0-indexed)
                        start_idx = max(0, expanded_start - 1)
                        end_idx = min(len(lines), expanded_end)

                        chunk_with_context = "\n".join(lines[start_idx:end_idx])
                        chunk_contents.append(chunk_with_context)

                    combined_chunks = "\n\n...\n\n".join(chunk_contents)
                    chunk_tokens = llm.estimate_tokens(combined_chunks)

                    # Check against overall token limit
                    if total_tokens + chunk_tokens <= budget_limit:
                        file_contents[file_path] = combined_chunks
                        total_tokens += chunk_tokens
                    else:
                        # Truncate to fit
                        remaining_tokens = budget_limit - total_tokens
                        if remaining_tokens > 500:
                            chars_to_include = remaining_tokens * 4
                            file_contents[file_path] = combined_chunks[
                                :chars_to_include
                            ]
                            total_tokens = budget_limit
                        break

            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

        # FAIL-FAST: Validate that at least some files were loaded if chunks were provided
        # This prevents silent data loss where searches find chunks but synthesis gets no code
        if chunks and not file_contents:
            raise RuntimeError(
                f"DATA LOSS DETECTED: Found {len(chunks)} chunks across {len(files_to_chunks)} files "
                f"but failed to read ANY file contents. "
                f"Possible causes: "
                f"(1) Token budget exhausted ({budget_limit:,} tokens insufficient), "
                f"(2) Files not found at base_directory: {base_dir}, "
                f"(3) All file read operations failed. "
                f"Check logs above for file-specific errors."
            )

        logger.debug(
            f"File reading complete: Loaded {len(file_contents)} files with {total_tokens:,} tokens "
            f"(limit: {budget_limit:,})"
        )
        return file_contents

    def _is_file_fully_read(self, file_content: str) -> bool:
        """Detect if file_content is full file vs partial chunks.

        Heuristic: Partial reads have "..." separator between chunks.

        Args:
            file_content: Content from file_contents dict

        Returns:
            True if full file was read, False if partial chunks
        """
        return "\n\n...\n\n" not in file_content

    def _get_chunk_expanded_range(self, chunk: dict[str, Any]) -> tuple[int, int]:
        """Get expanded line range for chunk.

        If expansion already computed and stored in chunk, return it.
        Otherwise, re-compute using _expand_to_natural_boundaries().

        Args:
            chunk: Chunk dictionary with metadata

        Returns:
            Tuple of (expanded_start_line, expanded_end_line) in 1-indexed format
        """
        # Check if already stored (after enhancement in _read_files_with_budget)
        if "expanded_start_line" in chunk and "expanded_end_line" in chunk:
            return (chunk["expanded_start_line"], chunk["expanded_end_line"])

        # Re-compute (fallback)
        file_path = chunk.get("file_path")
        start_line = chunk.get("start_line", 0)
        end_line = chunk.get("end_line", 0)

        if not file_path or not start_line or not end_line:
            return (start_line, end_line)

        # Read file lines
        try:
            base_dir = self._db_services.provider.get_base_directory()
            if Path(file_path).is_absolute():
                path = Path(file_path)
            else:
                path = base_dir / file_path

            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            logger.debug(f"Could not re-read file for expansion: {file_path}: {e}")
            return (start_line, end_line)

        expanded_start, expanded_end = self._expand_to_natural_boundaries(
            lines, start_line, end_line, chunk, file_path
        )

        # Fallback to original range if expansion fails
        if expanded_start == 0 and expanded_end == 0:
            logger.warning(
                f"Boundary expansion failed for {file_path}, using original range"
            )
            return (start_line, end_line)

        return (expanded_start, expanded_end)

    def _aggregate_all_findings(
        self, chunks: list[dict[str, Any]], file_contents: dict[str, str]
    ) -> dict[str, Any]:
        """Aggregate chunks from exploration into synthesis format.

        Deduplicates chunks by chunk_id and passes through pre-read file contents.

        Args:
            chunks: Flat list of chunks from exploration
            file_contents: Pre-read file contents from exploration strategy

        Returns:
            Dictionary with:
                - chunks: List of unique chunks (deduplicated by chunk_id)
                - files: Pre-read file contents
                - stats: Statistics about aggregation
        """
        logger.info(
            f"Aggregating {len(chunks)} chunks and {len(file_contents)} files from exploration"
        )

        # Deduplicate chunks by chunk_id
        chunks_map: dict[int | str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = get_chunk_id(chunk)
            if chunk_id and chunk_id not in chunks_map:
                chunks_map[chunk_id] = chunk

        unique_chunks = list(chunks_map.values())

        stats = {
            "unique_chunks": len(unique_chunks),
            "unique_files": len(file_contents),
            "total_chunks_input": len(chunks),
            "deduplication_ratio_chunks": (
                f"{len(chunks) / len(unique_chunks):.2f}x" if unique_chunks else "N/A"
            ),
        }

        logger.info(
            f"Aggregation complete: {stats['unique_chunks']} unique chunks from "
            f"{stats['unique_files']} files"
        )

        return {
            "chunks": unique_chunks,
            "files": file_contents,
            "stats": stats,
        }

    def _calculate_synthesis_budgets(self) -> dict[str, int]:
        """Calculate synthesis token budgets.

        Output budget is FIXED at 30k tokens for reasoning models (includes thinking + output).
        Input budget is determined by elbow detection (relevance-based filtering), not repo size.

        Returns:
            Dictionary with output_tokens (fixed at 30k for LLM output limit)
        """
        logger.debug(
            f"Synthesis budget: output={OUTPUT_TOKENS_WITH_REASONING:,} tokens (fixed)"
        )

        return {
            "output_tokens": OUTPUT_TOKENS_WITH_REASONING,
        }

    def _get_next_node_id(self) -> int:
        """Get next node ID for graph traversal."""
        self._node_counter += 1
        return self._node_counter
