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
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.clustering_service import ClusterGroup
from chunkhound.services.research.shared.citation_manager import CitationManager
from chunkhound.services.research.shared.evidence_ledger import (
    EvidenceLedger,
    extract_facts_with_clustering,
)
from chunkhound.services.research.shared.exploration import ExplorationStrategy
from chunkhound.services.research.shared.models import ResearchContext
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.v1.quality_validator import QualityValidator
from chunkhound.services.research.v1.question_generator import QuestionGenerator
from chunkhound.services.research.v1.synthesis_engine import SynthesisEngine

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay

# Constants
NODE_SIMILARITY_THRESHOLD = (
    0.2  # Reserved for future similarity-based deduplication (currently uses LLM)
)
MAX_FOLLOWUP_QUESTIONS = 3
MAX_SYMBOLS_TO_SEARCH = 5  # Top N symbols to search via regex (from spec)
QUERY_EXPANSION_ENABLED = True  # Enable LLM-powered query expansion for better recall
NUM_LLM_EXPANDED_QUERIES = 2  # LLM generates 2 queries, we prepend original = 3 total

# Adaptive token budgets (depth-dependent)
ENABLE_ADAPTIVE_BUDGETS = True  # Enable depth-based adaptive budgets

# File content budget range (input: what LLM sees for code)
FILE_CONTENT_TOKENS_MIN = 10_000  # Root nodes (synthesizing, need less raw code)
FILE_CONTENT_TOKENS_MAX = 50_000  # Leaf nodes (analyzing, need full implementations)

# LLM total input budget range (query + context + code)
LLM_INPUT_TOKENS_MIN = 15_000  # Root nodes
LLM_INPUT_TOKENS_MAX = 60_000  # Leaf nodes

# Leaf answer output budget (what LLM generates at leaves)
# NOTE: Reduced from 30k to balance cost vs quality. If you observe:
#   - Frequent "Missing: [detail]" statements
#   - Theoretical placeholders ("provide exact values")
#   - Incomplete analysis of complex components
# Consider increasing these values. Quality validation warnings will indicate budget pressure.
LEAF_ANSWER_TOKENS_BASE = (
    18_000  # Base budget for leaf nodes (was 30k, reduced for cost)
)
LEAF_ANSWER_TOKENS_BONUS = (
    3_000  # Additional tokens for deeper leaves (was 5k, reduced for cost)
)

# Internal synthesis output budget (what LLM generates at internal nodes)
# NOTE: Reduced from 17.5k/32k to balance cost vs quality. If root synthesis appears rushed or
# omits critical architectural details, consider increasing INTERNAL_ROOT_TARGET.
INTERNAL_ROOT_TARGET = 11_000  # Root synthesis target (was 17.5k, reduced for cost)
INTERNAL_MAX_TOKENS = (
    19_000  # Maximum for deep internal nodes (was 32k, reduced for cost)
)

# Follow-up question generation output budget (what LLM generates for follow-up questions)
# NOTE: High budgets needed for reasoning models (o1/o3/GPT-5) which use internal "thinking" tokens
# WHY: Reasoning models consume 5-15k tokens for internal reasoning before producing 100-500 tokens of output
# The actual generated questions are concise, but the model needs reasoning budget to evaluate relevance
FOLLOWUP_OUTPUT_TOKENS_MIN = (
    8_000  # Root/shallow nodes: simpler questions, less reasoning needed
)
FOLLOWUP_OUTPUT_TOKENS_MAX = (
    15_000  # Deep nodes: complex synthesis requires more reasoning depth
)

# Utility operation output budgets (for reasoning models like o1/o3/GPT-5)
# These operations use utility provider and don't vary by depth
# WHY: Each utility operation produces small output but requires reasoning budget for quality
QUERY_EXPANSION_TOKENS = (
    10_000  # Generate 2 queries (~200 output + ~8k reasoning to ensure diversity)
)
QUESTION_SYNTHESIS_TOKENS = (
    15_000  # Synthesize to 1-3 questions (~500 output + ~12k reasoning for quality)
)
QUESTION_FILTERING_TOKENS = (
    5_000  # Filter by relevance (~50 output + ~4k reasoning for accuracy)
)

# Legacy constants (used when ENABLE_ADAPTIVE_BUDGETS = False)
TOKEN_BUDGET_PER_FILE = 4000
EXTRA_CONTEXT_TOKENS = 1000
MAX_FILE_CONTENT_TOKENS = 3000
MAX_LLM_INPUT_TOKENS = 5000
MAX_LEAF_ANSWER_TOKENS = 400
MAX_SYNTHESIS_TOKENS = 600

# Single-pass synthesis constants (new architecture)
SINGLE_PASS_MAX_TOKENS = (
    150_000  # Total budget for single-pass synthesis (input + output)
)
OUTPUT_TOKENS_WITH_REASONING = 30_000  # Fixed output budget for reasoning models (18k output + 12k reasoning buffer)
SINGLE_PASS_OVERHEAD_TOKENS = 5_000  # Prompt template and overhead
SINGLE_PASS_TIMEOUT_SECONDS = 600  # 10 minutes timeout for large synthesis calls
# Available for code/chunks: Scales dynamically with repo size (30k-150k input tokens)

# Target output length (controlled via prompt instructions, not API token limits)
# WHY: OUTPUT_TOKENS_WITH_REASONING is FIXED at 30k for all queries (reasoning models need this)
# This allows reasoning models to use thinking tokens while producing appropriately sized output
# NOTE: Only INPUT budget scales dynamically based on repository size, output is fixed
TARGET_OUTPUT_TOKENS = 15_000  # Default target for standard research outputs

# NOTE: Synthesis input budget scaling (CHUNKS_TO_LOC_ESTIMATE, LOC_THRESHOLD_*, SYNTHESIS_INPUT_TOKENS_*)
# has been removed. Elbow detection now determines the relevance cutoff for chunks, providing
# data-driven filtering based on actual score distributions rather than repository size heuristics.
# See: chunkhound/services/research/shared/elbow_detection.py

# Output control
REQUIRE_CITATIONS = True  # Validate file:line format

# Map-reduce synthesis constants
MAX_TOKENS_PER_CLUSTER = 30_000  # Token budget per cluster for parallel synthesis
CLUSTER_OUTPUT_TOKEN_BUDGET = 15_000  # Max output tokens per cluster summary

# Pre-compiled regex patterns for citation processing
_CITATION_PATTERN = re.compile(r"\[\d+\]")  # Matches [N] citations
_CITATION_SEQUENCE_PATTERN = re.compile(
    r"(?:\[\d+\])+"
)  # Matches sequences like [1][2][3]

# Smart boundary detection for context-aware file reading
ENABLE_SMART_BOUNDARIES = True  # Expand to natural code boundaries (functions/classes)
MAX_BOUNDARY_EXPANSION_LINES = 300  # Maximum lines to expand for complete functions

# File-level reranking for synthesis budget allocation
# Prevents file diversity collapse where deep BFS exploration causes score accumulation in few files
MAX_CHUNKS_PER_FILE_REPR = (
    5  # Top chunks to include in file representative document for reranking
)
MAX_TOKENS_PER_FILE_REPR = 2000  # Token limit for file representative document


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
        """
        self._db_services = database_services
        self._embedding_manager = embedding_manager
        self._llm_manager = llm_manager
        self._tool_name = tool_name
        self._node_counter = 0
        self.progress = progress  # Store progress instance for event emission
        self._progress_lock: asyncio.Lock | None = (
            None  # Lazy init for concurrent progress updates
        )
        self._progress_lock_init = (
            threading.Lock()
        )  # Thread-safe guard for lock creation
        self._synthesis_engine = SynthesisEngine(llm_manager, database_services, self)
        self._question_generator = QuestionGenerator(llm_manager)
        self._citation_manager = CitationManager()
        self._quality_validator = QualityValidator(llm_manager)
        self._path_filter = path_filter
        self._unified_search_helper = UnifiedSearch(
            db_services=database_services,
            embedding_manager=embedding_manager,
            config=None,  # v1 doesn't use ResearchConfig
        )

        # Store exploration strategy (required - pluggable algorithm for chunk discovery)
        self._exploration_strategy = exploration_strategy

    async def _ensure_progress_lock(self) -> None:
        """Ensure progress lock exists (must be called in async event loop context).

        Lazy initialization pattern: Lock is created on first use to ensure it's created
        in the event loop context, avoiding RuntimeError from asyncio.Lock() in __init__.

        Uses double-checked locking to prevent race conditions where multiple concurrent
        tasks could create separate locks.
        """
        if self.progress and self._progress_lock is None:
            with self._progress_lock_init:  # Thread-safe initialization guard
                if self._progress_lock is None:  # Double-check inside lock
                    self._progress_lock = asyncio.Lock()

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
        await self._ensure_progress_lock()
        assert self._progress_lock is not None
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

        initial_chunks = await self._unified_search(
            query, context, node_id=0, depth=0
        )

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

        expanded_chunks, exploration_stats, file_contents = await self._exploration_strategy.explore(
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
        await self._emit_event("synthesis_start", "Aggregating findings from exploration")

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

        # Filter and prioritize files using elbow detection
        (
            prioritized_chunks,
            budgeted_files,
            selection_info,
        ) = await self._manage_token_budget_for_synthesis(
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
            answer = await self._single_pass_synthesis(
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
                    cluster_facts_context = evidence_ledger.get_facts_map_prompt_context(
                        cluster_files
                    )
                    return await self._map_synthesis_on_cluster(
                        cluster, query, prioritized_chunks, synthesis_budgets,
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

            answer = await self._reduce_synthesis(
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
        answer, quality_warnings = self._validate_output_quality(answer, target_tokens)
        if quality_warnings:
            logger.warning("Quality issues detected:\n" + "\n".join(quality_warnings))

        # Validate citations in answer
        answer = self._validate_citations(answer, expanded_chunks)

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
        schema = {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Array of exactly {NUM_LLM_EXPANDED_QUERIES} expanded search queries (semantically complete sentences)",
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
            num_queries=NUM_LLM_EXPANDED_QUERIES,
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

            # Validation: expect exactly 2 queries from LLM
            if not expanded or len(expanded) < NUM_LLM_EXPANDED_QUERIES:
                logger.warning(
                    f"LLM returned {len(expanded) if expanded else 0} queries, expected {NUM_LLM_EXPANDED_QUERIES}, using original query only"
                )
                return [query]

            # Filter empty strings
            expanded = [q.strip() for q in expanded if q and q.strip()]

            # PREPEND ORIGINAL QUERY (new logic)
            # Original query goes first for position bias in embedding models
            final_queries = [query] + expanded[:NUM_LLM_EXPANDED_QUERIES]

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
        if QUERY_EXPANSION_ENABLED:
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
            Tuple of (expanded_start_line, expanded_end_line) in 1-indexed format
        """
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
        chunks_map: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
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
        return await self._synthesis_engine._manage_token_budget_for_synthesis(
            chunks, files, root_query, synthesis_budgets
        )

    async def _single_pass_synthesis(
        self,
        root_query: str,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        context: ResearchContext,
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
        return await self._synthesis_engine._single_pass_synthesis(
            root_query, chunks, files, context, synthesis_budgets,
            constants_context=constants_context,
            facts_context=facts_context,
        )

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
        return await self._synthesis_engine._map_synthesis_on_cluster(
            cluster, root_query, chunks, synthesis_budgets, total_input_tokens,
            constants_context=constants_context,
            facts_context=facts_context,
        )

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
        return await self._synthesis_engine._reduce_synthesis(
            root_query, cluster_results, all_chunks, all_files, synthesis_budgets,
            constants_context=constants_context,
            facts_context=facts_context,
        )

    def _filter_verbosity(self, text: str) -> str:
        """Remove common LLM verbosity patterns from synthesis output.

        Acts as safety net even with good prompts. Strips defensive caveats,
        meta-commentary, and unnecessary qualifications.

        Args:
            text: Synthesis text to filter

        Returns:
            Filtered text with verbose patterns removed
        """
        import re

        # Patterns to remove (from research on LLM verbosity)
        patterns_to_remove = [
            r"It'?s important to note that\s+",
            r"It'?s worth noting that\s+",
            r"It should be noted that\s+",
            r"However, it should be mentioned that\s+",
            r"Please note that\s+",
            r"As mentioned (?:earlier|above|previously),?\s+",
            # Remove standalone "No information found" lines from body (keep in "Missing:" context)
            r"^No information (?:was )?found (?:for|about)[^\n]+\n",
            r"^Unfortunately, the (?:code|analysis) does not (?:show|provide)[^\n]+\n",
            # Remove vague precision statements
            r"The (?:exact|precise|specific) (?:implementation|details?|mechanism|values?) (?:is|are) not (?:provided|documented|shown|clear|available) in the (?:code|analysis)[,.]?\s*",
            r"(?:More|Additional) (?:research|investigation|analysis|context) (?:is|would be) (?:needed|required)[,.]?\s*",
        ]

        filtered = text
        for pattern in patterns_to_remove:
            filtered = re.sub(pattern, "", filtered, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive newlines left by removals (max 2 consecutive newlines)
        filtered = re.sub(r"\n{3,}", "\n\n", filtered)

        # Log if we actually filtered anything
        if filtered != text:
            chars_removed = len(text) - len(filtered)
            logger.debug(
                f"Verbosity filter removed {chars_removed} chars of meta-commentary"
            )

        return filtered

    def _validate_output_quality(
        self, answer: str, target_tokens: int
    ) -> tuple[str, list[str]]:
        """Validate output quality for conciseness and actionability.

        Args:
            answer: Synthesized answer to validate
            target_tokens: Target token count for this output

        Returns:
            Tuple of (validated_answer, list_of_warnings)
        """
        warnings = []
        llm = self._llm_manager.get_utility_provider()

        # Check 1: Detect theoretical placeholders
        theoretical_patterns = [
            "provide exact",
            "provide precise",
            "specify exact",
            "implementation-dependent",
            "precise line-level mappings",
            "exact numeric budgets",
            "provide the actual",
            "should specify",
            "need to determine",
            "requires clarification",
        ]

        for pattern in theoretical_patterns:
            if pattern.lower() in answer.lower():
                warnings.append(
                    f"QUALITY: Output contains theoretical placeholder: '{pattern}'. "
                    "This suggests lack of concrete information."
                )
                logger.warning(f"Output quality issue: contains '{pattern}'")

        # Check 2: Citation density (should have reasonable citations)
        citations = _CITATION_PATTERN.findall(answer)
        citation_count = len(citations)
        answer_tokens = llm.estimate_tokens(answer)

        if answer_tokens > 1000 and citation_count < 5:
            warnings.append(
                f"QUALITY: Low citation density ({citation_count} citations in {answer_tokens} tokens). "
                "Output may lack concrete code references."
            )
            logger.warning(
                f"Low citation density: {citation_count} citations in {answer_tokens} tokens"
            )

        # Check 3: Excessive length
        if answer_tokens > target_tokens * 1.5:
            warnings.append(
                f"QUALITY: Output is verbose ({answer_tokens:,} tokens vs {target_tokens:,} target). "
                "May need tighter prompting."
            )
            logger.warning(
                f"Verbose output: {answer_tokens:,} tokens (target: {target_tokens:,})"
            )

        # Check 4: Vague measurements (should use exact numbers)
        vague_patterns = [
            r"\b(several|many|few|some|various|multiple|numerous)\s+(seconds|minutes|items|entries|elements|chunks)",
            r"\b(around|approximately|roughly|about)\s+\d+",
            r"\bhundreds of\b",
            r"\bthousands of\b",
        ]

        for pattern in vague_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                warnings.append(
                    f"QUALITY: Vague measurement detected: {matches[0]}. "
                    "Should use exact values."
                )
                logger.warning(f"Vague measurement in output: {matches[0]}")
                break  # Only report first instance

        return answer, warnings

    def _validate_citations(self, answer: str, chunks: list[dict[str, Any]]) -> str:
        """Ensure answer contains numbered reference citations.

        Args:
            answer: Answer text to validate
            chunks: Chunks that were analyzed

        Returns:
            Answer with citations appended if missing
        """
        if not REQUIRE_CITATIONS:
            return answer

        # Check for citation pattern: [N] where N is a reference number
        citations = _CITATION_PATTERN.findall(answer)

        answer_length = len(answer.strip())
        answer_lines = answer.count("\n") + 1
        citation_count = len(citations)

        # Calculate citation density (citations per 100 lines of analysis)
        citation_density = (
            (citation_count / answer_lines * 100) if answer_lines > 0 else 0
        )

        # Log citation metrics
        logger.info(
            f"Citation metrics: {citation_count} citations found in {answer_length:,} chars "
            f"({answer_lines} lines), density={citation_density:.1f} citations/100 lines"
        )

        # Sample citations for debugging (show first 3)
        if citations:
            sample_citations = citations[:3]
            logger.debug(f"Sample citations: {', '.join(sample_citations)}")

        if not citations and chunks:
            # Answer missing inline citations - footer provides separate comprehensive listing
            # Enhanced warning with context
            if answer_length == 0:
                logger.warning(
                    "LLM answer is EMPTY - this indicates an LLM error. "
                    "Should have been caught by synthesis validation."
                )
            elif answer_length < 200:
                logger.warning(
                    f"LLM answer suspiciously short ({answer_length} chars) and missing "
                    f"reference citations [N] in analysis body"
                )
            else:
                logger.warning(
                    f"LLM answer missing reference citations [N] in analysis body "
                    f"(answer_length={answer_length} chars, {answer_lines} lines). "
                    f"Check if prompt citation examples are being followed."
                )
        elif citations and citation_density < 1.0:
            # Has some citations but density is low
            logger.warning(
                f"Low citation density: {citation_density:.1f} citations/100 lines "
                f"({citation_count} total citations). Consider reviewing prompt emphasis."
            )

        # Sort citation sequences for improved readability
        answer = self._sort_citation_sequences(answer)

        return answer

    def _sort_citation_sequences(self, text: str) -> str:
        """Sort inline citation sequences in ascending numerical order.

        Transforms sequences like [11][2][1][5] into [1][2][5][11] for improved
        readability. Only sorts consecutive citations - isolated citations and
        citations separated by text remain in their original positions.

        Args:
            text: Text containing citation sequences

        Returns:
            Text with sorted citation sequences

        Examples:
            >>> _sort_citation_sequences("Algorithm [11][2][1] uses BFS")
            "Algorithm [1][2][11] uses BFS"

            >>> _sort_citation_sequences("Timeout [5] and threshold [3][1][2]")
            "Timeout [5] and threshold [1][2][3]"
        """

        def sort_sequence(match):
            """Extract numbers, sort numerically, and reconstruct."""
            sequence = match.group(0)
            numbers = re.findall(r"\d+", sequence)
            sorted_numbers = sorted(int(n) for n in numbers)
            return "".join(f"[{n}]" for n in sorted_numbers)

        return _CITATION_SEQUENCE_PATTERN.sub(sort_sequence, text)

    def _build_file_reference_map(
        self, chunks: list[dict[str, Any]], files: dict[str, str]
    ) -> dict[str, int]:
        """Build mapping of file paths to reference numbers.

        Assigns sequential numbers to unique files in alphabetical order
        for deterministic, consistent numbering across synthesis steps.

        IMPORTANT: chunks must be pre-filtered to only include files present
        in the files dict. This ensures consistent numbering without gaps.

        Args:
            chunks: List of chunks (must be pre-filtered to match files dict)
            files: Dictionary of files used in synthesis

        Returns:
            Dictionary mapping file_path -> reference number (1-indexed)

        Examples:
            >>> files = {"src/main.py": "...", "tests/test.py": "..."}
            >>> chunks = []  # Empty or pre-filtered to match files
            >>> ref_map = service._build_file_reference_map(chunks, files)
            >>> ref_map
            {"src/main.py": 1, "tests/test.py": 2}
        """
        # Extract unique file paths from files dict
        # NOTE: chunks must be pre-filtered to only include files in the files dict
        # to ensure consistency between reference map, citations, and footer display
        file_paths = set(files.keys())

        # Sort alphabetically for deterministic numbering
        sorted_files = sorted(file_paths)

        # Assign sequential numbers (1-indexed)
        return {file_path: idx + 1 for idx, file_path in enumerate(sorted_files)}

    def _format_reference_table(self, file_reference_map: dict[str, int]) -> str:
        """Format file reference mapping as markdown table for LLM prompt.

        Args:
            file_reference_map: Dictionary mapping file_path -> reference number

        Returns:
            Formatted markdown table showing reference numbers

        Examples:
            >>> ref_map = {"src/main.py": 1, "tests/test.py": 2}
            >>> table = service._format_reference_table(ref_map)
            >>> print(table)
            ## Source References

            Use these reference numbers for citations:

            [1] src/main.py
            [2] tests/test.py
        """
        if not file_reference_map:
            return ""

        # Sort by reference number
        sorted_refs = sorted(file_reference_map.items(), key=lambda x: x[1])

        # Build table
        lines = [
            "## Source References",
            "",
            "Use these reference numbers for citations:",
            "",
        ]

        for file_path, ref_num in sorted_refs:
            lines.append(f"[{ref_num}] {file_path}")

        return "\n".join(lines)

    def _remap_cluster_citations(
        self,
        cluster_summary: str,
        cluster_file_map: dict[str, int],
        global_file_map: dict[str, int],
    ) -> str:
        """Remap cluster-local [N] citations to global reference numbers.

        Programmatically rewrites all [N] citations in the cluster summary to use
        global reference numbers instead of cluster-local numbers. This ensures
        consistent citations when combining multiple cluster summaries.

        Algorithm:
        1. Build reverse lookup: cluster_ref_num -> file_path
        2. For each file, get its global reference number
        3. Replace all [cluster_N] with [global_N] in the summary text

        Args:
            cluster_summary: Text with cluster-local [N] citations
            cluster_file_map: Mapping from file_path -> cluster-local reference number
            global_file_map: Mapping from file_path -> global reference number

        Returns:
            Summary text with remapped citations using global numbers

        Examples:
            >>> # Cluster 1 has: src/main.py=[1], tests/test.py=[2]
            >>> # Global has: src/main.py=[5], tests/test.py=[8]
            >>> cluster_summary = "Algorithm [1] calls helper [2]"
            >>> remapped = service._remap_cluster_citations(
            ...     cluster_summary,
            ...     {"src/main.py": 1, "tests/test.py": 2},
            ...     {"src/main.py": 5, "tests/test.py": 8}
            ... )
            >>> remapped
            "Algorithm [5] calls helper [8]"
        """
        # Build reverse lookup: cluster number -> file path
        cluster_num_to_file = {num: path for path, num in cluster_file_map.items()}

        # Build remapping table: cluster number -> global number
        remapping = {}
        for cluster_num, file_path in cluster_num_to_file.items():
            if file_path in global_file_map:
                global_num = global_file_map[file_path]
                remapping[cluster_num] = global_num
            else:
                logger.warning(
                    f"File {file_path} in cluster map but not in global map - "
                    f"citation [{cluster_num}] will not be remapped"
                )

        # Replace citations in order from highest to lowest number
        # This prevents issues like replacing [1] before [11] (which would break [11])
        remapped_summary = cluster_summary
        for cluster_num in sorted(remapping.keys(), reverse=True):
            global_num = remapping[cluster_num]
            # Replace [cluster_num] with [global_num]
            # Use word boundaries to avoid replacing [1] in [11]
            old_citation = f"[{cluster_num}]"
            new_citation = f"[{global_num}]"
            remapped_summary = remapped_summary.replace(old_citation, new_citation)

        logger.debug(
            f"Remapped {len(remapping)} citation references in cluster summary"
        )

        return remapped_summary

    def _validate_citation_references(
        self, text: str, file_reference_map: dict[str, int]
    ) -> list[int]:
        """Validate that all [N] citations exist in the file reference map.

        Checks that every citation [N] in the text corresponds to a valid
        file reference number. Invalid citations indicate bugs in remapping
        or LLM-generated citations.

        Args:
            text: Text containing [N] citations
            file_reference_map: Valid reference numbers (file_path -> number)

        Returns:
            List of invalid reference numbers (citations that don't exist in map)

        Examples:
            >>> text = "Algorithm [1] uses [2] but also [999]"
            >>> ref_map = {"src/main.py": 1, "tests/test.py": 2}
            >>> invalid = service._validate_citation_references(text, ref_map)
            >>> invalid
            [999]
        """
        # Extract all valid reference numbers from the map
        valid_refs = set(file_reference_map.values())

        # Find all [N] citations in text
        citations = _CITATION_PATTERN.findall(text)

        # Extract numbers from citations
        invalid_refs = []
        for citation in citations:
            # Extract number from [N]
            num = int(citation[1:-1])  # Remove [ and ]
            if num not in valid_refs:
                invalid_refs.append(num)

        return sorted(set(invalid_refs))  # Return unique sorted list

    def _build_sources_footer(
        self,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
        file_reference_map: dict[str, int] | None = None,
    ) -> str:
        """Build footer section with source file and chunk information.

        Creates a compact nested tree of analyzed files with chunk line ranges,
        optimized for token efficiency (using tabs) and readability.

        Args:
            chunks: List of chunks used in synthesis
            files: Dictionary of files used in synthesis (file_path -> content)

        Returns:
            Formatted markdown footer with source information

        Examples:
            >>> chunks = [
            ...     {"file_path": "src/main.py", "start_line": 10, "end_line": 25},
            ...     {"file_path": "src/main.py", "start_line": 50, "end_line": 75},
            ...     {"file_path": "tests/test.py", "start_line": 5, "end_line": 15}
            ... ]
            >>> files = {"src/main.py": "...", "tests/test.py": "..."}
            >>> footer = service._build_sources_footer(chunks, files)
            >>> print(footer)
            ---

            ## Sources

            **Files**: 2 | **Chunks**: 3

            ├── src/
            │	└── main.py (2 chunks: L10-25, L50-75)
            └── tests/
                └── test.py (1 chunks: L5-15)
        """
        if not files and not chunks:
            return ""

        # Group chunks by file
        chunks_by_file: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk.get("file_path", "unknown")
            if file_path not in chunks_by_file:
                chunks_by_file[file_path] = []
            chunks_by_file[file_path].append(chunk)

        # Build footer header
        footer_lines = [
            "---",
            "",
            "## Sources",
            "",
            f"**Files**: {len(files)} | **Chunks**: {len(chunks)}",
            "",
        ]

        # Build tree structure
        class TreeNode:
            def __init__(self, name: str):
                self.name = name
                self.children: dict[str, TreeNode] = {}
                self.is_file = False
                self.full_path = ""

        root = TreeNode("")

        for file_path in sorted(files.keys()):
            parts = file_path.split("/")
            current = root
            path_so_far = []

            for part in parts:
                path_so_far.append(part)
                if part not in current.children:
                    node = TreeNode(part)
                    node.full_path = "/".join(path_so_far)
                    current.children[part] = node
                current = current.children[part]

            current.is_file = True

        # Render tree recursively
        def render_node(node: TreeNode, prefix: str = "", is_last: bool = True) -> None:
            if node.name:  # Skip root
                # Build connector
                connector = "└── " if is_last else "├── "
                display_name = node.name

                # Add reference number for files (if map provided)
                if (
                    node.is_file
                    and file_reference_map
                    and node.full_path in file_reference_map
                ):
                    ref_num = file_reference_map[node.full_path]
                    display_name = f"[{ref_num}] {display_name}"

                # Add / suffix for directories
                if not node.is_file and node.children:
                    display_name += "/"

                line = f"{prefix}{connector}{display_name}"

                # Add chunk info for files
                if node.is_file:
                    if node.full_path in chunks_by_file:
                        file_chunks = chunks_by_file[node.full_path]
                        chunk_count = len(file_chunks)

                        # Get line ranges
                        ranges = []
                        for chunk in sorted(
                            file_chunks, key=lambda c: c.get("start_line", 0)
                        ):
                            start = chunk.get("start_line", "?")
                            end = chunk.get("end_line", "?")
                            ranges.append(f"L{start}-{end}")

                        # Compact format: show first 3 ranges + count if more
                        if len(ranges) <= 3:
                            range_str = ", ".join(ranges)
                        else:
                            range_str = (
                                f"{', '.join(ranges[:3])}, +{len(ranges) - 3} more"
                            )

                        line += f" ({chunk_count} chunks: {range_str})"
                    else:
                        # Full file analyzed without specific chunks
                        line += " (full file)"

                footer_lines.append(line)

            # Render children
            children_list = list(node.children.values())
            for idx, child in enumerate(children_list):
                is_last_child = idx == len(children_list) - 1

                # Build new prefix with tab indentation
                if node.name:  # Not root
                    if is_last:
                        new_prefix = prefix + "\t"
                    else:
                        new_prefix = prefix + "│\t"
                else:
                    new_prefix = ""

                render_node(child, new_prefix, is_last_child)

        render_node(root)

        return "\n".join(footer_lines)

    async def _filter_relevant_followups(
        self,
        questions: list[str],
        root_query: str,
        current_query: str,
        context: ResearchContext,
    ) -> list[str]:
        """Filter follow-ups by relevance to root query and architectural value.

        Args:
            questions: Candidate follow-up questions
            root_query: Original root query
            current_query: Current question being explored
            context: Research context

        Returns:
            Filtered list of most relevant follow-up questions
        """
        # Delegate to question generator
        return await self._question_generator.filter_relevant_followups(
            questions=questions,
            root_query=root_query,
            current_query=current_query,
            context=context,
        )

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

    def _get_adaptive_token_budgets(
        self, depth: int, max_depth: int, is_leaf: bool
    ) -> dict[str, int]:
        """Calculate adaptive token budgets based on node depth and tree position.

        Strategy (LLM×MapReduce Pyramid):
        - Leaves: Dense implementation details (10-12k tokens) - focused analysis
        - Internal nodes: Progressive compression toward root
        - Root: Concise synthesis (5-8k tokens target) - practical overview

        The deeper the node during expansion, the more detail needed.
        As we collapse upward during synthesis, we compress while maintaining quality.

        Args:
            depth: Current node depth (0 = root)
            max_depth: Maximum depth for this codebase (3-7 typically)
            is_leaf: Whether this is a leaf node

        Returns:
            Dictionary with adaptive token budgets for this node
        """
        if not ENABLE_ADAPTIVE_BUDGETS:
            # Fallback to legacy fixed budgets
            return {
                "file_content_tokens": MAX_FILE_CONTENT_TOKENS,
                "llm_input_tokens": MAX_LLM_INPUT_TOKENS,
                "answer_tokens": MAX_LEAF_ANSWER_TOKENS
                if is_leaf
                else MAX_SYNTHESIS_TOKENS,
            }

        # Normalize depth: 0.0 at root, 1.0 at max_depth
        depth_ratio = depth / max(max_depth, 1)

        # INPUT BUDGETS (what LLM sees - file content and total input)
        # ==============================================================

        # File content budget: Scales with depth (10k → 50k tokens)
        # Root needs LESS raw code (synthesizing), leaves need MORE (analyzing)
        file_content_tokens = int(
            FILE_CONTENT_TOKENS_MIN
            + (FILE_CONTENT_TOKENS_MAX - FILE_CONTENT_TOKENS_MIN) * depth_ratio
        )

        # LLM total input budget (query + context + code): 15k → 60k tokens
        llm_input_tokens = int(
            LLM_INPUT_TOKENS_MIN
            + (LLM_INPUT_TOKENS_MAX - LLM_INPUT_TOKENS_MIN) * depth_ratio
        )

        # OUTPUT BUDGETS (what LLM generates)
        # ====================================

        if is_leaf:
            # LEAVES: Dense, focused detail (10-12k tokens)
            # Scale slightly with depth to handle variable max_depth (3-7)
            answer_tokens = int(
                LEAF_ANSWER_TOKENS_BASE + LEAF_ANSWER_TOKENS_BONUS * depth_ratio
            )
        else:
            # INTERNAL NODES: Compress as we go UP the tree
            # Root (depth 0) gets concise output (5k)
            # Deeper internal nodes get more budget before compressing
            answer_tokens = int(
                INTERNAL_ROOT_TARGET
                + (INTERNAL_MAX_TOKENS - INTERNAL_ROOT_TARGET) * depth_ratio
            )

        # Follow-up question generation budget: Scales with depth (3k → 8k)
        # Deeper nodes have more context to analyze, need more output tokens
        followup_output_tokens = int(
            FOLLOWUP_OUTPUT_TOKENS_MIN
            + (FOLLOWUP_OUTPUT_TOKENS_MAX - FOLLOWUP_OUTPUT_TOKENS_MIN) * depth_ratio
        )

        logger.debug(
            f"Adaptive budgets for depth {depth}/{max_depth} ({'leaf' if is_leaf else 'internal'}): "
            f"file={file_content_tokens:,}, input={llm_input_tokens:,}, output={answer_tokens:,}, "
            f"followup={followup_output_tokens:,}"
        )

        return {
            "file_content_tokens": file_content_tokens,
            "llm_input_tokens": llm_input_tokens,
            "answer_tokens": answer_tokens,
            "followup_output_tokens": followup_output_tokens,
        }

    def _get_next_node_id(self) -> int:
        """Get next node ID for graph traversal."""
        self._node_counter += 1
        return self._node_counter
