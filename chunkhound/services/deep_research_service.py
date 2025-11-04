"""Deep Research Service for ChunkHound - BFS-based semantic exploration."""

import asyncio
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services import prompts
from chunkhound.services.clustering_service import ClusterGroup, ClusteringService
from chunkhound.services.research.budget_calculator import BudgetCalculator
from chunkhound.services.research.citation_manager import CitationManager
from chunkhound.services.research.context_manager import ContextManager
from chunkhound.services.research.file_reader import FileReader
from chunkhound.services.research.quality_validator import QualityValidator
from chunkhound.services.research.query_expander import QueryExpander
from chunkhound.services.research.question_generator import QuestionGenerator
from chunkhound.services.research.synthesis_engine import SynthesisEngine
from chunkhound.services.research.unified_search import UnifiedSearch

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay

# Constants
RELEVANCE_THRESHOLD = 0.5  # Lower threshold for better recall, reranking will filter
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

# Synthesis budget calculation (repository size scaling)
CHUNKS_TO_LOC_ESTIMATE = 20  # Rough estimation: 1 chunk ≈ 20 lines of code
LOC_THRESHOLD_TINY = 10_000  # Very small repos
LOC_THRESHOLD_SMALL = 100_000  # Small repos
LOC_THRESHOLD_MEDIUM = 1_000_000  # Medium repos
# Large repos: >= 1M LOC

SYNTHESIS_INPUT_TOKENS_TINY = 30_000  # Very small repos (< 10k LOC)
SYNTHESIS_INPUT_TOKENS_SMALL = 50_000  # Small repos (< 100k LOC)
SYNTHESIS_INPUT_TOKENS_MEDIUM = 80_000  # Medium repos (< 1M LOC)
SYNTHESIS_INPUT_TOKENS_LARGE = 150_000  # Large repos (>= 1M LOC)

# Output control
REQUIRE_CITATIONS = True  # Validate file:line format

# Map-reduce synthesis constants
MAX_TOKENS_PER_CLUSTER = 30_000  # Token budget per cluster for parallel synthesis
CLUSTER_OUTPUT_TOKEN_BUDGET = 15_000  # Max output tokens per cluster summary

# Pre-compiled regex patterns for citation processing
_CITATION_PATTERN = re.compile(r"\[\d+\]")  # Matches [N] citations
_CITATION_SEQUENCE_PATTERN = re.compile(r"(?:\[\d+\])+")  # Matches sequences like [1][2][3]

# Smart boundary detection for context-aware file reading
ENABLE_SMART_BOUNDARIES = True  # Expand to natural code boundaries (functions/classes)
MAX_BOUNDARY_EXPANSION_LINES = 300  # Maximum lines to expand for complete functions

# File-level reranking for synthesis budget allocation
# Prevents file diversity collapse where deep BFS exploration causes score accumulation in few files
MAX_CHUNKS_PER_FILE_REPR = (
    5  # Top chunks to include in file representative document for reranking
)
MAX_TOKENS_PER_FILE_REPR = 2000  # Token limit for file representative document


@dataclass
class BFSNode:
    """Node in the BFS research graph."""

    query: str
    parent: "BFSNode | None" = None
    depth: int = 0
    children: list["BFSNode"] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    file_contents: dict[str, str] = field(
        default_factory=dict
    )  # Full file contents for synthesis
    answer: str | None = None
    node_id: int = 0
    unanswered_aspects: list[str] = field(
        default_factory=list
    )  # Questions we couldn't answer
    token_budgets: dict[str, int] = field(
        default_factory=dict
    )  # Adaptive token budgets for this node
    task_id: int | None = None  # Progress task ID for TUI display

    # Termination tracking
    is_terminated_leaf: bool = False  # True if terminated due to no new information
    new_chunk_count: int = 0  # Count of truly new chunks
    duplicate_chunk_count: int = 0  # Count of duplicate chunks


@dataclass
class ResearchContext:
    """Context for research traversal."""

    root_query: str
    ancestors: list[str] = field(default_factory=list)
    traversal_path: list[str] = field(default_factory=list)


class DeepResearchService:
    """Service for performing deep research using BFS exploration."""

    def __init__(
        self,
        database_services: DatabaseServices,
        embedding_manager: EmbeddingManager,
        llm_manager: LLMManager,
        tool_name: str = "code_research",
        progress: "TreeProgressDisplay | None" = None,
    ):
        """Initialize deep research service.

        Args:
            database_services: Database services bundle
            embedding_manager: Embedding manager for semantic search
            llm_manager: LLM manager for generating follow-ups and synthesis
            tool_name: Name of the MCP tool (used in followup suggestions)
            progress: Optional TreeProgressDisplay instance for terminal UI (None for MCP)
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

        # Calculate dynamic synthesis budgets based on repository size
        stats = self._db_services.provider.get_stats()
        synthesis_budgets = self._calculate_synthesis_budgets(stats)
        logger.info(
            f"Synthesis budgets: input={synthesis_budgets['input_tokens']:,}, output={synthesis_budgets['output_tokens']:,}"
        )

        # Emit configuration info
        await self._emit_event(
            "main_info",
            f"Max depth: {max_depth}, synthesis budget: {synthesis_budgets['total_tokens'] // 1000}k tokens",
        )

        # Initialize BFS graph with root node
        root = BFSNode(query=query, depth=0, node_id=self._get_next_node_id())
        context = ResearchContext(root_query=query)

        # BFS traversal
        current_level = [root]
        all_nodes: list[BFSNode] = [root]

        # Global explored data: track ALL chunks/files discovered across entire BFS graph
        # This enables sibling nodes to detect duplicates, not just ancestors
        global_explored_data = {
            "files_fully_read": set(),
            "chunk_ranges": {},  # file_path -> list[(start, end)]
            "chunks": [],  # All chunks for building exploration gist
        }

        # BFS traversal: Process depth 0 (root node) through max_depth
        # Root node (depth 0) is already in current_level, so we start the loop at 0
        for depth in range(0, max_depth + 1):
            if not current_level:
                break

            logger.info(f"Processing BFS level {depth}, nodes: {len(current_level)}")

            # Emit depth start event
            await self._emit_event(
                "depth_start",
                f"Processing depth {depth}/{max_depth}",
                depth=depth,
                nodes=len(current_level),
                max_depth=max_depth,
            )

            # Process all nodes at this level concurrently (as per algorithm spec)
            # Each node gets its own context copy to avoid shared state issues
            node_contexts = []
            for node in current_level:
                # Create context copy WITHOUT adding current query to ancestors yet
                # (it will be added to global context AFTER processing)
                # This prevents redundancy in _build_search_query
                node_context = ResearchContext(
                    root_query=context.root_query,
                    ancestors=context.ancestors.copy(),  # Just copy, don't append node.query
                    traversal_path=context.traversal_path.copy(),
                )
                node_contexts.append((node, node_context))

            # Process all nodes concurrently
            node_tasks = [
                self._process_bfs_node(
                    node, node_ctx, depth, global_explored_data, max_depth
                )
                for node, node_ctx in node_contexts
            ]
            children_lists = await asyncio.gather(*node_tasks, return_exceptions=True)

            # Collect children and handle errors
            next_level: list[BFSNode] = []
            for (node, node_ctx), children_result in zip(node_contexts, children_lists):
                if isinstance(children_result, Exception):
                    logger.error(
                        f"BFS node failed for '{node.query}': {children_result}"
                    )
                    continue

                # Type narrowing: at this point children_result is list[BFSNode]
                assert isinstance(children_result, list)
                node.children.extend(children_result)
                next_level.extend(children_result)
                all_nodes.extend(children_result)

                # Update global explored data with this node's discoveries
                # Only update if node found new information (not terminated)
                if not node.is_terminated_leaf and node.chunks:
                    self._update_global_explored_data(global_explored_data, node)

            # Update global context with all processed queries
            for node, _ in node_contexts:
                if node.query not in context.ancestors:
                    context.ancestors.append(node.query)

            # Synthesize questions at this level if too many
            if len(next_level) > MAX_FOLLOWUP_QUESTIONS:
                next_level = await self._synthesize_questions(
                    next_level, context, MAX_FOLLOWUP_QUESTIONS
                )

            current_level = next_level

        # Aggregate all findings from BFS tree
        logger.info("BFS traversal complete, aggregating findings")

        # Emit aggregating event
        await self._emit_event("synthesis_start", "Aggregating findings from BFS tree")

        aggregated = self._aggregate_all_findings(root)

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

        # Manage token budget for single-pass synthesis
        (
            prioritized_chunks,
            budgeted_files,
            budget_info,
        ) = await self._manage_token_budget_for_synthesis(
            aggregated["chunks"], aggregated["files"], query, synthesis_budgets
        )

        # Emit synthesizing event
        await self._emit_event(
            "synthesis_start",
            f"Synthesizing final answer (input: {budget_info['used_tokens']:,}/{budget_info['available_tokens']:,} tokens, {budget_info['utilization']})",
            chunks=len(prioritized_chunks),
            files=len(budgeted_files),
            input_tokens_budget=budget_info["available_tokens"],
            input_tokens_used=budget_info["used_tokens"],
        )

        # Cluster sources for map-reduce synthesis
        cluster_groups, cluster_metadata = await self._cluster_sources_for_synthesis(
            prioritized_chunks, budgeted_files, synthesis_budgets
        )

        # If only 1 cluster, use single-pass (no benefit from map-reduce)
        if cluster_metadata["num_clusters"] == 1:
            logger.info("Single cluster detected - using single-pass synthesis")
            answer = await self._single_pass_synthesis(
                root_query=query,
                chunks=prioritized_chunks,
                files=budgeted_files,
                context=context,
                synthesis_budgets=synthesis_budgets,
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

            async def map_with_semaphore(cluster: ClusterGroup) -> dict[str, Any]:
                async with semaphore:
                    return await self._map_synthesis_on_cluster(
                        cluster, query, prioritized_chunks, synthesis_budgets
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

            answer = await self._reduce_synthesis(
                query,
                cluster_results,
                prioritized_chunks,
                budgeted_files,
                synthesis_budgets,
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
        answer = self._validate_citations(answer, root.chunks)

        # Calculate metadata
        metadata = {
            "depth_reached": max(node.depth for node in all_nodes),
            "nodes_explored": len(all_nodes),
            "chunks_analyzed": sum(len(node.chunks) for node in all_nodes),
            "aggregation_stats": aggregated["stats"],
            "token_budget": budget_info,
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

    async def _process_bfs_node(
        self,
        node: BFSNode,
        context: ResearchContext,
        depth: int,
        global_explored_data: dict[str, Any],
        max_depth: int,
    ) -> list[BFSNode]:
        """Process a single BFS node.

        Args:
            node: BFS node to process
            context: Research context
            depth: Current depth in graph
            global_explored_data: Global state tracking all explored chunks/files across entire BFS
            max_depth: Maximum depth for BFS traversal (used for adaptive budgets)

        Returns:
            List of child nodes (follow-up questions)
        """
        logger.debug(f"Processing node at depth {depth}: '{node.query}'")

        # Emit node start event
        query_preview = node.query[:60] + "..." if len(node.query) > 60 else node.query
        await self._emit_event(
            "node_start",
            query_preview,
            node_id=node.node_id,
            depth=depth,
        )

        # Calculate adaptive token budgets for this node (assume leaf initially)
        # Use max_depth passed from deep_research() to respect shallow mode
        node.token_budgets = self._get_adaptive_token_budgets(
            depth=depth, max_depth=max_depth, is_leaf=True
        )

        # Step 1: Combine query with BFS ancestors for semantic search
        search_query = self._build_search_query(node.query, context)

        # Step 2-6: Run unified search (semantic + symbol extraction + regex)
        chunks = await self._unified_search(
            search_query, context, node_id=node.node_id, depth=depth
        )
        node.chunks = chunks

        if not chunks:
            logger.warning(f"No chunks found for query: '{node.query}'")
            await self._emit_event(
                "node_complete",
                "No chunks found",
                node_id=node.node_id,
                depth=depth,
                chunks=0,
            )
            return []

        # Step 8: Read files with adaptive token budget
        await self._emit_event(
            "read_files", "Reading files", node_id=node.node_id, depth=depth
        )

        file_contents = await self._read_files_with_budget(
            chunks, max_tokens=node.token_budgets["file_content_tokens"]
        )
        node.file_contents = file_contents  # Store for later synthesis

        # Emit file reading results
        llm = self._llm_manager.get_utility_provider()
        total_tokens = sum(
            llm.estimate_tokens(content) for content in file_contents.values()
        )
        await self._emit_event(
            "read_files_complete",
            f"Read {len(file_contents)} files",
            node_id=node.node_id,
            depth=depth,
            files=len(file_contents),
            tokens=total_tokens,
        )

        # Step 8.5: Check for new information (termination rule)
        # Uses global explored data to detect duplicates across entire BFS graph, not just ancestors
        has_new_info, dedup_stats = self._detect_new_information(
            node, chunks, global_explored_data
        )
        node.new_chunk_count = dedup_stats["new_chunks"]
        node.duplicate_chunk_count = dedup_stats["duplicate_chunks"]

        if not has_new_info:
            logger.info(
                f"[Termination] Node '{node.query[:50]}...' at depth {depth} "
                f"found 0 new chunks ({dedup_stats['duplicate_chunks']} duplicates). "
                f"Marking as terminated leaf, skipping question generation."
            )
            node.is_terminated_leaf = True
            await self._emit_event(
                "node_terminated",
                "No new information found",
                node_id=node.node_id,
                depth=depth,
                duplicates=dedup_stats["duplicate_chunks"],
            )
            return []  # No children

        logger.debug(
            f"Node '{node.query[:50]}...' at depth {depth} "
            f"found {dedup_stats['new_chunks']} new chunks, "
            f"{dedup_stats['duplicate_chunks']} duplicates"
        )

        # Step 9: Generate follow-up questions using LLM with adaptive budget
        # Skip followup generation if we're at max depth (children would never be processed)
        # Use max_depth passed from deep_research() to respect shallow mode
        if depth >= max_depth:
            logger.debug(
                f"Node '{node.query[:50]}...' at max depth {depth}/{max_depth}, "
                f"skipping followup generation (no deeper exploration)"
            )
            await self._emit_event(
                "node_complete",
                "Complete (leaf at max depth)",
                node_id=node.node_id,
                depth=depth,
                files=len(file_contents),
                chunks=len(chunks),
                children=0,
            )
            return []

        await self._emit_event(
            "llm_followup",
            "Generating follow-up questions",
            node_id=node.node_id,
            depth=depth,
        )

        follow_ups = await self._generate_follow_up_questions(
            node.query,
            context,
            file_contents,
            chunks,
            global_explored_data,
            max_input_tokens=node.token_budgets["llm_input_tokens"],
            depth=depth,
            max_depth=max_depth,
        )

        # Emit follow-up generation results
        if follow_ups:
            questions_preview = "; ".join(
                q[:40] + "..." if len(q) > 40 else q for q in follow_ups[:2]
            )
            await self._emit_event(
                "llm_followup_complete",
                f"Generated {len(follow_ups)} follow-ups",
                node_id=node.node_id,
                depth=depth,
                followups=len(follow_ups),
            )
        else:
            await self._emit_event(
                "llm_followup_complete",
                "No follow-ups generated",
                node_id=node.node_id,
                depth=depth,
                followups=0,
            )

        # Create child nodes
        children = []
        for i, follow_up in enumerate(follow_ups[:MAX_FOLLOWUP_QUESTIONS]):
            child = BFSNode(
                query=follow_up,
                parent=node,
                depth=depth + 1,  # Children are one level deeper than current depth
                node_id=self._get_next_node_id(),
            )
            children.append(child)

        # Emit node completion
        await self._emit_event(
            "node_complete",
            "Complete",
            node_id=node.node_id,
            depth=depth,
            files=len(file_contents),
            chunks=len(chunks),
            children=len(children),
        )

        return children

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
        """Perform unified semantic + symbol-based regex search (Steps 2-6).

        Algorithm steps:
        1. Multi-hop semantic search with internal reranking (Step 2)
        2. Extract symbols from semantic results (Step 3)
        3. Select top N symbols (Step 4) - already in relevance order from reranked results
        4. Regex search for top symbols (Step 5)
        5. Unify results at chunk level (Step 6)

        Note: Multi-hop semantic search already performs reranking internally,
        so symbols are extracted from already-reranked results and no additional
        reranking is needed.

        Args:
            query: Search query
            context: Research context with root query and ancestors
            node_id: Optional BFS node ID for event emission
            depth: Optional BFS depth for event emission

        Returns:
            List of unified chunks
        """
        search_service = self._db_services.search_service

        # Step 2: Multi-hop semantic search with reranking (optionally with query expansion)
        if QUERY_EXPANSION_ENABLED:
            # Expand query into multiple diverse perspectives
            logger.debug("Step 2a: Expanding query for diverse semantic search")
            await self._emit_event(
                "query_expand", "Expanding query", node_id=node_id, depth=depth
            )

            expanded_queries = await self._expand_query_with_llm(query, context)
            logger.debug(
                f"Query expansion: 1 original + {len(expanded_queries) - 1} LLM-generated = {len(expanded_queries)} total: {expanded_queries}"
            )

            # Emit expanded queries event
            queries_preview = " | ".join(
                q[:40] + "..." if len(q) > 40 else q for q in expanded_queries[:3]
            )
            await self._emit_event(
                "query_expand_complete",
                f"Expanded to {len(expanded_queries)} queries",
                node_id=node_id,
                depth=depth,
                queries=len(expanded_queries),
            )

            # Run all semantic searches in parallel
            logger.debug(
                f"Step 2b: Running {len(expanded_queries)} parallel semantic searches"
            )
            search_tasks = [
                search_service.search_semantic(
                    query=expanded_q,
                    page_size=30,
                    threshold=RELEVANCE_THRESHOLD,
                    force_strategy="multi_hop",
                )
                for expanded_q in expanded_queries
            ]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Unify results: deduplicate by chunk_id (same pattern as semantic+regex unification)
            semantic_map = {}
            for result in search_results:
                if isinstance(result, Exception):
                    logger.warning(
                        f"Semantic search failed during query expansion: {result}"
                    )
                    continue
                # Validate tuple structure before unpacking
                if not isinstance(result, tuple) or len(result) != 2:
                    logger.error(
                        f"Unexpected search result structure: {type(result)}, skipping"
                    )
                    continue
                results, _ = result
                for chunk in results:
                    chunk_id = chunk.get("chunk_id") or chunk.get("id")
                    if chunk_id and chunk_id not in semantic_map:
                        semantic_map[chunk_id] = chunk

            semantic_results = list(semantic_map.values())
            logger.debug(
                f"Unified {sum(len(r[0]) if not isinstance(r, Exception) else 0 for r in search_results)} results from {len(expanded_queries)} searches -> {len(semantic_results)} unique chunks"
            )

            # Emit search results event
            await self._emit_event(
                "search_semantic",
                f"Found {len(semantic_results)} chunks",
                node_id=node_id,
                depth=depth,
                chunks=len(semantic_results),
            )
        else:
            # Original single-query approach (fallback)
            logger.debug(
                f"Step 2: Running multi-hop semantic search for query: '{query}'"
            )
            await self._emit_event(
                "search_semantic",
                "Searching semantically",
                node_id=node_id,
                depth=depth,
            )

            semantic_results, _ = await search_service.search_semantic(
                query=query,
                page_size=30,
                threshold=RELEVANCE_THRESHOLD,
                force_strategy="multi_hop",
            )
            logger.debug(f"Semantic search returned {len(semantic_results)} chunks")

            # Emit search results event
            await self._emit_event(
                "search_semantic",
                f"Found {len(semantic_results)} chunks",
                node_id=node_id,
                depth=depth,
                chunks=len(semantic_results),
            )

        # Steps 3-5: Symbol extraction, reranking, and regex search
        regex_results = []
        if semantic_results:
            # Step 3: Extract symbols from semantic results
            logger.debug("Step 3: Extracting symbols from semantic results")
            await self._emit_event(
                "extract_symbols", "Extracting symbols", node_id=node_id, depth=depth
            )

            symbols = await self._extract_symbols_from_chunks(semantic_results)

            if symbols:
                # Step 4: Select top symbols (already in relevance order from reranked semantic results)
                logger.debug(
                    f"Step 4: Selecting top {MAX_SYMBOLS_TO_SEARCH} symbols from {len(symbols)} extracted symbols"
                )
                top_symbols = symbols[:MAX_SYMBOLS_TO_SEARCH]

                # Emit symbol extraction results
                symbols_preview = ", ".join(top_symbols[:5])
                if len(top_symbols) > 5:
                    symbols_preview += "..."
                await self._emit_event(
                    "extract_symbols_complete",
                    f"Extracted {len(symbols)} symbols, searching top {len(top_symbols)}",
                    node_id=node_id,
                    depth=depth,
                    symbols=len(symbols),
                )

                if top_symbols:
                    # Step 5: Regex search for top symbols
                    logger.debug(
                        f"Step 5: Running regex search for {len(top_symbols)} top symbols"
                    )
                    await self._emit_event(
                        "search_regex",
                        "Running regex search",
                        node_id=node_id,
                        depth=depth,
                    )

                    regex_results = await self._search_by_symbols(top_symbols)

                    # Emit regex search results
                    await self._emit_event(
                        "search_regex_complete",
                        f"Found {len(regex_results)} additional chunks",
                        node_id=node_id,
                        depth=depth,
                        chunks=len(regex_results),
                    )

        # Step 6: Unify results at chunk level (deduplicate by chunk_id)
        logger.debug(
            f"Step 6: Unifying {len(semantic_results)} semantic + {len(regex_results)} regex results"
        )
        unified_map = {}

        # Add semantic results first (they have relevance scores from multi-hop)
        for chunk in semantic_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id:
                unified_map[chunk_id] = chunk

        # Add regex results (only new chunks not already found)
        for chunk in regex_results:
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if chunk_id and chunk_id not in unified_map:
                unified_map[chunk_id] = chunk

        unified_chunks = list(unified_map.values())
        logger.debug(f"Unified to {len(unified_chunks)} unique chunks")

        # Note: Multi-hop semantic search already reranked results, no need to rerank again
        return unified_chunks

    async def _extract_symbols_from_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> list[str]:
        """Extract symbols from already-parsed chunks (language-agnostic).

        Leverages existing chunk data from UniversalParser which already extracted
        symbols for all 25+ supported languages. No re-parsing needed!

        Args:
            chunks: List of chunks from semantic search

        Returns:
            Deduplicated list of symbol names
        """
        symbols = set()

        for chunk in chunks:
            # Primary: Extract symbol name (function/class/method name)
            # This field is populated by UniversalParser for all languages
            if symbol := chunk.get("symbol"):
                if symbol and symbol.strip():
                    symbols.add(symbol.strip())

            # Secondary: Extract parameters as potential searchable symbols
            # Many functions/methods have meaningful parameter names
            metadata = chunk.get("metadata", {})
            if params := metadata.get("parameters"):
                if isinstance(params, list):
                    symbols.update(p.strip() for p in params if p and p.strip())

            # Tertiary: Extract from chunk_type-specific metadata
            # Some chunks have additional symbol information
            if chunk_type := metadata.get("kind"):
                # Skip generic types, focus on specific symbols
                if chunk_type not in ("block", "comment", "unknown"):
                    if name := chunk.get("name"):
                        symbols.add(name.strip())

        # Filter out common noise (single chars, numbers, common keywords)
        filtered_symbols = [
            s
            for s in symbols
            if len(s) > 1
            and not s.isdigit()
            and s.lower() not in {"self", "cls", "this"}
        ]

        logger.debug(
            f"Extracted {len(filtered_symbols)} symbols from {len(chunks)} chunks"
        )
        return filtered_symbols

    async def _search_by_symbols(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Search codebase for top-ranked symbols using parallel async regex (Step 5).

        Uses async execution to avoid blocking the event loop, enabling better
        concurrency when searching for multiple symbols in parallel.

        Args:
            symbols: List of symbol names to search for

        Returns:
            List of chunks found via regex search
        """
        if not symbols:
            return []

        import re

        search_service = self._db_services.search_service

        async def search_symbol(symbol: str) -> list[dict[str, Any]]:
            """Search for a single symbol asynchronously."""
            try:
                # Escape special regex characters
                escaped = re.escape(symbol)
                # Match word boundaries to avoid partial matches
                # This works across all languages (identifier boundaries)
                pattern = rf"\b{escaped}\b"

                results, _ = await search_service.search_regex_async(
                    pattern=pattern,
                    page_size=10,  # Limit per symbol to avoid overwhelming results
                    offset=0,
                )

                logger.debug(f"Found {len(results)} chunks for symbol '{symbol}'")
                return results

            except Exception as e:
                logger.warning(f"Regex search failed for symbol '{symbol}': {e}")
                return []

        # Run all symbol searches concurrently
        results_per_symbol = await asyncio.gather(*[search_symbol(s) for s in symbols])

        # Flatten results
        all_results = []
        for results in results_per_symbol:
            all_results.extend(results)

        logger.debug(
            f"Parallel symbol regex search complete: {len(all_results)} total chunks from {len(symbols)} symbols"
        )
        return all_results

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

    def _collect_ancestor_data(self, node: BFSNode) -> dict[str, Any]:
        """Traverse parent chain and collect all ancestor chunks/files.

        NOTE: This method is now deprecated in favor of global_explored_data tracking.
        Kept for backward compatibility but not actively used in BFS duplicate detection.

        Args:
            node: Current BFS node

        Returns:
            Dictionary with:
                - files_fully_read: set[str] - Paths of fully-read files
                - chunk_ranges: dict[str, list[tuple[int, int]]] - file → [(start, end)]
        """
        files_fully_read: set[str] = set()
        chunk_ranges: dict[str, list[tuple[int, int]]] = {}

        current = node.parent
        while current:
            # Check which files were fully read
            for file_path, content in current.file_contents.items():
                if self._is_file_fully_read(content):
                    files_fully_read.add(file_path)

            # Collect expanded chunk ranges
            for chunk in current.chunks:
                file_path = chunk.get("file_path")
                if file_path:
                    expanded_range = self._get_chunk_expanded_range(chunk)
                    chunk_ranges.setdefault(file_path, []).append(expanded_range)

            current = current.parent

        return {
            "files_fully_read": files_fully_read,
            "chunk_ranges": chunk_ranges,
        }

    def _update_global_explored_data(
        self, global_explored_data: dict[str, Any], node: BFSNode
    ) -> None:
        """Update global explored data with discoveries from a single node.

        This allows sibling nodes and future nodes to detect duplicates across the entire BFS graph,
        not just their ancestor chain. Critical for preventing redundant exploration.

        Args:
            global_explored_data: Global state dict with files_fully_read, chunk_ranges, and chunks
            node: BFS node whose discoveries should be added to global state
        """
        # Add fully-read files
        for file_path, content in node.file_contents.items():
            if self._is_file_fully_read(content):
                global_explored_data["files_fully_read"].add(file_path)

        # Add expanded chunk ranges and chunks
        for chunk in node.chunks:
            file_path = chunk.get("file_path")
            if file_path:
                expanded_range = self._get_chunk_expanded_range(chunk)
                global_explored_data["chunk_ranges"].setdefault(file_path, []).append(
                    expanded_range
                )
                # Store chunk for building exploration gist
                global_explored_data["chunks"].append(chunk)

    def _build_exploration_gist(
        self, global_explored_data: dict[str, Any]
    ) -> str | None:
        """Build markdown tree view of explored files and chunks.

        Uses the same format as the final synthesis sources footer for consistency.
        Shows explored files in a tree structure with chunk line ranges.

        Args:
            global_explored_data: Global state with chunks list

        Returns:
            Markdown tree structure of explored files and chunks, or None if no exploration yet
        """
        chunks = global_explored_data["chunks"]
        if not chunks:
            return None  # No exploration yet - skip gist section entirely

        # Extract unique files from chunks (we don't need content, just the list)
        files = {
            chunk.get("file_path"): "" for chunk in chunks if chunk.get("file_path")
        }

        if not files:
            return None

        # Reuse the sources footer builder for consistent formatting
        # This creates a markdown tree structure with chunk line ranges
        footer = self._build_sources_footer(chunks, files)

        # Return just the tree portion (skip "---" separator at the start)
        # Keep the "## Sources" header and the tree
        return footer

    def _is_chunk_duplicate(
        self,
        chunk: dict[str, Any],
        chunk_expanded_range: tuple[int, int],
        explored_data: dict[str, Any],
    ) -> bool:
        """Check if chunk is 100% duplicate of any previously explored data in BFS graph.

        Returns True only if:
        1. Chunk's file was fully read by any previously explored node, OR
        2. Chunk's expanded range is 100% contained in any previously explored chunk

        Partial overlaps return False (counted as new information).

        Args:
            chunk: Chunk dictionary
            chunk_expanded_range: Expanded range for this chunk
            explored_data: Global explored data from entire BFS graph (not just ancestors)

        Returns:
            True if chunk is 100% duplicate, False otherwise
        """
        file_path = chunk.get("file_path")
        if not file_path:
            return False

        expanded_start, expanded_end = chunk_expanded_range

        # Check 1: File fully read by any previously explored node
        if file_path in explored_data["files_fully_read"]:
            return True

        # Check 2: 100% containment in any previously explored chunk
        for prev_start, prev_end in explored_data["chunk_ranges"].get(file_path, []):
            # Must be completely contained (100% overlap)
            if expanded_start >= prev_start and expanded_end <= prev_end:
                return True

        return False

    def _detect_new_information(
        self,
        node: BFSNode,
        chunks: list[dict[str, Any]],
        global_explored_data: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Detect if node has new information vs all previously explored nodes in BFS graph.

        Args:
            node: Current BFS node
            chunks: Chunks found for this node
            global_explored_data: Global state with files_fully_read and chunk_ranges from ALL processed nodes

        Returns:
            Tuple of (has_new_info, stats):
                - has_new_info: Boolean indicating if node has truly new chunks
                - stats: Dict with breakdown of new/duplicate counts
        """
        if not node.parent:
            # Root node always has new info
            return (
                True,
                {
                    "new_chunks": len(chunks),
                    "duplicate_chunks": 0,
                    "total_chunks": len(chunks),
                },
            )

        if not chunks:
            # No chunks at all
            return (False, {"new_chunks": 0, "duplicate_chunks": 0, "total_chunks": 0})

        # Check each chunk against global explored data (entire BFS graph, not just ancestors)
        new_count = 0
        duplicate_count = 0

        for chunk in chunks:
            # Get expanded range (from stored data or re-compute)
            expanded_range = self._get_chunk_expanded_range(chunk)

            is_duplicate = self._is_chunk_duplicate(
                chunk, expanded_range, global_explored_data
            )

            if is_duplicate:
                duplicate_count += 1
            else:
                new_count += 1

        has_new_info = new_count > 0

        stats = {
            "new_chunks": new_count,
            "duplicate_chunks": duplicate_count,
            "total_chunks": len(chunks),
        }

        return (has_new_info, stats)

    async def _generate_follow_up_questions(
        self,
        query: str,
        context: ResearchContext,
        file_contents: dict[str, str],
        chunks: list[dict[str, Any]],
        global_explored_data: dict[str, Any],
        max_input_tokens: int | None = None,
        depth: int = 0,
        max_depth: int = 1,
    ) -> list[str]:
        """Generate follow-up questions using LLM.

        Args:
            query: Current query
            context: Research context
            file_contents: File contents found
            chunks: Chunks found
            global_explored_data: Global state with all explored chunks/files
            max_input_tokens: Maximum tokens for LLM input (uses adaptive budget if provided)
            depth: Current depth in BFS traversal
            max_depth: Maximum depth for this codebase

        Returns:
            List of follow-up questions
        """
        # Build exploration gist to prevent redundant exploration
        exploration_gist = self._build_exploration_gist(global_explored_data)

        # Sync node counter to question generator
        self._question_generator.set_node_counter(self._node_counter)

        # Delegate to question generator
        return await self._question_generator.generate_follow_up_questions(
            query=query,
            context=context,
            file_contents=file_contents,
            chunks=chunks,
            global_explored_data=global_explored_data,
            exploration_gist=exploration_gist,
            max_input_tokens=max_input_tokens,
            depth=depth,
            max_depth=max_depth,
        )

    async def _synthesize_questions(
        self, nodes: list[BFSNode], context: ResearchContext, target_count: int
    ) -> list[BFSNode]:
        """Synthesize N new questions capturing unexplored aspects of input questions.

        Purpose: When BFS level has too many questions, synthesize them into
        fewer high-level questions that explore NEW areas for comprehensive coverage.

        Args:
            nodes: List of BFS nodes to synthesize
            context: Research context
            target_count: Target number of synthesized questions

        Returns:
            Fresh BFSNode objects with synthesized queries and empty metadata.
            These nodes will find their own chunks during processing.
        """
        # Sync node counter to question generator
        self._question_generator.set_node_counter(self._node_counter)

        # Delegate to question generator
        result = await self._question_generator.synthesize_questions(
            nodes=nodes,
            context=context,
            target_count=target_count,
        )

        # Sync node counter back from question generator
        self._node_counter = self._question_generator._node_counter

        return result

    def _aggregate_all_findings(self, root: BFSNode) -> dict[str, Any]:
        """Aggregate all chunks and files from entire BFS tree.

        Walks the complete BFS tree and collects all discovered chunks and files,
        deduplicating by chunk_id and file_path.

        Args:
            root: Root BFS node

        Returns:
            Dictionary with:
                - chunks: List of unique chunks (deduplicated by chunk_id)
                - files: Dict mapping file_path to content (deduplicated)
                - stats: Statistics about aggregation
        """
        logger.info("Aggregating all findings from BFS tree")

        # Collect all nodes via BFS traversal
        all_nodes: list[BFSNode] = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            all_nodes.append(node)
            queue.extend(node.children)

        # Aggregate chunks (deduplicate by chunk_id)
        chunks_map: dict[str, dict[str, Any]] = {}
        for node in all_nodes:
            for chunk in node.chunks:
                chunk_id = chunk.get("chunk_id") or chunk.get("id")
                if chunk_id and chunk_id not in chunks_map:
                    chunks_map[chunk_id] = chunk

        # Aggregate files (deduplicate by file_path)
        files_map: dict[str, str] = {}
        nodes_with_data_loss: list[str] = []  # Track nodes with chunks but no files

        for node in all_nodes:
            # Detect data loss: node has chunks but no file_contents
            if node.chunks and not node.file_contents:
                query_preview = (
                    node.query[:50] + "..." if len(node.query) > 50 else node.query
                )
                nodes_with_data_loss.append(query_preview)

            for file_path, content in node.file_contents.items():
                if file_path not in files_map:
                    files_map[file_path] = content

        # FAIL-FAST: If ALL nodes lost data (chunks exist but no files aggregated), raise error
        # This catches cascading failures from file reading errors
        if chunks_map and not files_map:
            logger.error(
                f"DATA LOSS: Aggregation found {len(chunks_map)} unique chunks "
                f"but ZERO files across {len(all_nodes)} nodes. "
                f"Nodes with data loss: {nodes_with_data_loss}"
            )
            raise RuntimeError(
                f"Complete data loss during aggregation: "
                f"Found {len(chunks_map)} chunks but failed to read ANY files. "
                f"{len(nodes_with_data_loss)} nodes had chunks but empty file_contents. "
                f"This indicates file reading failed for all nodes - check token budgets and file paths. "
                f"Failed queries: {', '.join(nodes_with_data_loss[:5])}"
                + (
                    f" and {len(nodes_with_data_loss) - 5} more"
                    if len(nodes_with_data_loss) > 5
                    else ""
                )
            )

        # WARN: If SOME nodes lost data (partial data loss), log warning but continue
        if nodes_with_data_loss:
            logger.warning(
                f"Partial data loss: {len(nodes_with_data_loss)}/{len(all_nodes)} nodes "
                f"found chunks but failed to read files. "
                f"Synthesis will proceed with {len(files_map)} files from successful nodes. "
                f"Failed queries: {', '.join(nodes_with_data_loss[:3])}"
                + (
                    f" and {len(nodes_with_data_loss) - 3} more"
                    if len(nodes_with_data_loss) > 3
                    else ""
                )
            )

        # Calculate statistics
        total_chunks_found = sum(len(node.chunks) for node in all_nodes)
        total_files_found = sum(len(node.file_contents) for node in all_nodes)

        stats = {
            "total_nodes": len(all_nodes),
            "unique_chunks": len(chunks_map),
            "unique_files": len(files_map),
            "total_chunks_found": total_chunks_found,
            "total_files_found": total_files_found,
            "deduplication_ratio_chunks": (
                f"{total_chunks_found / len(chunks_map):.2f}x" if chunks_map else "N/A"
            ),
            "deduplication_ratio_files": (
                f"{total_files_found / len(files_map):.2f}x" if files_map else "N/A"
            ),
        }

        logger.info(
            f"Aggregation complete: {stats['unique_chunks']} unique chunks, "
            f"{stats['unique_files']} unique files from {stats['total_nodes']} nodes"
        )

        return {
            "chunks": list(chunks_map.values()),
            "files": files_map,
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
        return await self._synthesis_engine._single_pass_synthesis(
            root_query, chunks, files, context, synthesis_budgets
        )

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
        return await self._synthesis_engine._cluster_sources_for_synthesis(
            chunks, files, synthesis_budgets
        )

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
        return await self._synthesis_engine._map_synthesis_on_cluster(
            cluster, root_query, chunks, synthesis_budgets
        )

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
        return await self._synthesis_engine._reduce_synthesis(
            root_query, cluster_results, all_chunks, all_files, synthesis_budgets
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
                if node.is_file and file_reference_map and node.full_path in file_reference_map:
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

    def _calculate_synthesis_budgets(
        self, repo_stats: dict[str, Any]
    ) -> dict[str, int]:
        """Calculate synthesis token budgets based on repository size.

        Output budget is FIXED at 30k tokens for reasoning models (includes thinking + output).
        Only INPUT budget scales with repo size from small repos (~65k total) to large repos
        (~185k total) using piecewise linear brackets with diminishing returns.

        Args:
            repo_stats: Repository statistics from get_stats() including chunk count

        Returns:
            Dictionary with input_tokens, output_tokens, overhead_tokens, total_tokens
        """
        total_chunks = repo_stats.get("chunks", 0)

        # Estimate LOC from chunk count
        estimated_loc = total_chunks * CHUNKS_TO_LOC_ESTIMATE

        # Scale INPUT budget based on repository size (piecewise linear brackets with diminishing returns)
        if estimated_loc < LOC_THRESHOLD_TINY:
            # Very small repos: minimal input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_TINY
        elif estimated_loc < LOC_THRESHOLD_SMALL:
            # Small repos: moderate input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_SMALL
        elif estimated_loc < LOC_THRESHOLD_MEDIUM:
            # Medium repos: standard input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_MEDIUM
        else:
            # Large repos (>= 1M LOC): maximum input context
            input_tokens = SYNTHESIS_INPUT_TOKENS_LARGE

        overhead_tokens = SINGLE_PASS_OVERHEAD_TOKENS
        total_tokens = input_tokens + OUTPUT_TOKENS_WITH_REASONING + overhead_tokens

        logger.debug(
            f"Synthesis budgets for ~{estimated_loc:,} LOC: "
            f"input={input_tokens:,}, output={OUTPUT_TOKENS_WITH_REASONING:,}, total={total_tokens:,}"
        )

        return {
            "input_tokens": input_tokens,
            "output_tokens": OUTPUT_TOKENS_WITH_REASONING,
            "overhead_tokens": overhead_tokens,
            "total_tokens": total_tokens,
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
