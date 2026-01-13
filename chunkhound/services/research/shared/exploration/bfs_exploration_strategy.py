"""BFS tree exploration strategy (V1 algorithm).

Uses breadth-first search with follow-up question generation to explore
codebase relationships iteratively.
"""

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.research_config import ResearchConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.chunk_dedup import get_chunk_id
from chunkhound.services.research.shared.exploration.elbow_filter import (
    filter_chunks_by_elbow,
    get_unified_score,
)
from chunkhound.services.research.shared.models import (
    FILE_CONTENT_TOKENS_MAX,
    FILE_CONTENT_TOKENS_MIN,
    MAX_CHUNKS_PER_FILE_REPR,
    MAX_FOLLOWUP_QUESTIONS,
    MAX_TOKENS_PER_FILE_REPR,
    ResearchContext,
)
from chunkhound.services.research.shared.unified_search import UnifiedSearch
from chunkhound.services.research.v1.question_generator import QuestionGenerator

# BFS-specific constants (not in shared/models.py)
MAX_DEPTH = 1  # Fixed shallow exploration (empirically optimal)


@dataclass
class BFSExplorationNode:
    """Node in the BFS exploration graph."""

    query: str
    parent: "BFSExplorationNode | None" = None
    depth: int = 0
    children: list["BFSExplorationNode"] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    file_contents: dict[str, str] = field(default_factory=dict)
    node_id: int = 0
    is_terminated_leaf: bool = False
    new_chunk_count: int = 0
    duplicate_chunk_count: int = 0


class BFSExplorationStrategy:
    """BFS tree exploration strategy (V1 algorithm).

    Implements ExplorationStrategy by using breadth-first search with
    follow-up question generation to iteratively explore codebase relationships.

    The strategy:
    1. Treats initial_chunks as root node discoveries
    2. Generates follow-up questions from root content
    3. Executes BFS traversal for follow-up exploration
    4. Tracks global explored state to prevent redundant exploration
    5. Aggregates all discovered chunks
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: EmbeddingManager,
        db_services: DatabaseServices,
        config: ResearchConfig | None = None,
        import_resolver: Any | None = None,
        import_context_service: Any | None = None,
        max_depth: int = MAX_DEPTH,
    ):
        """Initialize BFS exploration strategy.

        Args:
            llm_manager: LLM manager for follow-up question generation
            embedding_manager: Embedding manager for semantic search
            db_services: Database services bundle
            config: Optional research configuration
            import_resolver: Optional import resolver (unused, for interface compat)
            import_context_service: Optional import context service (unused)
            max_depth: Maximum BFS depth (default: 1)
        """
        self._llm_manager = llm_manager
        self._embedding_manager = embedding_manager
        self._db_services = db_services
        self._config = config
        self._max_depth = max_depth
        self._node_counter = 0

        self._unified_search = UnifiedSearch(
            db_services=db_services,
            embedding_manager=embedding_manager,
            config=config,
        )
        self._question_generator = QuestionGenerator(
            llm_manager, import_context_service=import_context_service
        )

    @property
    def name(self) -> str:
        """Strategy identifier."""
        return "bfs"

    def _get_next_node_id(self) -> int:
        """Get next unique node ID."""
        self._node_counter += 1
        return self._node_counter

    async def _traverse_bfs_tree(
        self,
        root: BFSExplorationNode,
        context: ResearchContext,
        global_explored_data: dict[str, Any],
        path_filter: str | None,
        constants_context: str,
        log_prefix: str = "BFSExplorationStrategy",
    ) -> list[BFSExplorationNode]:
        """Execute BFS traversal from root node.

        Performs breadth-first search with follow-up question generation,
        tracking global explored state to prevent redundant exploration.

        Args:
            root: Root node with initial chunks
            context: Research context for question generation
            global_explored_data: Global state tracking explored chunks/files
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts
            log_prefix: Prefix for log messages (for explore vs explore_raw)

        Returns:
            List of all visited nodes (including root) in BFS order
        """
        current_level = [root]
        all_nodes: list[BFSExplorationNode] = [root]

        for depth in range(0, self._max_depth + 1):
            if not current_level:
                break

            logger.info(
                f"{log_prefix}: Processing depth {depth}/{self._max_depth}, "
                f"nodes: {len(current_level)}"
            )

            # Process all nodes at this level concurrently
            node_contexts = []
            for node in current_level:
                node_context = ResearchContext(
                    root_query=context.root_query,
                    ancestors=context.ancestors.copy(),
                    traversal_path=context.traversal_path.copy(),
                )
                node_contexts.append((node, node_context))

            # Process nodes concurrently
            node_tasks = [
                self._process_node(
                    node,
                    node_ctx,
                    depth,
                    global_explored_data,
                    path_filter,
                    constants_context,
                )
                for node, node_ctx in node_contexts
            ]
            children_lists = await asyncio.gather(*node_tasks, return_exceptions=True)

            # Collect children
            next_level: list[BFSExplorationNode] = []
            for (node, node_ctx), children_result in zip(node_contexts, children_lists):
                if isinstance(children_result, Exception):
                    logger.error(
                        f"{log_prefix} node failed for '{node.query[:50]}...': {children_result}"
                    )
                    continue

                assert isinstance(children_result, list)
                node.children.extend(children_result)
                next_level.extend(children_result)
                all_nodes.extend(children_result)

                # Update global explored data
                if not node.is_terminated_leaf and node.chunks:
                    self._update_global_explored_data(global_explored_data, node)

            # Update global context
            for node, _ in node_contexts:
                if node.query not in context.ancestors:
                    context.ancestors.append(node.query)

            # Synthesize if too many questions
            if len(next_level) > MAX_FOLLOWUP_QUESTIONS:
                next_level = await self._synthesize_questions(
                    next_level, context, MAX_FOLLOWUP_QUESTIONS
                )

            current_level = next_level

        return all_nodes

    async def explore(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
        """Execute BFS exploration from initial chunks.

        Creates a virtual root node from initial_chunks, then generates
        follow-up questions and explores them via BFS traversal.

        Args:
            root_query: Original research query (injected in all LLM prompts)
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1 (unused in BFS)
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (expanded_chunks, exploration_stats, file_contents)
        """
        if not initial_chunks:
            logger.warning("BFSExplorationStrategy: No initial chunks to explore")
            return initial_chunks, {
                "nodes_explored": 0,
                "depth_reached": 0,
                "chunks_total": 0,
                "chunks_added": 0,
                "files_read": 0,
            }, {}

        logger.info(
            f"BFSExplorationStrategy: Starting BFS exploration "
            f"with {len(initial_chunks)} initial chunks, max_depth={self._max_depth}"
        )

        # Reset node counter for fresh exploration
        self._node_counter = 0

        # Create virtual root node from initial chunks
        root = BFSExplorationNode(
            query=root_query,
            depth=0,
            node_id=self._get_next_node_id(),
            chunks=initial_chunks,
        )

        # File reading deferred to synthesis
        root.file_contents = {}

        context = ResearchContext(root_query=root_query)

        # Initialize global explored data from initial chunks
        global_explored_data: dict[str, Any] = {
            "files_explored": set(),
            "chunk_ranges": {},
            "chunks": list(initial_chunks),
        }
        self._update_global_explored_data(global_explored_data, root)

        # BFS traversal
        all_nodes = await self._traverse_bfs_tree(
            root, context, global_explored_data, path_filter, constants_context,
            log_prefix="BFSExplorationStrategy"
        )

        # Aggregate all chunks and file contents from BFS tree
        all_chunks = self._aggregate_chunks(all_nodes)
        all_files = self._aggregate_files(all_nodes)

        # Apply elbow detection and reranking before returning
        filtered_chunks, filtered_files, filter_stats = (
            await self._filter_for_synthesis(all_chunks, root_query)
        )

        stats = {
            "nodes_explored": len(all_nodes),
            "depth_reached": max(n.depth for n in all_nodes) if all_nodes else 0,
            "chunks_total": len(all_chunks),
            "chunks_before": len(initial_chunks),
            "chunks_added": len(all_chunks) - len(initial_chunks),
            "files_read": len(all_files),
            **filter_stats,
        }

        logger.info(
            f"BFSExplorationStrategy: Complete. "
            f"Explored {stats['nodes_explored']} nodes, "
            f"added {stats['chunks_added']} chunks, read {stats['files_read']} files, "
            f"filtered to {len(filtered_chunks)} chunks/{len(filtered_files)} files"
        )

        return filtered_chunks, stats, filtered_files

    async def _process_node(
        self,
        node: BFSExplorationNode,
        context: ResearchContext,
        depth: int,
        global_explored_data: dict[str, Any],
        path_filter: str | None,
        constants_context: str,
    ) -> list[BFSExplorationNode]:
        """Process a single BFS node.

        Args:
            node: BFS node to process
            context: Research context
            depth: Current depth in graph
            global_explored_data: Global state tracking all explored chunks/files
            path_filter: Optional path filter for searches
            constants_context: Constants context for LLM prompts

        Returns:
            List of child nodes (follow-up questions)
        """
        logger.debug(f"Processing node at depth {depth}: '{node.query[:60]}...'")

        # Root node (depth 0) already has chunks from initial_chunks
        # For child nodes, perform search
        if depth > 0 or not node.chunks:
            search_query = self._build_search_query(node.query, context)
            chunks = await self._unified_search.unified_search(
                query=search_query,
                context=context,
                path_filter=path_filter,
            )
            node.chunks = chunks

            if not chunks:
                logger.debug(f"No chunks found for query: '{node.query[:50]}...'")
                return []

            # File reading deferred to synthesis
            node.file_contents = {}

        # Check for new information (termination rule)
        has_new_info, dedup_stats = self._detect_new_information(
            node, node.chunks, global_explored_data
        )
        node.new_chunk_count = dedup_stats["new_chunks"]
        node.duplicate_chunk_count = dedup_stats["duplicate_chunks"]

        if not has_new_info:
            logger.info(
                f"[Termination] Node '{node.query[:40]}...' at depth {depth} "
                f"found 0 new chunks. Marking as terminated leaf."
            )
            node.is_terminated_leaf = True
            return []

        # Skip follow-up generation at max depth
        if depth >= self._max_depth:
            logger.debug(
                f"Node at max depth {depth}/{self._max_depth}, skipping follow-ups"
            )
            return []

        # Generate follow-up questions
        exploration_gist = self._build_exploration_gist(global_explored_data)
        self._question_generator.set_node_counter(self._node_counter)

        follow_ups = await self._question_generator.generate_follow_up_questions(
            query=node.query,
            context=context,
            file_contents=node.file_contents,
            chunks=node.chunks,
            global_explored_data=global_explored_data,
            exploration_gist=exploration_gist,
            max_input_tokens=self._get_file_budget(depth),
            depth=depth,
            max_depth=self._max_depth,
            constants_context=constants_context,
        )

        # Sync node counter back
        self._node_counter = self._question_generator._node_counter

        # Create child nodes
        children = []
        for follow_up in follow_ups[:MAX_FOLLOWUP_QUESTIONS]:
            child = BFSExplorationNode(
                query=follow_up,
                parent=node,
                depth=depth + 1,
                node_id=self._get_next_node_id(),
            )
            children.append(child)

        return children

    def _build_search_query(self, query: str, context: ResearchContext) -> str:
        """Build search query combining input with BFS context."""
        if not context.ancestors:
            return query

        parent_context = (
            context.ancestors[-2:]
            if len(context.ancestors) >= 2
            else context.ancestors[-1:]
        )
        context_str = " → ".join(parent_context)
        return f"{query} | Context: {context_str}"

    def _get_file_budget(self, depth: int) -> int:
        """Get file content token budget based on depth."""
        depth_ratio = depth / max(self._max_depth, 1)
        return int(
            FILE_CONTENT_TOKENS_MIN
            + (FILE_CONTENT_TOKENS_MAX - FILE_CONTENT_TOKENS_MIN) * depth_ratio
        )

    async def _read_files_with_budget(
        self, chunks: list[dict[str, Any]], max_tokens: int | None = None
    ) -> dict[str, str]:
        """Read files containing chunks within optional token budget.

        When max_tokens is None, reads all files without budget constraint.
        """
        files_to_chunks: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk.get("file_path") or chunk.get("path", "")
            if file_path:
                files_to_chunks.setdefault(file_path, []).append(chunk)

        file_contents: dict[str, str] = {}
        total_tokens = 0
        llm = self._llm_manager.get_utility_provider()
        base_dir = self._db_services.provider.get_base_directory()

        for file_path, file_chunks in files_to_chunks.items():
            # Skip budget check if unlimited
            if max_tokens is not None and total_tokens >= max_tokens:
                break

            try:
                path = (
                    Path(file_path)
                    if Path(file_path).is_absolute()
                    else base_dir / file_path
                )
                if not path.exists():
                    continue

                content = path.read_text(encoding="utf-8", errors="ignore")
                estimated_tokens = llm.estimate_tokens(content)

                # Always include file if unlimited, else check budget
                if max_tokens is None or total_tokens + estimated_tokens <= max_tokens:
                    file_contents[file_path] = content
                    total_tokens += estimated_tokens
                else:
                    # Truncate to fit (only when budget is set)
                    remaining = max_tokens - total_tokens
                    if remaining > 500:
                        chars = remaining * 4
                        file_contents[file_path] = content[:chars]
                    break

            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")

        return file_contents

    async def _read_files_for_synthesis(
        self,
        chunks: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Read files for elbow-filtered chunks without token budget.

        Elbow detection already filtered chunks by relevance, so we read
        all files for the filtered set without artificial token caps.

        Args:
            chunks: Elbow-filtered chunks to read files for

        Returns:
            Dict mapping file_path -> content
        """
        return await self._read_files_with_budget(chunks, max_tokens=None)

    def _detect_new_information(
        self,
        node: BFSExplorationNode,
        chunks: list[dict[str, Any]],
        global_explored_data: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Detect if node has new information vs all previously explored nodes."""
        if not node.parent:
            # Root node always has new info
            return (
                True,
                {"new_chunks": len(chunks), "duplicate_chunks": 0, "total_chunks": len(chunks)},
            )

        if not chunks:
            return (False, {"new_chunks": 0, "duplicate_chunks": 0, "total_chunks": 0})

        new_count = 0
        duplicate_count = 0

        for chunk in chunks:
            expanded_range = self._get_chunk_expanded_range(chunk)
            if self._is_chunk_duplicate(chunk, expanded_range, global_explored_data):
                duplicate_count += 1
            else:
                new_count += 1

        return (
            new_count > 0,
            {"new_chunks": new_count, "duplicate_chunks": duplicate_count, "total_chunks": len(chunks)},
        )

    def _get_chunk_expanded_range(self, chunk: dict[str, Any]) -> tuple[int, int]:
        """Get expanded line range for chunk."""
        start = chunk.get("expanded_start_line") or chunk.get("start_line", 1)
        end = chunk.get("expanded_end_line") or chunk.get("end_line", 1)
        return (start, end)

    def _is_chunk_duplicate(
        self,
        chunk: dict[str, Any],
        chunk_expanded_range: tuple[int, int],
        explored_data: dict[str, Any],
    ) -> bool:
        """Check if chunk is 100% duplicate of previously explored data."""
        file_path = chunk.get("file_path")
        if not file_path:
            return False

        expanded_start, expanded_end = chunk_expanded_range

        # Check if file was already explored (50+ lines seen)
        if file_path in explored_data["files_explored"]:
            return True

        # Check for 100% containment
        for prev_start, prev_end in explored_data["chunk_ranges"].get(file_path, []):
            if expanded_start >= prev_start and expanded_end <= prev_end:
                return True

        return False

    def _update_global_explored_data(
        self, global_explored_data: dict[str, Any], node: BFSExplorationNode
    ) -> None:
        """Update global explored data with discoveries from a node."""
        # Track line coverage from chunks instead of file content length
        for chunk in node.chunks:
            file_path = chunk.get("file_path")
            if not file_path:
                continue
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", 1)

            # Track line coverage per file
            coverage = global_explored_data.setdefault("file_line_coverage", {})
            coverage.setdefault(file_path, set()).update(range(start_line, end_line + 1))

            # Mark as "explored" if we've seen 50+ lines (heuristic to prevent re-exploration)
            if len(coverage[file_path]) > 50:
                global_explored_data["files_explored"].add(file_path)

        for chunk in node.chunks:
            file_path = chunk.get("file_path")
            if file_path:
                expanded_range = self._get_chunk_expanded_range(chunk)
                global_explored_data["chunk_ranges"].setdefault(file_path, []).append(
                    expanded_range
                )
                global_explored_data["chunks"].append(chunk)

    def _build_exploration_gist(self, global_explored_data: dict[str, Any]) -> str | None:
        """Build summary of explored files for follow-up generation."""
        chunks = global_explored_data["chunks"]
        if not chunks:
            return None

        files = sorted({c.get("file_path") for c in chunks if c.get("file_path")})
        if not files:
            return None

        return "## Already Explored\n" + "\n".join(f"- {f}" for f in files[:20])

    async def _synthesize_questions(
        self,
        nodes: list[BFSExplorationNode],
        context: ResearchContext,
        target_count: int,
    ) -> list[BFSExplorationNode]:
        """Synthesize questions when too many are generated."""
        self._question_generator.set_node_counter(self._node_counter)

        # Convert to format expected by question generator
        from chunkhound.services.research.shared.models import BFSNode

        bfs_nodes = [
            BFSNode(
                query=n.query,
                depth=n.depth,
                node_id=n.node_id,
                chunks=n.chunks,
            )
            for n in nodes
        ]

        result = await self._question_generator.synthesize_questions(
            nodes=bfs_nodes,
            context=context,
            target_count=target_count,
        )

        self._node_counter = self._question_generator._node_counter

        # Convert back to exploration nodes
        return [
            BFSExplorationNode(
                query=n.query,
                depth=n.depth,
                node_id=n.node_id,
            )
            for n in result
        ]

    def _aggregate_chunks(
        self, all_nodes: list[BFSExplorationNode]
    ) -> list[dict[str, Any]]:
        """Aggregate all chunks from BFS tree, deduplicated by chunk_id.

        Keeps the highest-scoring version when the same chunk appears multiple times,
        consistent with v3 parallel strategy's merge behavior.
        """
        chunks_map: dict[int | str, dict[str, Any]] = {}
        for node in all_nodes:
            for chunk in node.chunks:
                chunk_id = get_chunk_id(chunk)
                if not chunk_id:
                    continue
                existing = chunks_map.get(chunk_id)
                if existing is None:
                    chunks_map[chunk_id] = chunk
                elif get_unified_score(chunk) > get_unified_score(existing):
                    chunks_map[chunk_id] = chunk
        return list(chunks_map.values())

    def _aggregate_files(
        self, all_nodes: list[BFSExplorationNode]
    ) -> dict[str, str]:
        """Aggregate all file contents from BFS tree.

        Merges file_contents from all nodes. If the same file appears in multiple
        nodes, the first occurrence (by node order) is kept.

        Args:
            all_nodes: All BFS nodes from exploration

        Returns:
            Dict mapping file_path -> content
        """
        all_files: dict[str, str] = {}
        for node in all_nodes:
            for file_path, content in node.file_contents.items():
                if file_path not in all_files:
                    all_files[file_path] = content
        return all_files

    async def explore_raw(
        self,
        root_query: str,
        initial_chunks: list[dict[str, Any]],
        phase1_threshold: float,
        path_filter: str | None = None,
        constants_context: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute BFS exploration without elbow filtering (for parallel mode).

        Returns raw aggregated chunks without elbow detection or file reading.
        The caller (ParallelExplorationStrategy) handles unified elbow detection
        and file reading after merging results from multiple strategies.

        Args:
            root_query: Original research query
            initial_chunks: Chunks from Phase 1 coverage retrieval
            phase1_threshold: Quality threshold floor from Phase 1 (unused in BFS)
            path_filter: Optional path filter for searches
            constants_context: Evidence ledger constants for LLM prompts

        Returns:
            Tuple of (raw_chunks, exploration_stats):
                - raw_chunks: All aggregated chunks without filtering
                - exploration_stats: Statistics about exploration process
        """
        if not initial_chunks:
            logger.warning("BFSExplorationStrategy.explore_raw: No initial chunks")
            return [], {
                "nodes_explored": 0,
                "depth_reached": 0,
                "chunks_total": 0,
                "chunks_added": 0,
            }

        logger.info(
            f"BFSExplorationStrategy.explore_raw: Starting BFS exploration "
            f"with {len(initial_chunks)} initial chunks, max_depth={self._max_depth}"
        )

        # Reset node counter for fresh exploration
        self._node_counter = 0

        # Create virtual root node from initial chunks
        root = BFSExplorationNode(
            query=root_query,
            depth=0,
            node_id=self._get_next_node_id(),
            chunks=initial_chunks,
        )
        root.file_contents = {}

        context = ResearchContext(root_query=root_query)

        # Initialize global explored data from initial chunks
        global_explored_data: dict[str, Any] = {
            "files_explored": set(),
            "chunk_ranges": {},
            "chunks": list(initial_chunks),
        }
        self._update_global_explored_data(global_explored_data, root)

        # BFS traversal
        all_nodes = await self._traverse_bfs_tree(
            root, context, global_explored_data, path_filter, constants_context,
            log_prefix="BFSExplorationStrategy.explore_raw"
        )

        # Aggregate all chunks (NO elbow filtering, NO file reading)
        all_chunks = self._aggregate_chunks(all_nodes)

        stats = {
            "nodes_explored": len(all_nodes),
            "depth_reached": max(n.depth for n in all_nodes) if all_nodes else 0,
            "chunks_total": len(all_chunks),
            "chunks_before": len(initial_chunks),
            "chunks_added": len(all_chunks) - len(initial_chunks),
        }

        logger.info(
            f"BFSExplorationStrategy.explore_raw: Complete. "
            f"Explored {stats['nodes_explored']} nodes, "
            f"found {stats['chunks_total']} total chunks"
        )

        return all_chunks, stats

    async def _rerank_files_in_batches(
        self,
        root_query: str,
        file_paths: list[str],
        file_documents: list[str],
    ) -> list[tuple[str, float]]:
        """Rerank files with explicit batch management at BFS layer.

        Args:
            root_query: User's research question
            file_paths: List of file paths (parallel to file_documents)
            file_documents: List of file representative documents

        Returns:
            List of (file_path, relevance_score) tuples, sorted by descending score
        """
        embedding_provider = self._embedding_manager.get_provider()
        if not embedding_provider:
            logger.warning("No embedding provider available for reranking files")
            return []

        # Get provider's max batch size
        max_batch = embedding_provider.get_max_rerank_batch_size()

        # Single batch - direct call
        if len(file_documents) <= max_batch:
            logger.debug(f"Reranking {len(file_documents)} files in single batch")
            try:
                rerank_results = await embedding_provider.rerank(
                    query=root_query, documents=file_documents, top_k=None
                )

                # Convert to (file_path, score) tuples
                results = []
                for result in rerank_results:
                    idx = result.index
                    score = result.score

                    # Validate index bounds
                    if idx < 0 or idx >= len(file_paths):
                        logger.warning(
                            f"Reranker returned invalid index {idx} "
                            f"(valid range: 0-{len(file_paths)-1}), skipping"
                        )
                        continue

                    results.append((file_paths[idx], float(score)))

                return results
            except Exception as e:
                logger.error(f"Single batch reranking failed: {e}")
                raise  # Re-raise since all batches failed

        # Multiple batches required
        num_batches = math.ceil(len(file_documents) / max_batch)
        logger.info(
            f"Reranking {len(file_documents)} files in {num_batches} batches "
            f"of {max_batch} ({embedding_provider.name})"
        )

        all_results: list[tuple[str, float]] = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_batch
            end_idx = min(start_idx + max_batch, len(file_documents))

            batch_documents = file_documents[start_idx:end_idx]
            batch_file_paths = file_paths[start_idx:end_idx]

            logger.debug(
                f"Reranking batch {batch_idx + 1}/{num_batches} "
                f"({len(batch_documents)} files)"
            )

            try:
                rerank_results = await embedding_provider.rerank(
                    query=root_query, documents=batch_documents, top_k=None
                )

                # Convert to (file_path, score) tuples with adjusted indices
                for result in rerank_results:
                    batch_relative_idx = result.index
                    score = result.score

                    # Validate index within batch
                    if (
                        batch_relative_idx < 0
                        or batch_relative_idx >= len(batch_file_paths)
                    ):
                        logger.warning(
                            f"Batch {batch_idx + 1} returned invalid index "
                            f"{batch_relative_idx} (valid range: 0-{len(batch_file_paths)-1}), "
                            f"skipping"
                        )
                        continue

                    all_results.append((batch_file_paths[batch_relative_idx], float(score)))

            except Exception as e:
                logger.error(
                    f"Batch {batch_idx + 1}/{num_batches} reranking failed: {e}, "
                    f"continuing with remaining batches"
                )
                # Continue processing other batches
                continue

        return all_results

    async def _filter_for_synthesis(
        self,
        chunks: list[dict[str, Any]],
        root_query: str,
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
        """Apply elbow detection and file reranking before synthesis.

        This method filters chunks using elbow detection on score distribution,
        then reranks files by relevance to the root query to prevent diversity
        collapse from multi-depth BFS score accumulation.

        Args:
            chunks: All chunks from BFS traversal
            root_query: Original research query (for reranking files)

        Returns:
            Tuple of (filtered_chunks, selected_files, filter_stats)
        """
        llm = self._llm_manager.get_utility_provider()

        logger.info(f"Filtering chunks for synthesis: {len(chunks)} chunks")

        # Apply elbow detection using shared utility (handles sorting internally)
        original_count = len(chunks)
        sorted_chunks, elbow_stats = filter_chunks_by_elbow(chunks, score_key=None)
        logger.info(
            f"Elbow detection: keeping {len(sorted_chunks)}/{original_count} chunks"
        )

        # Read files for elbow-filtered chunks (token budget applied here)
        elbow_filtered_files = await self._read_files_for_synthesis(sorted_chunks)
        logger.info(
            f"Read {len(elbow_filtered_files)} files for "
            f"{len(sorted_chunks)} elbow-filtered chunks"
        )

        # Build file-to-chunks mapping (use read files only)
        file_to_chunks: dict[str, list[dict[str, Any]]] = {}
        for chunk in sorted_chunks:
            file_path = chunk.get("file_path", "")
            if file_path and file_path in elbow_filtered_files:
                if file_path not in file_to_chunks:
                    file_to_chunks[file_path] = []
                file_to_chunks[file_path].append(chunk)

        # Create file representative documents for reranking
        file_paths = []
        file_documents = []

        for file_path, file_chunks in file_to_chunks.items():
            # Sort chunks by score and take top N chunks
            sorted_file_chunks = sorted(
                file_chunks, key=get_unified_score, reverse=True
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

        # Rerank files by relevance to root query using explicit batch management
        rerank_results = await self._rerank_files_in_batches(
            root_query, file_paths, file_documents
        )

        # Initialize file priorities dict
        file_priorities: dict[str, float] = {}

        if rerank_results:
            # Build priority map from rerank scores
            for file_path, score in rerank_results:
                file_priorities[file_path] = score

            logger.info(
                f"Reranked {len(file_priorities)} files for synthesis budget allocation"
            )
        else:
            # Fallback: Use accumulated chunk scores
            logger.warning("File reranking returned no results, falling back to chunk scores")
            for file_path, file_chunks in file_to_chunks.items():
                file_priorities[file_path] = sum(
                    get_unified_score(c) for c in file_chunks
                )

            logger.info(f"Using chunk score fallback for {len(file_priorities)} files")

        # Sort files by priority score (highest first)
        sorted_files = sorted(file_priorities.items(), key=lambda x: x[1], reverse=True)

        # Include ALL files from elbow-filtered chunks (no token budget cap)
        selected_files: dict[str, str] = {}
        total_tokens = 0

        missing_files = []
        for file_path, priority in sorted_files:
            if file_path not in elbow_filtered_files:
                missing_files.append(file_path)
                continue

            content = elbow_filtered_files[file_path]
            content_tokens = llm.estimate_tokens(content)
            selected_files[file_path] = content
            total_tokens += content_tokens

        if missing_files:
            logger.debug(
                f"Synthesis budget: {len(missing_files)} files exceeded budget "
                f"(kept {len(selected_files)} files)"
            )

        filter_stats = {
            "files_selected": len(selected_files),
            "total_tokens": total_tokens,
            "elbow_filtered_chunks": len(sorted_chunks),
            "original_chunks": original_count,
        }

        logger.info(
            f"File selection complete: {len(selected_files)} files, "
            f"{total_tokens:,} tokens "
            f"(elbow: {len(sorted_chunks)}/{original_count} chunks)"
        )

        return sorted_chunks, selected_files, filter_stats
