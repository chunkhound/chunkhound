"""Context management for deep research service.

This module handles tracking explored context during BFS traversal:
- Collecting ancestor data from parent nodes
- Updating global explored state across the BFS graph
- Building exploration summaries for follow-up generation
- Detecting duplicate chunks to avoid redundant exploration
- Detecting new information vs previously explored content
"""

from typing import TYPE_CHECKING, Any

from chunkhound.services.research.shared.models import BFSNode

# Threshold for considering a file "well covered" by explored chunks.
# Files with more than this many covered lines are marked as fully read.
WELL_COVERED_LINE_THRESHOLD = 50

if TYPE_CHECKING:
    from chunkhound.services.deep_research_service import DeepResearchService


class ContextManager:
    """Manages context tracking and duplicate detection during BFS research."""

    def __init__(self, service: "DeepResearchService"):
        """Initialize context manager.

        Args:
            service: Parent DeepResearchService instance (for accessing helper methods)
        """
        self._service = service

    def collect_ancestor_data(self, node: BFSNode) -> dict[str, Any]:
        """Traverse parent chain and collect all ancestor chunks/files.

        NOTE: This method is now deprecated in favor of global_explored_data tracking.
        Kept for backward compatibility but not actively used in BFS duplicate detection.

        Args:
            node: Current BFS node

        Returns:
            Dictionary with:
                - files_fully_read: set[str] - Paths of fully-read files
                - chunk_ranges: dict[str, list[tuple[int, int]]] - file → [(start, end)]
                - file_line_coverage: dict[str, set[int]] - file → set of covered lines
        """
        files_fully_read: set[str] = set()
        chunk_ranges: dict[str, list[tuple[int, int]]] = {}
        file_line_coverage: dict[str, set[int]] = {}

        current = node.parent
        while current:
            # Track line coverage from chunks (replaces file content-based detection)
            for chunk in current.chunks:
                file_path = chunk.get("file_path")
                if not file_path:
                    continue
                start_line = chunk.get("start_line", 1)
                end_line = chunk.get("end_line", 1)

                # Track line coverage per file
                file_line_coverage.setdefault(file_path, set()).update(
                    range(start_line, end_line + 1)
                )

                # Mark as "well covered" if substantial chunk coverage
                if len(file_line_coverage[file_path]) > WELL_COVERED_LINE_THRESHOLD:
                    files_fully_read.add(file_path)

                # Collect expanded chunk ranges
                expanded_range = self._service._get_chunk_expanded_range(chunk)
                chunk_ranges.setdefault(file_path, []).append(expanded_range)

            current = current.parent

        return {
            "files_fully_read": files_fully_read,
            "chunk_ranges": chunk_ranges,
            "file_line_coverage": file_line_coverage,
        }

    def update_global_explored_data(
        self, global_explored_data: dict[str, Any], node: BFSNode
    ) -> None:
        """Update global explored data with discoveries from a single node.

        This allows sibling nodes and future nodes to detect duplicates across the entire BFS graph,
        not just their ancestor chain. Critical for preventing redundant exploration.

        Args:
            global_explored_data: Global state dict with files_fully_read, chunk_ranges,
                chunks, and file_line_coverage
            node: BFS node whose discoveries should be added to global state
        """
        # Track line coverage from chunks instead of file content length
        for chunk in node.chunks:
            file_path = chunk.get("file_path")
            if not file_path:
                continue
            start_line = chunk.get("start_line", 1)
            end_line = chunk.get("end_line", 1)

            # Track line coverage per file
            coverage = global_explored_data.setdefault("file_line_coverage", {})
            lines = range(start_line, end_line + 1)
            coverage.setdefault(file_path, set()).update(lines)

            # Mark as "well covered" if substantial chunk coverage
            if len(coverage[file_path]) > WELL_COVERED_LINE_THRESHOLD:
                global_explored_data["files_fully_read"].add(file_path)

        # Add expanded chunk ranges and chunks
        for chunk in node.chunks:
            file_path = chunk.get("file_path")
            if file_path:
                expanded_range = self._service._get_chunk_expanded_range(chunk)
                global_explored_data["chunk_ranges"].setdefault(file_path, []).append(
                    expanded_range
                )
                # Store chunk for building exploration gist
                global_explored_data["chunks"].append(chunk)

    def build_exploration_gist(
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
        footer = self._service._citation_manager.build_sources_footer(chunks, files)

        # Return just the tree portion (skip "---" separator at the start)
        # Keep the "## Sources" header and the tree
        return footer

    def is_chunk_duplicate(
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

    def detect_new_information(
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
            expanded_range = self._service._get_chunk_expanded_range(chunk)

            is_duplicate = self.is_chunk_duplicate(
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
