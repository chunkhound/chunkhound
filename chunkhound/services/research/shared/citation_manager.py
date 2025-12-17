"""Citation management for deep research synthesis.

This module handles all citation-related operations for the deep research service,
including reference mapping, citation remapping, validation, and source footer generation.

Citation System Architecture:
- File Reference Map: Maps file paths to sequential reference numbers [1], [2], [3]...
- Cluster Citations: Local [N] citations within individual cluster summaries
- Global Citations: Remapped [N] citations in the final combined synthesis
- Source Footer: Tree-structured display of analyzed files with chunk line ranges

Citation Flow:
1. Filter chunks to only include files present in synthesis
2. Build file reference map (alphabetically sorted, 1-indexed)
3. Generate reference table for LLM prompt (tells LLM which [N] to use)
4. Remap cluster-local citations to global numbers (for map-reduce synthesis)
5. Validate all citations reference valid files
6. Build sources footer with file tree and chunk ranges

Usage:
    manager = CitationManager()

    # Build reference map for synthesis
    ref_map = manager.build_file_reference_map(chunks, files)

    # Generate reference table for LLM
    table = manager.format_reference_table(ref_map)

    # Remap cluster citations to global numbers
    remapped = manager.remap_cluster_citations(cluster_text, cluster_map, global_map)

    # Validate citations
    invalid = manager.validate_citation_references(text, ref_map)

    # Build sources footer
    footer = manager.build_sources_footer(chunks, files, ref_map)
"""

import re
from typing import Any

from loguru import logger

# Pre-compiled regex patterns for citation processing
_CITATION_PATTERN = re.compile(r"\[\d+\]")  # Matches [N] citations


class CitationManager:
    """Manages citation reference mapping and validation for research synthesis."""

    def filter_chunks_to_files(
        self,
        chunks: list[dict[str, Any]],
        files: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Filter chunks to only include those from specified files.

        Ensures consistency between reference map, citations, and footer by
        only including chunks from files that were actually synthesized.

        Args:
            chunks: List of all chunks from BFS traversal
            files: Dictionary of files used in synthesis

        Returns:
            Filtered list of chunks matching files dict

        Examples:
            >>> chunks = [
            ...     {"file_path": "src/main.py", "content": "..."},
            ...     {"file_path": "tests/test.py", "content": "..."},
            ...     {"file_path": "docs/readme.md", "content": "..."},
            ... ]
            >>> files = {"src/main.py": "...", "tests/test.py": "..."}
            >>> filtered = manager.filter_chunks_to_files(chunks, files)
            >>> len(filtered)
            2
        """
        return [c for c in chunks if c.get("file_path") in files]

    def build_file_reference_map(
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
            >>> ref_map = manager.build_file_reference_map(chunks, files)
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

    def format_reference_table(self, file_reference_map: dict[str, int]) -> str:
        """Format file reference mapping as markdown table for LLM prompt.

        Args:
            file_reference_map: Dictionary mapping file_path -> reference number

        Returns:
            Formatted markdown table showing reference numbers

        Examples:
            >>> ref_map = {"src/main.py": 1, "tests/test.py": 2}
            >>> table = manager.format_reference_table(ref_map)
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

    def remap_cluster_citations(
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
            >>> remapped = manager.remap_cluster_citations(
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

    def validate_citation_references(
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
            >>> invalid = manager.validate_citation_references(text, ref_map)
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

    def build_sources_footer(
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
            file_reference_map: Optional mapping of file_path -> reference number

        Returns:
            Formatted markdown footer with source information

        Examples:
            >>> chunks = [
            ...     {"file_path": "src/main.py", "start_line": 10, "end_line": 25},
            ...     {"file_path": "src/main.py", "start_line": 50, "end_line": 75},
            ...     {"file_path": "tests/test.py", "start_line": 5, "end_line": 15}
            ... ]
            >>> files = {"src/main.py": "...", "tests/test.py": "..."}
            >>> footer = manager.build_sources_footer(chunks, files)
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
