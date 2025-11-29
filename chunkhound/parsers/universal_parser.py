"""Universal parser that unifies all language mappings with cAST algorithm.

This module provides the UniversalParser class that brings together:
1. TreeSitterEngine - Universal tree-sitter parsing engine
2. ConceptExtractor - Universal semantic concept extraction
3. cAST Algorithm - Research-backed optimal semantic chunking
4. Language Mappings - All 21 supported language mappings

The parser applies the cAST (Code AST) algorithm which uses a split-then-merge
recursive approach to create chunks that:
- Preserve syntactic integrity by aligning with AST boundaries
- Maximize information density through greedy merging
- Maintain language invariance across all supported languages
- Ensure plug-and-play compatibility with existing systems
"""

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tree_sitter import Tree

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ByteOffset,
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.core.utils import estimate_tokens
from chunkhound.interfaces.language_parser import ParseResult
from chunkhound.utils.chunk_deduplication import deduplicate_chunks, get_chunk_specificity
from chunkhound.utils.normalization import normalize_content

from .concept_extractor import ConceptExtractor
from .mapping_adapter import MappingAdapter
from .mappings.base import BaseMapping
from .universal_engine import TreeSitterEngine, UniversalChunk, UniversalConcept


@dataclass
class CASTConfig:
    """Configuration for cAST algorithm.

    Based on research paper: "cAST: Enhancing Code Retrieval-Augmented Generation
    with Structural Chunking via Abstract Syntax Tree"
    """

    max_chunk_size: int = 1200  # Reduced from 2000 (non-whitespace chars)
    min_chunk_size: int = 50  # Minimum chunk size to avoid tiny fragments
    merge_threshold: float = (
        0.8  # Merge siblings if combined size < threshold * max_size
    )
    preserve_structure: bool = True  # Prioritize syntactic boundaries
    greedy_merge: bool = True  # Greedily merge adjacent sibling nodes
    safe_token_limit: int = 6000  # Conservative token limit (well under 8191 API limit)


@dataclass
class ChunkMetrics:
    """Metrics for measuring chunk quality and size."""

    non_whitespace_chars: int
    total_chars: int
    lines: int
    ast_depth: int

    @classmethod
    def from_content(cls, content: str, ast_depth: int = 0) -> "ChunkMetrics":
        """Calculate metrics from content string."""
        non_ws = len(re.sub(r"\s", "", content))
        total = len(content)
        lines = len(content.split("\n"))
        return cls(non_ws, total, lines, ast_depth)

    def estimated_tokens(self, ratio: float = 3.5) -> int:
        """Estimate token count using character-based ratio.

        Args:
            ratio: Chars-to-tokens ratio (conservative default 3.5)
        """
        # Fallback to conservative ratio-based estimation
        return int(self.non_whitespace_chars / ratio)


class UniversalParser:
    """Universal parser that works with all supported languages using cAST algorithm.

    This parser combines:
    - TreeSitterEngine for universal AST parsing
    - ConceptExtractor for semantic extraction using language mappings
    - cAST algorithm for optimal chunk boundaries
    - Compatibility layer for existing Chunk/ParseResult interfaces
    """

    def __init__(
        self,
        engine: TreeSitterEngine,
        mapping: BaseMapping,
        cast_config: CASTConfig | None = None,
    ):
        """Initialize universal parser.

        Args:
            engine: TreeSitterEngine for this language
            mapping: BaseMapping for this language (adapted if needed)
            cast_config: Configuration for cAST algorithm
        """
        self.engine = engine
        self.base_mapping = mapping

        # Convert BaseMapping to LanguageMapping if needed
        if isinstance(mapping, BaseMapping) and not hasattr(
            mapping, "get_query_for_concept"
        ):
            # Use adapter to bridge BaseMapping to LanguageMapping protocol
            adapted_mapping = MappingAdapter(mapping)
        else:
            # Assume it already implements LanguageMapping protocol
            adapted_mapping = mapping  # type: ignore

        self.mapping = adapted_mapping
        self.extractor = ConceptExtractor(engine, adapted_mapping)
        self.cast_config = cast_config or CASTConfig()

        # Statistics
        self._total_files_parsed = 0
        self._total_chunks_created = 0

    def _estimate_tokens(self, content: str) -> int:
        """Helper method to estimate tokens using centralized utility."""
        return estimate_tokens(content)

    @property
    def language_name(self) -> str:
        """Get the language name."""
        if self.engine:
            return self.engine.language_name
        elif self.base_mapping:
            return self.base_mapping.language.value
        else:
            return "unknown"

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        """Parse a file and extract semantic chunks using cAST algorithm.

        Args:
            file_path: Path to the file to parse
            file_id: Database file ID for chunk association

        Returns:
            List of Chunk objects with optimal boundaries

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file contains invalid encoding
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Special handling for PDF files - delegate to PDFMapping
        if file_path.suffix.lower() == ".pdf":
            content_bytes = file_path.read_bytes()
            if hasattr(self.base_mapping, "parse_pdf_content"):
                return self.base_mapping.parse_pdf_content(
                    content_bytes, file_path, file_id
                )
            # PDF files require a mapping with parse_pdf_content method
            raise RuntimeError(
                f"PDF parsing requires parse_pdf_content method, "
                f"got {type(self.base_mapping)}"
            )

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            # Try with fallback encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    content = file_path.read_text(encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(
                    "utf-8", b"", 0, 1, f"Could not decode file {file_path}"
                ) from e

        # Normalize content for consistent parsing and chunk comparison
        # Skip for binary/protocol files where CRLF is semantic
        if file_path.suffix.lower() not in [
            ".pdf",
            ".png",
            ".jpg",
            ".gif",
            ".zip",
            ".eml",
            ".http",
        ]:
            content = normalize_content(content)

        return self.parse_content(content, file_path, file_id)

    def parse_content(
        self,
        content: str,
        file_path: Path | None = None,
        file_id: FileId | None = None,
    ) -> list[Chunk]:
        """Parse content string and extract semantic chunks using cAST algorithm.

        Args:
            content: Source code content to parse
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunk association

        Returns:
            List of Chunk objects with optimal boundaries
        """
        if not content.strip():
            return []

        # Special handling for text files (no tree-sitter parsing)
        if self.engine is None:
            # Check if this is PDF content by looking at language mapping
            if (
                hasattr(self.base_mapping, "language")
                and self.base_mapping.language == Language.PDF
            ):
                # Convert string content back to bytes for PDF processing
                content_bytes = (
                    content.encode("utf-8") if isinstance(content, str) else content
                )
                if hasattr(self.base_mapping, "parse_pdf_content"):
                    return self.base_mapping.parse_pdf_content(
                        content_bytes, file_path, file_id
                    )
                # PDF files require a mapping with parse_pdf_content method
                raise RuntimeError(
                    f"PDF parsing requires parse_pdf_content method, "
                    f"got {type(self.base_mapping)}"
                )
            return self._parse_text_content(content, file_path, file_id)

        # Parse to AST using TreeSitterEngine
        ast_tree = self.engine.parse_to_ast(content)
        content_bytes = content.encode("utf-8")

        # Extract universal concepts using ConceptExtractor
        universal_chunks = self.extractor.extract_all_concepts(
            ast_tree.root_node, content_bytes
        )

        # Filter out whitespace-only chunks as secondary safety measure
        filtered_chunks = []
        for chunk in universal_chunks:
            normalized_code = normalize_content(chunk.content)
            if normalized_code:
                # Update chunk with normalized content
                chunk = replace(chunk, content=normalized_code)
                filtered_chunks.append(chunk)
        universal_chunks = filtered_chunks

        # Apply cAST algorithm for optimal chunking
        optimized_chunks = self._apply_cast_algorithm(
            universal_chunks, ast_tree, content
        )

        # Convert to standard Chunk format
        chunks = self._convert_to_chunks(optimized_chunks, content, file_path, file_id)

        # Update statistics
        self._total_files_parsed += 1
        self._total_chunks_created += len(chunks)

        return chunks

    def parse_with_result(self, file_path: Path, file_id: FileId) -> ParseResult:
        """Parse a file and return detailed result information.

        Args:
            file_path: Path to the file to parse
            file_id: Database file ID for chunk association

        Returns:
            ParseResult with chunks, metadata, and diagnostics
        """
        import time

        start_time = time.time()

        try:
            chunks = self.parse_file(file_path, file_id)
            parse_time = time.time() - start_time

            # Convert chunks to dict format for ParseResult
            chunk_dicts = [chunk.to_dict() for chunk in chunks]

            return ParseResult(
                chunks=chunk_dicts,
                language=Language.from_string(self.language_name),
                total_chunks=len(chunks),
                parse_time=parse_time,
                errors=[],
                warnings=[],
                metadata={
                    "parser_type": "universal_cast",
                    "cast_config": {
                        "max_chunk_size": self.cast_config.max_chunk_size,
                        "min_chunk_size": self.cast_config.min_chunk_size,
                        "merge_threshold": self.cast_config.merge_threshold,
                    },
                    "language_mapping": self.mapping.__class__.__name__,
                    "file_size": len(file_path.read_text(encoding="utf-8"))
                    if file_path.exists()
                    else 0,
                },
            )

        except Exception as e:
            parse_time = time.time() - start_time
            return ParseResult(
                chunks=[],
                language=Language.from_string(self.language_name),
                total_chunks=0,
                parse_time=parse_time,
                errors=[str(e)],
                warnings=[],
                metadata={"parser_type": "universal_cast", "error": str(e)},
            )

    def _apply_cast_algorithm(
        self, universal_chunks: list[UniversalChunk], ast_tree: Tree, content: str
    ) -> list[UniversalChunk]:
        """Apply cAST (Code AST) algorithm for optimal semantic chunking.

        The cAST algorithm uses a split-then-merge recursive approach:
        1. Parse source code into AST (already done)
        2. Apply recursive chunking with top-down traversal
        3. Fit large AST nodes into single chunks when possible
        4. Split nodes that exceed chunk size limit recursively
        5. Greedily merge adjacent sibling nodes to maximize information density
        6. Measure chunk size by non-whitespace characters

        Args:
            universal_chunks: Initial chunks extracted from concepts
            ast_tree: Full AST tree of the source code
            content: Original source content

        Returns:
            List of optimized chunks following cAST principles
        """
        if not universal_chunks:
            return []

        # NEW: Deduplicate chunks with identical content before processing
        # This prevents issues where the same source code is extracted multiple
        # times as different concepts (e.g., function as DEFINITION, COMMENT, STRUCTURE)
        universal_chunks = deduplicate_chunks(universal_chunks, self.language_name)

        # Group chunks by concept type for structured processing
        chunks_by_concept: dict[UniversalConcept, list[UniversalChunk]] = {}
        for chunk in universal_chunks:
            if chunk.concept not in chunks_by_concept:
                chunks_by_concept[chunk.concept] = []
            chunks_by_concept[chunk.concept].append(chunk)

        optimized_chunks = []

        # Process each concept type with appropriate chunking strategy
        for concept, concept_chunks in chunks_by_concept.items():
            if concept == UniversalConcept.DEFINITION:
                # Definitions (functions, classes) should remain intact when possible
                optimized_chunks.extend(
                    self._chunk_definitions(concept_chunks, content)
                )
            elif concept == UniversalConcept.BLOCK:
                # Blocks can be merged more aggressively
                optimized_chunks.extend(self._chunk_blocks(concept_chunks, content))
            elif concept == UniversalConcept.COMMENT:
                # Comments can be merged with nearby code
                optimized_chunks.extend(self._chunk_comments(concept_chunks, content))
            else:
                # Other concepts use default chunking
                optimized_chunks.extend(self._chunk_generic(concept_chunks, content))

        # Final pass: merge adjacent chunks that are below threshold
        if self.cast_config.greedy_merge:
            optimized_chunks = self._greedy_merge_pass(optimized_chunks, content)

        return optimized_chunks

    def _chunk_definitions(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to definition chunks (functions, classes, etc.).

        Definitions remain intact as complete semantic units.
        Only split if they exceed the maximum chunk size significantly.
        """
        result = []

        for chunk in chunks:
            # Always validate and split if needed
            split_chunks = self._validate_and_split_chunk(chunk, content)
            result.extend(split_chunks)

        return result

    def _validate_and_split_chunk(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Validate chunk size and split if necessary."""
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = self._estimate_tokens(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and estimated_tokens <= self.cast_config.safe_token_limit
        ):
            # Chunk fits within both limits
            return [chunk]
        else:
            # Don't split Makefile rules unless they're excessively large
            # Splitting would break the target/recipe relationship, but extremely
            # large rules (>1.2x limit) need to be split to avoid token limit issues
            if chunk.metadata.get("kind") == "rule":
                # Allow up to 1.2x the limit (matching cAST algorithm tolerance)
                # before forcing a split to maintain consistency with other chunks
                tolerance = 1.2
                if (
                    metrics.non_whitespace_chars
                    <= self.cast_config.max_chunk_size * tolerance
                ):
                    # Keep moderately large rules intact for semantic coherence
                    return [chunk]
                # For very large rules, use Makefile-specific splitting
                # that preserves semantic coherence (target + recipe relationship)
                return self._split_makefile_rule(chunk, content)

            # Too large, apply recursive splitting
            return self._recursive_split_chunk(chunk, content)

    def _split_makefile_rule(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Split a Makefile rule while preserving semantic coherence.

        Each split chunk will contain:
        - The target line (e.g., "install: all")
        - A subset of recipe lines that fit within the size limit

        This ensures that each chunk is semantically valid - recipe lines
        always have their associated target.

        Args:
            chunk: The Makefile rule chunk to split
            content: Original source content

        Returns:
            List of split chunks, each with target + recipe subset
        """
        lines = chunk.content.split("\n")

        # Find the target line (contains ':' and not a comment)
        target_line_idx = -1
        target_line = ""
        for i, line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                target_line_idx = i
                target_line = line
                break

        if target_line_idx == -1:
            # No target found - fall back to regular splitting
            return self._recursive_split_chunk(chunk, content)

        # Extract recipe lines (lines after target)
        recipe_lines = lines[target_line_idx + 1 :]

        if not recipe_lines:
            # No recipe - just return the target
            return [chunk]

        # Split recipe lines into groups that fit within size limit
        # Each group gets the target line prepended
        result_chunks = []
        current_recipe_group = []
        part_num = 1

        for recipe_line in recipe_lines:
            # Calculate size with target + current group + this line
            test_lines = [target_line] + current_recipe_group + [recipe_line]
            test_content = "\n".join(test_lines)
            test_metrics = ChunkMetrics.from_content(test_content)

            if test_metrics.non_whitespace_chars <= self.cast_config.max_chunk_size:
                # Fits - add to current group
                current_recipe_group.append(recipe_line)
            else:
                # Doesn't fit - finalize current group and start new one
                if current_recipe_group:
                    # Create chunk with target + current group
                    chunk_content = "\n".join([target_line] + current_recipe_group)
                    chunk_lines = len(current_recipe_group) + 1

                    result_chunks.append(
                        UniversalChunk(
                            concept=chunk.concept,
                            name=f"{chunk.name}_part{part_num}",
                            content=chunk_content,
                            start_line=chunk.start_line,
                            end_line=chunk.start_line + chunk_lines - 1,
                            metadata=chunk.metadata.copy(),
                            language_node_type=chunk.language_node_type,
                        )
                    )
                    part_num += 1

                # Start new group with this line
                current_recipe_group = [recipe_line]

        # Don't forget the last group
        if current_recipe_group:
            chunk_content = "\n".join([target_line] + current_recipe_group)
            chunk_lines = len(current_recipe_group) + 1

            result_chunks.append(
                UniversalChunk(
                    concept=chunk.concept,
                    name=f"{chunk.name}_part{part_num}",
                    content=chunk_content,
                    start_line=chunk.start_line,
                    end_line=chunk.start_line + chunk_lines - 1,
                    metadata=chunk.metadata.copy(),
                    language_node_type=chunk.language_node_type,
                )
            )

        return result_chunks if result_chunks else [chunk]

    def _chunk_blocks(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to block chunks.

        Blocks are more flexible and can be merged aggressively with siblings.
        """
        if not chunks:
            return []

        # Sort chunks by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_group = [sorted_chunks[0]]

        for chunk in sorted_chunks[1:]:
            # Check if we can merge with current group
            if self._can_merge_chunks(current_group, chunk, content):
                current_group.append(chunk)
            else:
                # Finalize current group and start new one
                merged = self._merge_chunk_group(current_group, content)
                result.extend(merged)
                current_group = [chunk]

        # Don't forget the last group
        if current_group:
            merged = self._merge_chunk_group(current_group, content)
            result.extend(merged)

        # Final validation: ensure all chunks meet size constraints
        validated_result = []
        for chunk in result:
            validated_result.extend(self._validate_and_split_chunk(chunk, content))

        return validated_result

    def _chunk_comments(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply cAST chunking to comment chunks.

        Comments are merged conservatively - only consecutive comments (gap <= 1)
        are merged together. This prevents merging comments that have code between
        them, preserving standalone comments while allowing multi-line comment blocks.
        """
        if not chunks:
            return []

        # Sort comments by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_group = [sorted_chunks[0]]

        for chunk in sorted_chunks[1:]:
            # Only merge if comments are consecutive or adjacent (gap <= 1)
            last_chunk = current_group[-1]
            line_gap = chunk.start_line - last_chunk.end_line

            if line_gap <= 1:
                # Comments are consecutive - can merge
                current_group.append(chunk)
            else:
                # Gap is too large - finalize current group and start new one
                merged = self._merge_chunk_group(current_group, content)
                result.extend(merged)
                current_group = [chunk]

        # Don't forget the last group
        if current_group:
            merged = self._merge_chunk_group(current_group, content)
            result.extend(merged)

        # Final validation: ensure all chunks meet size constraints
        validated_result = []
        for chunk in result:
            validated_result.extend(self._validate_and_split_chunk(chunk, content))

        return validated_result

    def _chunk_generic(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Apply generic cAST chunking to other chunk types."""
        return self._chunk_blocks(chunks, content)  # Use block strategy as default

    def _analyze_lines(self, lines: list[str]) -> tuple[bool, bool]:
        """Analyze line length statistics to choose optimal splitting strategy.

        Returns:
            (has_very_long_lines, is_regular_code)
        """
        if not lines:
            return False, False

        lengths = [len(line) for line in lines]
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)

        # 20% of chunk size threshold for detecting minified/concatenated code
        long_line_threshold = self.cast_config.max_chunk_size * 0.2
        has_very_long_lines = max_length > long_line_threshold

        # Regular code heuristics:
        # - >10 lines: meaningful code block, not snippet
        # - <200 chars: typical editor width
        # - <100 avg: normal code density
        is_regular_code = len(lines) > 10 and max_length < 200 and avg_length < 100.0

        return has_very_long_lines, is_regular_code

    def _recursive_split_chunk(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Smart content-aware splitting that chooses the optimal strategy.

        This implements the "split" part of the split-then-merge algorithm with
        content analysis to choose between line-based and character-based splitting.
        """
        # First: Check if we even need to split
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = self._estimate_tokens(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and estimated_tokens <= self.cast_config.safe_token_limit
        ):
            return [chunk]  # No splitting needed

        # Second: Analyze the content structure
        lines = chunk.content.split("\n")
        has_very_long_lines, is_regular_code = self._analyze_lines(lines)

        # Third: Choose splitting strategy based on content analysis
        if len(lines) <= 2 or has_very_long_lines:
            # Case 1: Single/few lines OR any line is very long
            # Use character-based emergency splitting
            return self._emergency_split_code(chunk, content)

        elif is_regular_code:
            # Case 2: Many short lines (normal code)
            # Use simple line-based splitting
            return self._split_by_lines_simple(chunk, lines)

        else:
            # Case 3: Mixed content - try line-based with emergency fallback
            return self._split_by_lines_with_fallback(chunk, lines, content)

    def _split_by_lines_simple(
        self, chunk: UniversalChunk, lines: list[str]
    ) -> list[UniversalChunk]:
        """Split chunk by lines for regular code with short lines."""
        if len(lines) <= 2:
            return [chunk]

        mid_point = len(lines) // 2

        # Create two sub-chunks
        chunk1_content = "\n".join(lines[:mid_point])
        chunk2_content = "\n".join(lines[mid_point:])

        # Simple line distribution based on content split
        chunk1_lines = len(lines[:mid_point])
        chunk1_end_line = chunk.start_line + chunk1_lines - 1
        chunk2_start_line = chunk1_end_line + 1

        # Ensure valid bounds
        chunk1_end_line = max(chunk.start_line, min(chunk1_end_line, chunk.end_line))
        chunk2_start_line = max(
            chunk.start_line, min(chunk2_start_line, chunk.end_line)
        )

        chunk1 = UniversalChunk(
            concept=chunk.concept,
            name=f"{chunk.name}_part1",
            content=chunk1_content,
            start_line=chunk.start_line,
            end_line=chunk1_end_line,
            metadata=chunk.metadata.copy(),
            language_node_type=chunk.language_node_type,
        )

        chunk2 = UniversalChunk(
            concept=chunk.concept,
            name=f"{chunk.name}_part2",
            content=chunk2_content,
            start_line=chunk2_start_line,
            end_line=chunk.end_line,
            metadata=chunk.metadata.copy(),
            language_node_type=chunk.language_node_type,
        )

        # Recursively check if sub-chunks still need splitting
        result = []
        for sub_chunk in [chunk1, chunk2]:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = self._estimate_tokens(sub_chunk.content)

            if (
                sub_metrics.non_whitespace_chars > self.cast_config.max_chunk_size
                or sub_tokens > self.cast_config.safe_token_limit
            ):
                result.extend(self._recursive_split_chunk(sub_chunk, sub_chunk.content))
            else:
                result.append(sub_chunk)

        return result

    def _split_by_lines_with_fallback(
        self, chunk: UniversalChunk, lines: list[str], content: str
    ) -> list[UniversalChunk]:
        """Split by lines but fall back to emergency split if needed."""
        # Try line-based splitting first
        line_split_result = self._split_by_lines_simple(chunk, lines)

        # Check if any chunks still exceed limits
        validated_result = []
        for sub_chunk in line_split_result:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = self._estimate_tokens(sub_chunk.content)

            # If still over limit, use emergency split
            if (
                sub_metrics.non_whitespace_chars > self.cast_config.max_chunk_size
                or sub_tokens > self.cast_config.safe_token_limit
            ):
                validated_result.extend(
                    self._emergency_split_code(sub_chunk, sub_chunk.content)
                )
            else:
                validated_result.append(sub_chunk)

        return validated_result

    def _emergency_split_code(
        self, chunk: UniversalChunk, content: str
    ) -> list[UniversalChunk]:
        """Smart code splitting for minified/large single-line files."""
        # Use the stricter limit: character limit or token-based limit
        # Calculate max chars based on token limit using provider-specific estimation
        estimated_tokens = self._estimate_tokens(chunk.content)
        if estimated_tokens > 0:
            # Calculate actual chars-to-token ratio for this content
            actual_ratio = len(chunk.content) / estimated_tokens
            max_chars_from_tokens = int(
                self.cast_config.safe_token_limit * actual_ratio * 0.8
            )
        else:
            # Fallback to conservative estimation
            max_chars_from_tokens = int(self.cast_config.safe_token_limit * 3.5 * 0.8)
        max_chars = min(self.cast_config.max_chunk_size, max_chars_from_tokens)

        metrics = ChunkMetrics.from_content(chunk.content)
        if (
            metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
            and len(chunk.content) <= max_chars_from_tokens
        ):
            return [chunk]

        # Smart split points for code (in order of preference)
        split_chars = [";", "}", "{", ",", " "]

        chunks = []
        remaining = chunk.content
        part_num = 1
        total_content_length = len(chunk.content)
        current_pos = (
            0  # Track position in original content for line number calculation
        )

        while remaining:
            remaining_metrics = ChunkMetrics.from_content(remaining)
            if (
                remaining_metrics.non_whitespace_chars
                <= self.cast_config.max_chunk_size
            ):
                chunks.append(
                    self._create_split_chunk(
                        chunk, remaining, part_num, current_pos, total_content_length
                    )
                )
                break

            # Find best split point within size limit
            best_split = 0
            for split_char in split_chars:
                # Search within character limit
                search_end = min(max_chars, len(remaining))
                pos = remaining.rfind(split_char, 0, search_end)

                if pos > best_split:
                    # Check if this split point gives us valid chunk size
                    test_content = remaining[: pos + 1]
                    test_metrics = ChunkMetrics.from_content(test_content)
                    if (
                        test_metrics.non_whitespace_chars
                        <= self.cast_config.max_chunk_size
                    ):
                        best_split = pos + 1  # Include the split character
                        break

            # If no good split found, force split at character limit
            if best_split == 0:
                best_split = max_chars

            chunks.append(
                self._create_split_chunk(
                    chunk,
                    remaining[:best_split],
                    part_num,
                    current_pos,
                    total_content_length,
                )
            )
            remaining = remaining[best_split:]
            current_pos += (
                best_split  # Update position tracker for next chunk's line calculation
            )
            part_num += 1

        return chunks

    def _create_split_chunk(
        self,
        original: UniversalChunk,
        content: str,
        part_num: int,
        content_start_pos: int = 0,
        total_content_length: int = 0,
    ) -> UniversalChunk:
        """Create a split chunk from emergency splitting with proportional lines."""

        # Simple proportional line calculation based on content position
        original_line_span = original.end_line - original.start_line + 1

        if total_content_length > 0 and content_start_pos >= 0:
            # Calculate proportional position and length
            position_ratio = content_start_pos / total_content_length
            content_ratio = len(content) / total_content_length

            # Distribute lines proportionally
            line_offset = int(position_ratio * original_line_span)
            line_span = max(1, int(content_ratio * original_line_span))

            start_line = original.start_line + line_offset
            end_line = min(original.end_line, start_line + line_span - 1)

            # Ensure valid bounds
            start_line = min(start_line, original.end_line)
            end_line = max(end_line, start_line)
        else:
            # Fallback to original bounds
            start_line = original.start_line
            end_line = original.end_line

        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part_num}",
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata=original.metadata.copy(),
            language_node_type=original.language_node_type,
        )

    def _can_merge_chunks(
        self,
        current_group: list[UniversalChunk],
        candidate: UniversalChunk,
        content: str,
    ) -> bool:
        """Check if a chunk can be merged with the current group.

        This implements the merge logic of the cAST algorithm.
        """
        if not current_group:
            return True

        # Calculate combined size
        total_content = (
            "\n".join(chunk.content for chunk in current_group)
            + "\n"
            + candidate.content
        )
        metrics = ChunkMetrics.from_content(total_content)

        # Check BOTH character and token constraints
        estimated_tokens = self._estimate_tokens(total_content)
        safe_token_limit = 6000

        if (
            metrics.non_whitespace_chars
            > self.cast_config.max_chunk_size * self.cast_config.merge_threshold
            or estimated_tokens > safe_token_limit * self.cast_config.merge_threshold
        ):
            return False

        # Check line proximity (chunks should be close to each other)
        last_chunk = current_group[-1]
        line_gap = candidate.start_line - last_chunk.end_line

        if line_gap > 5:  # Allow small gaps for related code
            return False

        # Check concept compatibility
        if last_chunk.concept != candidate.concept:
            # Only merge compatible concepts
            compatible_pairs = {
                (UniversalConcept.COMMENT, UniversalConcept.DEFINITION),
                (UniversalConcept.DEFINITION, UniversalConcept.COMMENT),
                (UniversalConcept.BLOCK, UniversalConcept.COMMENT),
                (UniversalConcept.COMMENT, UniversalConcept.BLOCK),
                (UniversalConcept.DEFINITION, UniversalConcept.STRUCTURE),
                (UniversalConcept.STRUCTURE, UniversalConcept.DEFINITION),
            }
            if (last_chunk.concept, candidate.concept) not in compatible_pairs:
                return False

        return True

    def _merge_chunk_group(
        self, group: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Merge a group of chunks into optimized chunks.

        This implements the "merge" part of the split-then-merge algorithm.
        """
        if len(group) <= 1:
            return group

        # Sort by line position
        sorted_group = sorted(group, key=lambda c: c.start_line)

        # Simple merge: combine content without duplication
        combined_content = sorted_group[0].content
        for chunk in sorted_group[1:]:
            # Only add content if not already included (prevent duplication)
            if chunk.content.strip() not in combined_content:
                combined_content += "\n" + chunk.content

        metrics = ChunkMetrics.from_content(combined_content)
        estimated_tokens = self._estimate_tokens(combined_content)

        # If combined chunk is too large, return original chunks
        if (
            metrics.non_whitespace_chars > self.cast_config.max_chunk_size
            or estimated_tokens > self.cast_config.safe_token_limit
        ):
            return group

        # Create merged chunk
        first_chunk = sorted_group[0]
        last_chunk = sorted_group[-1]

        # Combine names
        unique_names = list(dict.fromkeys(chunk.name for chunk in sorted_group))
        merged_name = (
            "_".join(unique_names) if len(unique_names) > 1 else unique_names[0]
        )

        # Combine metadata
        merged_metadata = first_chunk.metadata.copy()
        merged_metadata["merged_from"] = [chunk.name for chunk in sorted_group]
        merged_metadata["chunk_count"] = len(sorted_group)

        merged_chunk = UniversalChunk(
            concept=first_chunk.concept,  # Use primary concept
            name=merged_name,
            content=combined_content,
            start_line=first_chunk.start_line,
            end_line=last_chunk.end_line,
            metadata=merged_metadata,
            language_node_type=first_chunk.language_node_type,
        )

        return [merged_chunk]

    def _greedy_merge_pass(
        self, chunks: list[UniversalChunk], content: str
    ) -> list[UniversalChunk]:
        """Final greedy merge pass to maximize information density.

        This is the final optimization step of the cAST algorithm.
        """
        if len(chunks) <= 1:
            return chunks

        # Sort chunks by line position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_line)
        result = []
        current_chunk = sorted_chunks[0]

        for next_chunk in sorted_chunks[1:]:
            # Check concept compatibility before merging
            # Only merge chunks with compatible concept types to preserve
            # semantic boundaries
            if current_chunk.concept != next_chunk.concept:
                # Define compatible concept pairs that can be safely merged
                # These pairs preserve semantic meaning when combined:
                # - COMMENT + DEFINITION: Docstrings with functions/classes
                # - BLOCK + COMMENT: Comments within code blocks
                # - DEFINITION + STRUCTURE: Definitions with file structure
                compatible_pairs = {
                    (UniversalConcept.COMMENT, UniversalConcept.DEFINITION),
                    (UniversalConcept.DEFINITION, UniversalConcept.COMMENT),
                    (UniversalConcept.BLOCK, UniversalConcept.COMMENT),
                    (UniversalConcept.COMMENT, UniversalConcept.BLOCK),
                    (UniversalConcept.DEFINITION, UniversalConcept.STRUCTURE),
                    (UniversalConcept.STRUCTURE, UniversalConcept.DEFINITION),
                }

                # If concepts are not compatible, don't merge
                if (current_chunk.concept, next_chunk.concept) not in compatible_pairs:
                    result.append(current_chunk)
                    current_chunk = next_chunk
                    continue

            # Don't merge if next_chunk is nested inside current_chunk
            # True nesting means the entire range of next_chunk is within current_chunk
            # (e.g., methods inside classes, inner classes, functions in namespaces)
            # Adjacent chunks (where end_line == start_line) should still be mergeable
            is_nested = (
                next_chunk.start_line > current_chunk.start_line  # Strict inequality
                and next_chunk.end_line <= current_chunk.end_line
            )
            if is_nested:
                result.append(current_chunk)
                current_chunk = next_chunk
                continue

            # Don't merge if either chunk explicitly prevents merging
            # This respects language-specific metadata that marks chunks as semantically
            # independent (e.g., HCL attributes and blocks that should remain separate)
            current_prevents_merge = current_chunk.metadata.get("prevent_merge_across_concepts", False)
            next_prevents_merge = next_chunk.metadata.get("prevent_merge_across_concepts", False)
            if current_prevents_merge or next_prevents_merge:
                result.append(current_chunk)
                current_chunk = next_chunk
                continue

            # Simple merge logic: only if content is different and fits size limit
            if next_chunk.content.strip() not in current_chunk.content:
                combined_content = current_chunk.content + "\n" + next_chunk.content
            else:
                combined_content = current_chunk.content  # Skip duplicate content

            metrics = ChunkMetrics.from_content(combined_content)
            estimated_tokens = self._estimate_tokens(combined_content)

            # Check for semantic incompatibility within same concept type
            # For example, Makefile variables and rules are both DEFINITION but shouldn't merge
            semantic_mismatch = False
            if (
                current_chunk.concept
                == next_chunk.concept
                == UniversalConcept.DEFINITION
            ):
                # Check if both chunks have 'kind' metadata
                current_kind = current_chunk.metadata.get("kind")
                next_kind = next_chunk.metadata.get("kind")

                # Don't merge if kinds are different (e.g., variable vs rule)
                if current_kind and next_kind and current_kind != next_kind:
                    semantic_mismatch = True

                # Don't merge rules with each other - each rule is a discrete semantic unit
                # (but variables can merge with each other)
                if current_kind == "rule" and next_kind == "rule":
                    semantic_mismatch = True

            # Determine maximum allowed gap based on chunk types
            # For cross-concept merges involving COMMENT, require strict adjacency (gap <= 1)
            # to preserve standalone comments while allowing immediate docstrings to merge
            max_gap = 5  # Default: allow reasonable gaps for related code
            if current_chunk.concept != next_chunk.concept:
                # Cross-concept merge - check if either is COMMENT
                if (
                    current_chunk.concept == UniversalConcept.COMMENT
                    or next_chunk.concept == UniversalConcept.COMMENT
                ):
                    max_gap = 1  # Strict: only merge immediately adjacent comments/code

            # Simple merge condition: fits in size limit and close proximity
            can_merge = (
                not semantic_mismatch
                and metrics.non_whitespace_chars <= self.cast_config.max_chunk_size
                and estimated_tokens <= self.cast_config.safe_token_limit
                and next_chunk.start_line - current_chunk.end_line <= max_gap
            )

            if can_merge:
                # When merging chunks with different concepts, prefer the more specific one
                # (e.g., DEFINITION over COMMENT) for name and concept
                if current_chunk.concept != next_chunk.concept:
                    # Determine which chunk is more specific
                    current_spec = get_chunk_specificity(current_chunk)
                    next_spec = get_chunk_specificity(next_chunk)

                    if next_spec > current_spec:
                        # Next chunk is more specific - use its name and concept
                        merged_concept = next_chunk.concept
                        merged_name = next_chunk.name
                        merged_metadata = next_chunk.metadata.copy()
                        merged_language_node_type = next_chunk.language_node_type
                    else:
                        # Current chunk is more specific - keep its attributes
                        merged_concept = current_chunk.concept
                        merged_name = current_chunk.name
                        merged_metadata = current_chunk.metadata.copy()
                        merged_language_node_type = current_chunk.language_node_type
                else:
                    # Same concept - keep current chunk's attributes
                    merged_concept = current_chunk.concept
                    merged_name = current_chunk.name
                    merged_metadata = current_chunk.metadata.copy()
                    merged_language_node_type = current_chunk.language_node_type

                # Simple merge without complex metadata
                current_chunk = UniversalChunk(
                    concept=merged_concept,
                    name=merged_name,
                    content=combined_content,
                    start_line=current_chunk.start_line,
                    end_line=next_chunk.end_line,
                    metadata=merged_metadata,
                    language_node_type=merged_language_node_type,
                )
            else:
                # Cannot merge, finalize current chunk
                result.append(current_chunk)
                current_chunk = next_chunk

        # Don't forget the last chunk
        result.append(current_chunk)

        return result

    def _convert_to_chunks(
        self,
        universal_chunks: list[UniversalChunk],
        content: str,
        file_path: Path | None,
        file_id: FileId | None,
    ) -> list[Chunk]:
        """Convert UniversalChunk objects to standard Chunk format for compatibility.

        Args:
            universal_chunks: List of universal chunks to convert
            content: Original source content
            file_path: Optional file path
            file_id: Optional file ID

        Returns:
            List of standard Chunk objects
        """
        chunks = []

        for i, uc in enumerate(universal_chunks):
            # Map UniversalConcept to ChunkType
            chunk_type = self._map_concept_to_chunk_type(uc.concept, uc.metadata)

            # Calculate byte offsets if possible
            start_byte = None
            end_byte = None
            if content:
                lines_before = content.split("\n")[: uc.start_line - 1]
                start_byte = ByteOffset(
                    sum(len(line) + 1 for line in lines_before)
                )  # +1 for newlines
                end_byte = ByteOffset(start_byte + len(uc.content.encode("utf-8")))

            chunk = Chunk(
                symbol=uc.name,
                start_line=LineNumber(uc.start_line),
                end_line=LineNumber(uc.end_line),
                code=uc.content,
                chunk_type=chunk_type,
                file_id=file_id or FileId(0),  # Default to 0 if not provided
                language=Language.from_string(self.language_name),
                file_path=FilePath(str(file_path)) if file_path else None,
                start_byte=start_byte,
                end_byte=end_byte,
                metadata=uc.metadata,
            )

            chunks.append(chunk)

        return chunks

    def _map_concept_to_chunk_type(
        self, concept: UniversalConcept, metadata: dict[str, Any]
    ) -> ChunkType:
        """Map UniversalConcept to ChunkType for compatibility.

        Args:
            concept: Universal concept from extraction
            metadata: Additional metadata to help with mapping

        Returns:
            Appropriate ChunkType for the concept
        """
        # FIRST: Check for explicit chunk_type_hint in metadata (highest priority)
        # This allows language mappings to override default behavior
        chunk_type_hint = metadata.get("chunk_type_hint", "").lower()
        if chunk_type_hint:
            # Map string hints to ChunkType enum values
            hint_map = {
                "table": ChunkType.TABLE,
                "key_value": ChunkType.KEY_VALUE,
                "array": ChunkType.ARRAY,
                "object": ChunkType.OBJECT,
                "function": ChunkType.FUNCTION,
                "class": ChunkType.CLASS,
                "method": ChunkType.METHOD,
                "block": ChunkType.BLOCK,
                "comment": ChunkType.COMMENT,
                "namespace": ChunkType.NAMESPACE,
            }
            if chunk_type_hint in hint_map:
                return hint_map[chunk_type_hint]

        # SECOND: Fallback to concept-based mapping with metadata inspection
        if concept == UniversalConcept.DEFINITION:
            # Check metadata for more specific type information
            # Prefer "kind" field (semantic type) over "node_type" (AST node type)
            kind = metadata.get("kind", "").lower()
            node_type = metadata.get("node_type", "").lower()

            # Check kind first (more specific semantic information)
            if kind == "function" or "function" in node_type:
                return ChunkType.FUNCTION
            elif kind == "class" or "class" in node_type:
                return ChunkType.CLASS
            elif kind == "method" or "method" in node_type:
                return ChunkType.METHOD
            elif kind == "struct" or "struct" in node_type:
                # Structs map to CLASS in languages like Zig, Rust, Go
                return ChunkType.CLASS
            elif kind == "enum" or "enum" in node_type:
                return ChunkType.ENUM
            elif kind == "interface" or "interface" in node_type:
                return ChunkType.INTERFACE
            elif kind == "trait" or "trait" in node_type:
                return ChunkType.TRAIT
            else:
                return ChunkType.FUNCTION  # Default for definitions

        elif concept == UniversalConcept.BLOCK:
            return ChunkType.BLOCK

        elif concept == UniversalConcept.COMMENT:
            return ChunkType.COMMENT

        elif concept == UniversalConcept.IMPORT:
            return ChunkType.UNKNOWN  # No direct mapping for imports

        elif concept == UniversalConcept.STRUCTURE:
            return ChunkType.NAMESPACE

        else:
            return ChunkType.UNKNOWN

    def _parse_text_content(
        self, content: str, file_path: Path | None, file_id: FileId | None
    ) -> list[Chunk]:
        """Parse plain text content without tree-sitter.

        For text files, we simply chunk by paragraphs or fixed-size blocks.

        Args:
            content: Text content to parse
            file_path: Optional file path
            file_id: Optional file ID

        Returns:
            List of text chunks
        """
        chunks = []
        lines = content.split("\n")

        # Simple paragraph-based chunking for text
        current_paragraph: list[str] = []
        current_start_line = 1
        line_num = 1

        for line in lines:
            if line.strip():  # Non-empty line
                if not current_paragraph:
                    current_start_line = line_num
                current_paragraph.append(line)
            else:  # Empty line - end current paragraph
                if current_paragraph:
                    paragraph_content = "\n".join(current_paragraph)

                    # Only create chunk if it meets minimum size
                    metrics = ChunkMetrics.from_content(paragraph_content)
                    if metrics.non_whitespace_chars >= self.cast_config.min_chunk_size:
                        chunk = Chunk(
                            symbol=f"paragraph_{current_start_line}",
                            start_line=LineNumber(current_start_line),
                            end_line=LineNumber(line_num - 1),
                            code=paragraph_content,
                            chunk_type=ChunkType.PARAGRAPH,
                            file_id=file_id or FileId(0),
                            language=Language.TEXT,
                            file_path=FilePath(str(file_path)) if file_path else None,
                        )
                        chunks.append(chunk)

                    current_paragraph = []

            line_num += 1

        # Don't forget the last paragraph
        if current_paragraph:
            paragraph_content = "\n".join(current_paragraph)
            metrics = ChunkMetrics.from_content(paragraph_content)
            if metrics.non_whitespace_chars >= self.cast_config.min_chunk_size:
                chunk = Chunk(
                    symbol=f"paragraph_{current_start_line}",
                    start_line=LineNumber(current_start_line),
                    end_line=LineNumber(line_num - 1),
                    code=paragraph_content,
                    chunk_type=ChunkType.PARAGRAPH,
                    file_id=file_id or FileId(0),
                    language=Language.TEXT,
                    file_path=FilePath(str(file_path)) if file_path else None,
                )
                chunks.append(chunk)

        # Update statistics
        self._total_files_parsed += 1
        self._total_chunks_created += len(chunks)

        return chunks

    def get_statistics(self) -> dict[str, Any]:
        """Get parsing statistics.

        Returns:
            Dictionary with parsing statistics
        """
        return {
            "language": self.language_name,
            "total_files_parsed": self._total_files_parsed,
            "total_chunks_created": self._total_chunks_created,
            "cast_config": {
                "max_chunk_size": self.cast_config.max_chunk_size,
                "min_chunk_size": self.cast_config.min_chunk_size,
                "merge_threshold": self.cast_config.merge_threshold,
                "preserve_structure": self.cast_config.preserve_structure,
                "greedy_merge": self.cast_config.greedy_merge,
            },
        }

    def reset_statistics(self) -> None:
        """Reset parsing statistics."""
        self._total_files_parsed = 0
        self._total_chunks_created = 0
