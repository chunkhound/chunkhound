"""Chunk splitting logic for enforcing size limits.

This module provides the ChunkSplitter class that handles splitting chunks
that exceed the 1200 non-whitespace character limit. It extracts the splitting
logic from UniversalParser into a reusable component.

The splitter enforces cAST-inspired size constraints:
- Maximum 1200 non-whitespace characters per chunk
- Safe token limit of 6000 tokens per chunk
- Multiple splitting strategies based on content analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from chunkhound.core.types.common import ChunkType
from chunkhound.core.utils import estimate_tokens_chunking
from chunkhound.core.utils.token_utils import EMBEDDING_CHARS_PER_TOKEN

from .universal_engine import UniversalChunk, UniversalConcept

if TYPE_CHECKING:
    from chunkhound.core.models.chunk import Chunk
    from chunkhound.core.types.common import FileId, Language


@dataclass
class CASTConfig:
    """Configuration for cAST-inspired chunk splitting.

    Inspired by: "cAST: Enhancing Code Retrieval-Augmented Generation
    with Structural Chunking via Abstract Syntax Tree". Uses simplified
    char/token metrics rather than the full AST-depth approach.

    Note: max_chunk_size applies to content only. Embedding providers add
    headers (file path, language) on top of content when creating embeddings.
    """

    max_chunk_size: int = 1200  # Non-whitespace chars
    min_chunk_size: int = 50  # Minimum chunk size to avoid tiny fragments
    merge_threshold: float = (
        0.8  # Merge siblings if combined size < threshold * max_size
    )
    greedy_merge: bool = True  # Greedily merge adjacent sibling nodes
    safe_token_limit: int = 6000  # Conservative limit for embedding models

    def __post_init__(self) -> None:
        if self.min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be >= 1, got {self.min_chunk_size}")


@dataclass
class ChunkMetrics:
    """Metrics for measuring chunk quality and size."""

    non_whitespace_chars: int
    total_chars: int
    lines: int

    @classmethod
    def from_content(cls, content: str) -> ChunkMetrics:
        """Calculate metrics from content string."""
        non_ws = len(re.sub(r"\s", "", content))
        total = len(content)
        lines = len(content.split("\n"))
        return cls(non_ws, total, lines)


def compute_end_line(content: str, start_line: int) -> int:
    """Compute end line from content and start line.

    Handles trailing newline correctly: if content ends with newline,
    the last content line is before that newline.
    """
    newline_count = content.count("\n")
    if content.endswith("\n"):
        end_line = start_line + newline_count - 1
    else:
        end_line = start_line + newline_count
    return max(start_line, end_line)


class ChunkSplitter:
    """Handles splitting chunks that exceed size limits.

    This class enforces the 1200 non-whitespace character limit per chunk.
    It provides multiple splitting strategies based on content analysis:

    1. Line-based splitting for regular code with short lines
    2. Character-based emergency splitting for minified/single-line content

    Language-specific splitting (e.g., Makefile rules) is handled by dedicated
    parsers before reaching this class.

    The splitter uses the cAST algorithm's constraints:
    - max_chunk_size: Maximum non-whitespace characters (default 1200)
    - safe_token_limit: Maximum tokens per chunk (default 6000)
    """

    def __init__(self, config: CASTConfig | None = None):
        """Initialize chunk splitter.

        Args:
            config: Configuration for cAST algorithm. If None, uses defaults.
        """
        self.config = config or CASTConfig()

    def validate_and_split(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Validate chunk size and split if necessary.

        Args:
            chunk: The chunk to validate and potentially split

        Returns:
            List of chunks that all fit within size limits. Returns [chunk] if
            the input chunk already fits, otherwise returns multiple split chunks.

        Note:
            Subclasses may override this method for domain-specific splitting
            (e.g., MakefileChunkSplitter preserves target/recipe coherence).
        """
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = estimate_tokens_chunking(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimated_tokens <= self.config.safe_token_limit
        ):
            # Chunk fits within both limits
            return [chunk]

        # Too large, apply recursive splitting
        return self._recursive_split(chunk)

    def validate_text_content(
        self,
        content: str,
        name: str,
        start_line: int,
        end_line: int,
        concept: UniversalConcept = UniversalConcept.BLOCK,
    ) -> list[UniversalChunk]:
        """Validate text content and split if oversized.

        Convenience method for text-based parsers (PDF, TEXT) that don't use
        tree-sitter. Creates a UniversalChunk and validates it.

        Args:
            content: Text content to validate
            name: Name for the chunk
            start_line: Starting line number
            end_line: Ending line number
            concept: Universal concept type (default: BLOCK)

        Returns:
            List of validated UniversalChunks (may be split if oversized).
        """
        chunk = UniversalChunk(
            concept=concept,
            name=name,
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata={},
            language_node_type="text",
        )
        return self.validate_and_split(chunk)

    def validate_and_convert_text(
        self,
        content: str,
        name: str,
        start_line: int,
        end_line: int,
        file_path: Path | None,
        file_id: FileId | None,
        language: Language,
    ) -> list[Chunk]:
        """Validate text content, split if oversized, and convert to Chunks.

        Convenience method for text-based parsers (PDF, TEXT) that combines
        validation, splitting, and conversion in one call.

        Args:
            content: Text content to validate
            name: Name for the chunk
            start_line: Starting line number
            end_line: Ending line number
            file_path: Optional file path for the chunk
            file_id: Optional file ID for the chunk
            language: Language to assign to the chunk

        Returns:
            List of Chunk objects (may be split if content was oversized).
        """
        from chunkhound.core.models.chunk import Chunk
        from chunkhound.core.types.common import (
            FileId,
            FilePath,
            LineNumber,
        )

        validated = self.validate_text_content(
            content=content,
            name=name,
            start_line=start_line,
            end_line=end_line,
        )
        return [
            Chunk(
                symbol=vc.name,
                start_line=LineNumber(vc.start_line),
                end_line=LineNumber(vc.end_line),
                code=vc.content,
                chunk_type=ChunkType.PARAGRAPH,
                file_id=file_id or FileId(0),
                language=language,
                file_path=FilePath(str(file_path)) if file_path else None,
            )
            for vc in validated
        ]

    def _analyze_lines(self, lines: list[str]) -> bool:
        """Analyze line length statistics to choose optimal splitting strategy.

        Returns:
            has_very_long_lines: True if any line exceeds 20% of max_chunk_size
        """
        if not lines:
            return False

        max_length = max(len(line) for line in lines)

        # 20% of chunk size threshold for detecting minified/concatenated code
        long_line_threshold = self.config.max_chunk_size * 0.2
        return max_length > long_line_threshold

    def _recursive_split(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Smart content-aware splitting that chooses the optimal strategy.

        This implements the "split" part of the split-then-merge algorithm with
        content analysis to choose between line-based and character-based splitting.
        """
        # First: Check if we even need to split
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = estimate_tokens_chunking(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimated_tokens <= self.config.safe_token_limit
        ):
            return [chunk]  # No splitting needed

        # Second: Analyze the content structure
        lines = chunk.content.split("\n")
        has_very_long_lines = self._analyze_lines(lines)

        # Third: Choose splitting strategy based on content analysis
        line_span = chunk.end_line - chunk.start_line
        if len(lines) <= 2 or has_very_long_lines or line_span < 2:
            return self._emergency_split(chunk)

        return self._split_by_lines_simple(chunk, lines)

    def _split_by_lines_simple(
        self, chunk: UniversalChunk, lines: list[str]
    ) -> list[UniversalChunk]:
        """Split chunk by lines for regular code with short lines."""
        mid_point = len(lines) // 2

        # Create two sub-chunks
        chunk1_content = "\n".join(lines[:mid_point])
        chunk2_content = "\n".join(lines[mid_point:])

        # Simple line distribution based on content split
        chunk1_lines = len(lines[:mid_point])
        chunk1_end_line = chunk.start_line + chunk1_lines - 1
        chunk2_start_line = chunk1_end_line + 1

        # Ensure valid bounds — chunk2 must start strictly after chunk1 ends
        chunk1_end_line = max(chunk.start_line, min(chunk1_end_line, chunk.end_line))
        chunk2_start_line = max(
            chunk1_end_line + 1, min(chunk2_start_line, chunk.end_line)
        )

        # If metadata span is too narrow for two non-overlapping ranges, fall back
        if chunk2_start_line >= chunk.end_line:
            return self._emergency_split(chunk)

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

        # Recursively validate sub-chunks (splits further if needed)
        result = []
        for sub_chunk in [chunk1, chunk2]:
            result.extend(self._recursive_split(sub_chunk))

        return result

    def _emergency_split(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Smart code splitting for minified/large single-line files."""
        raw_search_window = self.config.safe_token_limit * EMBEDDING_CHARS_PER_TOKEN

        metrics = ChunkMetrics.from_content(chunk.content)
        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimate_tokens_chunking(chunk.content) <= self.config.safe_token_limit
        ):
            return [chunk]

        # Smart split points for code (in order of preference)
        split_chars = ["\n", ";", "}", "{", ",", " "]

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
                remaining_metrics.non_whitespace_chars <= self.config.max_chunk_size
                and estimate_tokens_chunking(remaining) <= self.config.safe_token_limit
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
                search_end = min(raw_search_window, len(remaining))
                pos = remaining.rfind(split_char, 0, search_end)

                if pos > best_split:
                    # Check if this split point gives us valid chunk size
                    test_content = remaining[: pos + 1]
                    test_metrics = ChunkMetrics.from_content(test_content)
                    if (
                        test_metrics.non_whitespace_chars <= self.config.max_chunk_size
                        and estimate_tokens_chunking(test_content) <= self.config.safe_token_limit
                    ):
                        best_split = pos + 1  # Include the split character
                        break

            # If no good split found, force split at raw window limit
            if best_split == 0:
                best_split = min(raw_search_window, len(remaining))
                # Shrink if force-split chunk exceeds non-ws limit
                test = ChunkMetrics.from_content(remaining[:best_split])
                while (
                    (
                        test.non_whitespace_chars > self.config.max_chunk_size
                        or estimate_tokens_chunking(remaining[:best_split]) > self.config.safe_token_limit
                    )
                    and best_split > self.config.min_chunk_size
                ):
                    best_split = best_split // 2
                    test = ChunkMetrics.from_content(remaining[:best_split])
                # Last-resort: chunk may still exceed limits if min_chunk_size reached
                slice_ = remaining[:best_split]
                if (
                    test.non_whitespace_chars > self.config.max_chunk_size
                    or estimate_tokens_chunking(slice_) > self.config.safe_token_limit
                ):
                    logger.warning(
                        "Chunk '{}' exceeds size limits after max splitting "
                        "(non_ws={}, limit={})",
                        chunk.name,
                        test.non_whitespace_chars,
                        self.config.max_chunk_size,
                    )

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


# Module-level mapping from ChunkType to UniversalConcept
# Used by chunk_to_universal() for central guard splitting
CHUNK_TYPE_TO_CONCEPT: dict[ChunkType, UniversalConcept] = {
    # Code structure -> DEFINITION
    ChunkType.FUNCTION: UniversalConcept.DEFINITION,
    ChunkType.METHOD: UniversalConcept.DEFINITION,
    ChunkType.CONSTRUCTOR: UniversalConcept.DEFINITION,
    ChunkType.PROPERTY: UniversalConcept.DEFINITION,
    ChunkType.FIELD: UniversalConcept.DEFINITION,
    ChunkType.CLOSURE: UniversalConcept.DEFINITION,
    ChunkType.EXTENSION_FUNCTION: UniversalConcept.DEFINITION,
    ChunkType.VARIABLE: UniversalConcept.DEFINITION,
    ChunkType.MACRO: UniversalConcept.DEFINITION,
    # Code structure -> STRUCTURE
    ChunkType.CLASS: UniversalConcept.STRUCTURE,
    ChunkType.INTERFACE: UniversalConcept.STRUCTURE,
    ChunkType.STRUCT: UniversalConcept.STRUCTURE,
    ChunkType.ENUM: UniversalConcept.STRUCTURE,
    ChunkType.NAMESPACE: UniversalConcept.STRUCTURE,
    ChunkType.TYPE_ALIAS: UniversalConcept.STRUCTURE,
    ChunkType.TRAIT: UniversalConcept.STRUCTURE,
    ChunkType.OBJECT: UniversalConcept.STRUCTURE,
    ChunkType.COMPANION_OBJECT: UniversalConcept.STRUCTURE,
    ChunkType.DATA_CLASS: UniversalConcept.STRUCTURE,
    ChunkType.TYPE: UniversalConcept.STRUCTURE,
    # Documentation -> COMMENT
    ChunkType.COMMENT: UniversalConcept.COMMENT,
    ChunkType.DOCSTRING: UniversalConcept.COMMENT,
    # Content blocks -> BLOCK (not DEFINITION, as these aren't named reusable units)
    ChunkType.SCRIPT: UniversalConcept.BLOCK,  # File-level scripts
    ChunkType.CODE_BLOCK: UniversalConcept.BLOCK,  # Markdown fenced blocks
    # Markdown structure -> BLOCK
    ChunkType.HEADER_1: UniversalConcept.BLOCK,
    ChunkType.HEADER_2: UniversalConcept.BLOCK,
    ChunkType.HEADER_3: UniversalConcept.BLOCK,
    ChunkType.HEADER_4: UniversalConcept.BLOCK,
    ChunkType.HEADER_5: UniversalConcept.BLOCK,
    ChunkType.HEADER_6: UniversalConcept.BLOCK,
    ChunkType.PARAGRAPH: UniversalConcept.BLOCK,
    ChunkType.TABLE: UniversalConcept.BLOCK,
    ChunkType.KEY_VALUE: UniversalConcept.BLOCK,
    ChunkType.ARRAY: UniversalConcept.BLOCK,
    ChunkType.BLOCK: UniversalConcept.BLOCK,
    ChunkType.EMBEDDED_SQL: UniversalConcept.BLOCK,
    ChunkType.UNKNOWN: UniversalConcept.BLOCK,
}


def universal_to_chunk(
    uc: UniversalChunk,
    file_path: Path | None,
    file_id: FileId | None,
    language: Language,
) -> Chunk:
    """Convert a UniversalChunk to a standard Chunk.

    Note: ChunkType restoration from language_node_type works correctly for
    chunks that went through chunk_to_universal() (central guard splitting).
    For parser-created UniversalChunks, language_node_type contains tree-sitter
    node types (e.g., "function_definition"), so type restoration falls back
    to concept mapping.

    Args:
        uc: The UniversalChunk to convert
        file_path: Optional file path for the chunk
        file_id: Optional file ID for the chunk
        language: Language to assign to the chunk

    Returns:
        Chunk instance with the same content and metadata
    """
    from chunkhound.core.models.chunk import Chunk
    from chunkhound.core.types.common import (
        FileId,
        FilePath,
        LineNumber,
    )

    # Try to restore original chunk_type from language_node_type first
    # This preserves types like METHOD, CLASS that were lost in chunk_to_universal()
    chunk_type = ChunkType.from_string(uc.language_node_type)
    if chunk_type == ChunkType.UNKNOWN:
        # Fallback to concept mapping for non-ChunkType language_node_types
        chunk_type_map = {
            UniversalConcept.DEFINITION: ChunkType.FUNCTION,
            UniversalConcept.BLOCK: ChunkType.BLOCK,
            UniversalConcept.COMMENT: ChunkType.COMMENT,
            UniversalConcept.IMPORT: ChunkType.BLOCK,
            UniversalConcept.STRUCTURE: ChunkType.BLOCK,
        }
        chunk_type = chunk_type_map.get(uc.concept, ChunkType.BLOCK)

    return Chunk(
        symbol=uc.name,
        start_line=LineNumber(uc.start_line),
        end_line=LineNumber(uc.end_line),
        code=uc.content,
        chunk_type=chunk_type,
        file_id=file_id or FileId(0),
        language=language,
        file_path=FilePath(str(file_path)) if file_path else None,
        metadata=dict(uc.metadata),
    )


def chunk_to_universal(chunk: Chunk) -> UniversalChunk:
    """Convert a Chunk to UniversalChunk for validation.

    Used by central guard to validate chunks from any parser.

    Args:
        chunk: The Chunk to convert

    Returns:
        UniversalChunk instance for validation/splitting
    """
    concept = CHUNK_TYPE_TO_CONCEPT.get(chunk.chunk_type, UniversalConcept.BLOCK)

    # Flat mapping loses DDL/DML distinction — restore from metadata["kind"].
    # This depends on metadata being preserved through the pipeline. A dedicated
    # ChunkType.EMBEDDED_SQL_DDL was considered but would require changes across
    # the DB schema, serialization, and all ChunkType consumers.
    if (
        chunk.chunk_type == ChunkType.EMBEDDED_SQL
        and chunk.metadata
        and chunk.metadata.get("kind") == "embedded_sql_ddl"
    ):
        concept = UniversalConcept.DEFINITION

    return UniversalChunk(
        concept=concept,
        name=chunk.symbol or "",
        content=chunk.code or "",
        start_line=int(chunk.start_line),
        end_line=int(chunk.end_line),
        metadata=dict(chunk.metadata) if chunk.metadata else {},
        language_node_type=str(chunk.chunk_type.value) if chunk.chunk_type else "block",
    )
