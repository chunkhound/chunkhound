"""Chunk size validation and splitting for cAST algorithm.

This module provides the ChunkSplitter class which enforces chunk size constraints
from the cAST (Code AST) research paper. It validates chunks against:
- 1200 non-whitespace character limit
- 6000 token limit (conservative, well under 8191 API limit)

The splitter is designed to be used by all parsers to ensure consistent
chunk sizing across the codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from chunkhound.core.utils import DEFAULT_CHARS_PER_TOKEN, estimate_tokens_embedding

from .universal_engine import UniversalChunk


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
    def from_content(cls, content: str, ast_depth: int = 0) -> ChunkMetrics:
        """Calculate metrics from content string."""
        non_ws = len(re.sub(r"\s", "", content))
        total = len(content)
        lines = len(content.split("\n"))
        return cls(non_ws, total, lines, ast_depth)


class ChunkSplitter:
    """Validates and splits chunks to meet cAST size constraints.

    This class provides the core chunk size enforcement for all parsers.
    It ensures no chunk exceeds the configured limits by recursively
    splitting oversized chunks.
    """

    def __init__(self, config: CASTConfig | None = None):
        """Initialize chunk splitter.

        Args:
            config: Configuration for cAST algorithm (uses defaults if None)
        """
        self.config = config or CASTConfig()

    def validate_and_split(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Validate chunk size and split if necessary.

        Args:
            chunk: UniversalChunk to validate

        Returns:
            List containing either the original chunk (if valid) or split chunks
        """
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = estimate_tokens_embedding(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimated_tokens <= self.config.safe_token_limit
        ):
            return [chunk]

        # Don't split Makefile rules unless they're excessively large
        if chunk.metadata.get("kind") == "rule":
            tolerance = 1.2
            if metrics.non_whitespace_chars <= self.config.max_chunk_size * tolerance:
                return [chunk]
            return self._split_makefile_rule(chunk)

        return self._recursive_split_chunk(chunk)

    def _split_makefile_rule(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Split a Makefile rule while preserving semantic coherence.

        Each split chunk will contain:
        - The target line (e.g., "install: all")
        - A subset of recipe lines that fit within the size limit
        """
        lines = chunk.content.split("\n")

        # Find the target line
        target_line_idx = -1
        target_line = ""
        for i, line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                target_line_idx = i
                target_line = line
                break

        if target_line_idx == -1:
            return self._recursive_split_chunk(chunk)

        recipe_lines = lines[target_line_idx + 1 :]
        if not recipe_lines:
            return [chunk]

        result_chunks = []
        current_recipe_group: list[str] = []
        part_num = 1

        for recipe_line in recipe_lines:
            test_lines = [target_line] + current_recipe_group + [recipe_line]
            test_content = "\n".join(test_lines)
            test_metrics = ChunkMetrics.from_content(test_content)

            if test_metrics.non_whitespace_chars <= self.config.max_chunk_size:
                current_recipe_group.append(recipe_line)
            else:
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
                    part_num += 1

                current_recipe_group = [recipe_line]

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

    def _analyze_lines(self, lines: list[str]) -> tuple[bool, bool]:
        """Analyze line length statistics to choose optimal splitting strategy."""
        if not lines:
            return False, False

        lengths = [len(line) for line in lines]
        max_length = max(lengths)
        avg_length = sum(lengths) / len(lengths)

        long_line_threshold = self.config.max_chunk_size * 0.2
        has_very_long_lines = max_length > long_line_threshold
        is_regular_code = len(lines) > 10 and max_length < 200 and avg_length < 100.0

        return has_very_long_lines, is_regular_code

    def _recursive_split_chunk(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Smart content-aware splitting that chooses the optimal strategy."""
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = estimate_tokens_embedding(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimated_tokens <= self.config.safe_token_limit
        ):
            return [chunk]

        lines = chunk.content.split("\n")
        has_very_long_lines, is_regular_code = self._analyze_lines(lines)

        if len(lines) <= 2 or has_very_long_lines:
            return self._emergency_split_code(chunk)
        elif is_regular_code:
            return self._split_by_lines_simple(chunk, lines)
        else:
            return self._split_by_lines_with_fallback(chunk, lines)

    def _split_by_lines_simple(
        self, chunk: UniversalChunk, lines: list[str]
    ) -> list[UniversalChunk]:
        """Split chunk by lines for regular code with short lines."""
        if len(lines) <= 2:
            return [chunk]

        mid_point = len(lines) // 2

        chunk1_content = "\n".join(lines[:mid_point])
        chunk2_content = "\n".join(lines[mid_point:])

        chunk1_lines = len(lines[:mid_point])
        chunk1_end_line = chunk.start_line + chunk1_lines - 1
        chunk2_start_line = chunk1_end_line + 1

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

        result = []
        for sub_chunk in [chunk1, chunk2]:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = estimate_tokens_embedding(sub_chunk.content)

            if (
                sub_metrics.non_whitespace_chars > self.config.max_chunk_size
                or sub_tokens > self.config.safe_token_limit
            ):
                result.extend(self._recursive_split_chunk(sub_chunk))
            else:
                result.append(sub_chunk)

        return result

    def _split_by_lines_with_fallback(
        self, chunk: UniversalChunk, lines: list[str]
    ) -> list[UniversalChunk]:
        """Split by lines but fall back to emergency split if needed."""
        line_split_result = self._split_by_lines_simple(chunk, lines)

        validated_result = []
        for sub_chunk in line_split_result:
            sub_metrics = ChunkMetrics.from_content(sub_chunk.content)
            sub_tokens = estimate_tokens_embedding(sub_chunk.content)

            if (
                sub_metrics.non_whitespace_chars > self.config.max_chunk_size
                or sub_tokens > self.config.safe_token_limit
            ):
                validated_result.extend(self._emergency_split_code(sub_chunk))
            else:
                validated_result.append(sub_chunk)

        return validated_result

    def _emergency_split_code(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Smart code splitting for minified/large single-line files."""
        estimated_tokens = estimate_tokens_embedding(chunk.content)
        if estimated_tokens > 0:
            actual_ratio = len(chunk.content) / estimated_tokens
            max_chars_from_tokens = int(
                self.config.safe_token_limit * actual_ratio * 0.8
            )
        else:
            max_chars_from_tokens = int(self.config.safe_token_limit * DEFAULT_CHARS_PER_TOKEN * 0.8)
        max_chars = min(self.config.max_chunk_size, max_chars_from_tokens)

        metrics = ChunkMetrics.from_content(chunk.content)
        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and len(chunk.content) <= max_chars_from_tokens
        ):
            return [chunk]

        split_chars = [";", "}", "{", ",", " "]

        chunks = []
        remaining = chunk.content
        part_num = 1
        total_content_length = len(chunk.content)
        current_pos = 0

        while remaining:
            remaining_metrics = ChunkMetrics.from_content(remaining)
            if remaining_metrics.non_whitespace_chars <= self.config.max_chunk_size:
                chunks.append(
                    self._create_split_chunk(
                        chunk, remaining, part_num, current_pos, total_content_length
                    )
                )
                break

            best_split = 0
            for split_char in split_chars:
                search_end = min(max_chars, len(remaining))
                pos = remaining.rfind(split_char, 0, search_end)

                if pos > best_split:
                    test_content = remaining[: pos + 1]
                    test_metrics = ChunkMetrics.from_content(test_content)
                    if test_metrics.non_whitespace_chars <= self.config.max_chunk_size:
                        best_split = pos + 1
                        break

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
            current_pos += best_split
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
        original_line_span = original.end_line - original.start_line + 1

        if total_content_length > 0 and content_start_pos >= 0:
            position_ratio = content_start_pos / total_content_length
            content_ratio = len(content) / total_content_length

            line_offset = int(position_ratio * original_line_span)
            line_span = max(1, int(content_ratio * original_line_span))

            start_line = original.start_line + line_offset
            end_line = min(original.end_line, start_line + line_span - 1)

            start_line = min(start_line, original.end_line)
            end_line = max(end_line, start_line)
        else:
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
