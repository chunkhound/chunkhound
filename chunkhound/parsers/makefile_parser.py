"""Specialized parser for Makefiles with size enforcement.

Makefile rules need special handling because:
1. Each split chunk must contain target line + recipe lines for semantic coherence
2. Splitting in the middle of a recipe breaks the target/recipe relationship
3. The parser handles oversized rules by splitting during parsing, never reaching
   ChunkSplitter's generic splitting

This follows the same pattern as VueParser and SvelteParser.
"""

from chunkhound.parsers.chunk_splitter import (
    CASTConfig,
    ChunkMetrics,
    ChunkSplitter,
)
from chunkhound.parsers.mappings.makefile import MakefileMapping
from chunkhound.parsers.universal_engine import TreeSitterEngine, UniversalChunk
from chunkhound.parsers.universal_parser import UniversalParser


class MakefileChunkSplitter(ChunkSplitter):
    """Chunk splitter with Makefile rule-aware splitting.

    Overrides validate_and_split to handle Makefile rules specially,
    preserving target/recipe semantic coherence.
    """

    def validate_and_split(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Validate chunk size and split if necessary.

        For Makefile rules, uses specialized splitting that preserves
        target/recipe relationships.
        """
        metrics = ChunkMetrics.from_content(chunk.content)
        estimated_tokens = self._estimate_tokens(chunk.content)

        if (
            metrics.non_whitespace_chars <= self.config.max_chunk_size
            and estimated_tokens <= self.config.safe_token_limit
        ):
            return [chunk]

        # Makefile rules use specialized splitting
        if chunk.metadata.get("kind") == "rule":
            return self._split_makefile_rule(chunk)

        # All other chunks use generic recursive splitting
        return self._recursive_split(chunk)

    def _split_makefile_rule(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Split Makefile rule preserving target/recipe coherence.

        Each split chunk will contain:
        - The target line (e.g., "install: all")
        - A subset of recipe lines that fit within the size limit

        This ensures that each chunk is semantically valid - recipe lines
        always have their associated target.
        """
        lines = chunk.content.split("\n")

        # Find target line (contains ':' and not a comment)
        target_line = ""
        target_idx = -1
        for i, line in enumerate(lines):
            if ":" in line and not line.strip().startswith("#"):
                target_line = line
                target_idx = i
                break

        if target_idx == -1:
            # No target found - fall back to generic splitting
            return self._recursive_split(chunk)

        recipe_lines = lines[target_idx + 1 :]
        if not recipe_lines:
            return [chunk]

        total_recipe_lines = len(recipe_lines)

        # Group recipe lines into size-limited chunks
        result: list[UniversalChunk] = []
        current_group: list[str] = []
        part = 1

        for recipe_line in recipe_lines:
            test_content = "\n".join([target_line] + current_group + [recipe_line])
            test_metrics = ChunkMetrics.from_content(test_content)

            if test_metrics.non_whitespace_chars <= self.config.max_chunk_size:
                current_group.append(recipe_line)
            else:
                if current_group:
                    result.append(
                        self._create_rule_chunk(
                            chunk, target_line, current_group, part, total_recipe_lines
                        )
                    )
                    part += 1
                current_group = [recipe_line]

        if current_group:
            result.append(
                self._create_rule_chunk(
                    chunk, target_line, current_group, part, total_recipe_lines
                )
            )

        return result if result else [chunk]

    def _create_rule_chunk(
        self,
        original: UniversalChunk,
        target: str,
        recipe_lines: list[str],
        part: int,
        total_recipe_lines: int,
    ) -> UniversalChunk:
        """Create a split rule chunk with target + recipe subset.

        Uses proportional line span calculation to maintain accurate
        end_line relative to the original chunk's span.
        """
        content = "\n".join([target] + recipe_lines)

        # Calculate proportional line span
        original_line_span = original.end_line - original.start_line + 1
        if total_recipe_lines > 0:
            recipe_ratio = len(recipe_lines) / total_recipe_lines
            line_span = max(1, int(recipe_ratio * original_line_span))
        else:
            line_span = 1
        end_line = min(original.end_line, original.start_line + line_span - 1)

        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part}",
            content=content,
            start_line=original.start_line,
            end_line=end_line,
            metadata=original.metadata.copy(),
            language_node_type=original.language_node_type,
        )


class MakefileParser(UniversalParser):
    """Parser for Makefiles with built-in size enforcement.

    Inherits from UniversalParser and uses MakefileChunkSplitter
    to handle oversized rules with target/recipe coherence.
    """

    def __init__(self, cast_config: CASTConfig | None = None):
        engine = self._create_makefile_engine()
        mapping = MakefileMapping()
        super().__init__(engine, mapping, cast_config)
        # Override with Makefile-aware chunk splitter
        self.chunk_splitter = MakefileChunkSplitter(self.cast_config)

    def _create_makefile_engine(self) -> TreeSitterEngine | None:
        """Create TreeSitterEngine for Makefile parsing."""
        try:
            import tree_sitter_make as ts_make
            from tree_sitter import Language as TSLanguage

            # Handle tree-sitter API - language() returns PyCapsule that needs wrapping
            lang_result = ts_make.language()
            if isinstance(lang_result, TSLanguage):
                ts_language = lang_result
            else:
                # In newer tree-sitter, need to wrap the capsule
                ts_language = TSLanguage(lang_result)

            return TreeSitterEngine("makefile", ts_language)
        except ImportError:
            return None
