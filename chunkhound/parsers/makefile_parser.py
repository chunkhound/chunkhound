"""Specialized parser for Makefiles with size enforcement.

Makefile rules need special handling because:
1. Each split chunk must contain target line + recipe lines for semantic coherence
2. Splitting in the middle of a recipe breaks the target/recipe relationship
3. The parser handles oversized rules by splitting during parsing, never reaching
   ChunkSplitter's generic splitting

This follows the same pattern as VueParser and SvelteParser.
"""

from pathlib import Path

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import FileId
from chunkhound.parsers.chunk_splitter import (
    CASTConfig,
    ChunkMetrics,
    ChunkSplitter,
)
from chunkhound.parsers.mappings.makefile import MakefileMapping
from chunkhound.parsers.universal_engine import UniversalChunk
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
                        self._create_rule_chunk(chunk, target_line, current_group, part)
                    )
                    part += 1
                current_group = [recipe_line]

        if current_group:
            result.append(
                self._create_rule_chunk(chunk, target_line, current_group, part)
            )

        return result if result else [chunk]

    def _create_rule_chunk(
        self,
        original: UniversalChunk,
        target: str,
        recipe_lines: list[str],
        part: int,
    ) -> UniversalChunk:
        """Create a split rule chunk with target + recipe subset."""
        content = "\n".join([target] + recipe_lines)
        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part}",
            content=content,
            start_line=original.start_line,
            end_line=original.start_line + len(recipe_lines),
            metadata=original.metadata.copy(),
            language_node_type=original.language_node_type,
        )


class MakefileParser:
    """Parser for Makefiles with built-in size enforcement.

    Delegates to UniversalParser but uses MakefileChunkSplitter
    to handle oversized rules with target/recipe coherence.
    """

    def __init__(self, cast_config: CASTConfig | None = None):
        self.mapping = MakefileMapping()
        self.cast_config = cast_config or CASTConfig()
        self._universal_parser = self._create_universal_parser()

    def _create_universal_parser(self) -> UniversalParser | None:
        """Create UniversalParser for Makefile content."""
        try:
            import tree_sitter_make as ts_make
            from tree_sitter import Language as TSLanguage

            from chunkhound.parsers.universal_engine import TreeSitterEngine

            # Handle tree-sitter API - language() returns PyCapsule that needs wrapping
            lang_result = ts_make.language()
            if isinstance(lang_result, TSLanguage):
                ts_language = lang_result
            else:
                # In newer tree-sitter, need to wrap the capsule
                ts_language = TSLanguage(lang_result)

            engine = TreeSitterEngine("makefile", ts_language)
            parser = UniversalParser(engine, self.mapping, self.cast_config)

            # Replace the default chunk_splitter with Makefile-aware version
            parser.chunk_splitter = MakefileChunkSplitter(self.cast_config)

            return parser
        except ImportError:
            return None

    def parse_file(self, file_path: Path, file_id: FileId) -> list[Chunk]:
        """Parse a Makefile.

        Args:
            file_path: Path to Makefile
            file_id: Database file ID

        Returns:
            List of chunks with size enforcement
        """
        if not self._universal_parser:
            return []
        return self._universal_parser.parse_file(file_path, file_id)

    def parse_content(
        self, content: str, file_path: Path | None, file_id: FileId | None
    ) -> list[Chunk]:
        """Parse Makefile content with size enforcement.

        Args:
            content: Makefile source
            file_path: Optional file path for metadata
            file_id: Optional file ID for chunks

        Returns:
            List of chunks with size enforcement
        """
        if not self._universal_parser:
            return []
        return self._universal_parser.parse_content(content, file_path, file_id)
