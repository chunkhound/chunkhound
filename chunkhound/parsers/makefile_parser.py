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
from chunkhound.parsers.universal_engine import (
    SetupError,
    TreeSitterEngine,
    UniversalChunk,
)
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
        recipe_offset = 0

        for recipe_line in recipe_lines:
            test_content = "\n".join([target_line] + current_group + [recipe_line])
            test_metrics = ChunkMetrics.from_content(test_content)
            test_tokens = self._estimate_tokens(test_content)

            if (
                test_metrics.non_whitespace_chars <= self.config.max_chunk_size
                and test_tokens <= self.config.safe_token_limit
            ):
                current_group.append(recipe_line)
            else:
                if current_group:
                    result.append(
                        self._create_rule_chunk(
                            chunk, target_line, current_group, part,
                            recipe_offset,
                        )
                    )
                    part += 1
                    recipe_offset += len(current_group)
                current_group = [recipe_line]

        if current_group:
            result.append(
                self._create_rule_chunk(
                    chunk, target_line, current_group, part,
                    recipe_offset,
                )
            )

        if not result:
            return [chunk]

        # Validate all result chunks - fall back to emergency split for any oversized
        # This handles the case where a single recipe line + target exceeds size limit
        validated_result: list[UniversalChunk] = []
        for rule_chunk in result:
            metrics = ChunkMetrics.from_content(rule_chunk.content)
            tokens = self._estimate_tokens(rule_chunk.content)
            if (
                metrics.non_whitespace_chars > self.config.max_chunk_size
                or tokens > self.config.safe_token_limit
            ):
                # Single recipe line exceeds limit - use emergency split
                validated_result.extend(self._emergency_split(rule_chunk))
            else:
                validated_result.append(rule_chunk)

        return validated_result if validated_result else [chunk]

    def _create_rule_chunk(
        self,
        original: UniversalChunk,
        target: str,
        recipe_lines: list[str],
        part: int,
        recipe_offset: int,
    ) -> UniversalChunk:
        """Create a split rule chunk with target + recipe subset.

        Each chunk includes the target line for semantic coherence.
        start_line matches the target line; end_line uses recipe_offset
        for distinct spans per chunk.
        """
        content = "\n".join([target] + recipe_lines)

        start_line = original.start_line
        recipe_start = original.start_line + 1 + recipe_offset
        end_line = min(original.end_line, recipe_start + len(recipe_lines) - 1)

        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part}",
            content=content,
            start_line=start_line,
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

    def _create_makefile_engine(self) -> TreeSitterEngine:
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
        except ImportError as exc:
            raise SetupError(
                parser="makefile",
                missing_dependency="tree-sitter-make",
                install_command="pip install tree-sitter-make",
                original_error=str(exc),
            ) from exc
