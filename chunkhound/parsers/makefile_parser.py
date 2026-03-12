"""Specialized parser for Makefiles with size enforcement.

Makefile rules need special handling because:
1. Part 1 contains the target + recipe lines; subsequent parts contain recipe lines only
2. Splitting in the middle of a recipe breaks the target/recipe relationship
3. Non-rule chunks and within-limit rules delegate to ChunkSplitter's generic path

"""

from loguru import logger

from chunkhound.core.utils import estimate_tokens_chunking
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
        if chunk.metadata.get("kind") == "rule":
            metrics = ChunkMetrics.from_content(chunk.content)
            estimated_tokens = estimate_tokens_chunking(chunk.content)
            if (
                metrics.non_whitespace_chars > self.config.max_chunk_size
                or estimated_tokens > self.config.safe_token_limit
            ):
                return self._split_makefile_rule(chunk)
            return [chunk]  # within-limit rule — skip redundant super() check
        return super().validate_and_split(chunk)

    def _split_makefile_rule(self, chunk: UniversalChunk) -> list[UniversalChunk]:
        """Split Makefile rule preserving target/recipe coherence.

        Part 1 contains the target line + a recipe subset.
        Parts > 1 contain only recipe lines; the target is stored in
        metadata["rule_target"] so consumers can reconstruct context.

        Exception: if a chunk exceeds the size limit after grouping, it is
        emergency-split by characters. Part 1 retains the target in both content
        and metadata["rule_target"]; parts > 1 retain it only in metadata["rule_target"].
        """
        lines = chunk.content.split("\n")

        # Tree-sitter places the target on the first line of a 'rule' node;
        # guard against unexpected structure defensively.
        target_line = lines[0]
        recipe_lines = lines[1:]
        if not recipe_lines:
            return self._recursive_split(chunk)

        # Group recipe lines into size-limited chunks
        result: list[UniversalChunk] = []
        current_group: list[str] = []
        part = 1
        recipe_offset = 0

        for recipe_line in recipe_lines:
            if part == 1:
                test_content = "\n".join([target_line] + current_group + [recipe_line])
            else:
                test_content = "\n".join(current_group + [recipe_line])
            test_metrics = ChunkMetrics.from_content(test_content)
            test_tokens = estimate_tokens_chunking(test_content)

            if (
                test_metrics.non_whitespace_chars <= self.config.max_chunk_size
                and test_tokens <= self.config.safe_token_limit
            ):
                current_group.append(recipe_line)
            else:
                if current_group:
                    result.append(
                        self._create_rule_chunk(
                            chunk,
                            target_line,
                            current_group,
                            part,
                            recipe_offset,
                        )
                    )
                    part += 1
                    recipe_offset += len(current_group)
                current_group = [recipe_line]

        if current_group:
            result.append(
                self._create_rule_chunk(
                    chunk,
                    target_line,
                    current_group,
                    part,
                    recipe_offset,
                )
            )

        # Validate all result chunks - fall back to emergency split for any oversized
        # This handles the case where a single recipe line + target exceeds size limit
        validated_result: list[UniversalChunk] = []
        for rule_chunk in result:
            metrics = ChunkMetrics.from_content(rule_chunk.content)
            tokens = estimate_tokens_chunking(rule_chunk.content)
            if (
                metrics.non_whitespace_chars > self.config.max_chunk_size
                or tokens > self.config.safe_token_limit
            ):
                # Single recipe line exceeds limit - use emergency split
                validated_result.extend(self._emergency_split(rule_chunk))
            else:
                validated_result.append(rule_chunk)

        return validated_result

    def _create_rule_chunk(
        self,
        original: UniversalChunk,
        target: str,
        recipe_lines: list[str],
        part: int,
        recipe_offset: int,
    ) -> UniversalChunk:
        """Create a split rule chunk with contiguous line ranges.

        Part 1 includes the target line + its recipe subset (contiguous from
        original.start_line). Parts > 1 contain only their recipe lines with
        start_line pointing to the first recipe line in the group; the target
        is stored in metadata["rule_target"] for semantic reference.
        """
        # +1 skips the single target line to reach the first recipe line
        recipe_start = original.start_line + 1 + recipe_offset
        recipe_end = recipe_start + len(recipe_lines) - 1

        if part == 1:
            content = "\n".join([target] + recipe_lines)
            start_line = original.start_line
        else:
            content = "\n".join(recipe_lines)
            start_line = recipe_start

        # Fires only for malformed input where end_line < start_line + content lines (parser bug)
        if recipe_end > original.end_line:
            logger.warning(
                "recipe_end {} exceeds original.end_line {} for chunk '{}'; clamping",
                recipe_end,
                original.end_line,
                original.name,
            )
            recipe_end = original.end_line
        end_line = recipe_end

        meta = original.metadata.copy()
        meta["rule_target"] = target

        return UniversalChunk(
            concept=original.concept,
            name=f"{original.name}_part{part}",
            content=content,
            start_line=start_line,
            end_line=end_line,
            metadata=meta,
            language_node_type=original.language_node_type,
        )


class MakefileParser(UniversalParser):
    """Parser for Makefiles with built-in size enforcement.

    Inherits from UniversalParser and uses MakefileChunkSplitter
    to handle oversized rules with target/recipe coherence.
    """

    def __init__(
        self, cast_config: CASTConfig | None = None, detect_embedded_sql: bool = True
    ):
        engine = self._create_makefile_engine()
        mapping = MakefileMapping()
        super().__init__(
            engine, mapping, cast_config, detect_embedded_sql=detect_embedded_sql
        )
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
