"""Tests for chunk_splitter module."""

from __future__ import annotations

from chunkhound.core.types.common import ChunkType
from chunkhound.parsers.chunk_splitter import (
    CHUNK_TYPE_TO_CONCEPT,
    CASTConfig,
    ChunkMetrics,
    ChunkSplitter,
    compute_end_line,
)
from chunkhound.parsers.makefile_parser import MakefileChunkSplitter
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept


def _make_chunk(
    content: str,
    name: str = "test",
    start_line: int = 1,
    *,
    concept: UniversalConcept = UniversalConcept.DEFINITION,
    metadata: dict[str, str] | None = None,
    language_node_type: str = "function_definition",
) -> UniversalChunk:
    """Create a UniversalChunk with sensible defaults for testing."""
    return UniversalChunk(
        concept=concept,
        name=name,
        content=content,
        start_line=start_line,
        end_line=compute_end_line(content, start_line),
        metadata=metadata or {},
        language_node_type=language_node_type,
    )


# --- Mapping completeness ---


def test_chunk_to_universal_covers_all_chunk_types() -> None:
    """Verify CHUNK_TYPE_TO_CONCEPT covers all ChunkType enum members."""
    all_types = set(ChunkType)
    mapped_types = set(CHUNK_TYPE_TO_CONCEPT.keys())
    missing = all_types - mapped_types

    assert not missing, (
        f"CHUNK_TYPE_TO_CONCEPT missing {len(missing)} types: "
        f"{sorted(t.name for t in missing)}"
    )


# --- CASTConfig defaults ---


def test_cast_config_defaults() -> None:
    """Verify CASTConfig defaults match documented values."""
    config = CASTConfig()
    assert config.max_chunk_size == 1200
    assert config.min_chunk_size == 50
    assert config.safe_token_limit == 6000


# --- ChunkMetrics ---


def test_chunk_metrics_non_whitespace_counting() -> None:
    metrics = ChunkMetrics.from_content("a b c")
    assert metrics.non_whitespace_chars == 3


def test_chunk_metrics_line_counting() -> None:
    metrics = ChunkMetrics.from_content("a\nb\nc\nd\ne")
    assert metrics.lines == 5


# --- validate_and_split ---


def test_validate_and_split_within_limit() -> None:
    """Chunk at exactly max_chunk_size returns unchanged."""
    splitter = ChunkSplitter()
    # "x = 1" has 3 non-whitespace chars (x, =, 1); 400 lines -> 1200 nws = exactly at limit
    content = "\n".join(["x = 1"] * 400)
    chunk = _make_chunk(content)
    result = splitter.validate_and_split(chunk)
    assert len(result) == 1
    assert result[0].content == content


def test_validate_and_split_exceeds_limit() -> None:
    """Chunk over max_chunk_size gets split."""
    splitter = ChunkSplitter()
    # 401 lines * 3 nws = 1203 -> over limit
    content = "\n".join(["x = 1"] * 401)
    chunk = _make_chunk(content)
    result = splitter.validate_and_split(chunk)
    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size


def test_validate_and_split_empty_content() -> None:
    """Empty content returns chunk unchanged."""
    splitter = ChunkSplitter()
    chunk = _make_chunk("")
    result = splitter.validate_and_split(chunk)
    assert len(result) == 1


# --- _emergency_split ---


def test_emergency_split_minified_single_line() -> None:
    """Single long line (minified code) splits without infinite loop."""
    splitter = ChunkSplitter()
    # "a=1;" has 4 nws chars; 500 reps = 2000 nws on a single line
    content = "a=1;" * 500
    chunk = _make_chunk(content)
    result = splitter._emergency_split(chunk)
    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size


def test_emergency_split_no_split_chars() -> None:
    """Content with no split characters force-splits at character limit."""
    splitter = ChunkSplitter()
    content = "a" * 3000
    chunk = _make_chunk(content)
    result = splitter._emergency_split(chunk)
    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size


def test_emergency_split_preserves_metadata() -> None:
    """Split chunks retain concept, language_node_type, and name prefix."""
    splitter = ChunkSplitter()
    content = "a=1;" * 500
    chunk = _make_chunk(
        content,
        name="my_func",
        concept=UniversalConcept.DEFINITION,
        language_node_type="function_definition",
    )
    result = splitter._emergency_split(chunk)
    assert len(result) > 1
    for part in result:
        assert part.concept == UniversalConcept.DEFINITION
        assert part.language_node_type == "function_definition"
        assert part.name.startswith("my_func")


# --- MakefileChunkSplitter ---


def test_makefile_within_limit_rule_passes_through() -> None:
    """Within-limit rule chunk returns unchanged (short-circuits without calling super)."""
    splitter = MakefileChunkSplitter()
    content = "install: all\n\tcp src /dst"
    chunk = _make_chunk(
        content,
        name="install",
        start_line=1,
        concept=UniversalConcept.DEFINITION,
        metadata={"kind": "rule"},
        language_node_type="rule",
    )

    result = splitter.validate_and_split(chunk)

    assert len(result) == 1
    assert result[0].content == content


def test_makefile_non_rule_oversized_delegates_to_parent() -> None:
    """Oversized non-rule chunk splits via parent's validate_and_split."""
    splitter = MakefileChunkSplitter()
    content = "\n".join(["x = 1"] * 401)  # 401 * 3 nws = 1203, over limit
    chunk = _make_chunk(content, metadata={"kind": "variable"})
    result = splitter.validate_and_split(chunk)
    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size


def test_makefile_split_produces_non_overlapping_line_ranges() -> None:
    """Parts 2+ start at their recipe range, not the original target line."""
    splitter = MakefileChunkSplitter()
    # Target line + 500 recipe lines — well over the 1200 nws char limit
    target = "install: all"
    recipe_lines = [f"\tcp file_{i}.txt /usr/local/bin/" for i in range(500)]
    content = "\n".join([target] + recipe_lines)
    chunk = _make_chunk(
        content,
        name="install",
        start_line=10,
        concept=UniversalConcept.DEFINITION,
        metadata={"kind": "rule"},
        language_node_type="rule",
    )

    result = splitter.validate_and_split(chunk)

    assert len(result) > 1
    assert result[0].start_line == 10  # part 1 owns the target line
    for part in result[1:]:
        assert part.start_line > 10  # parts 2+ start past the target
    # No two parts share a start_line
    start_lines = [p.start_line for p in result]
    assert len(start_lines) == len(set(start_lines))
    # Strict non-overlap: consecutive parts must not overlap
    for a, b in zip(result, result[1:]):
        assert a.end_line < b.start_line
    # Each recipe line fits within the chunk limit so normal splitting applies —
    # normal splits prepend the target to every part; no emergency split here.
    for part in result:
        assert part.content.startswith(target)


def test_makefile_split_target_only_oversized_rule() -> None:
    """Oversized rule with no recipe lines falls back to _recursive_split."""
    splitter = MakefileChunkSplitter()
    # Target-only content that exceeds the size limit (no recipe lines)
    target = "build: " + "dep_" * 500  # ~2000 nws chars, well over limit
    chunk = _make_chunk(
        target,
        name="build",
        start_line=1,
        concept=UniversalConcept.DEFINITION,
        metadata={"kind": "rule"},
        language_node_type="rule",
    )

    result = splitter.validate_and_split(chunk)

    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size


def test_makefile_split_falls_back_to_emergency_for_huge_recipe_line() -> None:
    """A single recipe line exceeding max_chunk_size triggers emergency split."""
    splitter = MakefileChunkSplitter()
    target = "install: all"
    huge_recipe = "\tcp " + "x" * 3000  # single line, ~3000 nws chars
    content = target + "\n" + huge_recipe
    chunk = _make_chunk(
        content,
        name="install",
        start_line=1,
        concept=UniversalConcept.DEFINITION,
        metadata={"kind": "rule"},
        language_node_type="rule",
    )

    result = splitter.validate_and_split(chunk)

    assert len(result) > 1
    for part in result:
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars <= splitter.config.max_chunk_size
