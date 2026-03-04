"""Tests for chunk_splitter module."""

from __future__ import annotations

from chunkhound.core.types.common import ChunkType
from chunkhound.core.utils.token_utils import estimate_tokens_chunking
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


def test_cast_config_rejects_zero_min_chunk_size() -> None:
    """min_chunk_size=0 would cause infinite loop in _emergency_split."""
    import pytest
    with pytest.raises(ValueError, match="min_chunk_size must be >= 1"):
        CASTConfig(min_chunk_size=0)


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


def test_emergency_split_respects_min_chunk_size() -> None:
    """The halving loop stops at min_chunk_size, not 1.

    Dense content with no split characters triggers force-split halving.
    All parts except possibly the last remainder must be at least min_chunk_size
    non-whitespace characters.
    """
    config = CASTConfig(max_chunk_size=100, min_chunk_size=20, safe_token_limit=400)
    splitter = ChunkSplitter(config)
    # 500 non-ws chars, no split chars → triggers _emergency_split force path
    content = "a" * 500
    chunk = _make_chunk(content)
    result = splitter._emergency_split(chunk)
    assert len(result) > 1
    for part in result[:-1]:  # last part may be a small remainder
        metrics = ChunkMetrics.from_content(part.content)
        assert metrics.non_whitespace_chars >= config.min_chunk_size


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


def test_split_by_lines_simple_no_overlap_when_span_narrow() -> None:
    """No overlapping line spans when metadata span << content line count.

    Exercises the edge case where end_line - start_line is tiny but the chunk
    has many content lines (metadata/content mismatch). The fixed clamping code
    falls back to _emergency_split, producing non-overlapping chunks.
    """
    splitter = ChunkSplitter()
    # 401 lines of content (well over limit) but metadata declares only 5 lines
    content = "\n".join(["x = 1"] * 401)
    chunk = UniversalChunk(
        concept=UniversalConcept.DEFINITION,
        name="narrow_span",
        content=content,
        start_line=10,
        end_line=14,  # span=4, far narrower than 401 content lines
        metadata={},
        language_node_type="function_definition",
    )
    result = splitter.validate_and_split(chunk)
    assert len(result) > 1
    # No two consecutive parts may overlap
    for a, b in zip(result, result[1:]):
        assert a.end_line < b.start_line


def test_emergency_split_respects_token_limit_for_whitespace_heavy_content() -> None:
    """All output chunks must satisfy the token limit even when non-ws chars are few.

    A chunk with sparse non-whitespace but enormous whitespace padding has
    non_whitespace_chars well within max_chunk_size while estimate_tokens_chunking
    (which counts all characters) exceeds safe_token_limit. The fixed code must
    keep halving until both constraints are satisfied.
    """
    splitter = ChunkSplitter()
    # ~40 non-ws chars total, but ~40000 total chars → ~13333 estimated tokens
    segment = "x" + " " * 19999
    content = segment + "\n" + segment
    chunk = _make_chunk(content, name="whitespace_heavy", start_line=1)

    result = splitter.validate_and_split(chunk)

    for part in result:
        assert estimate_tokens_chunking(part.content) <= splitter.config.safe_token_limit


def test_split_by_lines_boundary_chunk2_equals_end_line() -> None:
    """When chunk2_start_line would equal end_line, fall back to emergency split.

    Constructed so that mid_point == line_span, which causes the clamping logic
    to compute chunk2_start_line == chunk.end_line exactly. With the old `>` guard
    that case would proceed, producing chunk2 with start_line == end_line despite
    containing multiple lines of content. The `>=` fix correctly falls back to
    _emergency_split.

    Setup: span=3, 6 content lines → mid_point=3=span → chunk2_start_line=end_line.
    Lines are long enough (>1200 non-ws total) to trigger splitting but short
    enough (<240 chars each) to avoid the _emergency_split fast-path.
    """
    splitter = ChunkSplitter()
    # 6 lines × 231 non-ws chars = 1386 non-ws > 1200 (triggers split)
    # 233 chars per line < 240 (long_line_threshold) so _split_by_lines_simple is chosen
    line = "z = " + "a" * 229
    content = "\n".join([line] * 6)
    chunk = UniversalChunk(
        concept=UniversalConcept.DEFINITION,
        name="boundary_midpoint",
        content=content,
        start_line=10,
        end_line=13,  # span=3; mid_point=3 → chunk2_start_line=13=end_line
        metadata={},
        language_node_type="function_definition",
    )

    result = splitter.validate_and_split(chunk)

    assert len(result) > 1
    for part in result:
        assert not (part.start_line == part.end_line and "\n" in part.content), (
            f"Chunk '{part.name}' has single-line metadata span but multi-line content"
        )
