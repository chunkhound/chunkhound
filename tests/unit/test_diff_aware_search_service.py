"""Unit tests for DiffAwareSearchService and SearchServiceProtocol."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkId, ChunkType, FileId, Language, LineNumber
from chunkhound.embeddings import LocalEmbeddingResult
from chunkhound.services.diff_aware_search_service import (
    DiffAwareSearchService,
    SearchServiceProtocol,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    symbol: str,
    file_path: str = "src/foo.py",
    start_line: int = 1,
    end_line: int | None = None,
    code: str = "def foo(): pass",
) -> Chunk:
    # Ensure start_line is positive and end_line >= start_line
    sl = max(1, start_line)
    el = end_line if end_line is not None else sl + 4
    el = max(el, sl)
    return Chunk(
        symbol=symbol,
        start_line=LineNumber(sl),
        end_line=LineNumber(el),
        code=code,
        chunk_type=ChunkType.FUNCTION,
        file_id=FileId(1),
        language=Language.PYTHON,
        file_path=file_path,  # type: ignore[arg-type]
    )


def make_embedding_manager(embeddings: list[list[float]]) -> MagicMock:
    """Return a mock EmbeddingManager whose embed_texts returns controlled embeddings."""
    manager = MagicMock()

    async def _embed(texts):
        return LocalEmbeddingResult(
            embeddings=embeddings,
            model="test-model",
            provider="test",
            dims=len(embeddings[0]) if embeddings else 0,
        )

    manager.embed_texts = _embed
    return manager


def make_original(
    semantic_results: list[dict] | None = None,
    regex_results: list[dict] | None = None,
) -> MagicMock:
    """Return a mock original SearchService."""
    original = MagicMock()

    sem = semantic_results or []
    reg = regex_results or []

    original.search_semantic = AsyncMock(
        return_value=(sem, {"offset": 0, "page_size": 10, "has_more": False, "next_offset": None, "total": len(sem)})
    )
    original.search_regex = MagicMock(
        return_value=(reg, {"offset": 0, "page_size": 10, "has_more": False, "next_offset": None, "total": len(reg)})
    )
    original.search_regex_async = AsyncMock(
        return_value=(reg, {"offset": 0, "page_size": 10, "has_more": False, "next_offset": None, "total": len(reg)})
    )
    original.search_hybrid = AsyncMock(
        return_value=([], {"offset": 0, "page_size": 10, "has_more": False, "next_offset": None, "total": 0})
    )
    original.get_chunk_context = MagicMock(return_value={"context": "data"})
    original.get_file_chunks = MagicMock(return_value=[])
    return original


# ---------------------------------------------------------------------------
# Protocol check
# ---------------------------------------------------------------------------

def test_protocol_check():
    """DiffAwareSearchService should satisfy SearchServiceProtocol."""
    svc = DiffAwareSearchService(MagicMock(), [], [], "diff", MagicMock())
    assert isinstance(svc, SearchServiceProtocol), "Protocol check failed"


# ---------------------------------------------------------------------------
# vector_source == "db"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_db_mode_delegates_entirely():
    """In db mode, search_semantic is a pure passthrough; original is always called."""
    db_results = [{"file_path": "a.py", "content": "x", "start_line": 1, "score": 0.9}]
    original = make_original(semantic_results=db_results)
    manager = make_embedding_manager([[1.0, 0.0]])

    svc = DiffAwareSearchService(original, [], [], "db", manager)
    results, pagination = await svc.search_semantic("query", page_size=5)

    original.search_semantic.assert_called_once()
    assert results == db_results


# ---------------------------------------------------------------------------
# vector_source == "diff"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_diff_mode_returns_diff_results_not_original():
    """In diff mode, results come from diff chunks; original.search_semantic never called."""
    chunk = make_chunk("my_func", file_path="src/a.py")
    embeddings = [[1.0, 0.0, 0.0]]
    original = make_original()
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])

    svc = DiffAwareSearchService(original, [chunk], embeddings, "diff", manager)
    results, _ = await svc.search_semantic("query")

    original.search_semantic.assert_not_called()
    assert len(results) == 1
    assert results[0]["file_path"] == "src/a.py"
    assert results[0]["content"] == chunk.code


@pytest.mark.asyncio
async def test_diff_mode_empty_chunks_falls_back_to_original():
    """diff mode with empty diff_chunks falls back to original service."""
    db_results = [{"file_path": "b.py", "content": "y", "score": 0.5}]
    original = make_original(semantic_results=db_results)
    manager = make_embedding_manager([])

    svc = DiffAwareSearchService(original, [], [], "diff", manager)
    results, _ = await svc.search_semantic("query")

    original.search_semantic.assert_called_once()
    assert results == db_results


# ---------------------------------------------------------------------------
# vector_source == "both"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_mode_merges_and_sorts_by_score():
    """both mode merges diff + DB results sorted by score descending."""
    chunk_high = make_chunk("high_score", file_path="src/high.py", start_line=10)
    chunk_low = make_chunk("low_score", file_path="src/low.py", start_line=20)

    # chunk_high embedding is [1,0,0], query is [1,0,0] → score ~1.0
    # chunk_low embedding is [0,1,0], query is [1,0,0] → score ~0.0
    diff_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    original = make_original(
        semantic_results=[
            {"file_path": "src/db.py", "content": "db_result", "start_line": 30, "similarity": 0.5, "score": 0.5}
        ]
    )
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])

    svc = DiffAwareSearchService(
        original, [chunk_high, chunk_low], diff_embeddings, "both", manager
    )
    results, _ = await svc.search_semantic("query")

    # Should be sorted by score desc
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    file_paths = [r["file_path"] for r in results]
    assert "src/high.py" in file_paths
    assert "src/db.py" in file_paths


@pytest.mark.asyncio
async def test_both_mode_deduplicates_on_file_and_start_line():
    """both mode deduplicates (file_path, start_line); keeps highest-score entry."""
    chunk = make_chunk("func", file_path="src/a.py", start_line=1)
    diff_embeddings = [[1.0, 0.0, 0.0]]

    # DB also returns a result at the same (file_path, start_line) with lower score
    original = make_original(
        semantic_results=[
            {
                "file_path": "src/a.py",
                "content": "def func(): ...",
                "start_line": 1,
                "similarity": 0.3,
                "score": 0.3,
            }
        ]
    )
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])

    svc = DiffAwareSearchService(original, [chunk], diff_embeddings, "both", manager)
    results, _ = await svc.search_semantic("query")

    # Only one result for (src/a.py, 1)
    matching = [r for r in results if r["file_path"] == "src/a.py" and r["start_line"] == 1]
    assert len(matching) == 1
    # It should be the higher-score diff result (score ~1.0)
    assert matching[0]["score"] > 0.9


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_threshold_filters_low_scores():
    """Results below threshold are excluded in diff mode."""
    chunk_a = make_chunk("match", file_path="src/a.py", start_line=1)
    chunk_b = make_chunk("no_match", file_path="src/b.py", start_line=10)

    # chunk_a embedding [1,0,0] will score ~1.0 against query [1,0,0]
    # chunk_b embedding [0,1,0] will score ~0.0
    diff_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])
    original = make_original()

    svc = DiffAwareSearchService(
        original, [chunk_a, chunk_b], diff_embeddings, "diff", manager
    )
    results, _ = await svc.search_semantic("query", threshold=0.5)

    # Only chunk_a should survive the threshold
    assert len(results) == 1
    assert results[0]["file_path"] == "src/a.py"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pagination_offset_and_page_size():
    """offset and page_size correctly slice diff results."""
    chunks = [
        make_chunk(f"func_{i}", file_path=f"src/f{i}.py", start_line=i * 10)
        for i in range(5)
    ]
    # All identical embeddings → same score; order preserved by argsort stability
    embeddings = [[1.0, 0.0, 0.0]] * 5
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])
    original = make_original()

    svc = DiffAwareSearchService(original, chunks, embeddings, "diff", manager)
    results, pagination = await svc.search_semantic("query", page_size=2, offset=2)

    assert len(results) == 2
    assert pagination["offset"] == 2
    assert pagination["page_size"] == 2
    assert pagination["has_more"] is True
    assert pagination["next_offset"] == 4


# ---------------------------------------------------------------------------
# search_regex_async always delegates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_regex_async_delegates_regardless_of_vector_source():
    """search_regex_async always delegates to original, regardless of vector_source."""
    reg_results = [{"file_path": "src/x.py", "content": "match"}]
    original = make_original(regex_results=reg_results)
    manager = make_embedding_manager([])

    for source in ("diff", "db", "both"):
        original.search_regex_async.reset_mock()
        svc = DiffAwareSearchService(original, [], [], source, manager)
        results, _ = await svc.search_regex_async("pattern.*")
        original.search_regex_async.assert_called_once()
        assert results == reg_results


# ---------------------------------------------------------------------------
# Cosine similarity with known unit vectors (B1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cosine_similarity_known_unit_vectors():
    """q=[1,0,0], chunk embeddings [1,0,0] and [0,1,0]: first scores ~1.0, second ~0.0."""
    chunk_a = make_chunk("a", file_path="src/a.py", start_line=1)
    chunk_b = make_chunk("b", file_path="src/b.py", start_line=10)
    diff_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])
    original = make_original()

    svc = DiffAwareSearchService(original, [chunk_a, chunk_b], diff_embeddings, "diff", manager)
    results, _ = await svc.search_semantic("query")

    assert results[0]["file_path"] == "src/a.py"
    assert abs(results[0]["score"] - 1.0) < 1e-5
    assert results[1]["file_path"] == "src/b.py"
    assert abs(results[1]["score"] - 0.0) < 1e-5


@pytest.mark.asyncio
async def test_b1_non_unit_query_gives_same_ranking_as_unit_query():
    """Non-unit query [3,0,0] gives same ranking as unit query [1,0,0] (B1 normalization)."""
    chunk_a = make_chunk("a", file_path="src/a.py", start_line=1)
    chunk_b = make_chunk("b", file_path="src/b.py", start_line=10)
    diff_embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    original = make_original()

    # Unit query
    manager_unit = make_embedding_manager([[1.0, 0.0, 0.0]])
    svc_unit = DiffAwareSearchService(original, [chunk_a, chunk_b], diff_embeddings, "diff", manager_unit)
    results_unit, _ = await svc_unit.search_semantic("query")

    # Non-unit query (magnitude 3)
    manager_non_unit = make_embedding_manager([[3.0, 0.0, 0.0]])
    svc_non_unit = DiffAwareSearchService(
        original, [chunk_a, chunk_b], diff_embeddings, "diff", manager_non_unit
    )
    results_non_unit, _ = await svc_non_unit.search_semantic("query")

    # Same ranking order
    assert [r["file_path"] for r in results_unit] == [r["file_path"] for r in results_non_unit]
    # Scores should be equal (both normalised)
    for r_unit, r_non in zip(results_unit, results_non_unit):
        assert abs(r_unit["score"] - r_non["score"]) < 1e-4


# ---------------------------------------------------------------------------
# G1: path_filter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g1_path_filter_excludes_non_matching():
    """path_filter excludes chunks with non-matching file_path."""
    chunk_a = make_chunk("a", file_path="src/main.py", start_line=1)
    chunk_b = make_chunk("b", file_path="tests/test_main.py", start_line=1)
    diff_embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])
    original = make_original()

    svc = DiffAwareSearchService(original, [chunk_a, chunk_b], diff_embeddings, "diff", manager)
    results, _ = await svc.search_semantic("query", path_filter="src/")

    file_paths = [r["file_path"] for r in results]
    assert "src/main.py" in file_paths
    assert "tests/test_main.py" not in file_paths


# ---------------------------------------------------------------------------
# B3: "both" mode with distance-based DB results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_b3_both_mode_handles_distance_based_db_results():
    """both mode works when DB returns distance-based results (has 'distance' key)."""
    chunk = make_chunk("func", file_path="src/diff.py", start_line=1)
    diff_embeddings = [[1.0, 0.0, 0.0]]

    # DB returns distance-based results (no 'similarity' key)
    original = make_original(
        semantic_results=[
            {"file_path": "src/db.py", "content": "db_result", "start_line": 5, "distance": 0.2}
        ]
    )
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])

    svc = DiffAwareSearchService(original, [chunk], diff_embeddings, "both", manager)
    results, _ = await svc.search_semantic("query")

    # DB result should have been normalised: similarity = 1 - 0.2 = 0.8
    db_entry = next((r for r in results if r["file_path"] == "src/db.py"), None)
    assert db_entry is not None
    assert abs(db_entry["similarity"] - 0.8) < 1e-5
    assert abs(db_entry["score"] - 0.8) < 1e-5


# ---------------------------------------------------------------------------
# path_filter normalization — no partial directory name match
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_path_filter_does_not_match_partial_directory_name():
    """path_filter='src' must not match 'src_utils/foo.py'."""
    chunk_match = make_chunk("in_src", file_path="src/auth.py", start_line=1)
    chunk_no_match = make_chunk("in_src_utils", file_path="src_utils/helper.py", start_line=1)
    diff_embeddings = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])
    original = make_original()

    svc = DiffAwareSearchService(
        original, [chunk_match, chunk_no_match], diff_embeddings, "diff", manager
    )
    results, _ = await svc.search_semantic("query", path_filter="src")

    paths = [r["file_path"] for r in results]
    assert "src/auth.py" in paths
    assert "src_utils/helper.py" not in paths


# ---------------------------------------------------------------------------
# "both" mode multi-page pagination correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_mode_pagination_second_page_does_not_skip_results():
    """offset=2, page_size=1 in both mode returns the 3rd-ranked merged result."""
    # 3 diff chunks with known score order: chunk_a > chunk_b > chunk_c
    chunk_a = make_chunk("a", file_path="src/a.py", start_line=1)
    chunk_b = make_chunk("b", file_path="src/b.py", start_line=1)
    chunk_c = make_chunk("c", file_path="src/c.py", start_line=1)
    # query = [1,0,0]; embeddings score: a=1.0, b=0.5, c=0.0
    diff_embeddings = [
        [1.0, 0.0, 0.0],
        [0.5, 0.866, 0.0],
        [0.0, 1.0, 0.0],
    ]
    # DB returns nothing new
    original = make_original(semantic_results=[])
    manager = make_embedding_manager([[1.0, 0.0, 0.0]])

    svc = DiffAwareSearchService(
        original, [chunk_a, chunk_b, chunk_c], diff_embeddings, "both", manager
    )

    # Page 1 (offset=0, page_size=2): should be [a, b]
    page1, p1_meta = await svc.search_semantic("query", page_size=2, offset=0)
    assert [r["file_path"] for r in page1] == ["src/a.py", "src/b.py"]
    assert p1_meta["has_more"] is True

    # Page 2 (offset=2, page_size=2): should be [c]
    page2, p2_meta = await svc.search_semantic("query", page_size=2, offset=2)
    assert [r["file_path"] for r in page2] == ["src/c.py"]
    assert p2_meta["has_more"] is False


# ---------------------------------------------------------------------------
# Empty diff_embeddings with vector_source="both"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_both_mode_empty_diff_embeddings_still_returns_db_results():
    """With empty diff_embeddings, both mode still returns DB results."""
    db_results = [{"file_path": "src/db.py", "content": "result", "start_line": 1, "similarity": 0.7, "score": 0.7}]
    original = make_original(semantic_results=db_results)
    manager = make_embedding_manager([])

    svc = DiffAwareSearchService(original, [], [], "both", manager)
    results, _ = await svc.search_semantic("query")

    assert any(r["file_path"] == "src/db.py" for r in results)
