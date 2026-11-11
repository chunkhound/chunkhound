"""Contracts for diff-backed search across vector_source modes (db/diff/both)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.services.search_service_protocol import SearchServiceProtocol
from chunkhound.services.transient_search_service import TransientSearchService
from tests.unit.transient_search_helpers import (
    keyed_embedding_manager,
    make_chunk,
    make_original,
    stream_chunks,
)

QUERY = [1.0, 0.0, 0.0]


def diff_service(
    original: MagicMock,
    chunks: list,
    vectors: list[list[float]],
    vector_source: str,
    query_vector: list[float] | None = QUERY,
) -> TransientSearchService:
    return TransientSearchService(
        original,
        stream_chunks(chunks),
        vector_source,
        keyed_embedding_manager(query_vector, chunks, vectors),
    )


def paged(results: list[dict]):
    async def search(**kwargs):
        offset = kwargs["offset"]
        page_size = kwargs["page_size"]
        page = results[offset : offset + page_size]
        has_more = offset + page_size < len(results)
        return page, {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": len(results),
        }

    return search


def db_rows(count: int) -> list[dict]:
    return [
        {
            "chunk_id": f"db-{index}",
            "file_path": f"db-{index}.py",
            "start_line": 1,
            "content": "db",
            "score": 0.5 - index / 100,
            "similarity": 0.5 - index / 100,
        }
        for index in range(count)
    ]


def test_search_services_satisfy_protocol() -> None:
    from chunkhound.services.search_service import SearchService

    assert isinstance(SearchService(MagicMock()), SearchServiceProtocol)
    assert isinstance(diff_service(MagicMock(), [], [], "diff"), SearchServiceProtocol)


# --- vector_source isolation -------------------------------------------------


@pytest.mark.asyncio
async def test_db_mode_semantic_is_database_passthrough() -> None:
    db_results = [{"file_path": "a.py", "content": "x", "start_line": 1, "score": 0.9}]
    original = make_original(semantic_results=db_results)

    results, _ = await diff_service(original, [], [], "db").search_semantic(
        "query", page_size=5
    )

    original.search_semantic.assert_awaited_once()
    assert results == db_results


@pytest.mark.asyncio
async def test_diff_mode_semantic_never_queries_database() -> None:
    chunk = make_chunk("my_func", file_path="src/a.py")
    original = make_original(
        semantic_results=[{"file_path": "b.py", "content": "y", "score": 0.5}]
    )

    results, _ = await diff_service(original, [chunk], [QUERY], "diff").search_semantic(
        "query"
    )

    original.search_semantic.assert_not_called()
    assert [(r["file_path"], r["content"]) for r in results] == [
        ("src/a.py", chunk.code)
    ]


@pytest.mark.asyncio
async def test_diff_mode_empty_diff_returns_empty_not_database() -> None:
    original = make_original(
        semantic_results=[{"file_path": "b.py", "content": "y", "score": 0.5}]
    )

    results, pagination = await diff_service(
        original, [], [], "diff", query_vector=None
    ).search_semantic("query")

    original.search_semantic.assert_not_called()
    assert results == []
    assert pagination["total"] == 0


@pytest.mark.asyncio
async def test_both_mode_empty_diff_still_returns_database_results() -> None:
    original = make_original(
        semantic_results=[
            {
                "file_path": "src/db.py",
                "content": "result",
                "start_line": 1,
                "similarity": 0.7,
                "score": 0.7,
            }
        ]
    )

    results, _ = await diff_service(
        original, [], [], "both", query_vector=None
    ).search_semantic("query")

    assert any(r["file_path"] == "src/db.py" for r in results)


# --- scoring -----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("query_vector", [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
async def test_diff_scores_are_cosine_regardless_of_query_norm(
    query_vector: list[float],
) -> None:
    chunks = [
        make_chunk("a", file_path="src/a.py", start_line=1),
        make_chunk("b", file_path="src/b.py", start_line=10),
    ]
    service = diff_service(
        make_original(),
        chunks,
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "diff",
        query_vector=query_vector,
    )

    results, _ = await service.search_semantic("query")

    assert [r["file_path"] for r in results] == ["src/a.py", "src/b.py"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert results[1]["score"] == pytest.approx(0.0, abs=1e-5)


@pytest.mark.asyncio
async def test_diff_threshold_filters_low_scores() -> None:
    chunks = [
        make_chunk("match", file_path="src/a.py", start_line=1),
        make_chunk("no_match", file_path="src/b.py", start_line=10),
    ]
    service = diff_service(
        make_original(), chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "diff"
    )

    results, _ = await service.search_semantic("query", threshold=0.5)

    assert [r["file_path"] for r in results] == ["src/a.py"]


@pytest.mark.asyncio
async def test_semantic_path_filter_matches_directory_components() -> None:
    chunks = [
        make_chunk("a", file_path="src/main.py"),
        make_chunk("b", file_path="tests/test_main.py"),
        make_chunk("c", file_path="src_utils/helper.py"),
    ]
    service = diff_service(make_original(), chunks, [QUERY] * 3, "diff")

    for path_filter in ("src", "src/"):
        results, _ = await service.search_semantic("query", path_filter=path_filter)
        assert [r["file_path"] for r in results] == ["src/main.py"]


# --- both-mode merge ---------------------------------------------------------


@pytest.mark.asyncio
async def test_both_mode_merges_diff_and_database_by_score() -> None:
    chunks = [
        make_chunk("high_score", file_path="src/high.py", start_line=10),
        make_chunk("low_score", file_path="src/low.py", start_line=20),
    ]
    original = make_original(
        semantic_results=[
            {
                "file_path": "src/db.py",
                "content": "db_result",
                "start_line": 30,
                "similarity": 0.5,
                "score": 0.5,
            }
        ]
    )
    service = diff_service(original, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "both")

    results, _ = await service.search_semantic("query")

    assert [r["file_path"] for r in results] == [
        "src/high.py",
        "src/db.py",
        "src/low.py",
    ]


@pytest.mark.asyncio
async def test_both_mode_normalises_distance_based_database_results() -> None:
    original = make_original(
        semantic_results=[
            {
                "file_path": "src/db.py",
                "content": "db_result",
                "start_line": 5,
                "distance": 0.2,
            }
        ]
    )
    service = diff_service(
        original, [make_chunk("func", file_path="src/diff.py")], [QUERY], "both"
    )

    results, _ = await service.search_semantic("query")

    db_entry = next(r for r in results if r["file_path"] == "src/db.py")
    assert db_entry["similarity"] == pytest.approx(0.8, abs=1e-5)
    assert db_entry["score"] == pytest.approx(0.8, abs=1e-5)


def _single_line_fragments() -> list:
    return [
        make_chunk(f"big.js:1 (part {part})", file_path="big.js", end_line=1)
        for part in (1, 2, 3)
    ]


@pytest.mark.asyncio
async def test_single_line_split_fragments_survive_both_mode_dedup() -> None:
    """Char-split fragments share (file_path, start_line); all must survive."""
    service = diff_service(
        make_original(),
        _single_line_fragments(),
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        "both",
    )

    results, _ = await service.search_semantic("query", page_size=10)

    assert len([r for r in results if r["file_path"] == "big.js"]) == 3


@pytest.mark.asyncio
async def test_single_line_split_fragments_survive_hybrid_merge() -> None:
    service = diff_service(
        make_original(),
        _single_line_fragments(),
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        "diff",
    )

    results, _ = await service.search_hybrid("query", page_size=10)

    assert len([r for r in results if r.get("file_path") == "big.js"]) == 3


# --- both-mode pagination ----------------------------------------------------


@pytest.mark.asyncio
async def test_both_mode_second_page_continues_ranking() -> None:
    chunks = [make_chunk(name, file_path=f"src/{name}.py") for name in "abc"]
    service = diff_service(
        make_original(),
        chunks,
        [[1.0, 0.0, 0.0], [0.5, 0.866, 0.0], [0.0, 1.0, 0.0]],
        "both",
    )

    page1, meta1 = await service.search_semantic("query", page_size=2, offset=0)
    page2, meta2 = await service.search_semantic("query", page_size=2, offset=2)

    assert [r["file_path"] for r in page1] == ["src/a.py", "src/b.py"]
    assert meta1["has_more"] is True
    assert [r["file_path"] for r in page2] == ["src/c.py"]
    assert meta2["has_more"] is False


@pytest.mark.asyncio
async def test_both_mode_continuation_until_all_sources_are_consumed() -> None:
    chunks = [make_chunk(f"diff-{i}", file_path=f"diff-{i}.py") for i in range(3)]
    original = make_original()
    original.search_semantic = AsyncMock(side_effect=paged(db_rows(12)))
    service = diff_service(original, chunks, [QUERY] * 3, "both")

    seen: list[str] = []
    metadata = None
    for offset in range(0, 16, 2):
        results, metadata = await service.search_semantic(
            "query", page_size=2, offset=offset
        )
        seen.extend(str(result["chunk_id"]) for result in results)
        if not metadata["has_more"]:
            break

    assert len(seen) == len(set(seen)) == 15
    assert metadata is not None
    assert metadata["has_more"] is False
    assert metadata["next_offset"] is None


@pytest.mark.asyncio
async def test_both_mode_total_is_numeric_lower_bound_while_sources_have_more() -> None:
    """`total` is a renderable lower bound, never None (CLI prints it)."""
    chunks = [make_chunk(f"diff-{i}", file_path=f"diff-{i}.py") for i in range(3)]
    db_results = db_rows(12)
    original = make_original()
    original.search_semantic = AsyncMock(side_effect=paged(db_results))
    service = diff_service(original, chunks, [QUERY] * 3, "both")

    results, pagination = await service.search_semantic("query", page_size=2)

    assert len(results) == 2
    assert pagination["has_more"] is True
    assert pagination["next_offset"] == 2
    total = pagination["total"]
    assert isinstance(total, int)
    assert len(results) <= total <= len(chunks) + len(db_results)


# --- hybrid pagination -------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_pagination_does_not_skip_ranked_candidates() -> None:
    chunks = [make_chunk(f"semantic-{i}", file_path=f"sem-{i}.py") for i in range(3)]
    regex_results = [
        {"chunk_id": f"regex-{i}", "file_path": f"regex-{i}.py", "content": "match"}
        for i in range(3)
    ]
    service = diff_service(
        make_original(regex_results=regex_results),
        chunks,
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        "both",
    )

    seen: list[str] = []
    metadata = None
    for offset in range(0, 6, 2):
        results, metadata = await service.search_hybrid(
            "query", regex_pattern="match", page_size=2, offset=offset
        )
        seen.extend(str(result["chunk_id"]) for result in results)

    assert len(seen) == len(set(seen)) == 6
    assert metadata is not None
    assert metadata["has_more"] is False
    assert metadata["next_offset"] is None


@pytest.mark.asyncio
async def test_hybrid_pagination_handles_source_continuation() -> None:
    regex_results = [
        {"chunk_id": f"regex-{i}", "file_path": f"regex-{i}.py", "content": "match"}
        for i in range(30)
    ]
    original = make_original()

    async def search_regex(
        pattern, page_size=10, offset=0, path_filter=None, query=None
    ):
        return await paged(regex_results)(offset=offset, page_size=page_size)

    original.search_regex_async = AsyncMock(side_effect=search_regex)
    service = diff_service(
        original, [make_chunk("semantic", file_path="semantic.py")], [QUERY], "both"
    )

    seen: list[str] = []
    offset = 0
    while True:
        results, metadata = await service.search_hybrid(
            "query",
            regex_pattern="match",
            page_size=5,
            offset=offset,
            semantic_weight=0.4,
        )
        seen.extend(str(result["chunk_id"]) for result in results)
        if not metadata["has_more"]:
            break
        offset = metadata["next_offset"]

    assert len(seen) == len(set(seen)) == 31
    assert metadata["next_offset"] is None
    canonical, _ = await service.search_hybrid(
        "query", regex_pattern="match", page_size=100, semantic_weight=0.4
    )
    assert seen == [str(result["chunk_id"]) for result in canonical]


@pytest.mark.asyncio
async def test_hybrid_overlap_has_same_ranking_on_every_page() -> None:
    chunks = [make_chunk(f"x{i}", file_path=f"x{i}.py") for i in range(100)]
    regex_results = [
        {
            "chunk_id": f"diff:x{i}.py:1:x{i}",
            "file_path": f"x{i}.py",
            "content": "match",
        }
        for i in (91, 0)
    ]
    service = diff_service(
        make_original(regex_results=regex_results), chunks, [QUERY] * 100, "both"
    )

    async def page(page_size: int, offset: int = 0) -> list[dict]:
        results, _ = await service.search_hybrid(
            "query",
            regex_pattern="match",
            page_size=page_size,
            offset=offset,
            semantic_weight=0.4,
        )
        return results

    canonical = await page(200)
    first = await page(1, 0)
    second = await page(1, 1)

    assert [first[0]["chunk_id"], second[0]["chunk_id"]] == [
        canonical[0]["chunk_id"],
        canonical[1]["chunk_id"],
    ]


# --- regex source isolation --------------------------------------------------


@pytest.mark.asyncio
async def test_regex_diff_mode_never_queries_database() -> None:
    """Historical regex must not mix in working-tree chunks."""
    original = make_original(
        regex_results=[{"file_path": "current.py", "content": "target"}]
    )
    chunks = [make_chunk("historical", code="def target(): pass")]

    results, pagination = await diff_service(
        original, chunks, [QUERY], "diff"
    ).search_regex_async("target")

    original.search_regex_async.assert_not_awaited()
    assert [result["content"] for result in results] == ["def target(): pass"]
    assert pagination["total"] == 1


@pytest.mark.asyncio
async def test_regex_db_mode_delegates_to_database() -> None:
    reg_results = [{"file_path": "src/x.py", "content": "match"}]
    original = make_original(regex_results=reg_results)

    results, _ = await diff_service(original, [], [], "db").search_regex_async(
        "pattern.*"
    )

    original.search_regex_async.assert_awaited_once()
    assert results == reg_results


@pytest.mark.asyncio
async def test_regex_both_mode_merges_and_prefers_diff_location() -> None:
    chunks = [make_chunk("changed", file_path="shared.py", code="target in diff")]
    db_results = [
        {
            "chunk_id": "stale",
            "file_path": "shared.py",
            "start_line": 1,
            "content": "target in working tree",
        },
        {
            "chunk_id": "db-only",
            "file_path": "other.py",
            "start_line": 1,
            "content": "target elsewhere",
        },
    ]
    service = diff_service(
        make_original(regex_results=db_results), chunks, [QUERY], "both"
    )

    results, _ = await service.search_regex_async("target")

    assert [result["content"] for result in results] == [
        "target in diff",
        "target elsewhere",
    ]
