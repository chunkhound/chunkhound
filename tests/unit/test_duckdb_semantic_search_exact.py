"""DuckDB semantic search returns the exact ranking, scoped like the old query.

The query keeps vectors out of the chunk/file joins. These tests pin what that
must not change: the ranking equals a brute-force cosine ranking, an embedding
whose chunk no longer exists is never returned and never shortens a page or
fails the search, and the path filter, threshold and offset still apply.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator

_DIMS = 8
_PAGE = 3


def _source(prefix: str) -> str:
    return "\n\n".join(
        f"def {prefix}_{i}(value):\n"
        f"    total = value * {i + 2}\n"
        f"    for step in range({i + 3}):\n"
        f"        total += step\n"
        f"    return total\n"
        for i in range(8)
    )


@pytest.fixture
def indexed(tmp_path: Path):
    pytest.importorskip("duckdb")
    root = tmp_path / "repo"
    for sub in ("alpha", "beta"):
        (root / sub).mkdir(parents=True)
        (root / sub / f"{sub}.py").write_text(_source(sub))

    provider = DuckDBProvider(db_path=tmp_path / "db.duckdb", base_directory=root)
    provider.connect()
    coordinator = IndexingCoordinator(provider, root)
    paths: dict[int, str] = {}
    for sub in ("alpha", "beta"):
        result = asyncio.run(
            coordinator.process_file(root / sub / f"{sub}.py", skip_embeddings=True)
        )
        for chunk in provider.get_chunks_by_file_id(result["file_id"], as_model=False):
            paths[int(chunk["id"])] = sub
    assert sum(1 for p in paths.values() if p == "beta") > 2 * _PAGE, (
        "fixture needs more beta chunks than two pages"
    )

    rng = np.random.default_rng(5)
    vectors = {cid: rng.standard_normal(_DIMS).tolist() for cid in paths}
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": cid,
                "provider": "test",
                "model": "mini",
                "embedding": vec,
                "dims": _DIMS,
            }
            for cid, vec in vectors.items()
        ]
    )
    query = rng.standard_normal(_DIMS).tolist()
    # An embedding left behind by a deleted chunk, identical to the query, so
    # it would rank first if it were scored without the join.
    orphan_id = max(paths) + 1000
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": orphan_id,
                "provider": "test",
                "model": "mini",
                "embedding": query,
                "dims": _DIMS,
            }
        ]
    )
    try:
        yield provider, paths, vectors, query, orphan_id
    finally:
        provider.disconnect()


def _expected(vectors, query, keep=lambda cid: True):
    q = np.asarray(query)
    scored = [
        (cid, float(np.dot(v, q) / (np.linalg.norm(v) * np.linalg.norm(q))))
        for cid, v in ((cid, np.asarray(vec)) for cid, vec in vectors.items())
        if keep(cid)
    ]
    return sorted(scored, key=lambda item: item[1], reverse=True)


def _search(provider, query, **kwargs):
    results, pagination = provider.search_semantic(
        query, provider="test", model="mini", **kwargs
    )
    return [(int(r["chunk_id"]), r["similarity"]) for r in results], pagination


def _assert_same(actual, expected):
    assert [cid for cid, _ in actual] == [cid for cid, _ in expected]
    assert [s for _, s in actual] == pytest.approx([s for _, s in expected], abs=1e-5)


def test_ranking_is_exact_and_skips_orphaned_embeddings(indexed) -> None:
    provider, _, vectors, query, orphan_id = indexed
    expected = _expected(vectors, query)

    first, pagination = _search(provider, query, page_size=_PAGE)
    second, _ = _search(provider, query, page_size=_PAGE, offset=_PAGE)

    assert orphan_id not in [cid for cid, _ in first + second]
    _assert_same(first, expected[:_PAGE])
    _assert_same(second, expected[_PAGE : 2 * _PAGE])
    assert pagination["total"] == len(vectors)


def test_path_filter_and_threshold_still_apply(indexed) -> None:
    provider, paths, vectors, query, _ = indexed

    in_beta, _ = _search(provider, query, page_size=_PAGE, path_filter="beta")
    _assert_same(
        in_beta, _expected(vectors, query, lambda c: paths[c] == "beta")[:_PAGE]
    )

    floor = _expected(vectors, query)[4][1]
    above, _ = _search(provider, query, page_size=50, threshold=floor)
    _assert_same(above, [e for e in _expected(vectors, query) if e[1] >= floor - 1e-6])


def test_malformed_vector_outside_the_scope_does_not_empty_the_search(indexed) -> None:
    """A NaN vector is stored with NULL elements, on which cosine raises. One left
    behind by a deleted chunk, mid-way through the id range, must not fail an
    unscoped search or one scoped to another directory."""
    provider, paths, vectors, query, _ = indexed
    alpha = sorted(cid for cid, sub in paths.items() if sub == "alpha")
    gone = alpha[len(alpha) // 2]
    provider.delete_chunk(gone)
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": gone,
                "provider": "test",
                "model": "mini",
                "embedding": [float("nan")] * _DIMS,
                "dims": _DIMS,
            }
        ]
    )
    del vectors[gone]

    everything, pagination = _search(provider, query, page_size=50)
    in_beta, _ = _search(provider, query, page_size=50, path_filter="beta")

    _assert_same(everything, _expected(vectors, query))
    assert pagination["total"] == len(vectors)
    _assert_same(in_beta, _expected(vectors, query, lambda c: paths[c] == "beta"))


def test_total_counts_only_the_results_pages_return(indexed) -> None:
    """A live chunk whose vector is malformed is never returned, so the total must
    not count it either: every page offered holds results, the pages together
    hold exactly `total` results, and a page past the end still reports it."""
    provider, paths, vectors, query, _ = indexed
    beta = sorted(cid for cid, sub in paths.items() if sub == "beta")
    bad = beta[len(beta) // 2]
    provider.delete_embeddings_by_chunk_id(bad)
    provider.insert_embeddings_batch(
        [
            {
                "chunk_id": bad,
                "provider": "test",
                "model": "mini",
                "embedding": [float("nan")] * _DIMS,
                "dims": _DIMS,
            }
        ]
    )
    del vectors[bad]

    for path_filter, keep in (
        (None, lambda c: True),
        ("beta", lambda c: paths[c] == "beta"),
    ):
        expected = _expected(vectors, query, keep)
        paged: list[tuple[int, float]] = []
        offset = 0
        while offset is not None:
            page, pagination = _search(
                provider, query, page_size=_PAGE, offset=offset, path_filter=path_filter
            )
            assert page, f"empty page offered at offset {offset} ({path_filter=})"
            paged += page
            offset = pagination["next_offset"]

        _assert_same(paged, expected)
        assert pagination["total"] == len(expected)

        past_end, pagination = _search(
            provider,
            query,
            page_size=_PAGE,
            offset=len(expected),
            path_filter=path_filter,
        )
        assert past_end == []
        assert pagination["total"] == len(expected)
