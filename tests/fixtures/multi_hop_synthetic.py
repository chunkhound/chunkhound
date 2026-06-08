from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunkhound.interfaces.embedding_provider import RerankResult


@dataclass(frozen=True)
class SyntheticChunk:
    chunk_id: int
    content: str
    file_path: str
    similarity: float
    start_line: int = 1
    end_line: int = 1
    language: str = "python"


@dataclass(frozen=True)
class SyntheticMultiHopScenario:
    db: SyntheticGraphDatabase
    provider: DeterministicEmbeddingProvider
    qrels: dict[str, set[int]]


class DeterministicEmbeddingProvider:
    """Deterministic embedding/rerank provider with optional rerank failure."""

    def __init__(
        self,
        *,
        query_vectors: dict[str, list[float]],
        rerank_scores: dict[str, dict[str, float]],
        supports_reranking: bool = True,
        fail_rerank: bool = False,
        partial_rerank: bool = False,
        rerank_scores_after_expansion: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._query_vectors = query_vectors
        self._rerank_scores = rerank_scores
        self._rerank_scores_after_expansion = rerank_scores_after_expansion
        self._supports_reranking = supports_reranking
        self._fail_rerank = fail_rerank
        self._partial_rerank = partial_rerank

    @property
    def name(self) -> str:
        return "deterministic"

    @property
    def model(self) -> str:
        return "synthetic"

    @property
    def dims(self) -> int:
        return 1

    @property
    def distance(self) -> str:
        return "cosine"

    @property
    def batch_size(self) -> int:
        return 100

    @property
    def max_tokens(self) -> int:
        return 4096

    @property
    def config(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "dims": self.dims}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._query_vectors.get(text, [0.0]) for text in texts]

    async def embed_single(self, text: str) -> list[float]:
        return self._query_vectors.get(text, [0.0])

    async def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        return await self.embed(texts)

    async def rerank(
        self, query: str, documents: list[str], top_k: int | None = None
    ) -> list[RerankResult]:
        if self._fail_rerank:
            raise RuntimeError("synthetic rerank failure")

        # Expansion rerank sees more documents than the initial rerank in these
        # synthetic scenarios, so use that boundary to model post-expansion scores
        # without leaking state across repeated searches.
        if self._rerank_scores_after_expansion is not None and len(documents) > 5:
            score_map = self._rerank_scores_after_expansion[query]
        else:
            score_map = self._rerank_scores[query]

        results = [
            RerankResult(index=index, score=score_map.get(document, 0.0))
            for index, document in enumerate(documents)
        ]

        # Simulate a provider returning fewer rerank results than input documents
        if self._partial_rerank and len(results) > 2:
            results = results[:-1]

        results.sort(key=lambda item: item.score, reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

    def supports_reranking(self) -> bool:
        return self._supports_reranking


class SyntheticGraphDatabase:
    """Small deterministic retrieval graph for multi-hop contract tests."""

    def __init__(
        self,
        *,
        chunks: list[SyntheticChunk],
        query_results: dict[tuple[float, ...], list[int]],
        neighbors: dict[int, list[int]],
    ) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._query_results = query_results
        self._neighbors = neighbors

    def search_semantic(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        chunk_ids = self._query_results[tuple(query_embedding)]
        results = [
            self._chunk_to_result(self._chunks[chunk_id]) for chunk_id in chunk_ids
        ]
        results = self._apply_path_filter(results, path_filter)
        if threshold is not None:
            results = [r for r in results if r["similarity"] >= threshold]
        return self._paginate(results, page_size=page_size, offset=offset)

    def find_similar_chunks(
        self,
        chunk_id: int,
        provider: str,
        model: str,
        limit: int = 10,
        threshold: float | None = None,
        path_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        results = [
            self._chunk_to_result(self._chunks[neighbor_id])
            for neighbor_id in self._neighbors.get(chunk_id, [])
        ]
        results = self._apply_path_filter(results, path_filter)
        if threshold is not None:
            results = [r for r in results if r["similarity"] >= threshold]
        return results[:limit]

    def _chunk_to_result(self, chunk: SyntheticChunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "file_path": chunk.file_path,
            "similarity": chunk.similarity,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
        }

    def _apply_path_filter(
        self, results: list[dict[str, Any]], path_filter: str | None
    ) -> list[dict[str, Any]]:
        if path_filter is None:
            return results
        return [r for r in results if f"/{path_filter}" in f"/{r['file_path']}"]

    def _paginate(
        self, results: list[dict[str, Any]], *, page_size: int, offset: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        total = len(results)
        page = results[offset : offset + page_size]
        return page, {
            "offset": offset,
            "page_size": page_size,
            "has_more": offset + page_size < total,
            "next_offset": (
                offset + page_size if offset + page_size < total else None
            ),
            "total": total,
        }


def _build_scenario(
    *,
    query: str,
    query_vectors: dict[str, list[float]],
    rerank_scores: dict[str, dict[str, float]],
    chunks: list[SyntheticChunk],
    query_results: dict[tuple[float, ...], list[int]],
    neighbors: dict[int, list[int]],
    qrels: dict[str, set[int]],
    supports_reranking: bool = True,
    fail_rerank: bool = False,
    partial_rerank: bool = False,
    rerank_scores_after_expansion: dict[str, dict[str, float]] | None = None,
) -> SyntheticMultiHopScenario:
    return SyntheticMultiHopScenario(
        db=SyntheticGraphDatabase(
            chunks=chunks,
            query_results=query_results,
            neighbors=neighbors,
        ),
        provider=DeterministicEmbeddingProvider(
            query_vectors=query_vectors,
            rerank_scores=rerank_scores,
            supports_reranking=supports_reranking,
            fail_rerank=fail_rerank,
            partial_rerank=partial_rerank,
            rerank_scores_after_expansion=rerank_scores_after_expansion,
        ),
        qrels=qrels,
    )


def build_multi_hop_scenario(
    *,
    fail_rerank: bool = False,
    supports_reranking: bool = True,
    partial_rerank: bool = False,
) -> SyntheticMultiHopScenario:
    """Build a deterministic 8-chunk graph for multi-hop contract tests.

    Topology:
      chunks 1, 5, 6, 7, 8 are reachable from "security validation mechanisms"
      chunk 1 -> neighbor 2 (bridge) -> neighbors 3, 4 (targets)
      chunk 5 -> neighbor 2 (same bridge -- converging path test)
      chunks 6, 7, 8 have no neighbors

    Queries:
      "security validation mechanisms"  -> qrels {3, 4}  (needs 2 hops)
      "database connection auth"        -> qrels {3, 4}  (direct)
      "credential provider secret"      -> qrels {2, 3}  (direct)
    """
    query = "security validation mechanisms"
    direct_query = "database connection auth"
    bridge_query = "credential provider secret"

    chunks = [
        SyntheticChunk(
            1,
            "security validation policy checks credentials before access",
            "repo_a/security.py",
            0.90,
        ),
        SyntheticChunk(
            2,
            (
                "credential validator resolves provider secret "
                "for downstream authentication"
            ),
            "repo_a/bridge.py",
            0.70,
        ),
        SyntheticChunk(
            3,
            "load api_key for authentication provider connection",
            "repo_a/auth.py",
            0.50,
        ),
        SyntheticChunk(
            4,
            "database provider connection uses authentication token and api_key",
            "repo_a/database.py",
            0.48,
        ),
        SyntheticChunk(
            5,
            "validation mechanism checks schema shape for incoming payloads",
            "repo_a/schema.py",
            0.88,
        ),
        SyntheticChunk(
            6,
            "mechanism for audit logging and validation history retention",
            "repo_a/audit.py",
            0.86,
        ),
        SyntheticChunk(
            7,
            "security review workflow for change approval",
            "repo_a/review.py",
            0.84,
        ),
        SyntheticChunk(
            8,
            "mechanism registry for policy controls",
            "repo_b/controls.py",
            0.82,
        ),
    ]

    query_results: dict[tuple[float, ...], list[int]] = {
        (1.0,): [1, 5, 6, 7, 8],
        (2.0,): [4, 3, 2, 1, 5],
        (3.0,): [2, 1, 5, 6, 7],
    }
    neighbors = {
        1: [2],
        5: [2],
        6: [],
        7: [],
        8: [],
        2: [3, 4],
        3: [],
        4: [],
    }
    rerank_scores = {
        query: {
            chunks[0].content: 0.55,
            chunks[1].content: 0.88,
            chunks[2].content: 0.97,
            chunks[3].content: 0.95,
            chunks[4].content: 0.52,
            chunks[5].content: 0.51,
            chunks[6].content: 0.50,
            chunks[7].content: 0.49,
        },
        direct_query: {
            chunks[0].content: 0.45,
            chunks[1].content: 0.70,
            chunks[2].content: 0.94,
            chunks[3].content: 0.96,
            chunks[4].content: 0.20,
            chunks[5].content: 0.15,
            chunks[6].content: 0.14,
            chunks[7].content: 0.13,
        },
        bridge_query: {
            chunks[0].content: 0.60,
            chunks[1].content: 0.98,
            chunks[2].content: 0.90,
            chunks[3].content: 0.82,
            chunks[4].content: 0.30,
            chunks[5].content: 0.25,
            chunks[6].content: 0.24,
            chunks[7].content: 0.23,
        },
    }
    return _build_scenario(
        query=query,
        query_vectors={query: [1.0], direct_query: [2.0], bridge_query: [3.0]},
        rerank_scores=rerank_scores,
        chunks=chunks,
        query_results=query_results,
        neighbors=neighbors,
        qrels={
            query: {3, 4},
            direct_query: {3, 4},
            bridge_query: {2, 3},
        },
        supports_reranking=supports_reranking,
        fail_rerank=fail_rerank,
        partial_rerank=partial_rerank,
    )


def build_insufficient_candidates_scenario() -> SyntheticMultiHopScenario:
    """Build a graph where fewer than five positive scores block expansion."""
    query = "insufficient candidates"
    chunks = [
        SyntheticChunk(1, "candidate one", "repo_a/one.py", 0.95),
        SyntheticChunk(2, "candidate two", "repo_a/two.py", 0.94),
        SyntheticChunk(3, "candidate three", "repo_a/three.py", 0.93),
        SyntheticChunk(4, "candidate four", "repo_a/four.py", 0.92),
        SyntheticChunk(5, "candidate five", "repo_a/five.py", 0.91),
        SyntheticChunk(6, "unreached neighbor", "repo_a/six.py", 0.10),
    ]
    return _build_scenario(
        query=query,
        query_vectors={query: [4.0]},
        rerank_scores={
            query: {
                chunks[0].content: 0.90,
                chunks[1].content: 0.80,
                chunks[2].content: 0.70,
                chunks[3].content: 0.60,
                chunks[4].content: 0.0,
                chunks[5].content: 0.99,
            }
        },
        chunks=chunks,
        query_results={(4.0,): [1, 2, 3, 4, 5]},
        neighbors={1: [6], 2: [], 3: [], 4: [], 5: [], 6: []},
        qrels={query: {6}},
    )


def build_graph_exhaustion_scenario() -> SyntheticMultiHopScenario:
    """Build a graph where expansion stops once no new top candidates remain.

    Topology: chunks 1→5 initial, 1 has neighbor 6, 6 has neighbor 7, 7 has neighbor 8.
    Chunk 6 scores 0.60 after reranking, which falls below the top-5 expansion floor
    (lowest among tracked chunks is 0.70). Since ch6 never enters the top-5 expansion
    candidates, its neighbors (7 and 8) are never discovered. Round 1 adds ch6
    (total=6); round 2 finds no new top-5 candidate with unseen neighbors and
    terminates with "no new candidates found".

    Note: This exercises natural "no new candidates" exhaustion, not score-drop
    termination. Score-drop compares absolute per-document scores between rounds,
    which is invariant for a deterministic reranker.
    """
    query = "graph exhaustion"
    chunks = [
        SyntheticChunk(1, "core result one", "repo_a/one.py", 0.95),
        SyntheticChunk(2, "core result two", "repo_a/two.py", 0.94),
        SyntheticChunk(3, "core result three", "repo_a/three.py", 0.93),
        SyntheticChunk(4, "core result four", "repo_a/four.py", 0.92),
        SyntheticChunk(5, "core result five", "repo_a/five.py", 0.91),
        SyntheticChunk(6, "chain neighbor one", "repo_a/six.py", 0.50),
        SyntheticChunk(7, "chain neighbor two", "repo_a/seven.py", 0.40),
        SyntheticChunk(8, "chain neighbor three", "repo_a/eight.py", 0.30),
    ]
    return _build_scenario(
        query=query,
        query_vectors={query: [5.0]},
        rerank_scores={
            query: {
                chunks[0].content: 0.70,
                chunks[1].content: 0.90,
                chunks[2].content: 0.85,
                chunks[3].content: 0.80,
                chunks[4].content: 0.75,
                chunks[5].content: 0.60,
                chunks[6].content: 0.50,
                chunks[7].content: 0.40,
            }
        },
        chunks=chunks,
        query_results={(5.0,): [1, 2, 3, 4, 5]},
        neighbors={1: [6], 2: [], 3: [], 4: [], 5: [], 6: [7], 7: [8], 8: []},
        qrels={query: {6, 7, 8}},
    )


def build_score_drop_termination_scenario(
    *,
    round_two_chunk_one_score: float = 0.70,
    query: str = "score drop termination",
) -> SyntheticMultiHopScenario:
    """Build a graph where expansion can terminate via tracked score-drop.

    Topology: chunks 1->5 initial, 1 has neighbor 6, 6 has neighbor 7, 7 has neighbor 8.
    Initial rerank gives top-5 high scores [0.95, 0.90, 0.85, 0.80, 0.75].
    After expansion, chunk 1 reranks to ``round_two_chunk_one_score`` so tests can
    exercise threshold edges around SCORE_DROP_THRESHOLD.
    """
    chunks = [
        SyntheticChunk(1, "top result one", "repo_a/one.py", 0.95),
        SyntheticChunk(2, "top result two", "repo_a/two.py", 0.94),
        SyntheticChunk(3, "top result three", "repo_a/three.py", 0.93),
        SyntheticChunk(4, "top result four", "repo_a/four.py", 0.92),
        SyntheticChunk(5, "top result five", "repo_a/five.py", 0.91),
        SyntheticChunk(6, "chain neighbor", "repo_a/six.py", 0.60),
        SyntheticChunk(7, "chain neighbor two", "repo_a/seven.py", 0.40),
        SyntheticChunk(8, "chain neighbor three", "repo_a/eight.py", 0.30),
    ]
    rerank_scores_after_expansion = {
        query: {
            chunks[0].content: round_two_chunk_one_score,
            chunks[1].content: 0.90,  # ch2 -- stable
            chunks[2].content: 0.85,  # ch3 -- stable
            chunks[3].content: 0.80,  # ch4 -- stable
            chunks[4].content: 0.75,  # ch5 -- stable
            chunks[5].content: 0.60,  # ch6 (unchanged)
            chunks[6].content: 0.50,  # ch7
            chunks[7].content: 0.40,  # ch8
        }
    }
    return _build_scenario(
        query=query,
        query_vectors={query: [7.0]},
        rerank_scores={
            query: {
                chunks[0].content: 0.95,
                chunks[1].content: 0.90,
                chunks[2].content: 0.85,
                chunks[3].content: 0.80,
                chunks[4].content: 0.75,
                chunks[5].content: 0.60,
                chunks[6].content: 0.50,
                chunks[7].content: 0.40,
            }
        },
        chunks=chunks,
        query_results={(7.0,): [1, 2, 3, 4, 5]},
        neighbors={1: [6], 2: [], 3: [], 4: [], 5: [], 6: [7], 7: [8], 8: []},
        qrels={query: {6, 7, 8}},
        rerank_scores_after_expansion=rerank_scores_after_expansion,
    )


def build_min_score_termination_scenario() -> SyntheticMultiHopScenario:
    """Build a graph where expansion lowers the top-5 floor below 0.3.

    Topology: chunks 1→5 initial, 1 has neighbor 6, 6 has neighbor 7, 7 has
    neighbor 8. Without the min-score check, the loop would continue to at
    least 8 results. With it, the loop terminates at 6 once the top-five floor
    drops below MIN_RELEVANCE_FLOOR.
    """
    query = "minimum score termination"
    chunks = [
        SyntheticChunk(1, "marginal result one", "repo_a/one.py", 0.95),
        SyntheticChunk(2, "marginal result two", "repo_a/two.py", 0.94),
        SyntheticChunk(3, "marginal result three", "repo_a/three.py", 0.93),
        SyntheticChunk(4, "marginal result four", "repo_a/four.py", 0.92),
        SyntheticChunk(5, "marginal result five", "repo_a/five.py", 0.91),
        SyntheticChunk(6, "chain neighbor one", "repo_a/six.py", 0.50),
        SyntheticChunk(7, "chain neighbor two", "repo_a/seven.py", 0.40),
        SyntheticChunk(8, "chain neighbor three", "repo_a/eight.py", 0.30),
    ]
    # Rerank scores: ch4=0.29 and ch5=0.28 are already below MIN_RELEVANCE_FLOOR (0.3)
    # After expansion, ch6→ch7→ch8 adds more but min-score stops at 6
    return _build_scenario(
        query=query,
        query_vectors={query: [6.0]},
        rerank_scores={
            query: {
                chunks[0].content: 0.34,  # ch1
                chunks[1].content: 0.33,  # ch2
                chunks[2].content: 0.32,  # ch3
                chunks[3].content: 0.29,  # ch4 — below 0.3
                chunks[4].content: 0.28,  # ch5 — below 0.3
                chunks[5].content: 0.50,  # ch6
                chunks[6].content: 0.40,  # ch7
                chunks[7].content: 0.30,  # ch8
            }
        },
        chunks=chunks,
        query_results={(6.0,): [1, 2, 3, 4, 5]},
        neighbors={1: [6], 2: [], 3: [], 4: [], 5: [], 6: [7], 7: [8], 8: []},
        qrels={query: {6, 7, 8}},
    )
