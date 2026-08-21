"""Multi-hop semantic search with iterative expansion and reranking.

Iterative algorithm: initial search → rerank → expand via neighbors → rerank
all → repeat until a termination condition fires.

Termination conditions:
1. Time limit (default 5s, configurable)
2. Result limit (default 500, configurable)
3. Candidate quality: fewer than MIN_EXPANSION_CANDIDATES positive-score results
4. Score degradation: tracked top-5 score drops by >= SCORE_DROP_THRESHOLD
5. Minimum relevance: top-5 minimum score < MIN_RELEVANCE_FLOOR
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from chunkhound.core.exceptions import MaterializationLimitError
from chunkhound.interfaces.database_provider import DatabaseProvider
from chunkhound.interfaces.embedding_provider import EmbeddingProvider
from chunkhound.services.search.semantic_window import (
    normalize_semantic_window_cap,
    semantic_next_offset,
    semantic_total,
)

INITIAL_LIMIT_CAP_NORMAL = 100
INITIAL_LIMIT_CAP_EXHAUSTIVE = 500
NEIGHBORS_PER_CANDIDATE_NORMAL = 20
NEIGHBORS_PER_CANDIDATE_EXHAUSTIVE = 30
MIN_EXPANSION_CANDIDATES = 5
SCORE_DROP_THRESHOLD = 0.15
MIN_RELEVANCE_FLOOR = 0.3

SingleHopSearch = Callable[..., Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]]]


@dataclass
class _ExpansionRequest:
    query: str
    provider: str
    model: str
    path_filter: str | None
    result_limit: int | None


@dataclass
class _ExpansionState:
    results: list[dict[str, Any]]
    seen_ids: set[str]
    tracked_scores: dict[str, float]
    round_number: int = 0


@dataclass
class _SearchRequest:
    query: str
    page_size: int
    offset: int
    threshold: float | None
    provider: str
    model: str
    path_filter: str | None
    time_limit: float | None
    result_limit: int | None

    def expansion_request(self, result_limit: int | None) -> _ExpansionRequest:
        """Create the subset of request data needed during expansion."""
        return _ExpansionRequest(
            self.query, self.provider, self.model, self.path_filter, result_limit
        )


class MultiHopStrategy:
    """Dynamic multi-hop semantic search with relevance-based termination."""

    def __init__(
        self,
        database_provider: DatabaseProvider,
        embedding_provider: EmbeddingProvider,
        single_hop_search_fn: SingleHopSearch,
        config: Any = None,
    ):
        self._db = database_provider
        self._embedding_provider = embedding_provider
        self._single_hop_search = single_hop_search_fn
        self._config = config

    def _exhaustive_mode(self) -> bool:
        """True when the config enables exhaustive candidate collection."""
        return bool(self._config and self._config.exhaustive_mode)

    def _resolve_effective_limits(
        self, time_limit: float | None, result_limit: int | None
    ) -> tuple[float, int | None]:
        """Apply config defaults for time_limit and result_limit."""
        if self._config:
            effective_time_limit = (
                time_limit
                if time_limit is not None
                else self._config.get_effective_time_limit()
            )
            effective_result_limit = (
                result_limit
                if result_limit is not None
                else self._config.get_effective_result_limit()
            )
        else:
            effective_time_limit = time_limit if time_limit is not None else 5.0
            effective_result_limit = result_limit if result_limit is not None else 500
        return effective_time_limit, effective_result_limit

    def _materializable_cap(self, result_limit: int | None) -> int | None:
        """Return the strictest provider or configured result limit."""
        limits = [
            limit
            for limit in (
                normalize_semantic_window_cap(self._db.semantic_result_window_cap),
                result_limit,
            )
            if limit is not None
        ]
        return min(limits) if limits else None

    @staticmethod
    def _validate_materialization_window(
        requested_end: int, materializable_cap: int | None
    ) -> None:
        """Reject pages that cannot be fully materialized."""
        if materializable_cap is not None and requested_end > materializable_cap:
            raise MaterializationLimitError(
                f"Requested result window ends at {requested_end}, beyond the "
                f"multi-hop materialization limit of {materializable_cap}"
            )

    def _compute_initial_fetch_limit(
        self, page_size: int, offset: int, result_limit: int | None
    ) -> tuple[int, int | None]:
        """Compute the initial fetch size respecting materialization limits."""
        window_cap = normalize_semantic_window_cap(self._db.semantic_result_window_cap)
        materializable_cap = self._materializable_cap(result_limit)
        requested_end = offset + page_size
        self._validate_materialization_window(requested_end, materializable_cap)
        cap = (
            INITIAL_LIMIT_CAP_EXHAUSTIVE
            if self._exhaustive_mode()
            else INITIAL_LIMIT_CAP_NORMAL
        )
        requested_fetch = requested_end + 1
        if materializable_cap is not None:
            requested_fetch = min(requested_fetch, materializable_cap)
        initial_limit = max(min(page_size * 3, cap), requested_fetch)
        if materializable_cap is not None:
            initial_limit = min(initial_limit, materializable_cap)
        return initial_limit, window_cap

    @staticmethod
    def _seed_similarity_scores(results: list[dict[str, Any]]) -> None:
        """Provide deterministic fallback scores before reranking."""
        for result in results:
            if "score" not in result:
                result["score"] = result.get("similarity", 0.0)

    @staticmethod
    def _apply_rerank_scores(
        results: list[dict[str, Any]], rerank_results: list[Any]
    ) -> None:
        """Apply reranker scores to their source-document positions."""
        for rerank_result in rerank_results:
            if 0 <= rerank_result.index < len(results):
                results[rerank_result.index]["score"] = rerank_result.score

    async def _rerank_documents(
        self, query: str, results: list[dict[str, Any]]
    ) -> list[Any]:
        """Rerank every result document against the query."""
        assert hasattr(self._embedding_provider, "rerank")
        documents = [result["content"] for result in results]
        return await self._embedding_provider.rerank(
            query=query, documents=documents, top_k=len(documents)
        )

    async def _rerank_results(
        self, results: list[dict[str, Any]], query: str, label: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Rerank results, retaining deterministic fallback scores on failure."""
        self._seed_similarity_scores(results)
        try:
            reranked = await self._rerank_documents(query, results)
            self._apply_rerank_scores(results, reranked)
            logger.debug(
                f"{label} reranking: {len(reranked)}/{len(results)} results reranked"
            )
            reranked_ok = True
        except Exception as error:
            logger.warning(f"{label} reranking failed: {error}")
            reranked_ok = False
        return sorted(
            results, key=lambda item: item.get("score", 0.0), reverse=True
        ), reranked_ok

    def _budget_exhausted(
        self, start_time: float, time_limit: float, result_limit: int | None, count: int
    ) -> bool:
        """True when the time or result limit terminates expansion."""
        if time.perf_counter() - start_time >= time_limit:
            logger.debug(
                "Dynamic expansion terminated: "
                f"{time_limit:.1f} second time limit reached"
            )
            return True
        if result_limit is not None and count >= result_limit:
            logger.debug(
                f"Dynamic expansion terminated: {result_limit} result limit reached"
            )
            return True
        return False

    def _neighbor_limit(self) -> int:
        """Return the neighbor fan-out for the configured search mode."""
        return (
            NEIGHBORS_PER_CANDIDATE_EXHAUSTIVE
            if self._exhaustive_mode()
            else NEIGHBORS_PER_CANDIDATE_NORMAL
        )

    def _neighbors_for_candidate(
        self,
        candidate: dict[str, Any],
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Fetch one candidate's neighbors without aborting other expansion paths."""
        try:
            return self._db.find_similar_chunks(
                chunk_id=candidate["chunk_id"],
                provider=provider,
                model=model,
                limit=self._neighbor_limit(),
                threshold=None,
                path_filter=path_filter,
            )
        except Exception as error:
            logger.warning(f"Failed to expand chunk {candidate['chunk_id']}: {error}")
            return []

    def _find_new_candidates(
        self,
        top_candidates: list[dict[str, Any]],
        seen_ids: set[str],
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> list[dict[str, Any]]:
        """Collect unseen neighbors of the top candidates."""
        new_candidates: list[dict[str, Any]] = []
        new_ids: set[str] = set()
        for candidate in top_candidates:
            try:
                neighbors = self._neighbors_for_candidate(
                    candidate, provider, model, path_filter
                )
                for neighbor in neighbors:
                    chunk_id = neighbor["chunk_id"]
                    if chunk_id not in seen_ids and chunk_id not in new_ids:
                        new_candidates.append(neighbor)
                        new_ids.add(chunk_id)
            except Exception as error:
                logger.warning(
                    f"Failed to process neighbors for chunk "
                    f"{candidate['chunk_id']}: {error}"
                )
        return new_candidates

    def _truncate_to_budget(
        self, candidates: list[dict[str, Any]], result_limit: int | None, count: int
    ) -> list[dict[str, Any]]:
        """Cap expansion candidates to the remaining result budget."""
        if result_limit is None:
            return candidates
        remaining = result_limit - count
        if remaining <= 0:
            logger.debug(
                f"Dynamic expansion terminated: {result_limit} result limit reached"
            )
            return []
        if len(candidates) > remaining:
            logger.debug(
                f"result_limit={result_limit} truncates expansion "
                f"from {len(candidates)} to {remaining}"
            )
            return candidates[:remaining]
        return candidates

    @staticmethod
    def _top_positive_candidates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the highest-scoring positive candidates for expansion."""
        return [item for item in results if item.get("score", 0.0) > 0.0][
            :MIN_EXPANSION_CANDIDATES
        ]

    @staticmethod
    def _max_tracked_score_drop(
        results: list[dict[str, Any]], tracked_scores: dict[str, float]
    ) -> float | None:
        """Return the greatest degradation among previously tracked chunks."""
        scores = {item["chunk_id"]: item.get("score", 0.0) for item in results}
        drops = [
            previous - scores.get(chunk_id, 0.0)
            for chunk_id, previous in tracked_scores.items()
            if scores.get(chunk_id, 0.0) < previous
        ]
        return max(drops) if drops else None

    @staticmethod
    def _update_tracked_scores(
        tracked_scores: dict[str, float], results: list[dict[str, Any]]
    ) -> None:
        """Track scores of the current top candidates."""
        tracked_scores.clear()
        for result in results[:MIN_EXPANSION_CANDIDATES]:
            tracked_scores[result["chunk_id"]] = result.get("score", 0.0)

    def _score_termination_hit(
        self, results: list[dict[str, Any]], tracked_scores: dict[str, float]
    ) -> bool:
        """True when score degradation or the relevance floor stops expansion."""
        scores = [item.get("score", 0.0) for item in results[:MIN_EXPANSION_CANDIDATES]]
        max_drop = self._max_tracked_score_drop(results, tracked_scores)
        self._update_tracked_scores(tracked_scores, results)
        if max_drop is not None and max_drop >= SCORE_DROP_THRESHOLD:
            logger.debug(
                "Dynamic expansion terminated: tracked chunk score drop "
                f"{max_drop:.3f} >= {SCORE_DROP_THRESHOLD}"
            )
            return True
        if min(scores) < MIN_RELEVANCE_FLOOR:
            logger.debug(
                "Dynamic expansion terminated: minimum score "
                f"{min(scores):.3f} < {MIN_RELEVANCE_FLOOR}"
            )
            return True
        return False

    @staticmethod
    def _append_candidates(
        results: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        seen_ids: set[str],
    ) -> None:
        """Record and append accepted expansion candidates."""
        seen_ids.update(candidate["chunk_id"] for candidate in candidates)
        results.extend(candidates)

    def _add_expansion_candidates(
        self,
        results: list[dict[str, Any]],
        top_candidates: list[dict[str, Any]],
        seen_ids: set[str],
        request: _ExpansionRequest,
    ) -> bool:
        candidates = self._find_new_candidates(
            top_candidates,
            seen_ids,
            request.provider,
            request.model,
            request.path_filter,
        )
        if not candidates:
            logger.debug("Dynamic expansion terminated: no new candidates found")
            return False
        candidates = self._truncate_to_budget(
            candidates, request.result_limit, len(results)
        )
        if candidates:
            self._append_candidates(results, candidates, seen_ids)
        return bool(candidates)

    @staticmethod
    def _has_expansion_candidates(candidates: list[dict[str, Any]]) -> bool:
        """Require enough positive-score chunks to make expansion worthwhile."""
        if len(candidates) >= MIN_EXPANSION_CANDIDATES:
            return True
        logger.debug(
            f"Dynamic expansion terminated: only {len(candidates)} "
            f"high-scoring candidates, need at least {MIN_EXPANSION_CANDIDATES}"
        )
        return False

    @staticmethod
    def _expansion_state(results: list[dict[str, Any]]) -> _ExpansionState:
        """Initialize the deduplication and score state for expansion."""
        return _ExpansionState(
            results=list(results),
            seen_ids={result["chunk_id"] for result in results},
            tracked_scores={
                result["chunk_id"]: result.get("score", 0.0)
                for result in results[:MIN_EXPANSION_CANDIDATES]
            },
        )

    async def _run_expansion_round(
        self, state: _ExpansionState, request: _ExpansionRequest
    ) -> bool:
        """Add, rerank, and validate one expansion round."""
        candidates = self._top_positive_candidates(state.results)
        if not self._has_expansion_candidates(candidates):
            return False
        if not self._add_expansion_candidates(
            state.results, candidates, state.seen_ids, request
        ):
            return False
        state.results, reranked_ok = await self._rerank_results(
            state.results, request.query, f"Expansion round {state.round_number}"
        )
        return reranked_ok and not self._score_termination_hit(
            state.results, state.tracked_scores
        )

    async def _run_expansion_loop(
        self,
        initial_results: list[dict[str, Any]],
        request: _ExpansionRequest,
        time_limit: float,
        start_time: float,
    ) -> tuple[list[dict[str, Any]], int]:
        """Expand candidates until a budget, quality, or score condition stops it."""
        state = self._expansion_state(initial_results)
        while not self._budget_exhausted(
            start_time, time_limit, request.result_limit, len(state.results)
        ):
            if not await self._run_expansion_round(state, request):
                break
            state.round_number += 1
            logger.debug(
                f"Expansion round {state.round_number}: "
                f"{len(state.results)} total results"
            )
        return state.results, state.round_number

    @staticmethod
    def _filter_by_threshold(
        results: list[dict[str, Any]], threshold: float | None
    ) -> list[dict[str, Any]]:
        """Apply the final threshold to rerank scores."""
        return (
            results
            if threshold is None
            else [item for item in results if item.get("score", 0.0) >= threshold]
        )

    @staticmethod
    def _candidate_budget_exhausted(initial_pagination: dict[str, Any]) -> bool:
        """Propagate the initial semantic search's candidate-budget status."""
        return bool(initial_pagination.get("candidate_budget_exhausted", False))

    @staticmethod
    def _pagination(
        results: list[dict[str, Any]],
        offset: int,
        page_size: int,
        window_cap: int | None,
        initial_pagination: dict[str, Any],
    ) -> dict[str, Any]:
        next_offset = semantic_next_offset(
            offset, page_size, offset + page_size < len(results), window_cap
        )
        return {
            "offset": offset,
            "page_size": page_size,
            "has_more": next_offset is not None,
            "next_offset": next_offset,
            "total": semantic_total(len(results), window_cap),
            "candidate_budget_exhausted": MultiHopStrategy._candidate_budget_exhausted(
                initial_pagination
            ),
        }

    @staticmethod
    def _log_search_completion(
        page: list[dict[str, Any]],
        results: list[dict[str, Any]],
        round_number: int,
        start_time: float,
    ) -> None:
        """Log the completed search's result and timing summary."""
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Dynamic expansion search completed in {elapsed:.2f}s: "
            f"{len(page)} results returned ({len(results)} total candidates, "
            f"{round_number} expansion rounds)"
        )

    @staticmethod
    def _log_threshold(threshold: float, results: list[dict[str, Any]]) -> None:
        """Log the number of results surviving the final score threshold."""
        logger.debug(
            f"Applied rerank score threshold {threshold}, {len(results)} results remain"
        )

    def _apply_threshold_and_paginate(
        self,
        results: list[dict[str, Any]],
        threshold: float | None,
        offset: int,
        page_size: int,
        window_cap: int | None,
        initial_pagination: dict[str, Any],
        expansion_round: int,
        start_time: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Filter reranked results, paginate, and record completion metrics."""
        results = self._filter_by_threshold(results, threshold)
        if threshold is not None:
            self._log_threshold(threshold, results)
        page = results[offset : offset + page_size]
        self._log_search_completion(page, results, expansion_round, start_time)
        return page, self._pagination(
            results, offset, page_size, window_cap, initial_pagination
        )

    async def _initial_search(
        self,
        query: str,
        initial_limit: int,
        provider: str,
        model: str,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        results, pagination = await self._single_hop_search(
            query=query,
            page_size=initial_limit,
            offset=0,
            threshold=0.0,
            provider=provider,
            model=model,
            path_filter=path_filter,
        )
        reranked, reranked_ok = await self._rerank_results(results, query, "Initial")
        if not reranked_ok:
            logger.warning("Initial reranking failed, expansion may be limited")
            return results, pagination
        return reranked, pagination

    async def _initial_request_results(
        self, request: _SearchRequest, result_limit: int | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any], int | None]:
        """Compute the initial fetch and retrieve its reranked candidates."""
        initial_limit, window_cap = self._compute_initial_fetch_limit(
            request.page_size, request.offset, result_limit
        )
        results, pagination = await self._initial_search(
            request.query,
            initial_limit,
            request.provider,
            request.model,
            request.path_filter,
        )
        return results, pagination, window_cap

    def _finish_search(
        self,
        results: list[dict[str, Any]],
        request: _SearchRequest,
        window_cap: int | None,
        pagination: dict[str, Any],
        round_number: int,
        start_time: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply final thresholding and pagination to completed expansion."""
        return self._apply_threshold_and_paginate(
            results,
            request.threshold,
            request.offset,
            request.page_size,
            window_cap,
            pagination,
            round_number,
            start_time,
        )

    async def _search_request(
        self, request: _SearchRequest, start_time: float
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute a fully specified multi-hop search request."""
        time_limit, result_limit = self._resolve_effective_limits(
            request.time_limit, request.result_limit
        )
        initial_results, pagination, window_cap = await self._initial_request_results(
            request, result_limit
        )
        results, round_number = await self._run_expansion_loop(
            initial_results,
            request.expansion_request(result_limit),
            time_limit,
            start_time,
        )
        return self._finish_search(
            results, request, window_cap, pagination, round_number, start_time
        )

    async def search(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        provider: str,
        model: str,
        path_filter: str | None,
        time_limit: float | None = None,
        result_limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform dynamic multi-hop semantic search with reranking."""
        request = _SearchRequest(
            query=query,
            page_size=page_size,
            offset=offset,
            threshold=threshold,
            provider=provider,
            model=model,
            path_filter=path_filter,
            time_limit=time_limit,
            result_limit=result_limit,
        )
        return await self._search_request(request, time.perf_counter())
