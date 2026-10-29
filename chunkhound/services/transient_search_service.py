"""Streaming semantic search over transient chunks."""

import asyncio
import heapq
import math
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from numbers import Real
from typing import Any

import numpy as np
from loguru import logger

from chunkhound.services.search_service_protocol import SearchServiceProtocol
from chunkhound.services.vector_cache import VectorCache

OVERFETCH_FACTOR_ENV = "CHUNKHOUND_VECTOR_SEARCH_OVERFETCH_FACTOR"
_DEFAULT_OVERFETCH_FACTOR = 5
_MIN_OVERFETCH = 50
# Hard ceiling on (offset + page_size). Deep paging past this raises rather
# than growing the heap toward the corpus. Both-mode uses overfetch instead.
_MAX_RESULT_WINDOW = 2000
# Hybrid position scoring must use one fixed semantic snapshot across pages.
# Both-mode keeps this below the general paging window so it does not turn
# every hybrid query into a 2000-hit DB request.
_HYBRID_BOTH_SNAPSHOT_SIZE = 500


def _overfetch_factor() -> int:
    try:
        return max(1, int(os.environ.get(OVERFETCH_FACTOR_ENV, "")))
    except ValueError:
        return _DEFAULT_OVERFETCH_FACTOR


def _candidate_pool_size(page_size: int) -> int:
    """Bounded heap / DB fetch size for one transient search call.

    Independent of offset so paging the same query keeps a stable pool.
    Large enough to cover the requested page (and HNSW-style overfetch),
    small enough that both-mode does not pull the 2000-hit window on a
    10-result page.
    """
    return min(
        max(page_size, page_size * _overfetch_factor(), _MIN_OVERFETCH),
        _MAX_RESULT_WINDOW,
    )


# Wall-clock ceiling on a single diff/web search call: streaming the source
# plus embedding every uncached chunk it yields. This is a circuit breaker
# against a truly pathological source (an enormous commit range, an embedding
# backend that's stalled), not a tight SLA -- it's sized generously above the
# time a legitimate multi-thousand-chunk diff takes to embed sequentially in
# _TRANSIENT_BATCH_SIZE-sized batches. Without it, removing the old
# MAX_DIFF_CHUNKS cap traded silent truncation for an unbounded hang: a large
# enough source could block the calling MCP tool call indefinitely with no
# actionable error.
_TRANSIENT_SEARCH_TIMEOUT_SECONDS = 300

ChunkBatchStream = Callable[[], AsyncIterator[list[Any]]]


async def _await_pair(first: asyncio.Task[Any], second: asyncio.Task[Any]) -> Any:
    """Wait for two tasks and cancel the sibling if either fails."""
    try:
        return await asyncio.gather(first, second)
    except BaseException:
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        raise


class TransientSearchService:
    """Overlay bounded, streaming vector search onto a DB search service."""

    def __init__(
        self,
        original: SearchServiceProtocol,
        chunk_stream: ChunkBatchStream,
        vector_source: str,
        embedding_manager: Any,
        vector_cache: VectorCache | None = None,
    ) -> None:
        if vector_source not in {"db", "diff", "both"}:
            raise ValueError(f"Unsupported vector_source: {vector_source}")
        self._original = original
        self._chunk_stream = chunk_stream
        self._vector_source = vector_source
        self._embedding_manager = embedding_manager
        self._vector_cache = vector_cache if vector_cache is not None else VectorCache()
        self.peak_live_vectors = 0

    @staticmethod
    def _chunk_to_dict(chunk: Any, score: float) -> dict[str, Any]:
        return {
            "chunk_id": f"diff:{chunk.file_path}:{chunk.start_line}:{chunk.symbol}",
            "file_path": chunk.file_path,
            "content": chunk.code,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "language": (
                chunk.language.value
                if hasattr(chunk.language, "value")
                else chunk.language
            ),
            "chunk_type": (
                chunk.chunk_type.value
                if hasattr(chunk.chunk_type, "value")
                else chunk.chunk_type
            ),
            # Research combines these rows with ordinary DB rows in "both"
            # mode. Keep provenance on each row so historical diff evidence is
            # not confused with source that can safely be read from disk.
            "source_provenance": "transient",
            "similarity": score,
            "score": score,
        }

    @staticmethod
    def _normalise(vector: Sequence[float] | np.ndarray) -> np.ndarray:
        result = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(result)
        if norm > 1e-9:
            result = result / norm
        return result

    @staticmethod
    def _validate_path_filter(path_filter: str | None) -> str | None:
        if not path_filter:
            return None
        for danger in ("..", "~", "*", "?", "[", "]", "\0", "\n", "\r"):
            if danger in path_filter:
                raise ValueError(f"Path filter contains forbidden pattern: {danger!r}")
        return path_filter.replace("\\", "/").lstrip("/").rstrip("/") + "/"

    async def _embed_texts(self, texts: list[str], provider: str | None) -> Any:
        if provider is None:
            return await self._embedding_manager.embed_texts(texts)
        return await self._embedding_manager.embed_texts(texts, provider_name=provider)

    @staticmethod
    def _cache_namespace(result: Any) -> dict[str, Any]:
        identity = getattr(result, "embedding_space_identity", None)
        if not isinstance(identity, str) or not identity:
            raise RuntimeError(
                "Embedding result is missing embedding_space_identity; providers "
                "must stamp the identity used for each request"
            )
        return {
            "provider": result.provider,
            "model": result.model,
            "dims": result.dims,
            "endpoint": identity,
            "deployment": "",
        }

    @staticmethod
    def _validate_embedding_result(result: Any, expected: dict[str, Any]) -> None:
        actual = TransientSearchService._cache_namespace(result)
        if actual != expected:
            raise RuntimeError(
                "Embedding provider changed embedding space during transient search"
            )

    async def _search_transient(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
        provider: str | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            return await asyncio.wait_for(
                self._search_transient_impl(
                    query,
                    page_size,
                    offset,
                    threshold,
                    path_filter,
                    provider,
                    limit=limit,
                ),
                timeout=_TRANSIENT_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Diff/web search timed out after "
                f"{_TRANSIENT_SEARCH_TIMEOUT_SECONDS}s streaming and embedding "
                "transient content. Narrow the commit range or query, or use "
                "vector_source='db' to skip transient embedding."
            ) from None

    async def _search_transient_impl(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
        provider: str | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.peak_live_vectors = 0
        query_result = await self._embed_texts([query], provider)
        if not query_result.embeddings:
            return [], self._pagination(page_size, offset, 0)
        query_vector = self._normalise(query_result.embeddings[0])
        cache_ns = self._cache_namespace(query_result)
        path_prefix = self._validate_path_filter(path_filter)
        effective_limit = (
            limit
            if limit is not None
            else max(offset + page_size, _candidate_pool_size(page_size))
        )
        assert effective_limit >= offset + page_size, (
            "candidate pool must cover the requested offset+page_size slice"
        )
        heap: list[tuple[float, int, dict[str, Any]]] = []
        ordinal = 0
        qualifying_count = 0

        async for chunks in self._chunk_stream():
            filtered = [
                chunk
                for chunk in chunks
                if not path_prefix
                or (
                    chunk.file_path
                    and str(chunk.file_path).replace("\\", "/").startswith(path_prefix)
                )
            ]
            if not filtered:
                continue

            vectors: list[Sequence[float] | np.ndarray | None] = [
                self._vector_cache.get(chunk.code, **cache_ns) for chunk in filtered
            ]
            missing_indices = [
                index for index, vector in enumerate(vectors) if vector is None
            ]
            if missing_indices:
                texts = [filtered[index].code for index in missing_indices]
                embedded = await self._embed_texts(texts, provider)
                self._validate_embedding_result(embedded, cache_ns)
                for index, vector in zip(missing_indices, embedded.embeddings):
                    vectors[index] = vector
                    self._vector_cache.put(filtered[index].code, vector, **cache_ns)

            live = sum(1 for vector in vectors if vector is not None)
            self.peak_live_vectors = max(self.peak_live_vectors, len(heap) + live)

            for chunk, vector in zip(filtered, vectors):
                if vector is None:
                    continue
                score = float(np.dot(self._normalise(vector), query_vector))
                if threshold is not None and score < threshold:
                    continue
                qualifying_count += 1
                result = self._chunk_to_dict(chunk, score)
                # The root is the worst retained rank: lowest score, then
                # latest source position. Ties in output prefer earlier chunks.
                entry = (score, -ordinal, result)
                ordinal += 1
                if len(heap) < effective_limit:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)

        ranked = [
            entry[2] for entry in sorted(heap, key=lambda item: (-item[0], -item[1]))
        ]
        return ranked[offset : offset + page_size], self._pagination(
            page_size, offset, qualifying_count
        )

    async def _search_transient_many(
        self,
        queries: list[str],
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
        provider: str | None,
        *,
        limit: int | None = None,
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        """Score compatible queries during one pass over the transient source."""
        try:
            return await asyncio.wait_for(
                self._search_transient_many_impl(
                    queries,
                    page_size,
                    offset,
                    threshold,
                    path_filter,
                    provider,
                    limit=limit,
                ),
                timeout=_TRANSIENT_SEARCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Diff/web search timed out after "
                f"{_TRANSIENT_SEARCH_TIMEOUT_SECONDS}s streaming and embedding "
                "transient content. Narrow the commit range or queries, or use "
                "vector_source='db' to skip transient embedding."
            ) from None

    async def _search_transient_many_impl(
        self,
        queries: list[str],
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
        provider: str | None,
        *,
        limit: int | None = None,
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
        if not queries:
            return []
        query_result = await self._embed_texts(queries, provider)
        if not query_result.embeddings:
            empty_page = self._pagination(page_size, offset, 0)
            return [([], dict(empty_page)) for _ in queries]
        if len(query_result.embeddings) != len(queries):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of query vectors"
            )
        query_vectors = np.stack(
            [self._normalise(vector) for vector in query_result.embeddings]
        )
        cache_ns = self._cache_namespace(query_result)
        path_prefix = self._validate_path_filter(path_filter)
        effective_limit = (
            limit
            if limit is not None
            else max(offset + page_size, _candidate_pool_size(page_size))
        )
        heaps: list[list[tuple[float, int, dict[str, Any]]]] = [
            [] for _ in queries
        ]
        qualifying_counts = [0] * len(queries)
        ordinal = 0
        peak_live_vectors = 0

        async for chunks in self._chunk_stream():
            filtered = [
                chunk
                for chunk in chunks
                if not path_prefix
                or (
                    chunk.file_path
                    and str(chunk.file_path).replace("\\", "/").startswith(path_prefix)
                )
            ]
            if not filtered:
                continue
            vectors: list[Sequence[float] | np.ndarray | None] = [
                self._vector_cache.get(chunk.code, **cache_ns) for chunk in filtered
            ]
            missing_indices = [
                index for index, vector in enumerate(vectors) if vector is None
            ]
            if missing_indices:
                texts = [filtered[index].code for index in missing_indices]
                embedded = await self._embed_texts(texts, provider)
                self._validate_embedding_result(embedded, cache_ns)
                for index, vector in zip(missing_indices, embedded.embeddings):
                    vectors[index] = vector
                    self._vector_cache.put(filtered[index].code, vector, **cache_ns)

            available = [
                (chunk, self._normalise(vector))
                for chunk, vector in zip(filtered, vectors)
                if vector is not None
            ]
            if not available:
                continue
            chunk_matrix = np.stack([vector for _, vector in available])
            scores = chunk_matrix @ query_vectors.T
            peak_live_vectors = max(
                peak_live_vectors,
                sum(len(heap) for heap in heaps) + len(available),
            )
            for chunk_index, (chunk, _) in enumerate(available):
                for query_index, heap in enumerate(heaps):
                    score = float(scores[chunk_index, query_index])
                    if threshold is not None and score < threshold:
                        continue
                    qualifying_counts[query_index] += 1
                    entry = (score, -ordinal, self._chunk_to_dict(chunk, score))
                    if len(heap) < effective_limit:
                        heapq.heappush(heap, entry)
                    elif entry[:2] > heap[0][:2]:
                        heapq.heapreplace(heap, entry)
                ordinal += 1

        self.peak_live_vectors = peak_live_vectors
        outputs = []
        for heap, total in zip(heaps, qualifying_counts):
            ranked = [
                entry[2]
                for entry in sorted(heap, key=lambda item: (-item[0], -item[1]))
            ]
            outputs.append(
                (
                    ranked[offset : offset + page_size],
                    self._pagination(page_size, offset, total),
                )
            )
        return outputs

    @staticmethod
    def _pagination(page_size: int, offset: int, total: int) -> dict[str, Any]:
        has_more = (
            offset + page_size < total and offset + 2 * page_size <= _MAX_RESULT_WINDOW
        )
        return {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": total,
        }

    async def search_semantic_many(
        self,
        queries: list[str],
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        path_filter: str | None = None,
        force_strategy: str | None = None,
        time_limit: float | None = None,
        result_limit: int | None = None,
    ) -> list[tuple[list[dict[str, Any]], dict[str, Any]] | BaseException]:
        """Search compatible queries while scanning transient content once.

        ``time_limit`` only bounds DB multi-hop exploration. It does not bound
        transient streaming or the combined final rerank.
        """
        if not queries:
            return []
        if self._vector_source == "db":
            return await asyncio.gather(
                *(
                    self._original.search_semantic(
                        query=query,
                        page_size=page_size,
                        offset=offset,
                        threshold=threshold,
                        provider=provider,
                        model=model,
                        path_filter=path_filter,
                        force_strategy=force_strategy,
                        time_limit=time_limit,
                        result_limit=result_limit,
                    )
                    for query in queries
                ),
                return_exceptions=True,
            )
        if offset + page_size > _MAX_RESULT_WINDOW:
            raise ValueError(
                f"offset+page_size ({offset + page_size}) exceeds the maximum "
                f"supported window ({_MAX_RESULT_WINDOW}) for diff/web search"
            )
        if self._vector_source == "diff":
            transient_results = await self._search_transient_many(
                queries, page_size, offset, threshold, path_filter, provider
            )
            return list(transient_results)

        snapshot_size = _candidate_pool_size(page_size)
        transient_task = asyncio.create_task(
            self._search_transient_many(
                queries,
                snapshot_size,
                0,
                None,
                path_filter,
                provider,
                limit=snapshot_size,
            )
        )
        db_tasks = [
            asyncio.create_task(
                self._original.search_semantic(
                    query=query,
                    page_size=snapshot_size,
                    offset=0,
                    threshold=None,
                    provider=provider,
                    model=model,
                    path_filter=path_filter,
                    force_strategy=force_strategy,
                    time_limit=time_limit,
                    result_limit=result_limit,
                )
            )
            for query in queries
        ]
        try:
            transient_results, db_results = await asyncio.gather(
                transient_task,
                asyncio.gather(*db_tasks, return_exceptions=True),
            )
        except BaseException:
            transient_task.cancel()
            for task in db_tasks:
                task.cancel()
            await asyncio.gather(
                transient_task, *db_tasks, return_exceptions=True
            )
            raise

        outputs: list[
            tuple[list[dict[str, Any]], dict[str, Any]] | BaseException
        ] = []
        for query, transient_result, db_result in zip(
            queries, transient_results, db_results
        ):
            if isinstance(db_result, BaseException):
                outputs.append(db_result)
                continue
            try:
                merged = await self._finalize_both_results(
                    query=query,
                    transient_results=transient_result[0],
                    db_results=db_result[0],
                    snapshot_size=snapshot_size,
                    threshold=threshold,
                    provider=provider,
                    force_strategy=force_strategy,
                )
            except Exception as error:
                outputs.append(error)
                continue
            available = len(merged)
            outputs.append(
                (
                    merged[offset : offset + page_size],
                    self._pagination(page_size, offset, available),
                )
            )
        return outputs

    async def search_semantic(
        self,
        query: str,
        page_size: int = 10,
        offset: int = 0,
        threshold: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        path_filter: str | None = None,
        force_strategy: str | None = None,
        time_limit: float | None = None,
        result_limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search semantically across the selected vector sources.

        ``time_limit`` only bounds DB multi-hop exploration. It does not bound
        transient streaming or the combined final rerank.
        """
        if self._vector_source == "db":
            return await self._original.search_semantic(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=threshold,
                provider=provider,
                model=model,
                path_filter=path_filter,
                force_strategy=force_strategy,
                time_limit=time_limit,
                result_limit=result_limit,
            )
        if offset + page_size > _MAX_RESULT_WINDOW:
            raise ValueError(
                f"offset+page_size ({offset + page_size}) exceeds the maximum "
                f"supported window ({_MAX_RESULT_WINDOW}) for diff/web search "
                "pagination; narrow the query or commit range, or use "
                "vector_source='db' for deep pagination over the indexed corpus."
            )
        if self._vector_source == "diff":
            return await self._search_transient(
                query, page_size, offset, threshold, path_filter, provider
            )

        # Overfetch-sized heaps from both sources, independent of offset.
        # Not the 2000-hit paging ceiling — that only rejects deep offsets.
        snapshot_size = _candidate_pool_size(page_size)
        merged = await self._search_both_snapshot(
            query=query,
            snapshot_size=snapshot_size,
            threshold=threshold,
            path_filter=path_filter,
            provider=provider,
            model=model,
            force_strategy=force_strategy,
            time_limit=time_limit,
            result_limit=result_limit,
        )
        available = len(merged)
        page = self._pagination(page_size, offset, available)
        return merged[offset : offset + page_size], page

    async def _search_both_snapshot(
        self,
        *,
        query: str,
        snapshot_size: int,
        threshold: float | None,
        path_filter: str | None,
        provider: str | None = None,
        model: str | None = None,
        force_strategy: str | None = None,
        time_limit: float | None = None,
        result_limit: int | None = None,
        final_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch and combine equal bounded snapshots from both sources."""
        transient_task = asyncio.create_task(
            self._search_transient(
                query,
                snapshot_size,
                0,
                None,
                path_filter,
                provider,
                limit=snapshot_size,
            )
        )
        db_task = asyncio.create_task(
            self._original.search_semantic(
                query=query,
                page_size=snapshot_size,
                offset=0,
                threshold=None,
                provider=provider,
                model=model,
                path_filter=path_filter,
                force_strategy=force_strategy,
                time_limit=time_limit,
                result_limit=result_limit,
            )
        )
        (transient_results, _), (db_results, _) = await _await_pair(
            transient_task, db_task
        )
        if not final_rerank:
            # Hybrid position fusion intentionally retains its existing source
            # scoring behavior and larger fixed snapshot.
            return self._merge(transient_results, db_results, threshold)
        return await self._finalize_both_results(
            query=query,
            transient_results=transient_results,
            db_results=db_results,
            snapshot_size=snapshot_size,
            threshold=threshold,
            provider=provider,
            force_strategy=force_strategy,
        )

    def _uses_final_rerank(
        self, provider: str | None, force_strategy: str | None
    ) -> tuple[bool, Any]:
        embedding_provider = self._embedding_manager.get_provider(provider)
        supports_reranking = getattr(embedding_provider, "supports_reranking", None)
        supported = callable(supports_reranking) and supports_reranking() is True
        return force_strategy != "single_hop" and supported, embedding_provider

    async def _finalize_both_results(
        self,
        *,
        query: str,
        transient_results: list[dict[str, Any]],
        db_results: list[dict[str, Any]],
        snapshot_size: int,
        threshold: float | None,
        provider: str | None,
        force_strategy: str | None,
    ) -> list[dict[str, Any]]:
        """Put retained both-mode candidates onto one validated score scale."""
        candidates = self._merge(transient_results, db_results, None)
        use_rerank, embedding_provider = self._uses_final_rerank(
            provider, force_strategy
        )
        if not use_rerank:
            if threshold is not None:
                candidates = [
                    result
                    for result in candidates
                    if float(result.get("score", 0.0)) >= threshold
                ]
            return candidates[:snapshot_size]
        if not candidates:
            return []

        documents = [str(result.get("content", "")) for result in candidates]
        rerank_results = await embedding_provider.rerank(
            query=query, documents=documents, top_k=len(documents)
        )
        scores: dict[int, float] = {}
        for rerank_result in rerank_results:
            index = rerank_result.index
            score = rerank_result.score
            if type(index) is not int or not 0 <= index < len(candidates):
                raise RuntimeError(
                    "Combined final rerank returned an invalid candidate index"
                )
            if index in scores:
                raise RuntimeError(
                    "Combined final rerank returned a duplicate candidate index"
                )
            if isinstance(score, bool) or not isinstance(score, Real):
                raise RuntimeError("Combined final rerank returned a non-numeric score")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RuntimeError("Combined final rerank returned a non-finite score")
            scores[index] = numeric_score
        if set(scores) != set(range(len(candidates))):
            raise RuntimeError(
                "Combined final rerank did not score every candidate exactly once"
            )

        # Build new dictionaries only after complete validation so callers can
        # never observe a mixture of cosine and reranker scores.
        ranked = [
            {**result, "score": scores[index]}
            for index, result in enumerate(candidates)
        ]
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        ranked = ranked[:snapshot_size]
        if threshold is not None:
            ranked = [
                result for result in ranked if float(result["score"]) >= threshold
            ]
        return ranked

    @staticmethod
    def _merge(
        transient_results: list[dict[str, Any]],
        db_results: list[dict[str, Any]],
        threshold: float | None,
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        transient_locations: set[tuple[Any, Any]] = set()
        seen_transient_ids: set[str] = set()
        for result in transient_results:
            chunk_id = str(result.get("chunk_id", ""))
            if chunk_id in seen_transient_ids:
                continue
            seen_transient_ids.add(chunk_id)
            transient_locations.add((result.get("file_path"), result.get("start_line")))
            merged.append(result)

        seen_db_locations: set[tuple[Any, Any]] = set()
        for raw_result in db_results:
            result = dict(raw_result)
            if "distance" in result and "similarity" not in result:
                result["similarity"] = 1.0 - float(result["distance"])
            if "similarity" in result and "score" not in result:
                result["score"] = result["similarity"]
            location = (result.get("file_path"), result.get("start_line"))
            if location in transient_locations or location in seen_db_locations:
                continue
            seen_db_locations.add(location)
            merged.append(result)

        merged.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if threshold is not None:
            merged = [
                result
                for result in merged
                if float(result.get("score", 0.0)) >= threshold
            ]
        return merged

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._original.search_regex(
            pattern, page_size=page_size, offset=offset, path_filter=path_filter
        )

    async def _search_transient_regex(
        self,
        pattern: str,
        page_size: int,
        offset: int,
        path_filter: str | None,
        *,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Search streamed chunks without consulting the indexed working tree."""
        if offset + page_size > _MAX_RESULT_WINDOW:
            raise ValueError(
                f"offset+page_size ({offset + page_size}) exceeds the maximum "
                f"supported window ({_MAX_RESULT_WINDOW}) for diff/web search"
            )
        expression = re.compile(pattern)
        path_prefix = self._validate_path_filter(path_filter)
        effective_limit = limit if limit is not None else offset + page_size
        matches: list[dict[str, Any]] = []
        total = 0

        async def scan() -> None:
            nonlocal total
            async for chunks in self._chunk_stream():
                for chunk in chunks:
                    path = str(chunk.file_path).replace("\\", "/")
                    if path_prefix and not path.startswith(path_prefix):
                        continue
                    if expression.search(chunk.code) is None:
                        continue
                    total += 1
                    if len(matches) < effective_limit:
                        matches.append(self._chunk_to_dict(chunk, 0.0))

        try:
            await asyncio.wait_for(scan(), timeout=_TRANSIENT_SEARCH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Diff/web regex search timed out after "
                f"{_TRANSIENT_SEARCH_TIMEOUT_SECONDS}s streaming transient content. "
                "Narrow the commit range or pattern, or use vector_source='db'."
            ) from None
        return matches[offset : offset + page_size], self._pagination(
            page_size, offset, total
        )

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
        query: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if self._vector_source == "db":
            return await self._original.search_regex_async(
                pattern,
                page_size=page_size,
                offset=offset,
                path_filter=path_filter,
                query=query,
            )
        if self._vector_source == "diff":
            return await self._search_transient_regex(
                pattern, page_size, offset, path_filter
            )

        snapshot_size = min(
            max(offset + page_size, _candidate_pool_size(page_size)),
            _MAX_RESULT_WINDOW,
        )
        transient_task = asyncio.create_task(
            self._search_transient_regex(
                pattern, snapshot_size, 0, path_filter, limit=snapshot_size
            )
        )
        db_task = asyncio.create_task(
            self._original.search_regex_async(
                pattern,
                page_size=snapshot_size,
                offset=0,
                path_filter=path_filter,
                query=query,
            )
        )
        (transient_results, _), (db_results, _) = await _await_pair(
            transient_task, db_task
        )
        merged = self._merge_regex(transient_results, db_results)
        return merged[offset : offset + page_size], self._pagination(
            page_size, offset, len(merged)
        )

    @staticmethod
    def _merge_regex(
        transient_results: list[dict[str, Any]],
        db_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge regex snapshots while treating diff locations as authoritative."""
        merged = list(transient_results)
        seen_ids = {str(result.get("chunk_id", "")) for result in transient_results}
        transient_locations = {
            (result.get("file_path"), result.get("start_line"))
            for result in transient_results
        }
        for result in db_results:
            chunk_id = str(result.get("chunk_id", ""))
            location = (result.get("file_path"), result.get("start_line"))
            if chunk_id in seen_ids or location in transient_locations:
                continue
            seen_ids.add(chunk_id)
            merged.append(result)
        return merged

    async def search_hybrid(
        self,
        query: str,
        regex_pattern: str | None = None,
        page_size: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from chunkhound.services.search.result_enhancer import ResultEnhancer

        if self._vector_source == "both":
            logger.warning(
                "Hybrid search with vector_source='both' may process up to {} "
                "semantic candidates from each source. This can be slow or fail "
                "if the configured reranker cannot handle the combined workload.",
                _HYBRID_BOTH_SNAPSHOT_SIZE,
            )

        hybrid_window = (
            _HYBRID_BOTH_SNAPSHOT_SIZE
            if self._vector_source == "both"
            else _MAX_RESULT_WINDOW
        )
        outside_window = (
            offset >= hybrid_window
            if self._vector_source == "both"
            else offset + page_size > hybrid_window
        )
        if self._vector_source != "db" and outside_window:
            raise ValueError(
                f"requested page at offset {offset} exceeds the maximum supported "
                f"window ({hybrid_window}) for diff/web hybrid search"
            )
        candidate_limit = offset + page_size + 1
        # Hybrid ranking uses list position, so every page of a query must
        # freeze the same semantic/regex lists. Cap at the paging window,
        # not overfetch(page_size): page_size=1 vs page_size=200 would
        # otherwise rank overlapping hits differently.
        transient_snapshot = _MAX_RESULT_WINDOW

        async def frozen_snapshot(
            fetch: Callable[
                [int],
                Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]],
            ],
        ) -> list[dict[str, Any]]:
            if self._vector_source != "db":
                results, _ = await fetch(transient_snapshot)
                return results
            results, pagination = await fetch(candidate_limit)
            if not pagination.get("has_more"):
                return results

            initial_total = pagination.get("total")
            snapshot_size = (
                initial_total
                if isinstance(initial_total, int) and initial_total > candidate_limit
                else candidate_limit * 2
            )
            results, _ = await fetch(snapshot_size)
            return results

        async def fetch_semantic(
            fetch_size: int,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            if self._vector_source == "db":
                return await self.search_semantic(
                    query,
                    page_size=fetch_size,
                    offset=0,
                    threshold=threshold,
                )
            if self._vector_source == "both":
                merged = await self._search_both_snapshot(
                    query=query,
                    snapshot_size=_HYBRID_BOTH_SNAPSHOT_SIZE,
                    threshold=threshold,
                    path_filter=None,
                    final_rerank=False,
                )
                available = min(len(merged), _HYBRID_BOTH_SNAPSHOT_SIZE)
                return merged, self._pagination(fetch_size, 0, available)
            return await self._search_transient(
                query, fetch_size, 0, threshold, None, limit=fetch_size
            )

        async def fetch_regex(
            fetch_size: int,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            if regex_pattern is None:
                return [], {"has_more": False, "total": 0}
            return await self.search_regex_async(
                regex_pattern, page_size=fetch_size, offset=0
            )

        semantic_task = asyncio.create_task(frozen_snapshot(fetch_semantic))
        if regex_pattern is None:
            semantic_results = await semantic_task
            regex_results: list[dict[str, Any]] = []
        else:
            regex_task = asyncio.create_task(frozen_snapshot(fetch_regex))
            semantic_results, regex_results = await _await_pair(
                semantic_task, regex_task
            )
        if self._vector_source == "both":
            # A diff result owns its source location. Preserve regex credit when
            # the current diff content matched, but do not reintroduce a stale
            # DB copy under a different chunk ID.
            diff_by_location = {
                (result.get("file_path"), result.get("start_line")): result
                for result in semantic_results
                if str(result.get("chunk_id", "")).startswith("diff:")
            }
            reconciled_regex: list[dict[str, Any]] = []
            for result in regex_results:
                location = (result.get("file_path"), result.get("start_line"))
                semantic_result = diff_by_location.get(location)
                if semantic_result is None:
                    reconciled_regex.append(result)
                elif str(result.get("chunk_id", "")).startswith("diff:"):
                    reconciled_regex.append(
                        {**result, "chunk_id": semantic_result["chunk_id"]}
                    )
            regex_results = reconciled_regex
        combined = ResultEnhancer().combine_search_results(
            semantic_results=semantic_results,
            regex_results=regex_results,
            semantic_weight=semantic_weight,
            limit=len(semantic_results) + len(regex_results),
        )
        available = (
            len(combined)
            if self._vector_source == "db"
            else min(len(combined), hybrid_window)
        )
        has_more = available > offset + page_size
        return combined[offset : offset + page_size], {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": available if self._vector_source != "db" else len(combined),
        }

    async def get_chunk_similarities_async(
        self,
        chunk_ids: list[int],
        query_embedding: list[float],
        provider: str,
        model: str,
    ) -> dict[int, float]:
        return await self._original.get_chunk_similarities_async(
            chunk_ids, query_embedding, provider, model
        )

    def get_chunk_context(
        self, chunk_id: Any, context_lines: int = 5
    ) -> dict[str, Any]:
        return self._original.get_chunk_context(chunk_id, context_lines)

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]:
        return self._original.get_file_chunks(file_path)
