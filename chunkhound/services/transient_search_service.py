"""Streaming semantic search over transient chunks."""

import asyncio
import heapq
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import numpy as np

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
        cache_ns = {
            "provider": query_result.provider,
            "model": query_result.model,
            "dims": query_result.dims,
        }
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
        available = min(len(merged), snapshot_size)
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
    ) -> list[dict[str, Any]]:
        """Fetch and merge equal bounded snapshots from transient and DB sources."""
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
        return self._merge(transient_results, db_results, threshold)

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

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
        query: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return await self._original.search_regex_async(
            pattern,
            page_size=page_size,
            offset=offset,
            path_filter=path_filter,
            query=query,
        )

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
            # A diff result owns its source location. Do not let a DB regex hit
            # reintroduce the stale indexed copy under a different chunk_id.
            diff_locations = {
                (result.get("file_path"), result.get("start_line"))
                for result in semantic_results
                if str(result.get("chunk_id", "")).startswith("diff:")
            }
            regex_results = [
                result
                for result in regex_results
                if (result.get("file_path"), result.get("start_line"))
                not in diff_locations
            ]
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
