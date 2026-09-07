"""Streaming semantic search over transient chunks."""

import asyncio
import heapq
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import numpy as np

from chunkhound.services.search_service_protocol import SearchServiceProtocol
from chunkhound.services.vector_cache import VectorCache

_OVERFETCH_FACTOR = 5

# Hard ceiling on (offset + page_size) for a single transient (diff/web) search
# call. The top-K heap must hold at least offset + page_size entries to return
# an exact page, so heap size -- and therefore peak memory -- scales with this
# window. Capping it keeps worst-case memory a known constant regardless of
# corpus size (a 500-chunk diff and a 500,000-chunk diff have the same bound),
# instead of silently growing toward O(corpus size) as callers page deeper.
_MAX_RESULT_WINDOW = 2000

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
        limit = max(offset + page_size, page_size * _OVERFETCH_FACTOR)
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
                entry = (score, ordinal, result)
                ordinal += 1
                if len(heap) < limit:
                    heapq.heappush(heap, entry)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        ranked = [
            entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))
        ]
        return ranked[offset : offset + page_size], self._pagination(
            page_size, offset, qualifying_count
        )

    @staticmethod
    def _pagination(page_size: int, offset: int, total: int) -> dict[str, Any]:
        has_more = offset + page_size < total
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

        fetch_size = max(offset + page_size, page_size * _OVERFETCH_FACTOR)
        transient_task = asyncio.create_task(
            self._search_transient(query, fetch_size, 0, None, path_filter, provider)
        )
        db_task = asyncio.create_task(
            self._original.search_semantic(
                query=query,
                page_size=fetch_size,
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
        (transient_results, transient_page), (db_results, db_page) = await _await_pair(
            transient_task, db_task
        )
        merged = self._merge(transient_results, db_results, threshold)
        has_more = (
            len(merged) > offset + page_size
            or bool(transient_page.get("has_more"))
            or bool(db_page.get("has_more"))
        )
        page = self._pagination(page_size, offset, len(merged))
        if has_more:
            page["has_more"] = True
            page["next_offset"] = offset + page_size
        # `total` stays len(merged): a lower bound while either source still has
        # results to give, exact once both are exhausted. Never None -- callers
        # render it directly (see api/cli/commands/search.py).
        return merged[offset : offset + page_size], page

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

        candidate_limit = offset + page_size + 1

        async def frozen_snapshot(
            fetch: Callable[
                [int],
                Awaitable[tuple[list[dict[str, Any]], dict[str, Any]]],
            ],
        ) -> list[dict[str, Any]]:
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
            return await self.search_semantic(
                query,
                page_size=fetch_size,
                offset=0,
                threshold=threshold,
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
        combined = ResultEnhancer().combine_search_results(
            semantic_results=semantic_results,
            regex_results=regex_results,
            semantic_weight=semantic_weight,
            limit=len(semantic_results) + len(regex_results),
        )
        has_more = len(combined) > offset + page_size
        return combined[offset : offset + page_size], {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": len(combined),
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
