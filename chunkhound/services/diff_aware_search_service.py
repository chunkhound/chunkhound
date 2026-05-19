"""Diff-aware search service for ChunkHound git history summarization.

This module provides DiffAwareSearchService, a decorator-style class that wraps
an existing SearchService (duck-typed) and merges in-memory diff chunk results
with the DB-backed results. It also defines SearchServiceProtocol for static
type-checking compatibility.
"""

import asyncio
from typing import Any, Protocol, runtime_checkable

import numpy as np
from loguru import logger


@runtime_checkable
class SearchServiceProtocol(Protocol):
    """Protocol matching the public API of SearchService for duck-type checks."""

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
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    async def search_hybrid(
        self,
        query: str,
        regex_pattern: str | None = None,
        page_size: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...

    def get_chunk_context(
        self, chunk_id: Any, context_lines: int = 5
    ) -> dict[str, Any]: ...

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]: ...


class DiffAwareSearchService:
    """Search service decorator that overlays diff-chunk results onto a base service.

    Supports three ``vector_source`` modes:

    - ``"db"``   — pure passthrough to the original SearchService.
    - ``"diff"`` — search only the in-memory diff chunks; falls back to the
                   original service when diff_chunks is empty.
    - ``"both"`` — run diff search and DB search concurrently, merge by score.
    """

    def __init__(
        self,
        original: Any,  # SearchService duck-type
        diff_chunks: list,  # list[Chunk]
        diff_embeddings: list[list[float]],
        vector_source: str,  # "diff", "db", or "both"
        embedding_manager: Any,  # EmbeddingManager
    ) -> None:
        self._original = original
        self._diff_chunks = diff_chunks
        self._vector_source = vector_source
        self._embedding_manager = embedding_manager

        if diff_embeddings:
            # Guard: embedding provider may return more/fewer embeddings than inputs
            # (batching boundary bug observed in Qwen3 provider). Clamp both to min.
            M = min(len(diff_embeddings), len(diff_chunks))
            self._diff_chunks = diff_chunks[:M]
            mat = np.array(diff_embeddings[:M], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.clip(norms, np.float32(1e-9), None)  # avoid upcast via float32
            self._norm_matrix: np.ndarray | None = mat / norms
        else:
            self._norm_matrix = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chunk_to_dict(self, chunk: Any, score: float) -> dict[str, Any]:
        """Convert a Chunk domain object to a search result dict."""
        return {
            "file_path": chunk.file_path,
            "content": chunk.code,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "language": chunk.language.value
            if hasattr(chunk.language, "value")
            else chunk.language,
            "chunk_type": chunk.chunk_type.value
            if hasattr(chunk.chunk_type, "value")
            else chunk.chunk_type,
            "similarity": score,
            "score": score,
        }

    async def _search_diff(
        self,
        query: str,
        page_size: int,
        offset: int,
        threshold: float | None,
        path_filter: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Perform cosine-similarity search over in-memory diff chunks."""
        if not self._diff_chunks or self._norm_matrix is None:
            logger.warning(
                "DiffAwareSearchService: diff mode requested but no diff chunks "
                "available; falling back to original search service."
            )
            result: tuple[list[dict[str, Any]], dict[str, Any]] = await self._original.search_semantic(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=threshold,
                path_filter=path_filter,
            )
            return result

        # 1. Embed the query
        embed_result = await self._embedding_manager.embed_texts([query])
        q_vec = np.array(embed_result.embeddings[0], dtype=np.float32)

        # B1 — normalize query vector
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-9:
            q_vec = q_vec / q_norm

        # 2. Cosine similarities (matrix is already row-normalised)
        scores: np.ndarray = self._norm_matrix @ q_vec  # shape (N,)

        N = len(self._diff_chunks)

        # 3. Sort indices by score descending
        sorted_indices = np.argsort(scores)[::-1].tolist()

        # G1 — path_filter
        if path_filter:
            sorted_indices = [
                i
                for i in sorted_indices
                if self._diff_chunks[i].file_path
                and str(self._diff_chunks[i].file_path).startswith(path_filter)
            ]

        # 4. Apply threshold
        if threshold is not None:
            sorted_indices = [
                i for i in sorted_indices if float(scores[i]) >= threshold
            ]

        total_after_filter = len(sorted_indices)

        # 5. Paginate
        paged = sorted_indices[offset : offset + page_size]

        results = [
            self._chunk_to_dict(self._diff_chunks[i], float(scores[i]))
            for i in paged
        ]

        has_more = (offset + page_size) < total_after_filter
        pagination: dict[str, Any] = {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": N,
        }
        return results, pagination

    # ------------------------------------------------------------------
    # Primary search method
    # ------------------------------------------------------------------

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
        """Semantic search with configurable vector_source."""

        if self._vector_source == "db":
            db_only: tuple[list[dict[str, Any]], dict[str, Any]] = await self._original.search_semantic(
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
            return db_only

        if self._vector_source == "diff":
            return await self._search_diff(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=threshold,
                path_filter=path_filter,
            )

        # vector_source == "both"
        diff_task = asyncio.ensure_future(
            self._search_diff(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=None,  # filter after merge
                path_filter=path_filter,
            )
        )
        db_task = asyncio.ensure_future(
            self._original.search_semantic(
                query=query,
                page_size=page_size,
                offset=offset,
                threshold=None,  # filter after merge
                provider=provider,
                model=model,
                path_filter=path_filter,
                force_strategy=force_strategy,
                time_limit=time_limit,
                result_limit=result_limit,
            )
        )

        (diff_results, _), (db_results, _) = await asyncio.gather(diff_task, db_task)

        # B3 — normalise DB results that use distance-based scoring
        normalised_db: list[dict[str, Any]] = []
        for r in db_results:
            r = dict(r)
            if "distance" in r and "similarity" not in r:
                r["similarity"] = 1.0 - float(r["distance"])
                r["score"] = r["similarity"]
            elif "similarity" in r and "score" not in r:
                r["score"] = r["similarity"]
            normalised_db.append(r)

        # Merge and sort by score descending
        all_results = list(diff_results) + normalised_db
        all_results.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

        # Deduplicate on (file_path, start_line) — keep first (highest score)
        seen: set[tuple[Any, Any]] = set()
        deduped: list[dict[str, Any]] = []
        for r in all_results:
            key = (r.get("file_path"), r.get("start_line"))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        # Apply threshold
        if threshold is not None:
            deduped = [
                r for r in deduped if float(r.get("score", 0.0)) >= threshold
            ]

        total = len(deduped)
        paged = deduped[offset : offset + page_size]
        has_more = (offset + page_size) < total
        pagination: dict[str, Any] = {
            "offset": offset,
            "page_size": page_size,
            "has_more": has_more,
            "next_offset": offset + page_size if has_more else None,
            "total": total,
        }
        return paged, pagination

    # ------------------------------------------------------------------
    # Delegation methods — all other public SearchService methods
    # ------------------------------------------------------------------

    def search_regex(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._original.search_regex(  # type: ignore[no-any-return]
            pattern, page_size=page_size, offset=offset, path_filter=path_filter
        )

    async def search_regex_async(
        self,
        pattern: str,
        page_size: int = 10,
        offset: int = 0,
        path_filter: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        regex_result: tuple[list[dict[str, Any]], dict[str, Any]] = await self._original.search_regex_async(
            pattern, page_size=page_size, offset=offset, path_filter=path_filter
        )
        return regex_result

    async def search_hybrid(
        self,
        query: str,
        regex_pattern: str | None = None,
        page_size: int = 10,
        offset: int = 0,
        semantic_weight: float = 0.7,
        threshold: float | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        hybrid_result: tuple[list[dict[str, Any]], dict[str, Any]] = await self._original.search_hybrid(
            query,
            regex_pattern=regex_pattern,
            page_size=page_size,
            offset=offset,
            semantic_weight=semantic_weight,
            threshold=threshold,
        )
        return hybrid_result

    def get_chunk_context(
        self, chunk_id: Any, context_lines: int = 5
    ) -> dict[str, Any]:
        return self._original.get_chunk_context(chunk_id, context_lines)  # type: ignore[no-any-return]

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]:
        return self._original.get_file_chunks(file_path)  # type: ignore[no-any-return]
