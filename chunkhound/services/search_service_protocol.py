"""Structural protocol shared by database and transient search services."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SearchServiceProtocol(Protocol):
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
        query: str | None = None,
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

    async def get_chunk_similarities_async(
        self,
        chunk_ids: list[int],
        query_embedding: list[float],
        provider: str,
        model: str,
    ) -> dict[int, float]: ...

    def get_chunk_context(
        self, chunk_id: Any, context_lines: int = 5
    ) -> dict[str, Any]: ...

    def get_file_chunks(self, file_path: str) -> list[dict[str, Any]]: ...
