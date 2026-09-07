"""In-process research composition for transient fetched web pages."""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, cast

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import (
    ChunkType,
    FileId,
    FilePath,
    Language,
    LineNumber,
)
from chunkhound.database_factory import DatabaseServices
from chunkhound.parsers.mappings.pdf import PDFMapping
from chunkhound.services.research.factory import ResearchServiceFactory
from chunkhound.services.transient_search_service import TransientSearchService
from chunkhound.services.vector_cache import VectorCache

_PAGE_CHUNK_CHARS = 10_000
_BATCH_SIZE = 100
_WEB_VECTOR_CACHE = VectorCache()

Page = tuple[str, str, str | bytes]


class _EmptySearchService:
    async def search_semantic(self, **kwargs: Any) -> tuple[list, dict]:
        page_size = kwargs.get("page_size", 10)
        offset = kwargs.get("offset", 0)
        return [], {
            "offset": offset,
            "page_size": page_size,
            "has_more": False,
            "next_offset": None,
            "total": 0,
        }

    def search_regex(self, *args: Any, **kwargs: Any) -> tuple[list, dict]:
        return [], {"total": 0, "has_more": False, "next_offset": None}

    async def search_regex_async(self, *args: Any, **kwargs: Any) -> tuple[list, dict]:
        return self.search_regex()

    async def search_hybrid(self, *args: Any, **kwargs: Any) -> tuple[list, dict]:
        return self.search_regex()

    async def get_chunk_similarities_async(
        self, *args: Any, **kwargs: Any
    ) -> dict[int, float]:
        return {}

    def get_chunk_context(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def get_file_chunks(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


class _TransientProvider:
    def __init__(self) -> None:
        self._files: dict[str, str] = {}

    def store_page_text(self, source_url: str, text: str) -> None:
        self._files[source_url] = text

    def get_transient_file_content(self, file_path: str) -> str | None:
        return self._files.get(file_path)

    def get_base_directory(self) -> Path:
        return Path.cwd()

    def get_chunks_in_range(self, *args: Any, **kwargs: Any) -> list:
        return []

    def get_file_by_path(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_chunks_by_file_id(self, *args: Any, **kwargs: Any) -> list:
        return []


def _is_unusable_pdf_chunk(chunk: Chunk) -> bool:
    return chunk.symbol in {"pdf_unavailable", "pdf_parse_error"} or any(
        message in (chunk.code or "")
        for message in ("PDF parsing not available", "Error parsing PDF:")
    )


def _text_chunks(source_url: str, content: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    lines = content.splitlines(keepends=True)
    current: list[str] = []
    current_chars = 0
    start_line = 1
    line_number = 1

    def flush() -> None:
        nonlocal current, current_chars, start_line
        if not current:
            return
        chunks.append(
            Chunk(
                symbol=f"page:{len(chunks) + 1}",
                start_line=LineNumber(start_line),
                end_line=LineNumber(max(start_line, line_number - 1)),
                code="".join(current),
                chunk_type=ChunkType.BLOCK,
                file_id=FileId(0),
                language=Language.MARKDOWN,
                file_path=FilePath(source_url),
            )
        )
        current = []
        current_chars = 0
        start_line = line_number

    for line in lines:
        if current and current_chars + len(line) > _PAGE_CHUNK_CHARS:
            flush()
        if len(line) > _PAGE_CHUNK_CHARS:
            for start in range(0, len(line), _PAGE_CHUNK_CHARS):
                current = [line[start : start + _PAGE_CHUNK_CHARS]]
                current_chars = len(current[0])
                flush()
            line_number += 1
            start_line = line_number
            continue
        current.append(line)
        current_chars += len(line)
        line_number += 1
    flush()
    return chunks


def pages_to_chunks(pages: Sequence[Page]) -> list[Chunk]:
    """Convert fetched page bodies to embedding-safe chunks."""
    chunks: list[Chunk] = []
    for source_url, extension, content in pages:
        chunks.extend(_page_to_chunks(source_url, extension, content))
    return chunks


def _page_to_chunks(
    source_url: str, extension: str, content: str | bytes
) -> list[Chunk]:
    if extension == ".pdf" and isinstance(content, bytes):
        parsed = PDFMapping().parse_pdf_content(content, None, FileId(0))
        return [
            Chunk(
                symbol=chunk.symbol,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                code=chunk.code,
                chunk_type=chunk.chunk_type,
                file_id=chunk.file_id,
                language=chunk.language,
                file_path=FilePath(source_url),
                parent_header=chunk.parent_header,
                start_byte=chunk.start_byte,
                end_byte=chunk.end_byte,
                metadata=chunk.metadata,
            )
            for chunk in parsed
            if not _is_unusable_pdf_chunk(chunk)
        ]
    if isinstance(content, str):
        return _text_chunks(source_url, content)
    return []


async def _iter_pages(
    pages: AsyncIterator[Page] | Sequence[Page],
) -> AsyncIterator[Page]:
    if isinstance(pages, Sequence):
        for page in pages:
            yield page
        return
    async for page in pages:
        yield page


class _ReplayableChunkStream:
    """Convert a one-shot page stream once and replay batches to every search."""

    def __init__(
        self,
        pages: AsyncIterator[Page] | Sequence[Page],
        provider: _TransientProvider,
        warning_callback: Callable[[str], None] | None,
    ) -> None:
        self._pages = pages
        self._provider = provider
        self._warning_callback = warning_callback
        self._batches: list[list[Chunk]] = []
        self._condition = asyncio.Condition()
        self._producer: asyncio.Task[None] | None = None
        self._complete = False
        self._closing = False
        self._error: BaseException | None = None
        self.usable_page_count = 0

    async def _append(self, batch: list[Chunk]) -> None:
        async with self._condition:
            self._batches.append(batch)
            self._condition.notify_all()

    async def _produce(self) -> None:
        batch: list[Chunk] = []
        try:
            async for source_url, extension, content in _iter_pages(self._pages):
                page_chunks = _page_to_chunks(source_url, extension, content)
                if not page_chunks:
                    if self._warning_callback is not None:
                        self._warning_callback(
                            f"No usable content parsed from {source_url}"
                        )
                    continue

                self.usable_page_count += 1
                if isinstance(content, str):
                    self._provider.store_page_text(source_url, content)
                else:
                    self._provider.store_page_text(
                        source_url, "".join(chunk.code for chunk in page_chunks)
                    )
                batch.extend(page_chunks)
                while len(batch) >= _BATCH_SIZE:
                    await self._append(batch[:_BATCH_SIZE])
                    batch = batch[_BATCH_SIZE:]
            if batch:
                await self._append(batch)
        except BaseException as exc:
            if not (self._closing and isinstance(exc, asyncio.CancelledError)):
                self._error = exc
        finally:
            async with self._condition:
                self._complete = True
                self._condition.notify_all()

    async def __call__(self) -> AsyncIterator[list[Chunk]]:
        if self._producer is None:
            self._producer = asyncio.create_task(self._produce())

        index = 0
        while True:
            async with self._condition:
                await self._condition.wait_for(
                    lambda: index < len(self._batches) or self._complete
                )
                if index < len(self._batches):
                    batch = self._batches[index]
                    index += 1
                else:
                    if self._error is not None:
                        raise self._error
                    return
            yield batch

    async def close(self) -> None:
        if self._producer is not None and not self._producer.done():
            self._closing = True
            self._producer.cancel()
        if self._producer is not None:
            await asyncio.gather(self._producer, return_exceptions=True)

    def raise_producer_error(self) -> None:
        if self._error is not None:
            raise self._error


async def research_web_pages(
    query: str,
    pages: AsyncIterator[Page] | Sequence[Page],
    config: Any,
    embedding_manager: Any,
    llm_manager: Any,
    progress: Any = None,
    warning_callback: Callable[[str], None] | None = None,
    previous_query: str | None = None,
) -> dict[str, Any]:
    """Run the configured research strategy over fetched pages in process."""
    provider = _TransientProvider()
    chunk_stream = _ReplayableChunkStream(pages, provider, warning_callback)

    search_service = TransientSearchService(
        original=cast(Any, _EmptySearchService()),
        chunk_stream=chunk_stream,
        vector_source="diff",
        embedding_manager=embedding_manager,
        vector_cache=_WEB_VECTOR_CACHE,
    )
    services = DatabaseServices(
        provider=cast(Any, provider),
        indexing_coordinator=cast(Any, None),
        search_service=search_service,
        embedding_service=cast(Any, None),
    )
    research_service = ResearchServiceFactory.create(
        config=config,
        db_services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        tool_name="websearch",
        progress=progress,
        path_filter=None,
    )
    try:
        result = await research_service.deep_research(
            query, previous_query=previous_query
        )
    finally:
        await chunk_stream.close()
    chunk_stream.raise_producer_error()
    if chunk_stream.usable_page_count == 0:
        raise ValueError("No usable page content was produced")
    return result
