"""In-process research composition for transient fetched web pages."""

import asyncio
import copy
import hashlib
import sys
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
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
from chunkhound.utils.websearch_core import fetch_pages

# Count all codepoints, including whitespace: each needs at most four UTF-8
# bytes, hence at most 4,800 byte-level tokens (below OpenAI's 8,191 limit).
# A non-whitespace count or len(text)//3 estimate is unsafe for CJK/emoji.
_PAGE_CHUNK_CHARS = 1_200
_BATCH_SIZE = 100
_SOURCE_CACHE_BYTES = 32 * 1024 * 1024
_MAX_CONCURRENT_SCANS = 4
_WEB_VECTOR_CACHE = VectorCache.from_env()

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
        # An explicit synthesis contract, not a claim that sources fit this size.
        self.transient_source_budget_bytes = 1_000_000
        self._sources: OrderedDict[str, tuple[Page, str, int]] = OrderedDict()
        self._source_bytes = 0
        self._load_source: Callable[[str], Awaitable[str | None]] | None = None

    def _cached(self, source_url: str) -> tuple[Page, str, int] | None:
        entry = self._sources.get(source_url)
        if entry is not None:
            self._sources.move_to_end(source_url)
        return entry

    def _store(self, page: Page, text: str) -> None:
        source_url, extension, content = page
        previous = self._sources.pop(source_url, None)
        if previous is not None:
            self._source_bytes -= previous[2]
        # Count raw and normalized text separately even when they share storage.
        # Include generous per-entry overhead for the mapping and tuple nodes.
        size = (
            sum(
                sys.getsizeof(item)
                for item in (page, source_url, extension, content, text)
            )
            + 512
        )
        if size > _SOURCE_CACHE_BYTES:
            return
        while self._sources and self._source_bytes + size > _SOURCE_CACHE_BYTES:
            _, evicted = self._sources.popitem(last=False)
            self._source_bytes -= evicted[2]
        self._sources[source_url] = (page, text, size)
        self._source_bytes += size

    def get_transient_file_content(self, file_path: str) -> str | None:
        """Compatibility-only cache lookup; never performs blocking network I/O."""
        entry = self._cached(file_path)
        return entry[1] if entry is not None else None

    async def get_transient_file_content_async(self, file_path: str) -> str | None:
        """Resolve an evicted source by live refetch, rejecting changed content."""
        if self._load_source is None:
            return None
        return await self._load_source(file_path)

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
        chunks: list[Chunk] = []
        for chunk in parsed:
            if _is_unusable_pdf_chunk(chunk):
                continue
            chunk = replace(chunk, file_path=FilePath(source_url))
            if len(chunk.code) <= _PAGE_CHUNK_CHARS:
                chunks.append(chunk)
                continue
            # Enforce the same hard bound even if the PDF parser's semantic
            # splitter admits oversized text (e.g. whitespace-heavy chunks).
            # Keep the original source span/metadata: extracted PDF text does
            # not map directly to byte offsets in the PDF document.
            for part, start in enumerate(
                range(0, len(chunk.code), _PAGE_CHUNK_CHARS), start=1
            ):
                chunks.append(
                    replace(
                        chunk,
                        symbol=f"{chunk.symbol}:part:{part}",
                        code=chunk.code[start : start + _PAGE_CHUNK_CHARS],
                    )
                )
        return chunks
    if isinstance(content, str):
        return _text_chunks(source_url, content)
    return []


@dataclass(frozen=True)
class _PageDescriptor:
    url: str
    fingerprint: bytes
    usable: bool


def _normalized_page(page: Page) -> tuple[str, list[Chunk]]:
    chunks = _page_to_chunks(*page)
    text = page[2] if isinstance(page[2], str) else "".join(c.code for c in chunks)
    return text, chunks


def _fingerprint(text: str) -> bytes:
    # Incremental encoding avoids another whole-page UTF-8 allocation.
    digest = hashlib.sha256()
    for start in range(0, len(text), 65536):
        digest.update(text[start : start + 65536].encode("utf-8"))
    return digest.digest()


class _BoundedPageStream:
    """Replay URL descriptors, never a corpus of chunks or source bodies.

    The one-shot input advances only on demand under a lock. Each scan derives
    fresh chunks, with at most four scans active. Evicted sources are live
    refetched (no disk snapshot); normalized text must match its first-read
    fingerprint or research fails rather than mixing revisions. A caller-owned
    Sequence is not copied, and its memory remains the caller's responsibility.
    """

    def __init__(
        self,
        pages: AsyncIterator[Page] | Sequence[Page],
        provider: _TransientProvider,
        warning_callback: Callable[[str], None] | None,
    ) -> None:
        self._pages = pages
        self._provider = provider
        self._warning_callback = warning_callback
        self._descriptors: list[_PageDescriptor] = []
        self._by_url: dict[str, _PageDescriptor] = {}
        self._lock = asyncio.Lock()
        self._scans = asyncio.Semaphore(_MAX_CONCURRENT_SCANS)
        self._complete = False
        self._closed = False
        self._error: BaseException | None = None
        self.usable_page_count = 0
        provider._load_source = self.load_source

    def _record_error(self, exc: BaseException) -> None:
        # Keep failures sticky without retaining frames holding page bodies.
        if self._error is None:
            self._error = copy.copy(exc).with_traceback(None)
            self._error.__context__ = None
            self._error.__cause__ = None
        self._complete = True

    async def _refetch(self, descriptor: _PageDescriptor) -> tuple[str, list[Chunk]]:
        self.raise_stream_error()
        try:
            return await self._refetch_page(descriptor)
        except Exception as exc:
            self._record_error(exc)
            raise

    async def _refetch_page(
        self, descriptor: _PageDescriptor
    ) -> tuple[str, list[Chunk]]:
        entry = self._provider._cached(descriptor.url)
        if entry is not None:
            return entry[1], _page_to_chunks(*entry[0])
        # Each refetch owns and closes its fetch iterator/browser. The shared
        # lock bounds refetches to one, including asynchronous source readers.
        fetched = fetch_pages([descriptor.url], warning_callback=self._warning_callback)
        try:
            page = await anext(fetched, None)
            if page is None:
                raise RuntimeError(f"Unable to refetch web source: {descriptor.url}")
            text, chunks = _normalized_page(page)
            if _fingerprint(text) != descriptor.fingerprint:
                raise RuntimeError(
                    f"Web source changed since first read: {descriptor.url}"
                )
            self._provider._store(page, text)
            return text, chunks
        finally:
            await fetched.aclose()

    async def load_source(self, url: str) -> str | None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Web research source session is closed")
            descriptor = self._by_url.get(url)
            if descriptor is None:
                return None
            text, _ = await self._refetch(descriptor)
            return text

    async def _page_chunks(self, index: int) -> list[Chunk] | None:
        async with self._lock:
            if self._closed:
                raise RuntimeError("Web research source session is closed")
            if index < len(self._descriptors):
                descriptor = self._descriptors[index]
                if not descriptor.usable:
                    return []
                _, chunks = await self._refetch(descriptor)
                return chunks
            if self._complete:
                self.raise_stream_error()
                return None
            try:
                if isinstance(self._pages, Sequence):
                    page = self._pages[index] if index < len(self._pages) else None
                else:
                    page = await anext(self._pages, None)
                if page is None:
                    self._complete = True
                    return None
                text, chunks = _normalized_page(page)
                descriptor = _PageDescriptor(page[0], _fingerprint(text), bool(chunks))
                previous = self._by_url.get(page[0])
                if (
                    previous is not None
                    and previous.fingerprint != descriptor.fingerprint
                ):
                    raise RuntimeError(
                        f"Web source changed since first read: {page[0]}"
                    )
                self._descriptors.append(descriptor)
                self._by_url[page[0]] = descriptor
                if chunks:
                    self.usable_page_count += 1
                    self._provider._store(page, text)
                elif self._warning_callback is not None:
                    self._warning_callback(f"No usable content parsed from {page[0]}")
                return chunks
            except BaseException as exc:
                self._record_error(exc)
                raise

    async def __call__(self) -> AsyncIterator[list[Chunk]]:
        async with self._scans:
            index = 0
            batch: list[Chunk] = []
            while True:
                chunks = await self._page_chunks(index)
                if chunks is None:
                    break
                index += 1
                for chunk in chunks:
                    batch.append(chunk)
                    if len(batch) == _BATCH_SIZE:
                        yield batch
                        batch = []
                # Do not carry the preceding page into the next network await.
                del chunks
            if batch:
                yield batch

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            close = getattr(self._pages, "aclose", None)
            if close is not None:
                await close()
            self._provider._sources.clear()
            self._provider._source_bytes = 0

    def raise_stream_error(self) -> None:
        if self._error is not None:
            raise copy.copy(self._error).with_traceback(None)


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
    chunk_stream = _BoundedPageStream(pages, provider, warning_callback)

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
    chunk_stream.raise_stream_error()
    if chunk_stream.usable_page_count == 0:
        raise ValueError("No usable page content was produced")
    return result
