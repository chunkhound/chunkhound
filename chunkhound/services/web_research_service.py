"""In-process research composition for transient fetched web pages."""

import asyncio
import copy
import hashlib
import json
import multiprocessing
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
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
_PDF_PARSE_TIMEOUT_SECONDS = 30.0
_PDF_MAX_EXTRACTED_CHARS = 1_000_000
_PDF_MAX_CHUNKS = 1_000
_PDF_MAX_SYMBOL_CHARS = 512
_PDF_MAX_HEADER_CHARS = 4_096
_PDF_MAX_IPC_BYTES = 8 * 1024 * 1024
_PDF_PROCESS_JOIN_SECONDS = 0.5
_WEB_VECTOR_CACHE = VectorCache.from_env()

Page = tuple[str, str, str | bytes]
_PDFChunkWire = tuple[str, int, int, str, str, str | None]


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
        self._sources: OrderedDict[
            str, tuple[Page, str, tuple[Chunk, ...] | None, int]
        ] = OrderedDict()
        self._source_bytes = 0
        self._load_source: Callable[[str], Awaitable[str | None]] | None = None

    def _cached(
        self, source_url: str
    ) -> tuple[Page, str, tuple[Chunk, ...] | None, int] | None:
        entry = self._sources.get(source_url)
        if entry is not None:
            self._sources.move_to_end(source_url)
        return entry

    def _store(
        self,
        page: Page,
        text: str,
        chunks: Sequence[Chunk] | None = None,
    ) -> None:
        source_url, extension, content = page
        previous = self._sources.pop(source_url, None)
        if previous is not None:
            self._source_bytes -= previous[3]
        cached_chunks = tuple(chunks) if chunks is not None else None
        cached_page = (
            (source_url, extension, b"")
            if extension == ".pdf" and cached_chunks is not None
            else page
        )
        # Do not retain raw PDF bytes after bounded normalization. The source
        # fingerprint remains in the descriptor for live-refetch validation.
        content = cached_page[2]
        # Count raw and normalized text separately even when they share storage.
        # Include chunk text again because it may not share string storage with
        # the normalized synthesis text, plus generous container overhead.
        chunk_size = sum(
            sys.getsizeof(chunk) + sys.getsizeof(chunk.code)
            for chunk in cached_chunks or ()
        )
        size = (
            sum(
                sys.getsizeof(item)
                for item in (cached_page, source_url, extension, content, text)
            )
            + chunk_size
            + 512
        )
        if size > _SOURCE_CACHE_BYTES:
            return
        while self._sources and self._source_bytes + size > _SOURCE_CACHE_BYTES:
            _, evicted = self._sources.popitem(last=False)
            self._source_bytes -= evicted[3]
        self._sources[source_url] = (cached_page, text, cached_chunks, size)
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


def _bounded_pdf_chunks(parsed: Sequence[Chunk]) -> tuple[list[Chunk], bool]:
    """Split and bound PDF output before it crosses a process boundary."""
    chunks: list[Chunk] = []
    extracted_chars = 0
    truncated = False
    for source in parsed:
        if _is_unusable_pdf_chunk(source):
            continue
        for part, start in enumerate(
            range(0, len(source.code), _PAGE_CHUNK_CHARS), start=1
        ):
            if len(chunks) >= _PDF_MAX_CHUNKS:
                return chunks, True
            remaining = _PDF_MAX_EXTRACTED_CHARS - extracted_chars
            if remaining <= 0:
                return chunks, True
            code = source.code[start : start + min(_PAGE_CHUNK_CHARS, remaining)]
            if not code:
                return chunks, True
            symbol = (
                source.symbol
                if len(source.code) <= _PAGE_CHUNK_CHARS
                else f"{source.symbol}:part:{part}"
            )
            chunks.append(
                replace(
                    source,
                    symbol=str(symbol)[:_PDF_MAX_SYMBOL_CHARS],
                    code=code,
                    parent_header=(
                        source.parent_header[:_PDF_MAX_HEADER_CHARS]
                        if source.parent_header is not None
                        else None
                    ),
                )
            )
            extracted_chars += len(code)
            if len(code) < min(_PAGE_CHUNK_CHARS, len(source.code) - start):
                truncated = True
                return chunks, truncated
    return chunks, truncated


def pages_to_chunks(pages: Sequence[Page]) -> list[Chunk]:
    """Synchronously convert pages; production PDF research uses isolation."""
    chunks: list[Chunk] = []
    for source_url, extension, content in pages:
        if extension == ".pdf" and isinstance(content, bytes):
            parsed = PDFMapping().parse_pdf_content(content, None, FileId(0))
            bounded, _ = _bounded_pdf_chunks(parsed)
            chunks.extend(
                replace(chunk, file_path=FilePath(source_url)) for chunk in bounded
            )
        elif isinstance(content, str):
            chunks.extend(_text_chunks(source_url, content))
    return chunks


def _redirect_stdout_to_stderr() -> None:
    """Keep worker output off the parent's stdout.

    The spawned worker inherits fd 1, which is the JSON-RPC channel under
    ``chunkhound mcp`` stdio. MuPDF writes diagnostics for malformed PDFs
    straight to fd 1, bypassing ``sys.stdout``, so redirect at the fd level.
    """
    if sys.stdout is not None:
        sys.stdout.flush()
    try:
        target = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        target = os.open(os.devnull, os.O_WRONLY)
    os.dup2(target, 1)


def _pdf_parse_worker(content: bytes, connection: Connection) -> None:
    """Parse one PDF in a disposable process and send bounded JSON bytes."""
    try:
        _redirect_stdout_to_stderr()
        parsed = PDFMapping().parse_pdf_content(content, None, FileId(0))
        bounded, truncated = _bounded_pdf_chunks(parsed)
        wire: list[_PDFChunkWire] = [
            (
                str(chunk.symbol),
                int(chunk.start_line),
                int(chunk.end_line),
                chunk.code,
                chunk.chunk_type.value,
                chunk.parent_header,
            )
            for chunk in bounded
        ]
        payload = json.dumps(
            ["ok", wire, truncated], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        if len(payload) > _PDF_MAX_IPC_BYTES:
            raise ValueError("PDF parser result exceeds the IPC safety limit")
        connection.send_bytes(payload)
    except Exception as error:
        try:
            payload = json.dumps(
                ["error", str(error)[:500], False], separators=(",", ":")
            ).encode("utf-8")
            connection.send_bytes(payload)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _stop_pdf_process(process: Any) -> None:
    if process.is_alive():
        process.terminate()
        process.join(_PDF_PROCESS_JOIN_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_PDF_PROCESS_JOIN_SECONDS)
    else:
        process.join(0)
    process.close()


def _run_pdf_parser_process(
    content: bytes,
    stop: threading.Event,
) -> bytes:
    """Own the blocking spawn/IPC lifecycle outside the asyncio event loop."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_pdf_parse_worker, args=(content, child))
    started = False
    try:
        try:
            process.start()
            started = True
        except Exception as error:
            raise RuntimeError(f"Unable to start PDF parser: {error}") from error
        finally:
            child.close()
        expires = time.monotonic() + _PDF_PARSE_TIMEOUT_SECONDS
        while not stop.is_set() and time.monotonic() < expires:
            if parent.poll(0.01):
                try:
                    return parent.recv_bytes(_PDF_MAX_IPC_BYTES)
                except OSError as error:
                    raise RuntimeError(
                        "PDF parser result exceeds the IPC safety limit"
                    ) from error
            if not process.is_alive():
                if parent.poll():
                    try:
                        return parent.recv_bytes(_PDF_MAX_IPC_BYTES)
                    except OSError as error:
                        raise RuntimeError(
                            "PDF parser result exceeds the IPC safety limit"
                        ) from error
                raise RuntimeError(
                    f"PDF parser exited without a result (exit {process.exitcode})"
                )
        if stop.is_set():
            raise RuntimeError("PDF parsing cancelled")
        raise TimeoutError(
            f"PDF parsing timed out after {_PDF_PARSE_TIMEOUT_SECONDS:.0f}s"
        )
    finally:
        parent.close()
        if started:
            _stop_pdf_process(process)
        else:
            process.close()


def _decode_pdf_payload(payload_bytes: bytes, source_url: str) -> list[Chunk]:
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("PDF parser returned an invalid result") from error
    if not (
        isinstance(payload, list)
        and len(payload) == 3
        and isinstance(payload[0], str)
        and isinstance(payload[2], bool)
    ):
        raise RuntimeError("PDF parser returned an invalid result")
    status, result, truncated = payload
    if status != "ok" or not isinstance(result, list):
        raise RuntimeError(f"PDF parsing failed: {str(result)[:500]}")
    if len(result) > _PDF_MAX_CHUNKS:
        raise RuntimeError("PDF parser returned too many chunks")
    chunks: list[Chunk] = []
    total_chars = 0
    for item in result:
        if not (isinstance(item, list) and len(item) == 6):
            raise RuntimeError("PDF parser returned an invalid chunk")
        symbol, start_line, end_line, code, chunk_type, parent_header = item
        if not (
            isinstance(symbol, str)
            and isinstance(start_line, int)
            and not isinstance(start_line, bool)
            and isinstance(end_line, int)
            and not isinstance(end_line, bool)
            and isinstance(code, str)
            and isinstance(chunk_type, str)
            and chunk_type in {member.value for member in ChunkType}
            and (parent_header is None or isinstance(parent_header, str))
            and 1 <= start_line <= end_line
            and len(symbol) <= _PDF_MAX_SYMBOL_CHARS
            and len(code) <= _PAGE_CHUNK_CHARS
            and (parent_header is None or len(parent_header) <= _PDF_MAX_HEADER_CHARS)
        ):
            raise RuntimeError("PDF parser returned an invalid chunk")
        total_chars += len(code)
        if total_chars > _PDF_MAX_EXTRACTED_CHARS:
            raise RuntimeError("PDF parser returned too much extracted text")
        chunks.append(
            Chunk(
                symbol=symbol,
                start_line=LineNumber(start_line),
                end_line=LineNumber(end_line),
                code=code,
                chunk_type=ChunkType.from_string(chunk_type),
                file_id=FileId(0),
                language=Language.PDF,
                file_path=FilePath(source_url),
                parent_header=parent_header,
            )
        )
    if truncated:
        raise ValueError("PDF extracted content exceeds the websearch safety limit")
    return chunks


async def _parse_pdf_isolated(content: bytes, source_url: str) -> list[Chunk]:
    """Parse PDF bytes with a hard timeout and cancellation-safe cleanup."""
    stop = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(_run_pdf_parser_process, content, stop)
    )
    try:
        payload = await asyncio.wait_for(
            asyncio.shield(task), timeout=_PDF_PARSE_TIMEOUT_SECONDS + 1.5
        )
    except BaseException:
        stop.set()
        try:
            await asyncio.shield(task)
        except BaseException:
            pass
        raise
    return _decode_pdf_payload(payload, source_url)


@dataclass(frozen=True)
class _PageDescriptor:
    url: str
    fingerprint: bytes
    usable: bool
    is_pdf: bool = False


async def _normalized_page(
    page: Page,
    warning_callback: Callable[[str], None] | None,
) -> tuple[str, list[Chunk], bytes, bool]:
    source_url, extension, content = page
    if extension == ".pdf" and isinstance(content, bytes):
        fingerprint = hashlib.sha256(content).digest()
        try:
            chunks = await _parse_pdf_isolated(content, source_url)
        except (TimeoutError, RuntimeError, ValueError) as error:
            if warning_callback is not None:
                warning_callback(f"Failed to parse {source_url}: {error}")
            return "", [], fingerprint, True
        return "".join(chunk.code for chunk in chunks), chunks, fingerprint, False
    if isinstance(content, str):
        return content, _text_chunks(source_url, content), _fingerprint(content), False
    return "", [], hashlib.sha256(b"").digest(), False


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
            cached_chunks = entry[2]
            if cached_chunks is not None:
                return entry[1], list(cached_chunks)
            return entry[1], _text_chunks(descriptor.url, entry[1])
        # Each refetch owns and closes its fetch iterator/browser. The shared
        # lock bounds refetches to one, including asynchronous source readers.
        fetched = fetch_pages([descriptor.url], warning_callback=self._warning_callback)
        try:
            page = await anext(fetched, None)
            if page is None:
                raise RuntimeError(f"Unable to refetch web source: {descriptor.url}")
            if descriptor.is_pdf:
                content = page[2]
                fingerprint = (
                    hashlib.sha256(content).digest()
                    if isinstance(content, bytes)
                    else _fingerprint(content)
                )
                if fingerprint != descriptor.fingerprint:
                    raise RuntimeError(
                        f"Web source changed since first read: {descriptor.url}"
                    )
            text, chunks, fingerprint, warned = await _normalized_page(
                page, self._warning_callback
            )
            if fingerprint != descriptor.fingerprint:
                raise RuntimeError(
                    f"Web source changed since first read: {descriptor.url}"
                )
            if warned or (descriptor.usable and not chunks):
                raise RuntimeError(
                    f"Unable to replay previously usable web source: "
                    f"{descriptor.url}"
                )
            self._provider._store(page, text, chunks)
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
                text, chunks, fingerprint, warned = await _normalized_page(
                    page, self._warning_callback
                )
                is_pdf = page[1] == ".pdf" and isinstance(page[2], bytes)
                descriptor = _PageDescriptor(
                    page[0], fingerprint, bool(chunks), is_pdf
                )
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
                    self._provider._store(page, text, chunks)
                elif self._warning_callback is not None and not warned:
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
