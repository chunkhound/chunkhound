"""In-process research composition for transient fetched web pages."""

from collections.abc import AsyncIterator
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
    def get_base_directory(self) -> Path:
        return Path.cwd()

    def get_chunks_in_range(self, *args: Any, **kwargs: Any) -> list:
        return []

    def get_file_by_path(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_chunks_by_file_id(self, *args: Any, **kwargs: Any) -> list:
        return []


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


def pages_to_chunks(pages: list[tuple[str, str, str | bytes]]) -> list[Chunk]:
    """Convert fetched page bodies to embedding-safe chunks."""
    chunks: list[Chunk] = []
    for source_url, extension, content in pages:
        if extension == ".pdf" and isinstance(content, bytes):
            parsed = PDFMapping().parse_pdf_content(content, None, FileId(0))
            chunks.extend(
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
            )
        elif isinstance(content, str):
            chunks.extend(_text_chunks(source_url, content))
    return chunks


async def research_web_pages(
    query: str,
    pages: list[tuple[str, str, str | bytes]],
    config: Any,
    embedding_manager: Any,
    llm_manager: Any,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the configured research strategy over fetched pages in process."""
    chunks = pages_to_chunks(pages)

    async def chunk_stream() -> AsyncIterator[list[Chunk]]:
        for start in range(0, len(chunks), _BATCH_SIZE):
            yield chunks[start : start + _BATCH_SIZE]

    search_service = TransientSearchService(
        original=cast(Any, _EmptySearchService()),
        chunk_stream=chunk_stream,
        vector_source="diff",
        embedding_manager=embedding_manager,
        vector_cache=_WEB_VECTOR_CACHE,
    )
    services = DatabaseServices(
        provider=cast(Any, _TransientProvider()),
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
    return await research_service.deep_research(query)
