"""File reading utilities for deep research service.

This module provides token-budget-aware file reading functionality for the deep research service.
It handles:
- Reading files within token budgets
- Smart boundary expansion to complete function/class definitions
- Detection of fully read vs partial files
- Chunk range calculation with context expansion

The FileReader class is responsible for efficiently loading file contents while respecting
token limits and ensuring code completeness for better synthesis quality.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.database_factory import DatabaseServices
from chunkhound.services.research.shared.chunk_range import (
    expand_to_natural_boundaries,
    get_chunk_expanded_range,
)
from chunkhound.services.research.shared.models import (
    FILE_CONTENT_TOKENS_MAX,
    TOKEN_BUDGET_PER_FILE,
)


def resolve_source_text(provider: Any, file_path: str) -> str | None:
    """Return full source text from a transient store or the filesystem."""
    getter = getattr(provider, "get_transient_file_content", None)
    if callable(getter):
        stored = getter(file_path)
        if isinstance(stored, str) and stored:
            return stored

    base_dir = provider.get_base_directory()
    path = Path(file_path) if Path(file_path).is_absolute() else base_dir / file_path
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


async def resolve_source_text_async(provider: Any, file_path: str) -> str | None:
    """Load transient sources asynchronously (evicted web pages may refetch)."""
    getter = getattr(provider, "get_transient_file_content_async", None)
    if callable(getter):
        content = await getter(file_path)
        return content if isinstance(content, str) else None
    return resolve_source_text(provider, file_path)


class BudgetedSourceExcerpt(str):
    """Authoritative web evidence: synthesis must not rebuild it from chunks.

    The marker lives on each value so aggregation and clustering can copy or
    partition the source mapping without losing its budget semantics. Ordinary
    code source strings retain their existing chunk/line-marker rendering.
    """


async def read_transient_sources_with_budget(
    provider: Any,
    chunks: list[dict[str, Any]],
    llm_manager: Any,
    max_tokens: int | None,
) -> dict[str, str] | None:
    """Retain bounded selected evidence, not the fetched corpus, for synthesis.

    Only providers explicitly advertising a byte budget opt in. Large sources
    contribute the actual matching chunk text rather than a prefix of the page
    or an unbounded line (a whole page may be a single line). Source loading
    verifies that refetched content still matches the searched revision.
    """
    byte_budget = getattr(provider, "transient_source_budget_bytes", None)
    if not isinstance(byte_budget, int) or byte_budget <= 0:
        return None
    llm = llm_manager.get_utility_provider()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        path = chunk.get("file_path") or chunk.get("path")
        if path:
            grouped.setdefault(path, []).append(chunk)
    contents: dict[str, str] = {}
    remaining_tokens = max_tokens
    chunks_are_source = bool(getattr(provider, "transient_chunks_are_source", False))
    for path, matches in grouped.items():
        if byte_budget <= 0 or (remaining_tokens is not None and remaining_tokens <= 0):
            break
        per_file = min(byte_budget, 64_000)
        max_chars = per_file // 4
        if chunks_are_source:
            # Git history results already contain the requested revision's diff
            # hunks. They are authoritative evidence: consulting the working
            # tree here could substitute newer code or lose deleted files.
            pieces: list[str] = []
            remaining_chars = max_chars
            for match in matches:
                text = match.get("content") or match.get("code") or ""
                piece = text[:remaining_chars]
                if piece:
                    pieces.append(piece)
                    remaining_chars -= len(piece) + 7
                if remaining_chars <= 0:
                    break
            excerpt = "\n\n...\n\n".join(pieces)[:max_chars]
        else:
            # Let refetch failures propagate: silently using a changed revision
            # would invalidate citations, even if other sources were readable.
            source = await resolve_source_text_async(provider, path)
            if not source:
                continue
            # Four bytes per Unicode code point is a conservative UTF-8 bound.
            if len(source) * 4 <= per_file:
                excerpt = source
            else:
                pieces = []
                remaining_chars = max_chars
                for match in matches:
                    text = match.get("content") or match.get("code") or ""
                    piece = text[:remaining_chars]
                    if piece:
                        pieces.append(piece)
                        remaining_chars -= len(piece) + 7
                    if remaining_chars <= 0:
                        break
                excerpt = "\n\n...\n\n".join(pieces)[:max_chars]
            del source
        if remaining_tokens is not None:
            low, high = 0, len(excerpt)
            while low < high:
                middle = (low + high + 1) // 2
                if llm.estimate_tokens(excerpt[:middle]) <= remaining_tokens:
                    low = middle
                else:
                    high = middle - 1
            excerpt = excerpt[:low]
        if excerpt:
            contents[path] = BudgetedSourceExcerpt(excerpt)
            byte_budget -= len(excerpt.encode("utf-8"))
            if remaining_tokens is not None:
                remaining_tokens -= llm.estimate_tokens(excerpt)
    if chunks and not contents:
        raise RuntimeError(
            "No readable transient source evidence within the synthesis budget"
        )
    return contents


class FileReader:
    """Handles token-budget-aware file reading for deep research."""

    def __init__(self, db_services: DatabaseServices):
        """Initialize file reader.

        Args:
            db_services: Database services bundle for accessing file paths
        """
        self._db_services = db_services

    async def read_files_with_budget(
        self,
        chunks: list[dict[str, Any]],
        llm_manager: Any,
        max_tokens: int | None = None,
    ) -> dict[str, str]:
        """Read files containing chunks within optional token budget.

        When max_tokens is None (default), reads all files without budget constraint.
        Use this after elbow detection has already filtered chunks by relevance.

        Args:
            chunks: List of chunks
            llm_manager: LLM manager for token estimation
            max_tokens: Maximum tokens for file contents (None = unlimited)

        Returns:
            Dictionary mapping file paths to contents
        """
        transient = await read_transient_sources_with_budget(
            self._db_services.provider, chunks, llm_manager, max_tokens
        )
        if transient is not None:
            return transient

        # Group chunks by file
        files_to_chunks: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk.get("file_path") or chunk.get("path", "")
            if file_path:
                if file_path not in files_to_chunks:
                    files_to_chunks[file_path] = []
                files_to_chunks[file_path].append(chunk)

        budget_limit = max_tokens  # None means unlimited

        # Read files with budget (track total tokens per algorithm spec)
        file_contents: dict[str, str] = {}
        total_tokens = 0
        llm = llm_manager.get_utility_provider()

        for file_path, file_chunks in files_to_chunks.items():
            # Check if we've hit the overall token limit (skip if unlimited)
            if budget_limit is not None and total_tokens >= budget_limit:
                logger.debug(
                    f"Reached token limit ({budget_limit:,}), stopping file reading"
                )
                break

            try:
                content = await resolve_source_text_async(
                    self._db_services.provider, file_path
                )
                if content is None:
                    logger.warning(f"File not found: {file_path}")
                    continue

                # Calculate token budget for this file (capped to prevent bloat)
                num_chunks = len(file_chunks)
                raw_budget = TOKEN_BUDGET_PER_FILE * num_chunks
                budget = min(raw_budget, FILE_CONTENT_TOKENS_MAX)

                # Estimate tokens
                estimated_tokens = llm.estimate_tokens(content)

                if estimated_tokens <= budget:
                    # File fits in budget, check against overall limit (skip if unlimited)
                    if (
                        budget_limit is None
                        or total_tokens + estimated_tokens <= budget_limit
                    ):
                        file_contents[file_path] = content
                        total_tokens += estimated_tokens
                    else:
                        # Truncate to fit within overall limit
                        remaining_tokens = budget_limit - total_tokens
                        if remaining_tokens > 500:  # Only include if meaningful
                            chars_to_include = remaining_tokens * 4
                            file_contents[file_path] = content[:chars_to_include]
                            total_tokens = budget_limit
                        break
                else:
                    # File too large, extract chunks with smart boundary detection
                    chunk_contents = []
                    lines = content.split("\n")  # Pre-split for all chunks in this file

                    for chunk in file_chunks:
                        start_line = chunk.get("start_line", 1)
                        end_line = chunk.get("end_line", 1)

                        # Use smart boundary detection to expand to complete functions/classes
                        expanded_start, expanded_end = expand_to_natural_boundaries(
                            lines, start_line, end_line, chunk, file_path
                        )

                        if expanded_start == 0 and expanded_end == 0:
                            logger.warning(
                                f"Skipping chunk with invalid boundaries in {file_path}: "
                                f"start_line={start_line}, end_line={end_line} "
                                f"(file has {len(lines)} lines)"
                            )
                            continue  # (0,0) = out-of-bounds chunk; skip rather than emit garbled slice

                        # Store expanded range in chunk for later deduplication
                        chunk["expanded_start_line"] = expanded_start
                        chunk["expanded_end_line"] = expanded_end

                        # Extract chunk with smart boundaries (convert 1-indexed to 0-indexed)
                        start_idx = max(0, expanded_start - 1)
                        end_idx = min(len(lines), expanded_end)

                        chunk_with_context = "\n".join(lines[start_idx:end_idx])
                        chunk_contents.append(chunk_with_context)

                    combined_chunks = "\n\n...\n\n".join(chunk_contents)
                    chunk_tokens = llm.estimate_tokens(combined_chunks)

                    # Check against overall token limit (skip if unlimited)
                    if (
                        budget_limit is None
                        or total_tokens + chunk_tokens <= budget_limit
                    ):
                        file_contents[file_path] = combined_chunks
                        total_tokens += chunk_tokens
                    else:
                        # Truncate to fit
                        remaining_tokens = budget_limit - total_tokens
                        if remaining_tokens > 500:
                            chars_to_include = remaining_tokens * 4
                            file_contents[file_path] = combined_chunks[
                                :chars_to_include
                            ]
                            total_tokens = budget_limit
                        break

            except Exception as e:
                logger.warning(f"Failed to read file {file_path}: {e}")
                continue

        # FAIL-FAST: Validate that at least some files were loaded if chunks were provided
        # This prevents silent data loss where searches find chunks but synthesis gets no code
        if chunks and not file_contents:
            budget_desc = (
                "unlimited" if budget_limit is None else f"{budget_limit:,} tokens"
            )
            raise RuntimeError(
                f"DATA LOSS DETECTED: Found {len(chunks)} chunks across {len(files_to_chunks)} files "
                f"but failed to read ANY file contents. "
                f"Possible causes: "
                f"(1) Token budget exhausted ({budget_desc}), "
                f"(2) Files not found on disk or in the transient store, "
                f"(3) All file read operations failed. "
                f"Check logs above for file-specific errors."
            )

        limit_desc = "unlimited" if budget_limit is None else f"{budget_limit:,}"
        logger.debug(
            f"File reading complete: Loaded {len(file_contents)} files with {total_tokens:,} tokens "
            f"(limit: {limit_desc})"
        )
        return file_contents

    def is_file_fully_read(self, file_content: str) -> bool:
        """Detect if file_content is full file vs partial chunks.

        Heuristic: Partial reads have "..." separator between chunks.

        Args:
            file_content: Content from file_contents dict

        Returns:
            True if full file was read, False if partial chunks
        """
        return "\n\n...\n\n" not in file_content

    def get_chunk_expanded_range(self, chunk: dict[str, Any]) -> tuple[int, int]:
        """Delegate to shared chunk_range.get_chunk_expanded_range."""
        return get_chunk_expanded_range(chunk, self._db_services)
