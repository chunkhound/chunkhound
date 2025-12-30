"""Shared chunk context building utilities for research services.

This module provides utilities for building LLM context from chunk data
without requiring pre-read files. Used across synthesis pipelines to
format chunks for LLM consumption.

The ChunkContextBuilder class provides two main capabilities:
1. Building navigation summaries (chunk previews with line numbers)
2. Building code context with import headers

Usage:
    builder = ChunkContextBuilder(import_context_service, llm_manager)

    # Build navigation summary
    summary = builder.build_chunk_summary(chunks, max_chunks=5)

    # Build code context with imports
    context = builder.build_code_context_with_imports(chunks, max_tokens=10000)
"""

from collections import defaultdict
from typing import Any

from loguru import logger

from chunkhound.llm_manager import LLMManager
from chunkhound.services.research.shared.import_context import ImportContextService


def get_chunk_text(chunk: dict) -> str:
    """Extract text content from chunk, handling different field names.

    Chunks may store content in different fields depending on source:
    - "content": Standard field for semantic chunks
    - "code": Used by regex search and some parsers
    - "text": Alternative field name

    Args:
        chunk: Dictionary containing chunk data

    Returns:
        Text content from the chunk, or empty string if no text field found
    """
    return chunk.get("content") or chunk.get("code") or chunk.get("text") or ""


class ChunkContextBuilder:
    """Build LLM context from chunks without requiring pre-read files.

    This service extracts and formats chunk content for LLM consumption,
    handling import extraction and token budget enforcement.
    """

    def __init__(
        self,
        import_context_service: ImportContextService | None = None,
        llm_manager: LLMManager | None = None,
    ):
        """Initialize chunk context builder.

        Args:
            import_context_service: Service for extracting imports from chunk content.
                If None, import headers are skipped.
            llm_manager: LLM manager for token estimation. If None, uses character-based
                estimation (4 chars per token).
        """
        self._import_context_service = import_context_service
        self._llm_manager = llm_manager

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses LLM provider's token estimation if available, otherwise falls
        back to character-based estimation (4 chars per token).

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if self._llm_manager:
            try:
                provider = self._llm_manager.get_utility_provider()
                return provider.estimate_tokens(text)
            except Exception:
                pass
        # Fallback: ~4 chars per token
        return len(text) // 4

    def build_chunk_summary(
        self,
        chunks: list[dict[str, Any]],
        max_chunks: int = 5,
    ) -> str:
        """Build navigation summary from chunks.

        Creates a compact summary of chunks showing line ranges, symbols,
        and content previews. Useful for providing LLM with file context
        without consuming excessive tokens.

        Format:
            - Lines X-Y (symbol): preview_text...

        Args:
            chunks: List of chunk dictionaries with fields:
                - start_line: Starting line number
                - end_line: Ending line number
                - symbol: Optional symbol name (function, class, etc.)
                - content: Chunk content
            max_chunks: Maximum number of chunks to include (default 5)

        Returns:
            Formatted summary string, or "(no chunks)" if empty

        Examples:
            >>> builder = ChunkContextBuilder()
            >>> chunks = [
            ...     {"start_line": 1, "end_line": 10, "symbol": "main", "content": "def main():..."}
            ... ]
            >>> builder.build_chunk_summary(chunks)
            '- Lines 1-10 (main): def main():...'
        """
        if not chunks:
            return "(no chunks)"

        chunk_summaries = []
        for chunk in chunks[:max_chunks]:
            start = chunk.get("start_line", "?")
            end = chunk.get("end_line", "?")
            symbol = chunk.get("symbol", "")
            content = get_chunk_text(chunk)
            content_preview = content[:200]

            summary = f"- Lines {start}-{end}"
            if symbol:
                summary += f" ({symbol})"
            summary += f": {content_preview}..."
            chunk_summaries.append(summary)

        return "\n".join(chunk_summaries) if chunk_summaries else "(no chunks)"

    def build_code_context_with_imports(
        self,
        chunks: list[dict[str, Any]],
        max_tokens: int,
    ) -> str:
        """Build code context with import headers.

        Groups chunks by file and prepends import statements extracted from
        each file's content. Respects token budget by truncating if needed.

        Format:
            File: /path/to/file.py
            # Imports:
            import os
            from pathlib import Path

            <chunk content>

        Args:
            chunks: List of chunk dictionaries with fields:
                - file_path: Path to the source file
                - content: Chunk content
            max_tokens: Maximum tokens for the output

        Returns:
            Formatted code context string with import headers

        Examples:
            >>> builder = ChunkContextBuilder(import_service, llm_manager)
            >>> chunks = [{"file_path": "main.py", "content": "def main(): pass"}]
            >>> context = builder.build_code_context_with_imports(chunks, 1000)
        """
        if not chunks:
            return ""

        # Group chunks by file for import extraction
        chunks_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            file_path = chunk.get("file_path", "unknown")
            chunks_by_file[file_path].append(chunk)

        # Extract imports per file
        file_imports: dict[str, list[str]] = {}
        if self._import_context_service:
            for file_path, file_chunks in chunks_by_file.items():
                if not file_chunks:
                    continue
                # Use first chunk's content for import extraction
                first_chunk = file_chunks[0]
                content = get_chunk_text(first_chunk)
                try:
                    imports = self._import_context_service.get_file_imports(
                        file_path, content
                    )
                    if imports:
                        file_imports[file_path] = imports
                except Exception as e:
                    logger.debug(f"Failed to extract imports from {file_path}: {e}")

        # Build code context
        code_lines: list[str] = []
        current_tokens = 0

        for chunk in chunks:
            file_path = chunk.get("file_path", "unknown")
            code = get_chunk_text(chunk)

            # Build chunk entry
            imports_header = ""
            if file_path in file_imports:
                imports_header = (
                    "# Imports:\n" + "\n".join(file_imports[file_path]) + "\n\n"
                )

            entry = f"File: {file_path}\n{imports_header}{code}\n"
            entry_tokens = self._estimate_tokens(entry)

            # Check token budget
            if current_tokens + entry_tokens > max_tokens:
                remaining = max_tokens - current_tokens
                if remaining > 100:
                    # Truncate entry to fit remaining budget
                    # Rough estimation: 4 chars per token
                    max_chars = remaining * 4
                    entry = entry[:max_chars] + "\n... (truncated)"
                    code_lines.append(entry)
                break

            code_lines.append(entry)
            current_tokens += entry_tokens

        result = "\n".join(code_lines)
        logger.debug(
            f"Built code context: {len(chunks)} chunks, ~{current_tokens} tokens"
        )
        return result

    def build_file_grouped_context(
        self,
        chunks: list[dict[str, Any]],
        max_tokens: int,
        include_imports: bool = True,
    ) -> dict[str, str]:
        """Build context grouped by file path.

        Groups chunks by file and returns a dictionary mapping file paths
        to their formatted content. Useful when you need to process files
        separately.

        Args:
            chunks: List of chunk dictionaries
            max_tokens: Maximum tokens per file
            include_imports: Whether to include import headers

        Returns:
            Dictionary mapping file paths to formatted content
        """
        if not chunks:
            return {}

        # Group chunks by file
        chunks_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            file_path = chunk.get("file_path", "unknown")
            chunks_by_file[file_path].append(chunk)

        result: dict[str, str] = {}

        for file_path, file_chunks in chunks_by_file.items():
            # Extract imports if requested
            imports_header = ""
            if include_imports and self._import_context_service and file_chunks:
                first_chunk = file_chunks[0]
                content = get_chunk_text(first_chunk)
                try:
                    imports = self._import_context_service.get_file_imports(
                        file_path, content
                    )
                    if imports:
                        imports_header = (
                            "# Imports:\n" + "\n".join(imports) + "\n\n"
                        )
                except Exception as e:
                    logger.debug(f"Failed to extract imports from {file_path}: {e}")

            # Build file content
            file_content_parts: list[str] = []
            current_tokens = 0

            if imports_header:
                file_content_parts.append(imports_header)
                current_tokens += self._estimate_tokens(imports_header)

            for chunk in file_chunks:
                code = get_chunk_text(chunk)
                code_tokens = self._estimate_tokens(code)

                if current_tokens + code_tokens > max_tokens:
                    remaining = max_tokens - current_tokens
                    if remaining > 10:
                        max_chars = remaining * 4
                        file_content_parts.append(code[:max_chars] + "\n... (truncated)")
                    break

                file_content_parts.append(code)
                current_tokens += code_tokens

            result[file_path] = "\n".join(file_content_parts)

        return result
