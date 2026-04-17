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

        # Get base directory for path resolution
        base_dir = self._db_services.provider.get_base_directory()

        for file_path, file_chunks in files_to_chunks.items():
            # Check if we've hit the overall token limit (skip if unlimited)
            if budget_limit is not None and total_tokens >= budget_limit:
                logger.debug(
                    f"Reached token limit ({budget_limit:,}), stopping file reading"
                )
                break

            try:
                # Resolve path relative to base directory
                if Path(file_path).is_absolute():
                    path = Path(file_path)
                else:
                    path = base_dir / file_path

                if not path.exists():
                    logger.warning(f"File not found (expected at {path}): {file_path}")
                    continue

                # Calculate token budget for this file (capped to prevent bloat)
                num_chunks = len(file_chunks)
                raw_budget = TOKEN_BUDGET_PER_FILE * num_chunks
                budget = min(raw_budget, FILE_CONTENT_TOKENS_MAX)

                # Read file
                content = path.read_text(encoding="utf-8", errors="ignore")

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
                        expanded_start, expanded_end = (
                            expand_to_natural_boundaries(
                                lines, start_line, end_line, chunk, file_path
                            )
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
                f"(2) Files not found at base_directory: {base_dir}, "
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
