"""Chunk line-range resolution using natural code boundaries.

Expands raw start/end lines to encompass complete language constructs
(functions, classes, blocks) using indentation heuristics for Python
and brace-matching for C-family languages.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from chunkhound.database_factory import DatabaseServices

from chunkhound.services.research.shared.models import (
    ENABLE_SMART_BOUNDARIES,
    EXTRA_CONTEXT_TOKENS,
    MAX_BOUNDARY_EXPANSION_LINES,
)


def expand_to_natural_boundaries(
    lines: list[str],
    start_line: int,
    end_line: int,
    chunk: dict[str, Any],
    file_path: str,
) -> tuple[int, int]:
    """Expand chunk boundaries to complete function/class definitions.

    Uses existing chunk metadata (symbol, kind) and language-specific heuristics
    to detect natural code boundaries instead of using fixed 50-line windows.

    Args:
        lines: File content split by lines
        start_line: Original chunk start line (1-indexed)
        end_line: Original chunk end line (1-indexed)
        chunk: Chunk metadata with symbol, kind fields
        file_path: File path for language detection

    Returns:
        Tuple of (expanded_start_line, expanded_end_line) in 1-indexed format.
        Returns (0, 0) if inputs are invalid (out-of-bounds or start > end).
    """
    if start_line < 1 or start_line > end_line or end_line > len(lines):
        return (0, 0)

    if not ENABLE_SMART_BOUNDARIES:
        # Fallback to legacy fixed-window behavior
        context_lines = EXTRA_CONTEXT_TOKENS // 20  # ~50 lines
        start_idx = max(1, start_line - context_lines)
        end_idx = min(len(lines), end_line + context_lines)
        return start_idx, end_idx

    # Check if chunk metadata indicates this is already a complete unit
    metadata = chunk.get("metadata", {})
    chunk_kind = metadata.get("kind") or chunk.get("symbol_type", "")

    # If this chunk is marked as a complete function/class/method, use its exact boundaries
    if chunk_kind in (
        "function",
        "method",
        "class",
        "interface",
        "struct",
        "enum",
        # TwinCAT kinds (parser produces complete units)
        "program",
        "function_block",
        "action",
        "property",
    ):
        # Chunk is already a complete unit - just add small padding for context
        padding = 3  # A few lines for docstrings/decorators/comments
        start_idx = max(1, start_line - padding)
        end_idx = min(len(lines), end_line + padding)
        logger.debug(
            f"Using complete {chunk_kind} boundaries: {file_path}:{start_idx}-{end_idx}"
        )
        return start_idx, end_idx

    # For non-complete chunks, expand to natural boundaries
    # Detect language from file extension for language-specific logic
    file_path_lower = file_path.lower()
    is_python = file_path_lower.endswith((".py", ".pyw"))
    is_brace_lang = file_path_lower.endswith(
        (
            ".c",
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".rs",
            ".go",
            ".java",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".cs",
            ".swift",
            ".kt",
            ".scala",
        )
    )

    # Convert to 0-indexed for array access
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines) - 1, end_line - 1)

    # Expand backward to find function/class start
    expanded_start = start_idx
    if is_python:
        # Look for def/class keywords at start of line with proper indentation
        for i in range(start_idx - 1, max(0, start_idx - 200), -1):
            line = lines[i].strip()
            if line.startswith(("def ", "class ", "async def ")):
                expanded_start = i
                break
            # Stop at empty lines followed by significant dedents (module boundary)
            if not line and i > 0:
                next_line = lines[i + 1].lstrip() if i + 1 < len(lines) else ""
                if next_line and not next_line.startswith((" ", "\t")):
                    break
    elif is_brace_lang:
        # Look for opening braces and function signatures
        brace_depth = 0
        for i in range(start_idx, max(0, start_idx - 200), -1):
            line = lines[i]
            # Count braces
            open_braces = line.count("{")
            close_braces = line.count("}")
            brace_depth += close_braces - open_braces

            # Found matching opening brace
            if brace_depth > 0 and "{" in line:
                # Look backward for function signature
                for j in range(i, max(0, i - 10), -1):
                    sig_line = lines[j].strip()
                    # Heuristic: function signatures often have (...) or start with keywords
                    if "(" in sig_line and (")" in sig_line or j < i):
                        expanded_start = j
                        break
                if expanded_start != start_idx:
                    break

    # Expand forward to find function/class end
    expanded_end = end_idx
    if is_python:
        # Find end by detecting dedentation back to original level
        if expanded_start < len(lines):
            start_indent = len(lines[expanded_start]) - len(
                lines[expanded_start].lstrip()
            )
            for i in range(end_idx + 1, min(len(lines), end_idx + 200)):
                line = lines[i]
                if line.strip():  # Non-empty line
                    line_indent = len(line) - len(line.lstrip())
                    # Dedented to same or less indentation = end of block
                    if line_indent <= start_indent:
                        expanded_end = i - 1
                        break
            else:
                # Reached search limit, use current position
                expanded_end = min(len(lines) - 1, end_idx + 50)
    elif is_brace_lang:
        # Find matching closing brace
        brace_depth = 0
        for i in range(expanded_start, min(len(lines), end_idx + 200)):
            line = lines[i]
            open_braces = line.count("{")
            close_braces = line.count("}")
            brace_depth += open_braces - close_braces

            # Found matching closing brace
            if brace_depth == 0 and i > expanded_start and "}" in line:
                expanded_end = i
                break

    # Safety: Don't expand beyond max limit
    if expanded_end - expanded_start > MAX_BOUNDARY_EXPANSION_LINES:
        logger.debug(
            f"Boundary expansion too large ({expanded_end - expanded_start} lines), "
            f"limiting to {MAX_BOUNDARY_EXPANSION_LINES}"
        )
        expanded_end = expanded_start + MAX_BOUNDARY_EXPANSION_LINES

    # Convert back to 1-indexed
    final_start = expanded_start + 1
    final_end = expanded_end + 1

    logger.debug(
        f"Expanded boundaries: {file_path}:{start_line}-{end_line} -> "
        f"{final_start}-{final_end} ({final_end - final_start} lines)"
    )

    return final_start, final_end


def get_chunk_expanded_range(
    chunk: dict[str, Any],
    db_services: "DatabaseServices",
) -> tuple[int, int]:
    """Get expanded line range for chunk.

    If expansion already computed and stored in chunk, return it.
    Otherwise, re-compute using expand_to_natural_boundaries().

    Args:
        chunk: Chunk dictionary with metadata
        db_services: Database services bundle for accessing file paths

    Returns:
        Tuple of (expanded_start_line, expanded_end_line) in 1-indexed format
    """
    # Check if already stored (after enhancement in read_files_with_budget)
    if "expanded_start_line" in chunk and "expanded_end_line" in chunk:
        return (chunk["expanded_start_line"], chunk["expanded_end_line"])

    # Re-compute (fallback)
    file_path = chunk.get("file_path")
    start_line = chunk.get("start_line", 0)
    end_line = chunk.get("end_line", 0)

    if not file_path or not start_line or not end_line:
        return (start_line, end_line)

    # Read file lines
    try:
        base_dir = db_services.provider.get_base_directory()
        if Path(file_path).is_absolute():
            path = Path(file_path)
        else:
            path = base_dir / file_path

        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        logger.debug(f"Could not re-read file for expansion: {file_path}: {e}")
        return (start_line, end_line)

    expanded_start, expanded_end = expand_to_natural_boundaries(
        lines, start_line, end_line, chunk, file_path
    )

    if expanded_start == 0 and expanded_end == 0:
        logger.warning(
            f"Boundary expansion failed for {file_path}, using original range"
        )
        return (start_line, end_line)

    return (expanded_start, expanded_end)
