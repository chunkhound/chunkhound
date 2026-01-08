from __future__ import annotations

from typing import Any

from chunkhound.utils.text import slugify_kebab


def derive_heading_from_point(point: str) -> str:
    """Derive a short section heading from a point-of-interest bullet."""
    text = point.strip()

    # Strip leading markdown emphasis markers
    if text.startswith("**") and "**" in text[2:]:
        end = text.find("**", 2)
        if end != -1:
            text = text[2:end].strip()

    # Split on colon or dash if present
    for sep in (":", " - ", " — "):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break

    # Fallback: truncate long headings
    if len(text) > 80:
        text = text[:77].rstrip() + "..."
    return text or "Point of Interest"


def slugify_heading(heading: str) -> str:
    """Convert a heading into a filesystem-friendly slug."""
    return slugify_kebab(heading, max_length=60)


def merge_sources_metadata(
    results: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], int | None, int | None]:
    """Merge sources metadata from multiple deep-research calls."""
    unified_source_files: dict[str, str] = {}
    chunk_keys: set[tuple[str, int | None, int | None]] = set()
    unified_chunks_dedup: list[dict[str, Any]] = []

    # Track best-effort database totals (we use the last non-zero stats)
    total_files_indexed: int | None = None
    total_chunks_indexed: int | None = None

    for result in results:
        metadata = result.get("metadata") or {}
        sources = metadata.get("sources") or {}

        for file_path in sources.get("files") or []:
            if file_path:
                unified_source_files.setdefault(str(file_path), "")

        for chunk in sources.get("chunks") or []:
            file_path = chunk.get("file_path")
            if not file_path:
                continue
            start_line = chunk.get("start_line")
            end_line = chunk.get("end_line")
            key = (
                str(file_path),
                int(start_line) if isinstance(start_line, int) else None,
                int(end_line) if isinstance(end_line, int) else None,
            )
            if key not in chunk_keys:
                chunk_keys.add(key)
                unified_chunks_dedup.append(
                    {
                        "file_path": str(file_path),
                        "start_line": key[1],
                        "end_line": key[2],
                    }
                )

        stats = metadata.get("aggregation_stats") or {}
        files_total = stats.get("files_total")
        chunks_total = stats.get("chunks_total")
        if isinstance(files_total, int) and files_total > 0:
            total_files_indexed = files_total
        if isinstance(chunks_total, int) and chunks_total > 0:
            total_chunks_indexed = chunks_total

    return (
        unified_source_files,
        unified_chunks_dedup,
        total_files_indexed,
        total_chunks_indexed,
    )


def is_empty_research_result(result: dict[str, Any]) -> bool:
    """Return True when a deep-research result carries no useful content."""
    answer = str(result.get("answer") or "").strip()
    if not answer:
        return True

    metadata = result.get("metadata") or {}
    if metadata.get("skipped_synthesis"):
        return True

    first_line = answer.splitlines()[0].strip()
    if first_line.startswith("No relevant code context found for:"):
        return True

    return False

