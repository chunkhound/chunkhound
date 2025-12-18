from __future__ import annotations

from typing import Any

from chunkhound.autodoc.coverage import compute_db_scope_stats
from chunkhound.autodoc.models import AgentDocMetadata


def format_metadata_block(meta: AgentDocMetadata) -> str:
    """Render the metadata comment block."""

    def _emit_value(lines: list[str], key: str, value: Any, indent: int) -> None:
        pad = " " * indent
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            for sub_k, sub_v in value.items():
                _emit_value(lines, str(sub_k), sub_v, indent + 2)
            return
        if isinstance(value, list):
            lines.append(f"{pad}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    for sub_k, sub_v in item.items():
                        _emit_value(lines, str(sub_k), sub_v, indent + 4)
                else:
                    lines.append(f"{pad}  - {item}")
            return
        lines.append(f"{pad}{key}: {value}")

    lines = [
        "<!--",
        "agent_doc_metadata:",
    ]

    if meta.created_from_sha != "NO_GIT_HEAD":
        lines.append(f"  created_from_sha: {meta.created_from_sha}")

    lines.append(f"  generated_at: {meta.generated_at}")
    if meta.llm_config:
        lines.append("  llm_config:")
        for key, value in meta.llm_config.items():
            lines.append(f"    {key}: {value}")
    if meta.generation_stats:
        lines.append("  generation_stats:")
        for key, value in meta.generation_stats.items():
            _emit_value(lines, str(key), value, indent=4)
    lines.append("-->")
    return "\n".join(lines) + "\n\n"


def build_generation_stats(
    *,
    generator_mode: str,
    total_research_calls: int,
    unified_source_files: dict[str, str],
    unified_chunks_dedup: list[dict[str, Any]],
    services: Any,
    scope_label: str,
) -> dict[str, Any]:
    """Build minimal generation stats for autodoc metadata."""
    stats: dict[str, Any] = {
        "generator_mode": generator_mode,
        "total_research_calls": str(total_research_calls),
        "referenced_files": str(len(unified_source_files)),
        "referenced_chunks": str(len(unified_chunks_dedup)),
    }

    scope_total_files, scope_total_chunks, _scoped_files = compute_db_scope_stats(
        services, scope_label
    )
    if scope_total_files:
        stats["scope_total_files_indexed"] = str(scope_total_files)
    if scope_total_chunks:
        stats["scope_total_chunks_indexed"] = str(scope_total_chunks)

    prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"
    referenced_in_scope = 0
    for path in unified_source_files:
        norm = str(path).replace("\\", "/")
        if not norm:
            continue
        if prefix and not norm.startswith(prefix):
            continue
        referenced_in_scope += 1
    stats["referenced_files_in_scope"] = str(referenced_in_scope)
    if scope_total_files:
        stats["scope_unreferenced_files_count"] = str(
            max(0, scope_total_files - referenced_in_scope)
        )

    return stats
