from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.models import AgentDocMetadata
from chunkhound.code_mapper.utils import compute_scope_prefix
from chunkhound.database_factory import DatabaseServices


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


@dataclass
class CoverageContext:
    referenced_files: int
    referenced_chunks: int
    files_denominator: int | None
    chunks_denominator: int | None
    scope_total_files: int
    scope_total_chunks: int
    files_basis: str
    chunks_basis: str
    referenced_files_in_scope: int
    unreferenced_files_in_scope: int | None


def build_generation_stats_with_coverage(
    *,
    generator_mode: str,
    total_research_calls: int,
    unified_source_files: dict[str, str],
    unified_chunks_dedup: list[dict[str, Any]],
    scope_label: str,
    scope_total_files: int,
    scope_total_chunks: int,
    total_files_global: int | None = None,
    total_chunks_global: int | None = None,
) -> tuple[dict[str, Any], CoverageContext]:
    """Build generation stats and coverage context for Code Mapper metadata."""
    referenced_files = len(unified_source_files)
    referenced_chunks = len(unified_chunks_dedup)

    files_denominator: int | None = scope_total_files or None
    chunks_denominator: int | None = scope_total_chunks or None
    if files_denominator is None and isinstance(total_files_global, int):
        files_denominator = total_files_global
    if chunks_denominator is None and isinstance(total_chunks_global, int):
        chunks_denominator = total_chunks_global

    files_basis = "scope" if scope_total_files else "unknown"
    if not scope_total_files and files_denominator:
        files_basis = "database"

    chunks_basis = "scope" if scope_total_chunks else "unknown"
    if not scope_total_chunks and chunks_denominator:
        chunks_basis = "database"

    prefix = compute_scope_prefix(scope_label)
    referenced_in_scope = 0
    for path in unified_source_files:
        norm = str(path).replace("\\", "/")
        if not norm:
            continue
        if prefix and not norm.startswith(prefix):
            continue
        referenced_in_scope += 1

    unreferenced_in_scope: int | None = None
    if scope_total_files:
        unreferenced_in_scope = max(0, scope_total_files - referenced_in_scope)

    generation_stats: dict[str, Any] = {
        "generator_mode": generator_mode,
        "total_research_calls": str(total_research_calls),
        "code_mapper_comprehensiveness": "unknown",
        "files": {
            "referenced": referenced_files,
            "total_indexed": files_denominator or 0,
            "basis": files_basis,
            "coverage": (
                f"{(referenced_files / files_denominator) * 100.0:.2f}%"
                if files_denominator
                else None
            ),
            "referenced_in_scope": referenced_in_scope,
            "unreferenced_in_scope": unreferenced_in_scope,
        },
        "chunks": {
            "referenced": referenced_chunks,
            "total_indexed": chunks_denominator or 0,
            "basis": chunks_basis,
            "coverage": (
                f"{(referenced_chunks / chunks_denominator) * 100.0:.2f}%"
                if chunks_denominator
                else None
            ),
        },
    }

    coverage = CoverageContext(
        referenced_files=referenced_files,
        referenced_chunks=referenced_chunks,
        files_denominator=files_denominator,
        chunks_denominator=chunks_denominator,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
        files_basis=files_basis,
        chunks_basis=chunks_basis,
        referenced_files_in_scope=referenced_in_scope,
        unreferenced_files_in_scope=unreferenced_in_scope,
    )
    return generation_stats, coverage


def build_generation_stats(
    *,
    generator_mode: str,
    total_research_calls: int,
    unified_source_files: dict[str, str],
    unified_chunks_dedup: list[dict[str, Any]],
    services: DatabaseServices,
    scope_label: str,
) -> dict[str, Any]:
    """Build minimal generation stats for Code Mapper metadata."""
    scope_total_files, scope_total_chunks, _scoped_files = compute_db_scope_stats(
        services, scope_label
    )
    stats, _coverage = build_generation_stats_with_coverage(
        generator_mode=generator_mode,
        total_research_calls=total_research_calls,
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        scope_label=scope_label,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
    )
    return stats
