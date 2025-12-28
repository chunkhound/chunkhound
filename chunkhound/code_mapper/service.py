from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.models import CodeMapperPOI
from chunkhound.code_mapper.pipeline import (
    _derive_heading_from_point,
    _is_empty_research_result,
    _merge_sources_metadata,
    _run_code_mapper_overview_hyde,
)
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


class CodeMapperNoPointsError(RuntimeError):
    """Raised when Code Mapper overview produces no points of interest."""

    def __init__(self, overview_answer: str) -> None:
        super().__init__("Code Mapper overview produced no points of interest.")
        self.overview_answer = overview_answer


@dataclass
class CodeMapperPipelineResult:
    overview_result: dict[str, Any]
    poi_sections: list[tuple[CodeMapperPOI, dict[str, Any]]]
    total_points_of_interest: int
    unified_source_files: dict[str, str]
    unified_chunks_dedup: list[dict[str, Any]]
    total_files_global: int | None
    total_chunks_global: int | None
    scope_total_files: int
    scope_total_chunks: int


def _normalize_taint(taint: str | None) -> str:
    normalized = (taint or "").strip().lower()
    if normalized in ("1", "technical"):
        return "technical"
    if normalized in ("3", "end-user", "end_user", "enduser"):
        return "end-user"
    if normalized in ("2", "balanced", ""):
        return "balanced"
    return "balanced"


def _taint_guidance_lines(*, taint: str) -> list[str]:
    normalized = _normalize_taint(taint)
    if normalized == "technical":
        return [
            "Audience: technical (software engineers).",
            "Prefer implementation details, key types, and precise terminology.",
        ]
    if normalized == "end-user":
        return [
            "Audience: end-user (less technical).",
            "Prefer practical workflows and plain language; explain code identifiers briefly when needed.",
            "De-emphasize internal implementation details unless essential to user outcomes.",
        ]
    return []


async def run_code_mapper_overview_only(
    *,
    llm_manager: LLMManager | None,
    target_dir: Path,
    scope_path: Path,
    scope_label: str,
    max_points: int,
    comprehensiveness: str,
    out_dir: Path | None,
    assembly_provider: LLMProvider | None,
    indexing_cfg: IndexingConfig | None,
) -> tuple[str, list[CodeMapperPOI]]:
    """Run overview-only Code Mapper and return the answer + points."""
    overview_answer, points_of_interest = await _run_code_mapper_overview_hyde(
        llm_manager=llm_manager,
        target_dir=target_dir,
        scope_path=scope_path,
        scope_label=scope_label,
        max_points=max_points,
        comprehensiveness=comprehensiveness,
        out_dir=out_dir,
        assembly_provider=assembly_provider,
        indexing_cfg=indexing_cfg,
    )

    if not points_of_interest:
        raise CodeMapperNoPointsError(overview_answer)

    return overview_answer, points_of_interest


async def run_code_mapper_pipeline(
    *,
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager,
    target_dir: Path,
    scope_path: Path,
    scope_label: str,
    path_filter: str | None,
    comprehensiveness: str,
    max_points: int,
    out_dir: Path | None,
    assembly_provider: LLMProvider | None,
    indexing_cfg: IndexingConfig | None,
    progress: TreeProgressDisplay | None,
    taint: str = "balanced",
    log_info: Callable[[str], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
    log_error: Callable[[str], None] | None = None,
) -> CodeMapperPipelineResult:
    """Run Code Mapper overview + per-point deep research and compute coverage."""
    overview_answer, points_of_interest = await _run_code_mapper_overview_hyde(
        llm_manager=llm_manager,
        target_dir=target_dir,
        scope_path=scope_path,
        scope_label=scope_label,
        max_points=max_points,
        comprehensiveness=comprehensiveness,
        out_dir=out_dir,
        assembly_provider=assembly_provider,
        indexing_cfg=indexing_cfg,
    )

    overview_result: dict[str, Any] = {
        "answer": overview_answer,
        "metadata": {
            "sources": {
                "files": [],
                "chunks": [],
            },
            "aggregation_stats": {},
        },
    }

    if not points_of_interest:
        raise CodeMapperNoPointsError(overview_answer)

    total_points_of_interest = len(points_of_interest)
    poi_sections: list[tuple[CodeMapperPOI, dict[str, Any]]] = []
    taint_lines = _taint_guidance_lines(taint=taint)
    taint_block = ("\n".join(f"- {line}" for line in taint_lines) + "\n\n") if taint_lines else ""
    for idx, poi in enumerate(points_of_interest, start=1):
        poi_text = poi.text
        heading = _derive_heading_from_point(poi_text)
        if log_info:
            log_info(
                f"[Code Mapper] Processing point of interest {idx}/"
                f"{len(points_of_interest)}: {heading}"
            )

        if poi.mode == "operational":
            section_query = (
                "Expand the following OPERATIONAL point of interest into a "
                "detailed, operator/runbook-style documentation section for the "
                f"scoped folder '{scope_label}'.\n\n"
                f"{taint_block}"
                "Focus on step-by-step workflows and 'how to run this end-to-end' "
                "guidance grounded in the code:\n"
                "- Setup and local run path (commands only when supported by repo "
                "evidence)\n"
                "- Configuration (env vars, config files) only when supported by "
                "repo evidence\n"
                "- Common workflows/recipes\n"
                "- Troubleshooting/common failure modes and fixes\n\n"
                "Point of interest:\n"
                f"{poi_text}\n\n"
                "Use markdown headings and bullet lists as needed. It is acceptable "
                "for this section to be long and detailed as long as it remains "
                "grounded in the code."
            )
        else:
            section_query = (
                "Expand the following ARCHITECTURAL point of interest into a "
                "detailed, agent-facing documentation section for the scoped folder "
                f"'{scope_label}'. Explain how the relevant code and configuration "
                "implement this behavior, including responsibilities, key types, "
                "important flows, and operational constraints.\n\n"
                f"{taint_block}"
                "Point of interest:\n"
                f"{poi_text}\n\n"
                "Use markdown headings and bullet lists as needed. It is acceptable "
                "for this section to be long and detailed as long as it remains "
                "grounded in the code."
            )

        try:
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=section_query,
                progress=progress,
                path=path_filter,
            )
            if _is_empty_research_result(result):
                if log_warning:
                    log_warning(
                        f"[Code Mapper] Skipping point of interest {idx} because "
                        "deep research returned no usable content."
                    )
                continue
            poi_sections.append((poi, result))
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            if log_error:
                log_error(f"Code Mapper deep research failed for point {idx}: {exc}")
            logger.exception(f"Code Mapper deep research failed for point {idx}.")

    all_results: list[dict[str, Any]] = [overview_result] + [
        result for _, result in poi_sections
    ]
    (
        unified_source_files,
        unified_chunks_dedup,
        total_files_global,
        total_chunks_global,
    ) = _merge_sources_metadata(all_results)

    scope_total_files, scope_total_chunks, _scoped_files = compute_db_scope_stats(
        services, scope_label
    )

    return CodeMapperPipelineResult(
        overview_result=overview_result,
        poi_sections=poi_sections,
        total_points_of_interest=total_points_of_interest,
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        total_files_global=total_files_global,
        total_chunks_global=total_chunks_global,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
    )
