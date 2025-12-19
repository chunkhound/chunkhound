from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.pipeline import (
    _derive_heading_from_point,
    _is_empty_research_result,
    _merge_sources_metadata,
    _run_code_mapper_overview_hyde,
)
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl


class CodeMapperNoPointsError(RuntimeError):
    """Raised when Code Mapper overview produces no points of interest."""

    def __init__(self, overview_answer: str) -> None:
        super().__init__("Code Mapper overview produced no points of interest.")
        self.overview_answer = overview_answer


@dataclass
class CodeMapperPipelineResult:
    overview_result: dict[str, Any]
    poi_sections: list[tuple[str, dict[str, Any]]]
    unified_source_files: dict[str, str]
    unified_chunks_dedup: list[dict[str, Any]]
    total_files_global: int | None
    total_chunks_global: int | None
    scope_total_files: int
    scope_total_chunks: int


async def run_code_mapper_overview_only(
    *,
    llm_manager: LLMManager | None,
    target_dir: Any,
    scope_path: Any,
    scope_label: str,
    max_points: int,
    comprehensiveness: str,
    out_dir: Any,
    assembly_provider: Any | None,
    indexing_cfg: Any | None,
) -> tuple[str, list[str]]:
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
    services: Any,
    embedding_manager: Any,
    llm_manager: Any,
    target_dir: Any,
    scope_path: Any,
    scope_label: str,
    path_filter: str | None,
    comprehensiveness: str,
    max_points: int,
    out_dir: Any,
    assembly_provider: Any | None,
    indexing_cfg: Any | None,
    progress: Any,
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

    poi_sections: list[tuple[str, dict[str, Any]]] = []
    for idx, poi in enumerate(points_of_interest, start=1):
        heading = _derive_heading_from_point(poi)
        if log_info:
            log_info(
                f"[Code Mapper] Processing point of interest {idx}/"
                f"{len(points_of_interest)}: {heading}"
            )

        section_query = (
            "Expand the following point of interest into a detailed, "
            "agent-facing documentation section for the scoped folder "
            f"'{scope_label}'. Explain how the relevant code and "
            "configuration implement this behavior, including "
            "responsibilities, key types, "
            "important flows, and operational constraints.\n\n"
            "Point of interest:\n"
            f"{poi}\n\n"
            "Use markdown headings and bullet lists as needed. It is "
            "acceptable for this section to be long and detailed as "
            "long as it remains "
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
        except Exception as exc:
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
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        total_files_global=total_files_global,
        total_chunks_global=total_chunks_global,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
    )
