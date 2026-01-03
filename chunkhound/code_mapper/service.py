from __future__ import annotations

import asyncio
import os
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.code_mapper.coverage import compute_db_scope_stats
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
    poi_sections: list[tuple[str, dict[str, Any]]]
    poi_sections_indexed: list[tuple[int, str, dict[str, Any]]]
    failed_poi_sections: list[tuple[int, str, str]]
    total_points_of_interest: int
    unified_source_files: dict[str, str]
    unified_chunks_dedup: list[dict[str, Any]]
    total_files_global: int | None
    total_chunks_global: int | None
    scope_total_files: int
    scope_total_chunks: int


def _resolve_poi_concurrency(total_points: int) -> int:
    raw = os.getenv("CH_CODE_MAPPER_POI_CONCURRENCY", "").strip()
    if raw:
        try:
            parsed = int(raw)
            if parsed < 1:
                return 1
            return min(parsed, max(total_points, 1))
        except ValueError:
            return 1
    if total_points <= 1:
        return 1
    return min(4, total_points)


class _PoiProgressProxy:
    def __init__(
        self,
        progress: TreeProgressDisplay,
        *,
        depth_offset: int,
        node_id_offset: int,
    ) -> None:
        self._progress = progress
        self._depth_offset = depth_offset
        self._node_id_offset = node_id_offset

    async def emit_event(
        self,
        event_type: str,
        message: str,
        node_id: int | None = None,
        depth: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if depth is None:
            mapped_depth = self._depth_offset
        else:
            mapped_depth = depth + self._depth_offset
        mapped_node_id = None if node_id is None else self._node_id_offset + node_id

        await self._progress.emit_event(
            event_type,
            message,
            node_id=mapped_node_id,
            depth=mapped_depth,
            metadata=metadata,
        )


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
    poi_jobs: int | None,
    progress: TreeProgressDisplay | None,
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
    if poi_jobs is not None:
        if poi_jobs < 1:
            raise ValueError("poi_jobs must be >= 1")
        poi_concurrency = min(poi_jobs, max(total_points_of_interest, 1))
    else:
        poi_concurrency = _resolve_poi_concurrency(total_points_of_interest)
    if log_info and poi_concurrency > 1:
        log_info(
            "[Code Mapper] Running PoI deep research with "
            f"concurrency={poi_concurrency}"
        )
    if log_warning and poi_concurrency >= 8:
        log_warning(
            "[Code Mapper] High PoI concurrency may overwhelm your LLM provider. "
            f"jobs={poi_concurrency}"
        )

    def _failure_markdown(
        *,
        idx: int,
        poi: str,
        heading: str,
        first_error: str,
        retry_error: str | None,
    ) -> str:
        lines: list[str] = [f"# {heading} (failed)", ""]
        lines.append(
            "This point of interest failed to generate content after a retry."
        )
        lines.append("")
        lines.append(f"- Point of interest ({idx}/{total_points_of_interest}): {poi}")
        lines.append(f"- First attempt: {first_error}")
        if retry_error is not None:
            lines.append(f"- Retry attempt: {retry_error}")
        lines.append("")
        return "\n".join(lines)

    backoff_to_serial = asyncio.Event()
    pending = deque(
        [
            (idx, poi, _derive_heading_from_point(poi))
            for idx, poi in enumerate(points_of_interest, start=1)
        ]
    )
    pending_lock = asyncio.Lock()
    successful_sections: list[tuple[int, str, dict[str, Any]]] = []
    retry_candidates: dict[int, tuple[str, str, str]] = {}
    failed_poi_sections: list[tuple[int, str, str]] = []

    def _poi_node_id(idx: int) -> int:
        return idx * 1_000_000

    def _poi_progress(idx: int) -> Any:
        if progress is None:
            return None
        poi_id = _poi_node_id(idx)
        return _PoiProgressProxy(
            progress,
            depth_offset=1,
            node_id_offset=poi_id + 1,
        )

    async def _emit_poi_start(idx: int, heading: str) -> None:
        if progress is None:
            return
        await progress.emit_event(
            "poi_start",
            f"PoI {idx}/{total_points_of_interest}: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _emit_poi_complete(idx: int, heading: str) -> None:
        if progress is None:
            return
        await progress.emit_event(
            "poi_complete",
            f"PoI {idx}/{total_points_of_interest} complete: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _emit_poi_failed(idx: int, heading: str) -> None:
        if progress is None:
            return
        await progress.emit_event(
            "poi_failed",
            f"PoI {idx}/{total_points_of_interest} failed: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _next_pending() -> tuple[int, str, str] | None:
        async with pending_lock:
            if not pending:
                return None
            return pending.popleft()

    async def _run_point_once(
        *,
        idx: int,
        poi: str,
        heading: str,
        poi_progress: Any,
    ) -> tuple[str, dict[str, Any] | None, str | None, bool]:
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
            if log_info:
                log_info(
                    f"[Code Mapper] Processing point of interest {idx}/"
                    f"{len(points_of_interest)}: {heading}"
                )
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=section_query,
                progress=poi_progress,
                path=path_filter,
            )
            if _is_empty_research_result(result):
                if log_warning:
                    log_warning(
                        f"[Code Mapper] Point of interest {idx} returned no usable "
                        "content (will retry)."
                    )
                return heading, None, "empty result", False
            return heading, result, None, False
        except Exception as exc:
            if log_error:
                log_error(f"Code Mapper deep research failed for point {idx}: {exc}")
            logger.exception(f"Code Mapper deep research failed for point {idx}.")
            return heading, None, f"{type(exc).__name__}: {exc}", True

    async def _worker(worker_id: int) -> None:
        while True:
            if backoff_to_serial.is_set() and worker_id != 0:
                return

            item = await _next_pending()
            if item is None:
                return
            idx, poi, heading = item
            await _emit_poi_start(idx, heading)
            poi_progress = _poi_progress(idx)
            heading, result, error_summary, should_backoff = await _run_point_once(
                idx=idx,
                poi=poi,
                heading=heading,
                poi_progress=poi_progress,
            )
            if result is not None:
                successful_sections.append((idx, poi, result))
                await _emit_poi_complete(idx, heading)
                continue

            if should_backoff:
                backoff_to_serial.set()
            retry_candidates[idx] = (poi, heading, error_summary or "unknown error")

    workers = [asyncio.create_task(_worker(i)) for i in range(poi_concurrency)]
    await asyncio.gather(*workers)

    if retry_candidates:
        for idx in sorted(retry_candidates.keys()):
            poi, heading, first_error = retry_candidates[idx]
            poi_progress = _poi_progress(idx)
            if poi_progress is not None:
                try:
                    await poi_progress.emit_event(
                        "main_info",
                        "Retrying after error",
                        node_id=None,
                        depth=0,
                    )
                except Exception:
                    pass
            retry_delay = 1.0
            await asyncio.sleep(random.uniform(0.0, retry_delay))
            retry_heading, retry_result, retry_error, _should_backoff = (
                await _run_point_once(
                    idx=idx,
                    poi=poi,
                    heading=heading,
                    poi_progress=poi_progress,
                )
            )
            if retry_result is not None:
                successful_sections.append((idx, poi, retry_result))
                await _emit_poi_complete(idx, heading)
                continue

            failed_poi_sections.append(
                (
                    idx,
                    poi,
                    _failure_markdown(
                        idx=idx,
                        poi=poi,
                        heading=retry_heading or heading,
                        first_error=first_error,
                        retry_error=retry_error,
                    ),
                )
            )
            await _emit_poi_failed(idx, heading)

    successful_sections.sort(key=lambda item: item[0])
    poi_sections_indexed = [
        (idx, poi, result) for idx, poi, result in successful_sections
    ]
    poi_sections = [(poi, result) for _, poi, result in successful_sections]

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
        poi_sections_indexed=poi_sections_indexed,
        failed_poi_sections=sorted(failed_poi_sections, key=lambda item: item[0]),
        total_points_of_interest=total_points_of_interest,
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        total_files_global=total_files_global,
        total_chunks_global=total_chunks_global,
        scope_total_files=scope_total_files,
        scope_total_chunks=scope_total_chunks,
    )
