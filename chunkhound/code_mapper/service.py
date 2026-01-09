from __future__ import annotations

import asyncio
import os
import random
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.code_mapper.coverage import compute_db_scope_stats
from chunkhound.code_mapper.models import AgentDocMetadata, CodeMapperPOI
from chunkhound.code_mapper.pipeline import run_code_mapper_overview_hyde
from chunkhound.code_mapper.public_utils import (
    derive_heading_from_point,
    is_empty_research_result,
    merge_sources_metadata,
)
from chunkhound.core.audience import normalize_audience
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.database_factory import DatabaseServices
from chunkhound.embeddings import EmbeddingManager
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.services.deep_research_service import run_deep_research

if TYPE_CHECKING:
    from chunkhound.api.cli.utils.tree_progress import TreeProgressDisplay


class CodeMapperNoPointsError(RuntimeError):
    """Raised when Code Mapper overview produces no points of interest."""

    def __init__(self, overview_answer: str) -> None:
        super().__init__("Code Mapper overview produced no points of interest.")
        self.overview_answer = overview_answer


class CodeMapperInvalidConcurrencyError(ValueError):
    """Raised when concurrency configuration is invalid."""


@dataclass
class CodeMapperPipelineResult:
    overview_result: dict[str, Any]
    poi_sections: list[tuple[CodeMapperPOI, dict[str, Any]]]
    poi_sections_indexed: list[tuple[int, CodeMapperPOI, dict[str, Any]]]
    failed_poi_sections: list[tuple[int, CodeMapperPOI, str]]
    total_points_of_interest: int
    unified_source_files: dict[str, str]
    unified_chunks_dedup: list[dict[str, Any]]
    total_files_global: int | None
    total_chunks_global: int | None
    scope_total_files: int
    scope_total_chunks: int


def _audience_guidance_lines(*, audience: str) -> list[str]:
    normalized = normalize_audience(audience)
    if normalized == "technical":
        return [
            "Audience: technical (software engineers).",
            "Prefer implementation details, key types, and precise terminology.",
        ]
    if normalized == "end-user":
        return [
            "Audience: end-user (less technical).",
            (
                "Prefer practical workflows and plain language; explain code "
                "identifiers briefly when needed."
            ),
            (
                "De-emphasize internal implementation details unless essential to user "
                "outcomes."
            ),
        ]
    return []


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        key = id(current)
        if key in seen:
            break
        seen.add(key)
        yield current
        current = current.__cause__ or current.__context__


_RETRYABLE_POI_ERROR_SUBSTRINGS = (
    "llm completion failed",
    "llm structured completion failed",
    "llm returned empty response",
    "llm response truncated",
    "token limit",
    "rate limit",
    "429",
    "timeout",
    "timed out",
    "overloaded",
    "content filter",
    "responses api",
)


def _is_retryable_poi_error(exc: Exception) -> bool:
    for chained in _iter_exception_chain(exc):
        if isinstance(chained, (asyncio.TimeoutError, TimeoutError, OSError)):
            return True

    message = str(exc).strip().lower()
    if not message:
        return False
    return any(token in message for token in _RETRYABLE_POI_ERROR_SUBSTRINGS)


def _resolve_poi_concurrency(total_points: int) -> int:
    raw = os.getenv("CH_CODE_MAPPER_POI_CONCURRENCY", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            raise CodeMapperInvalidConcurrencyError(
                "CH_CODE_MAPPER_POI_CONCURRENCY must be an integer >= 1."
            ) from None
        if parsed < 1:
            raise CodeMapperInvalidConcurrencyError(
                "CH_CODE_MAPPER_POI_CONCURRENCY must be >= 1."
            )
        return min(parsed, max(total_points, 1))
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
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        self._progress = progress
        self._depth_offset = depth_offset
        self._node_id_offset = node_id_offset
        self._on_error = on_error

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

        try:
            await self._progress.emit_event(
                event_type,
                message,
                node_id=mapped_node_id,
                depth=mapped_depth,
                metadata=metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(event_type, exc)
            else:
                logger.debug(
                    "[Code Mapper] Progress emit failed for event_type="
                    f"{event_type}: {exc}"
                )


async def run_code_mapper_overview_only(
    *,
    llm_manager: LLMManager | None,
    target_dir: Path,
    scope_path: Path,
    scope_label: str,
    meta: AgentDocMetadata | None = None,
    context: str | None = None,
    max_points: int,
    comprehensiveness: str,
    out_dir: Path | None,
    map_hyde_provider: LLMProvider | None,
    indexing_cfg: IndexingConfig | None,
) -> tuple[str, list[CodeMapperPOI]]:
    """Run overview-only Code Mapper and return the answer + points."""
    overview_answer, points_of_interest = await run_code_mapper_overview_hyde(
        llm_manager=llm_manager,
        target_dir=target_dir,
        scope_path=scope_path,
        scope_label=scope_label,
        meta=meta,
        context=context,
        max_points=max_points,
        comprehensiveness=comprehensiveness,
        out_dir=out_dir,
        persist_prompt=True,
        map_hyde_provider=map_hyde_provider,
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
    meta: AgentDocMetadata | None = None,
    context: str | None = None,
    comprehensiveness: str,
    max_points: int,
    out_dir: Path | None,
    map_hyde_provider: LLMProvider | None = None,
    indexing_cfg: IndexingConfig | None = None,
    poi_jobs: int | None = None,
    progress: TreeProgressDisplay | None = None,
    audience: str = "balanced",
    log_info: Callable[[str], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
    log_error: Callable[[str], None] | None = None,
) -> CodeMapperPipelineResult:
    """Run Code Mapper overview + per-point deep research and compute coverage."""
    overview_answer, points_of_interest = await run_code_mapper_overview_hyde(
        llm_manager=llm_manager,
        target_dir=target_dir,
        scope_path=scope_path,
        scope_label=scope_label,
        meta=meta,
        context=context,
        max_points=max_points,
        comprehensiveness=comprehensiveness,
        out_dir=out_dir,
        map_hyde_provider=map_hyde_provider,
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

    audience_lines = _audience_guidance_lines(audience=audience)
    audience_block = (
        ("\n".join(f"- {line}" for line in audience_lines) + "\n\n")
        if audience_lines
        else ""
    )

    progress_failures: set[str] = set()

    def _on_progress_error(event_type: str, exc: Exception) -> None:
        if event_type in progress_failures:
            return
        progress_failures.add(event_type)
        logger.debug(
            "[Code Mapper] Progress emit failed for event_type="
            f"{event_type}: {exc}"
        )

    async def _safe_progress_emit(
        event_type: str,
        message: str,
        *,
        node_id: int | None = None,
        depth: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if progress is None:
            return
        try:
            await progress.emit_event(
                event_type,
                message,
                node_id=node_id,
                depth=depth,
                metadata=metadata,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _on_progress_error(event_type, exc)

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
        poi: CodeMapperPOI,
        heading: str,
        first_error: str,
        retry_error: str | None,
    ) -> str:
        lines: list[str] = [f"# {heading} (failed)", ""]
        lines.append(
            "This point of interest failed to generate content after a retry."
        )
        lines.append("")
        lines.append(
            f"- Point of interest ({idx}/{total_points_of_interest}): {poi.text}"
        )
        lines.append(f"- First attempt: {first_error}")
        if retry_error is not None:
            lines.append(f"- Retry attempt: {retry_error}")
        lines.append("")
        return "\n".join(lines)

    backoff_to_serial = asyncio.Event()
    pending = deque(
        [
            (idx, poi, derive_heading_from_point(poi.text))
            for idx, poi in enumerate(points_of_interest, start=1)
        ]
    )
    pending_lock = asyncio.Lock()
    pending_cond = asyncio.Condition(pending_lock)
    in_flight = 0
    successful_sections: list[tuple[int, CodeMapperPOI, dict[str, Any]]] = []
    retry_candidates: dict[int, tuple[CodeMapperPOI, str, str]] = {}
    failed_poi_sections: list[tuple[int, CodeMapperPOI, str]] = []

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
            on_error=_on_progress_error,
        )

    async def _emit_poi_start(idx: int, heading: str) -> None:
        await _safe_progress_emit(
            "poi_start",
            f"PoI {idx}/{total_points_of_interest}: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _emit_poi_complete(idx: int, heading: str) -> None:
        await _safe_progress_emit(
            "poi_complete",
            f"PoI {idx}/{total_points_of_interest} complete: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _emit_poi_failed(idx: int, heading: str) -> None:
        await _safe_progress_emit(
            "poi_failed",
            f"PoI {idx}/{total_points_of_interest} failed: {heading}",
            node_id=_poi_node_id(idx),
            depth=0,
        )

    async def _next_pending(worker_id: int) -> tuple[int, CodeMapperPOI, str] | None:
        nonlocal in_flight
        async with pending_cond:
            while True:
                if not pending:
                    return None
                if backoff_to_serial.is_set():
                    if worker_id != 0:
                        return None
                    while in_flight > 0:
                        await pending_cond.wait()
                item = pending.popleft()
                in_flight += 1
                return item

    async def _mark_item_done() -> None:
        nonlocal in_flight
        async with pending_cond:
            in_flight -= 1
            if in_flight == 0:
                pending_cond.notify_all()

    async def _run_point_once(
        *,
        idx: int,
        poi: CodeMapperPOI,
        heading: str,
        poi_progress: Any,
    ) -> tuple[str, dict[str, Any] | None, str | None, bool]:
        poi_text = poi.text
        if poi.mode == "operational":
            section_query = (
                "Expand the following OPERATIONAL point of interest into a "
                "detailed, operator/runbook-style documentation section for the "
                f"scoped folder '{scope_label}'.\n\n"
                f"{audience_block}"
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
                f"{audience_block}"
                "Point of interest:\n"
                f"{poi_text}\n\n"
                "Use markdown headings and bullet lists as needed. It is acceptable "
                "for this section to be long and detailed as long as it remains "
                "grounded in the code."
            )

        try:
            if log_info:
                log_info(
                    f"[Code Mapper] Processing point of interest {idx}/"
                    f"{len(points_of_interest)}: {heading}"
                )
            result = await run_deep_research(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=section_query,
                tool_name="code_research",
                progress=poi_progress,
                path=path_filter,
            )
            if is_empty_research_result(result):
                if log_warning:
                    log_warning(
                        f"[Code Mapper] Point of interest {idx} returned no usable "
                        "content (will retry)."
                    )
                return heading, None, "empty result", False
            return heading, result, None, False
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not _is_retryable_poi_error(exc):
                raise

            if log_error:
                log_error(f"Code Mapper deep research failed for point {idx}: {exc}")
            logger.exception(f"Code Mapper deep research failed for point {idx}.")
            return heading, None, f"{type(exc).__name__}: {exc}", True

    async def _process_item(idx: int, poi: CodeMapperPOI, heading: str) -> None:
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
            return

        if should_backoff:
            backoff_to_serial.set()
        retry_candidates[idx] = (poi, heading, error_summary or "unknown error")

    async def _worker(worker_id: int) -> None:
        while True:
            item = await _next_pending(worker_id)
            if item is None:
                return
            idx, poi, heading = item
            try:
                await _process_item(idx, poi, heading)
            finally:
                await _mark_item_done()

    workers = [asyncio.create_task(_worker(i)) for i in range(poi_concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker in workers:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

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
                except asyncio.CancelledError:
                    raise
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
    ) = merge_sources_metadata(all_results)

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
