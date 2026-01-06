from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from chunkhound.code_mapper.hyde import build_hyde_scope_prompt, run_hyde_only_query
from chunkhound.code_mapper.models import AgentDocMetadata, CodeMapperPOI, HydeConfig
from chunkhound.code_mapper.scope import collect_scope_files
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.text import safe_scope_label


class CodeMapperHyDEError(RuntimeError):
    """Raised when HyDE planning fails (distinct from empty PoI results)."""

    def __init__(self, hyde_message: str) -> None:
        super().__init__(hyde_message)
        self.hyde_message = hyde_message


def _extract_points_of_interest(text: str, max_points: int = 10) -> list[str]:
    """Extract up to max_points points of interest from a markdown list."""
    seen: set[str] = set()
    unique_points: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        candidate = ""
        # Numbered list: "1. heading ..." or "1) heading ..."
        if stripped[0].isdigit():
            idx = stripped.find(".")
            if idx == -1:
                idx = stripped.find(")")
            if idx != -1:
                candidate = stripped[idx + 1 :].strip()

        # Bullet list: "- text" or "* text"
        if not candidate and (stripped.startswith("- ") or stripped.startswith("* ")):
            candidate = stripped[2:].strip()

        if not candidate:
            continue

        if candidate not in seen:
            seen.add(candidate)
            unique_points.append(candidate)

        if len(unique_points) >= max_points:
            break

    return unique_points[:max_points]


def _coverage_summary_lines(
    *,
    referenced_files: int,
    referenced_chunks: int,
    files_denominator: int | None,
    chunks_denominator: int | None,
    scope_total_files: int,
    scope_total_chunks: int,
) -> list[str]:
    lines: list[str] = ["## Coverage Summary", ""]

    if files_denominator and files_denominator > 0:
        file_cov = (referenced_files / files_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_files else "this database"
        lines.append(
            f"- Referenced files: {referenced_files} / {files_denominator} "
            f"({file_cov:.2f}% of indexed files in {scope_label_display})."
        )
    else:
        lines.append(
            f"- Referenced files: {referenced_files} (database totals unavailable)."
        )

    if chunks_denominator and chunks_denominator > 0:
        chunk_cov = (referenced_chunks / chunks_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_chunks else "this database"
        lines.append(
            f"- Referenced chunks: {referenced_chunks} / {chunks_denominator} "
            f"({chunk_cov:.2f}% of indexed chunks in {scope_label_display})."
        )
    else:
        lines.append(
            f"- Referenced chunks: {referenced_chunks} (database totals unavailable)."
        )

    return lines


def _operational_poi_budget(comprehensiveness: str) -> int:
    if comprehensiveness == "minimal":
        return 1
    if comprehensiveness == "low":
        return 2
    if comprehensiveness == "medium":
        return 3
    if comprehensiveness == "high":
        return 4
    if comprehensiveness == "ultra":
        return 5
    return 3


def _ensure_operational_quickstart(points: list[str], max_points: int) -> list[str]:
    normalized = [p.strip().lower() for p in points if p.strip()]
    for item in normalized:
        if (
            "quickstart" in item
            or "getting started" in item
            or "local run" in item
            or "run locally" in item
        ):
            return points[:max_points]

    injected = (
        "**Quickstart / Local run**: How to install, configure, and run this "
        "project end-to-end in a local development environment."
    )
    return [injected, *points][:max_points]


async def run_code_mapper_overview_hyde(
    llm_manager: LLMManager | None,
    target_dir: Path,
    scope_path: Path,
    scope_label: str,
    meta: AgentDocMetadata | None = None,
    context: str | None = None,
    max_points: int = 10,
    comprehensiveness: str = "medium",
    out_dir: Path | None = None,
    persist_prompt: bool = False,
    map_hyde_provider: LLMProvider | None = None,
    indexing_cfg: IndexingConfig | None = None,
) -> tuple[str, list[CodeMapperPOI]]:
    """Run a HyDE-style overview pass to identify points of interest."""
    hyde_cfg = HydeConfig.from_env()

    # Code Mapper overview should stay well below Codex CLI argv/stdin limits.
    # Callers can override the snippet token budget explicitly via
    # CH_CODE_MAPPER_HYDE_SNIPPET_TOKENS; otherwise, we map the CLI
    # comprehensiveness level to a budget that controls how much of the
    # *code* is sampled while keeping the file list at full scope.
    override_tokens = os.getenv("CH_CODE_MAPPER_HYDE_SNIPPET_TOKENS")
    if override_tokens:
        try:
            parsed = int(override_tokens)
            if parsed > 0:
                hyde_cfg.max_snippet_tokens = parsed
        except ValueError:
            pass
    else:
        # Map comprehensiveness to a proportion of the underlying HyDE snippet
        # budget. This only affects how much *code* is sampled for planning,
        # not which files are considered in scope.
        if comprehensiveness == "minimal":
            target_tokens = 2_000
        elif comprehensiveness == "low":
            target_tokens = 10_000
        elif comprehensiveness == "medium":
            target_tokens = 20_000
        elif comprehensiveness == "high":
            target_tokens = 35_000
        elif comprehensiveness == "ultra":
            target_tokens = 50_000
        else:
            target_tokens = 20_000

        if hyde_cfg.max_snippet_tokens > target_tokens:
            hyde_cfg.max_snippet_tokens = target_tokens

    # Hard cap the HyDE file list. This keeps the scope prompt readable and
    # prevents huge repos from spending most of the context window on file paths.
    if comprehensiveness == "minimal":
        scope_file_cap = 200
    elif comprehensiveness == "low":
        scope_file_cap = 500
    elif comprehensiveness == "medium":
        scope_file_cap = 2000
    elif comprehensiveness == "high":
        scope_file_cap = 3000
    elif comprehensiveness == "ultra":
        scope_file_cap = 5000
    else:
        scope_file_cap = 2000

    # Respect an explicit env override for CH_AGENT_DOC_HYDE_MAX_SCOPE_FILES,
    # but still enforce the hard cap.
    if os.getenv("CH_AGENT_DOC_HYDE_MAX_SCOPE_FILES"):
        if hyde_cfg.max_scope_files > scope_file_cap:
            hyde_cfg.max_scope_files = scope_file_cap
    else:
        hyde_cfg.max_scope_files = scope_file_cap

    include_patterns = None
    indexing_excludes = None
    ignore_sources = None
    gitignore_backend = "python"
    workspace_root_only_gitignore: bool | None = None
    try:
        if indexing_cfg is not None:
            include_patterns = list(getattr(indexing_cfg, "include", None) or [])
            get_exc = getattr(indexing_cfg, "get_effective_config_excludes", None)
            if callable(get_exc):
                indexing_excludes = list(get_exc())
            get_sources = getattr(indexing_cfg, "resolve_ignore_sources", None)
            if callable(get_sources):
                ignore_sources = list(get_sources())
            gitignore_backend = str(
                getattr(indexing_cfg, "gitignore_backend", "python")
            )
            workspace_root_only_gitignore = getattr(
                indexing_cfg, "workspace_gitignore_nonrepo", None
            )
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug(f"Code Mapper: failed to read indexing config overrides: {exc}")
        include_patterns = None
        indexing_excludes = None
        ignore_sources = None
        gitignore_backend = "python"
        workspace_root_only_gitignore = None

    context_text = context.strip() if context is not None else ""

    file_paths: list[str] = []
    if not context_text:
        file_paths = collect_scope_files(
            scope_path=scope_path,
            project_root=target_dir,
            hyde_cfg=hyde_cfg,
            include_patterns=include_patterns,
            indexing_excludes=indexing_excludes,
            ignore_sources=ignore_sources,
            gitignore_backend=gitignore_backend,
            workspace_root_only_gitignore=workspace_root_only_gitignore,
        )

    prompt_meta = meta or AgentDocMetadata(
        created_from_sha="CODE_MAPPER",
        previous_target_sha="CODE_MAPPER",
        target_sha="CODE_MAPPER",
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_config={},
        generation_stats={"overview_mode": "hyde_scope_only"},
    )

    hyde_scope_prompt = build_hyde_scope_prompt(
        meta=prompt_meta,
        scope_label=scope_label,
        file_paths=file_paths,
        hyde_cfg=hyde_cfg,
        context=context_text or None,
        project_root=target_dir,
        mode="architectural",
    )

    ops_hyde_scope_prompt = build_hyde_scope_prompt(
        meta=prompt_meta,
        scope_label=scope_label,
        file_paths=file_paths,
        hyde_cfg=hyde_cfg,
        context=context_text or None,
        project_root=target_dir,
        mode="operational",
    )

    ops_budget = _operational_poi_budget(comprehensiveness)

    def _build_overview_prompt(*, scope_prompt: str, mode: str, budget: int) -> str:
        focus = (
            "architectural areas first"
            if mode == "architectural"
            else "operational workflows (setup, local run, troubleshooting) first"
        )
        poi_target_line = (
            f"- Identify up to {budget} points of interest for this scoped project. "
            f"Prioritize the most important {focus}, but you may include slightly "
            "less critical topics to use the full budget when appropriate.\n"
        )
        context_guard = ""
        if context_text:
            context_guard = (
                "- Use ONLY the user-provided context above; do not infer or assume "
                "details from repository code.\n"
            )
        return (
            f"{scope_prompt}\n\n"
            "HyDE objective (override for Code Mapper):\n"
            "- Ignore any earlier 'HyDE objective' and 'Output format' instructions "
            "in the scope prompt.\n"
            f"{context_guard}"
            "- Instead of writing a full documentation, do a concise planning pass "
            "for deep code research.\n"
            f"{poi_target_line}\n"
            "Output format:\n"
            "- Produce ONLY a numbered markdown list (1., 2., 3., ...).\n"
            "- Each item MUST follow this exact shape:\n"
            "  - `N. **Short Title** — 1–2 sentences. Key files: `path`, `path` "
            "(optional).`\n"
            "- Requirements for `**Short Title**`:\n"
            "  - 3–8 words (max 60 characters).\n"
            "  - MUST be human-readable and general (no file paths, no module "
            "names-as-titles).\n"
            "  - MUST NOT include backticks, parentheses, slashes, or file "
            "extensions like `.py`.\n"
            "- If you include key files:\n"
            "  - Put them after the 1–2 sentence summary as `Key files: ...`.\n"
            "  - Use backticks around each path and include at most 3 files.\n"
            "- Do not include any other sections or prose; just the numbered list.\n"
        )

    arch_overview_prompt = _build_overview_prompt(
        scope_prompt=hyde_scope_prompt,
        mode="architectural",
        budget=max_points,
    )
    ops_overview_prompt = _build_overview_prompt(
        scope_prompt=ops_hyde_scope_prompt,
        mode="operational",
        budget=ops_budget,
    )

    # Optional debugging/traceability: when out_dir is provided and explicitly
    # enabled via CH_CODE_MAPPER_WRITE_HYDE_PROMPT, persist the exact PoI-generation
    # prompt (scope + HyDE objective).
    write_prompt = persist_prompt or (
        os.getenv("CH_CODE_MAPPER_WRITE_HYDE_PROMPT", "").strip().lower()
        in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
    )
    if out_dir is not None and write_prompt:
        try:
            safe_scope = safe_scope_label(scope_label)
            arch_prompt_path = out_dir / f"hyde_scope_prompt_arch_{safe_scope}.md"
            arch_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            arch_prompt_path.write_text(arch_overview_prompt, encoding="utf-8")

            ops_prompt_path = out_dir / f"hyde_scope_prompt_ops_{safe_scope}.md"
            ops_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            ops_prompt_path.write_text(ops_overview_prompt, encoding="utf-8")
        except OSError as exc:
            logger.debug(f"Code Mapper: failed to persist HyDE prompt: {exc}")

    arch_overview_answer, ok = await run_hyde_only_query(
        llm_manager=llm_manager,
        prompt=arch_overview_prompt,
        provider_override=map_hyde_provider,
        hyde_cfg=hyde_cfg,
    )
    if not ok:
        raise CodeMapperHyDEError(arch_overview_answer)

    ops_overview_answer, ok = await run_hyde_only_query(
        llm_manager=llm_manager,
        prompt=ops_overview_prompt,
        provider_override=map_hyde_provider,
        hyde_cfg=hyde_cfg,
    )
    if not ok:
        raise CodeMapperHyDEError(ops_overview_answer)

    # Persist the HyDE overview plan itself (the PoI list and any surrounding
    # context) alongside the prompt when an output directory is available.
    if out_dir is not None and arch_overview_answer and arch_overview_answer.strip():
        try:
            safe_scope = safe_scope_label(scope_label)
            arch_plan_path = out_dir / f"hyde_plan_arch_{safe_scope}.md"
            arch_plan_path.parent.mkdir(parents=True, exist_ok=True)
            arch_plan_path.write_text(arch_overview_answer, encoding="utf-8")
        except OSError as exc:
            logger.debug(f"Code Mapper: failed to persist HyDE plan: {exc}")

    if out_dir is not None and ops_overview_answer and ops_overview_answer.strip():
        try:
            safe_scope = safe_scope_label(scope_label)
            ops_plan_path = out_dir / f"hyde_plan_ops_{safe_scope}.md"
            ops_plan_path.parent.mkdir(parents=True, exist_ok=True)
            ops_plan_path.write_text(ops_overview_answer, encoding="utf-8")
        except OSError as exc:
            logger.debug(f"Code Mapper: failed to persist HyDE plan: {exc}")

    arch_points = _extract_points_of_interest(
        arch_overview_answer, max_points=max_points
    )
    ops_points = _extract_points_of_interest(ops_overview_answer, max_points=ops_budget)
    ops_points = _ensure_operational_quickstart(ops_points, ops_budget)

    combined_overview_answer = (
        "\n".join(
            [
                "## Architectural Map (HyDE)",
                "",
                arch_overview_answer.strip(),
                "",
                "## Operational Map (HyDE)",
                "",
                ops_overview_answer.strip(),
                "",
            ]
        ).strip()
        + "\n"
    )

    points_of_interest: list[CodeMapperPOI] = [
        *[CodeMapperPOI(mode="architectural", text=p) for p in arch_points],
        *[CodeMapperPOI(mode="operational", text=p) for p in ops_points],
    ]
    return combined_overview_answer, points_of_interest
