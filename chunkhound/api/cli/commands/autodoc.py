"""Autodoc command module - generates scoped architecture/operations docs.

This command uses a two-phase pipeline:
1. Run a shallow deep-research call to identify points of interest for the
   requested scope (overview plan). The count depends on the chosen
   comprehensiveness setting.
2. For each point of interest, run a dedicated deep-research pass and assemble
   the results into a single flowing document, along with a simple coverage
   summary based on referenced files and chunks.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.api.cli.utils import verify_database_exists
from chunkhound.autodoc.coverage import compute_db_scope_stats
from chunkhound.autodoc.hyde import build_hyde_scope_prompt, run_hyde_only_query
from chunkhound.autodoc.llm import build_llm_metadata_and_assembly
from chunkhound.autodoc.metadata import build_generation_stats, format_metadata_block
from chunkhound.autodoc.models import AgentDocMetadata, HydeConfig
from chunkhound.autodoc.scope import collect_scope_files
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


def _get_head_sha(project_root: Path) -> str:
    """Return the current HEAD SHA for the project or a stable placeholder.

    Autodoc treats the workspace root as the project for metadata purposes.
    For non-git workspaces, we fall back to a neutral placeholder so that
    generation remains robust even when commit metadata is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        # Best-effort only; downstream logic treats this as "no git metadata".
        pass
    return "NO_GIT_HEAD"


def _compute_scope_label(target_dir: Path, scope_path: Path) -> str:
    """Compute a human-readable scope label relative to the target directory."""
    try:
        rel = scope_path.resolve().relative_to(target_dir.resolve())
        label = str(rel).replace(os.sep, "/")
        if not label or label == ".":
            return "/"
        return label
    except ValueError:
        # Scope is outside target_dir; fall back to basename
        return scope_path.name or "/"


def _compute_path_filter(target_dir: Path, scope_path: Path) -> str | None:
    """Compute a database path filter relative to the target directory.

    Returns:
        Relative POSIX-style path string suitable for ChunkHound path filters,
        or None to indicate no additional scoping.
    """
    try:
        rel = scope_path.resolve().relative_to(target_dir.resolve())
    except ValueError:
        return None

    rel_str = str(rel).replace(os.sep, "/")
    if not rel_str or rel_str == ".":
        return None
    return rel_str


def _extract_points_of_interest(text: str, max_points: int = 10) -> list[str]:
    """Extract up to max_points points of interest from a markdown list.

    The overview deep-research call is instructed to return a numbered list,
    but this helper is defensive and also handles bullet lists.
    """
    points: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # Numbered list: "1. heading ..." or "1) heading ..."
        if stripped[0].isdigit():
            idx = stripped.find(".")
            if idx == -1:
                idx = stripped.find(")")
            if idx != -1:
                candidate = stripped[idx + 1 :].strip()
                if candidate:
                    points.append(candidate)
                    continue

        # Bullet list: "- text" or "* text"
        if stripped.startswith("- ") or stripped.startswith("* "):
            candidate = stripped[2:].strip()
            if candidate:
                points.append(candidate)

        if len(points) >= max_points:
            break

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_points: list[str] = []
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)

    # Ensure we have at most max_points
    return unique_points[:max_points]


async def _run_autodoc_overview_hyde(
    llm_manager: LLMManager | None,
    target_dir: Path,
    scope_path: Path,
    scope_label: str,
    max_points: int = 10,
    comprehensiveness: str = "medium",
    out_dir: Path | None = None,
    assembly_provider: Any | None = None,
    indexing_cfg: Any | None = None,
) -> tuple[str, list[str]]:
    """Run a HyDE-style overview pass to identify points of interest.

    This reuses the same HyDE scope prompt machinery as the standalone
    deep_doc pipeline but narrows the objective to a numbered list of
    points of interest instead of a full documentation draft.
    """
    hyde_cfg = HydeConfig.from_env()

    # Autodoc overview should stay well below Codex CLI argv/stdin limits.
    # Callers can override the snippet token budget explicitly via
    # CH_AUTODOC_HYDE_SNIPPET_TOKENS; otherwise, we map the CLI
    # comprehensiveness level to a budget that controls how much of the
    # *code* is sampled while keeping the file list at full scope.
    override_tokens = os.getenv("CH_AUTODOC_HYDE_SNIPPET_TOKENS")
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
    except Exception:
        include_patterns = None
        indexing_excludes = None
        ignore_sources = None
        gitignore_backend = "python"
        workspace_root_only_gitignore = None

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

    meta = AgentDocMetadata(
        created_from_sha="AUTODOC",
        previous_target_sha="AUTODOC",
        target_sha="AUTODOC",
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_config={},
        generation_stats={"overview_mode": "hyde_scope_only"},
    )

    hyde_scope_prompt = build_hyde_scope_prompt(
        meta=meta,
        scope_label=scope_label,
        file_paths=file_paths,
        hyde_cfg=hyde_cfg,
        project_root=target_dir,
    )

    # Provide an explicit, comprehensiveness-aware target for the number of
    # points of interest so HyDE can bias its list length accordingly. We use
    # max_points directly as the upper bound, but allow the model to include
    # slightly less critical items if needed to reach that number.
    poi_target_line = (
        f"- Identify up to {max_points} points of interest for this scoped project. "
        "Prioritize the most important architectural or operational areas first, "
        "but you may include slightly less critical topics to use the full budget "
        "when appropriate.\n"
    )

    overview_prompt = (
        f"{hyde_scope_prompt}\n\n"
        "HyDE objective (override for autodoc):\n"
        "- Instead of writing a full deep-wiki style documentation, focus on a "
        "concise\n"
        "  planning pass for deep code research.\n"
        f"{poi_target_line}\n"
        "Output format:\n"
        "- Produce ONLY a numbered markdown list (1., 2., 3., ...).\n"
        "- For each item, include:\n"
        "  - A short heading (you may use bold), and\n"
        "  - 1–2 sentences summarizing why this area is important.\n"
        "- Do not include any other sections or prose; just the numbered list.\n"
    )

    # Optional debugging/traceability: when out_dir is provided, persist the
    # exact PoI-generation prompt (scope + HyDE objective) so that code_research
    # runs can be inspected alongside the generated autodoc topics.
    if out_dir is not None:
        try:
            safe_scope = scope_label.replace("/", "_") or "root"
            prompt_path = out_dir / f"hyde_scope_prompt_{safe_scope}.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(overview_prompt, encoding="utf-8")
        except Exception:
            # Prompt persistence is best-effort; never break main generation.
            pass

    overview_answer = await run_hyde_only_query(
        llm_manager=llm_manager,
        prompt=overview_prompt,
        provider_override=assembly_provider,
        hyde_cfg=hyde_cfg,
    )

    # Persist the HyDE overview plan itself (the PoI list and any surrounding
    # context) alongside the prompt when an output directory is available.
    if out_dir is not None and overview_answer and overview_answer.strip():
        try:
            safe_scope = scope_label.replace("/", "_") or "root"
            plan_path = out_dir / f"hyde_plan_{safe_scope}.md"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(overview_answer, encoding="utf-8")
        except Exception:
            # Plan persistence is also best-effort.
            pass

    points_of_interest = _extract_points_of_interest(
        overview_answer, max_points=max_points
    )
    return overview_answer, points_of_interest


def _derive_heading_from_point(point: str) -> str:
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


def _slugify_heading(heading: str) -> str:
    """Convert a heading into a filesystem-friendly slug."""
    text = heading.strip().lower()
    # Replace non-alphanumeric characters with dashes
    slug_chars: list[str] = []
    prev_dash = False
    for ch in text:
        if ch.isalnum():
            slug_chars.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                slug_chars.append("-")
                prev_dash = True
    slug = "".join(slug_chars).strip("-")
    if not slug:
        slug = "topic"
    if len(slug) > 60:
        slug = slug[:60].rstrip("-")
    return slug


def _merge_sources_metadata(
    results: list[dict[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]], int | None, int | None]:
    """Merge sources metadata from multiple deep-research calls.

    Returns:
        Tuple of (
            unified_source_files,
            unified_chunks_dedup,
            total_files_indexed,
            total_chunks_indexed,
        ) where:
        - unified_source_files is a dict keyed by file path (values unused),
        - unified_chunks_dedup is a list of unique chunk dicts with file/line info,
        - totals are best-effort database-wide counts taken from aggregation_stats.
    """
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
        # Use stats totals if they look plausible; this is best-effort only.
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


def _is_empty_research_result(result: dict[str, Any]) -> bool:
    """Return True when a deep-research result carries no useful content.

    This treats both structurally empty answers and the DeepResearchService
    \"no relevant code context\" guidance path as effectively empty so that
    autodoc does not emit placeholder sections for them.
    """
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


async def autodoc_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the autodoc command using deep code research."""
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Resolve workspace root from explicit config file when provided.
    # This supports a multi-project workspace where a single
    # /workspaces/.chunkhound.json and /workspaces/.chunkhound/db index
    # serve multiple subprojects (e.g., /workspaces/arguseek) while
    # autodoc scopes a specific folder.
    cfg_override: Path | None = None
    if getattr(args, "config", None):
        try:
            cfg_override = Path(args.config).expanduser().resolve()
        except Exception:
            cfg_override = None
    if cfg_override is None:
        config_file_env = os.getenv("CHUNKHOUND_CONFIG_FILE")
        if config_file_env:
            try:
                cfg_override = Path(config_file_env).expanduser().resolve()
            except Exception:
                cfg_override = None

    if cfg_override is not None:
        workspace_root = cfg_override.parent
        if workspace_root.exists():
            # For autodoc, treat the directory containing the config as the
            # workspace root so that scopes like "arguseek" line up with a shared
            # workspace index.
            config.target_dir = workspace_root

            # Do NOT override an explicitly configured database path.
            # Only fall back to the conventional workspace DB if the config file
            # didn't specify one and the CLI didn't override it.
            explicit_db_cli = bool(
                getattr(args, "db", None) or getattr(args, "database_path", None)
            )
            explicit_db_in_file = False
            try:
                import json

                raw = json.loads(cfg_override.read_text(encoding="utf-8"))
                db = raw.get("database") if isinstance(raw, dict) else None
                explicit_db_in_file = isinstance(db, dict) and bool(db.get("path"))
            except Exception:
                explicit_db_in_file = False

            if not explicit_db_cli and not explicit_db_in_file:
                config.database.path = workspace_root / ".chunkhound" / "db"

    # Autodoc always writes artifacts; keep the CLI contract explicit.
    out_dir_arg = getattr(args, "out_dir", None)
    if out_dir_arg is None:
        formatter.error(
            "Autodoc requires --out-dir so it can write an index and per-topic files."
        )
        sys.exit(2)

    # Overview-only mode should be lightweight: only HyDE planning + stdout,
    # plus best-effort prompt persistence under --out-dir.
    if getattr(args, "overview_only", False):
        try:
            out_dir = Path(out_dir_arg).resolve()
        except Exception:
            out_dir = None

        llm_manager: LLMManager | None = None
        try:
            if config.llm:
                utility_config, synthesis_config = config.llm.get_provider_configs()
                llm_manager = LLMManager(utility_config, synthesis_config)
        except Exception:
            llm_manager = None

        target_dir = config.target_dir or Path(".").resolve()
        raw_scope = Path(args.path)
        scope_path = (
            raw_scope.resolve()
            if raw_scope.is_absolute()
            else (target_dir / raw_scope).resolve()
        )
        scope_label = _compute_scope_label(target_dir, scope_path)

        llm_meta, assembly_provider = build_llm_metadata_and_assembly(
            config=config,
            llm_manager=llm_manager,
        )
        meta = AgentDocMetadata(
            created_from_sha=_get_head_sha(scope_path),
            previous_target_sha=_get_head_sha(scope_path),
            target_sha=_get_head_sha(scope_path),
            generated_at=datetime.now(timezone.utc).isoformat(),
            llm_config=llm_meta,
            generation_stats={"overview_only": "true"},
        )

        comprehensiveness = getattr(args, "comprehensiveness", "medium")
        if comprehensiveness == "minimal":
            max_points = 1
        elif comprehensiveness == "low":
            max_points = 5
        elif comprehensiveness == "medium":
            max_points = 10
        elif comprehensiveness == "high":
            max_points = 15
        elif comprehensiveness == "ultra":
            max_points = 20
        else:
            max_points = 10

        overview_answer, points_of_interest = await _run_autodoc_overview_hyde(
            llm_manager=llm_manager,
            target_dir=target_dir,
            scope_path=scope_path,
            scope_label=scope_label,
            max_points=max_points,
            comprehensiveness=comprehensiveness,
            out_dir=out_dir,
            assembly_provider=assembly_provider,
            indexing_cfg=getattr(config, "indexing", None),
        )

        meta.generation_stats["autodoc_comprehensiveness"] = comprehensiveness
        print(format_metadata_block(meta).rstrip("\n"))
        print(f"# AutoDoc Overview for {scope_label}")
        print("")
        print(overview_answer.strip())
        print("")
        return

    # Verify database exists and get paths
    try:
        verify_database_exists(config)
        db_path = config.database.path
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    # Initialize embedding manager (required for deep research)
    embedding_manager = EmbeddingManager()
    try:
        if config.embedding:
            provider = EmbeddingProviderFactory.create_provider(config.embedding)
            embedding_manager.register_provider(provider, set_default=True)
        else:
            raise ValueError("No embedding provider configured for autodoc")
    except ValueError as e:
        formatter.error(f"Embedding provider setup failed: {e}")
        formatter.info(
            "Configure an embedding provider via:\n"
            "1. Create a .chunkhound.json config file with embeddings, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Initialize LLM manager (required for deep research)
    llm_manager: LLMManager | None = None
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
        else:
            raise ValueError("No LLM provider configured for autodoc")
    except ValueError as e:
        formatter.error(f"LLM provider setup failed: {e}")
        formatter.info(
            "Configure an LLM provider via:\n"
            "1. Create a .chunkhound.json config file with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up LLM provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Create services using unified factory (exactly like MCP/CLI research)
    try:
        services = create_services(
            db_path=db_path,
            config=config,
            embedding_manager=embedding_manager,
        )
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    target_dir = config.target_dir or Path(".").resolve()
    raw_scope = Path(args.path)
    if raw_scope.is_absolute():
        scope_path = raw_scope.resolve()
    else:
        # Interpret relative scopes (e.g., "arguseek") as folders under the
        # workspace root so that path filters line up with the shared DB.
        scope_path = (target_dir / raw_scope).resolve()
    scope_label = _compute_scope_label(target_dir, scope_path)
    path_filter = _compute_path_filter(target_dir, scope_path)

    comprehensiveness = getattr(args, "comprehensiveness", "medium")
    if comprehensiveness == "minimal":
        max_points = 1
    elif comprehensiveness == "low":
        max_points = 5
    elif comprehensiveness == "medium":
        max_points = 10
    elif comprehensiveness == "high":
        max_points = 15
    elif comprehensiveness == "ultra":
        max_points = 20
    else:
        max_points = 10

    # Capture Git and LLM configuration metadata for the run so the output
    # document can be treated as a proper agent doc. Prefer the Git HEAD for
    # the scoped folder (so per-project roots in a workspace get accurate
    # SHAs), but fall back to the workspace root when the scope is not inside
    # a Git repository.
    created_from_sha = _get_head_sha(scope_path)
    if created_from_sha == "NO_GIT_HEAD" and scope_path != target_dir:
        created_from_sha = _get_head_sha(target_dir)
    llm_meta, assembly_provider = build_llm_metadata_and_assembly(
        config=config,
        llm_manager=llm_manager,
    )
    meta = AgentDocMetadata(
        created_from_sha=created_from_sha,
        previous_target_sha=created_from_sha,
        target_sha=created_from_sha,
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_config=llm_meta,
        generation_stats={},
    )

    # Phase 1 + 2: run overview (HyDE-based) and per-point deep research with a shared
    # TUI.
    overview_result: dict[str, Any]
    poi_sections: list[tuple[str, dict[str, Any]]] = []
    points_of_interest: list[str] = []

    # Optional depth override (currently disabled by default): wiring for
    # experiments where ultra mode may increase BFS depth beyond the global
    # default. For now we keep depth at the standard value unless an explicit
    # environment override is provided so coverage remains focused on breadth.
    original_depth_env = os.getenv("CH_CODE_RESEARCH_MAX_DEPTH")
    depth_overridden = False
    if (
        comprehensiveness == "ultra"
        and os.getenv("CH_AUTODOC_ULTRA_USE_DEPTH", "0") == "1"
    ):
        os.environ["CH_CODE_RESEARCH_MAX_DEPTH"] = "2"
        depth_overridden = True

    try:
        with TreeProgressDisplay() as tree_progress:
            try:
                # Phase 1: overview / points of interest via HyDE-only synthesis.
                overview_answer, points_of_interest = await _run_autodoc_overview_hyde(
                    llm_manager=llm_manager,
                    target_dir=target_dir,
                    scope_path=scope_path,
                    scope_label=scope_label,
                    max_points=max_points,
                    comprehensiveness=comprehensiveness,
                    out_dir=Path(out_dir_arg),
                    assembly_provider=assembly_provider,
                    indexing_cfg=getattr(config, "indexing", None),
                )

                overview_result = {
                    "answer": overview_answer,
                    # HyDE-only overview does not hit the database, so it has no
                    # sources metadata. We keep an explicit empty structure so
                    # downstream coverage merging remains robust.
                    "metadata": {
                        "sources": {
                            "files": [],
                            "chunks": [],
                        },
                        "aggregation_stats": {},
                    },
                }
            except Exception as e:
                formatter.error(f"Autodoc overview research failed: {e}")
                logger.exception("Full error details:")
                sys.exit(1)

            if not points_of_interest:
                formatter.error(
                    "Autodoc could not extract any points of interest from the "
                    "overview."
                )
                print("\n--- Overview answer ---\n")
                print(overview_answer)
                sys.exit(1)

            # Phase 2: deep dives per point of interest (unless overview-only).
            if not getattr(args, "overview_only", False):
                for idx, poi in enumerate(points_of_interest, start=1):
                    heading = _derive_heading_from_point(poi)
                    formatter.info(
                        f"[Autodoc] Processing point of interest {idx}/"
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
                            progress=tree_progress,
                            path=path_filter,
                        )
                        if _is_empty_research_result(result):
                            formatter.warning(
                                f"[Autodoc] Skipping point of interest {idx} because "
                                "deep research returned no usable content."
                            )
                            continue
                        poi_sections.append((poi, result))
                    except Exception as e:
                        formatter.error(
                            f"Autodoc deep research failed for point {idx}: {e}"
                        )
                        logger.exception("Full error details:")
                        # Continue with remaining points to salvage partial
                        # documentation.
    finally:
        # Restore original depth override so other commands/tests are not
        # affected by the ultra-mode setting.
        if depth_overridden:
            if original_depth_env is None:
                os.environ.pop("CH_CODE_RESEARCH_MAX_DEPTH", None)
            else:
                os.environ["CH_CODE_RESEARCH_MAX_DEPTH"] = original_depth_env

    # Phase 3 (part 2): compute coverage and render the final document with metadata.
    poi_results = [r for _, r in poi_sections]
    all_results: list[dict[str, Any]] = [overview_result] + poi_results
    (
        unified_source_files,
        unified_chunks_dedup,
        total_files_global,
        total_chunks_global,
    ) = _merge_sources_metadata(all_results)

    # Compute scope-level database stats for coverage and generation metadata.
    scope_total_files, scope_total_chunks, _scoped_files = compute_db_scope_stats(
        services, scope_label
    )

    referenced_files = len(unified_source_files)
    referenced_chunks = len(unified_chunks_dedup)

    # Prefer scope-level totals; fall back to global aggregation stats when
    # scope stats are unavailable.
    files_denominator: int | None = scope_total_files or None
    chunks_denominator: int | None = scope_total_chunks or None
    if files_denominator is None and isinstance(total_files_global, int):
        files_denominator = total_files_global
    if chunks_denominator is None and isinstance(total_chunks_global, int):
        chunks_denominator = total_chunks_global

    files_basis = (
        "scope" if scope_total_files else ("database" if files_denominator else "unknown")
    )
    chunks_basis = (
        "scope" if scope_total_chunks else ("database" if chunks_denominator else "unknown")
    )

    prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"
    referenced_files_in_scope = 0
    for path in unified_source_files:
        norm = str(path).replace("\\", "/")
        if not norm:
            continue
        if prefix and not norm.startswith(prefix):
            continue
        referenced_files_in_scope += 1
    unreferenced_files_in_scope = (
        max(0, scope_total_files - referenced_files_in_scope) if scope_total_files else None
    )

    total_research_calls = len(poi_sections)
    generation_stats: dict[str, Any] = {
        "generator_mode": "code_research",
        "total_research_calls": str(total_research_calls),
        "autodoc_comprehensiveness": comprehensiveness,
        "files": {
            "referenced": referenced_files,
            "total_indexed": files_denominator or 0,
            "basis": files_basis,
            "coverage": (
                f"{(referenced_files / files_denominator) * 100.0:.2f}%"
                if files_denominator
                else None
            ),
            "referenced_in_scope": referenced_files_in_scope,
            "unreferenced_in_scope": unreferenced_files_in_scope,
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

    # Attach generation statistics so the metadata header reflects coverage and
    # records the CLI-level comprehensiveness knob that shaped this run.
    meta.generation_stats = generation_stats

    # Render metadata header once we have all stats, then the document body.
    metadata_block = format_metadata_block(meta)
    print(metadata_block, end="")

    print(f"# AutoDoc for {scope_label}")
    print("")
    print("## Coverage Summary")
    print("")
    if files_denominator and files_denominator > 0:
        file_cov = (referenced_files / files_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_files else "this database"
        print(
            f"- Referenced files: {referenced_files} / {files_denominator} "
            f"({file_cov:.2f}% of indexed files in {scope_label_display})."
        )
    else:
        print(f"- Referenced files: {referenced_files} (database totals unavailable).")

    if chunks_denominator and chunks_denominator > 0:
        chunk_cov = (referenced_chunks / chunks_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_chunks else "this database"
        print(
            f"- Referenced chunks: {referenced_chunks} / {chunks_denominator} "
            f"({chunk_cov:.2f}% of indexed chunks in {scope_label_display})."
        )
    else:
        print(
            f"- Referenced chunks: {referenced_chunks} (database totals unavailable)."
        )

    print("")
    print("## Points of Interest Overview")
    print("")
    print(overview_result.get("answer", "").strip())
    print("")

    # Emit detailed sections after the overview, one per point of interest.
    if not getattr(args, "overview_only", False):
        for idx, (poi, result) in enumerate(poi_sections, start=1):
            heading = _derive_heading_from_point(poi)
            print(f"## {idx}. {heading}")
            print("")
            section_body = result.get("answer", "").strip()
            print(section_body)
            print("")

    # Final coverage summary (identical counts to the header section for
    # ease of visual inspection at the end of the document).
    print("## Coverage Summary")
    print("")
    if files_denominator and files_denominator > 0:
        file_cov = (referenced_files / files_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_files else "this database"
        print(
            f"- Referenced files: {referenced_files} / {files_denominator} "
            f"({file_cov:.2f}% of indexed files in {scope_label_display})."
        )
    else:
        print(f"- Referenced files: {referenced_files} (database totals unavailable).")

    if chunks_denominator and chunks_denominator > 0:
        chunk_cov = (referenced_chunks / chunks_denominator) * 100.0
        scope_label_display = "this scope" if scope_total_chunks else "this database"
        print(
            f"- Referenced chunks: {referenced_chunks} / {chunks_denominator} "
            f"({chunk_cov:.2f}% of indexed chunks in {scope_label_display})."
        )
    else:
        print(
            f"- Referenced chunks: {referenced_chunks} (database totals unavailable)."
        )

    # Optional disk outputs: when out-dir is provided and full deep research
    # was performed, emit an index file plus one markdown file per topic.
    if out_dir_arg is not None and not getattr(args, "overview_only", False):
        out_dir = Path(out_dir_arg).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        safe_scope = scope_label.replace("/", "_") or "root"
        index_path = out_dir / f"{safe_scope}_autodoc_index.md"

        # Optional: compute unreferenced file list for the scope and write it to
        # a separate artifact (can be large).
        unref_filename: str | None = None
        try:
            provider = getattr(services, "provider", None)
            get_scope_file_paths = getattr(provider, "get_scope_file_paths", None)
            if callable(get_scope_file_paths):
                prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"
                all_files = get_scope_file_paths(prefix)
                referenced_set = {
                    str(p).replace("\\", "/")
                    for p in unified_source_files
                    if p and (not prefix or str(p).replace("\\", "/").startswith(prefix))
                }
                unreferenced = [
                    str(p).replace("\\", "/")
                    for p in all_files
                    if p and str(p).replace("\\", "/") not in referenced_set
                ]
                unref_filename = f"{safe_scope}_scope_unreferenced_files.txt"
                (out_dir / unref_filename).write_text(
                    "\n".join(unreferenced) + ("\n" if unreferenced else ""),
                    encoding="utf-8",
                )
                try:
                    files_stats = meta.generation_stats.get("files")
                    if isinstance(files_stats, dict):
                        files_stats["unreferenced_list_file"] = unref_filename
                except Exception:
                    pass
        except Exception:
            unref_filename = None

        # Build index content with the same metadata header for traceability.
        index_lines: list[str] = []
        index_lines.append(format_metadata_block(meta).rstrip("\n"))
        index_lines.append(f"# AutoDoc Topics for {scope_label}")
        index_lines.append("")
        index_lines.append(
            "This index lists the per-topic autodoc sections generated for this scope."
        )
        if unref_filename is not None:
            index_lines.append(
                f"- Unreferenced files in scope: [{unref_filename}]({unref_filename})"
            )
        index_lines.append("")
        index_lines.append("## Topics")
        index_lines.append("")

        for idx, (poi, result) in enumerate(poi_sections, start=1):
            heading = _derive_heading_from_point(poi)
            slug = _slugify_heading(heading)
            topic_filename = f"{safe_scope}_topic_{idx:02d}_{slug}.md"
            topic_path = out_dir / topic_filename
            index_lines.append(f"{idx}. [{heading}]({topic_filename})")

            section_body = result.get("answer", "").strip()
            topic_lines = [f"# {heading}", "", section_body, ""]
            topic_path.write_text("\n".join(topic_lines), encoding="utf-8")

        index_lines.append("")
        index_path.write_text("\n".join(index_lines), encoding="utf-8")
