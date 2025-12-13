"""Autodoc command module - generates scoped architecture/operations docs.

This command uses a two-phase pipeline:
1. Run a shallow deep-research call to identify 5-10 points of interest for the
   requested scope (overview plan).
2. For each point of interest, run a dedicated deep-research pass and assemble
   the results into a single flowing document, along with a simple coverage
   summary based on referenced files and chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.api.cli.utils import verify_database_exists
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


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
    """Extract 5-10 discrete points of interest from a markdown list.

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


def _merge_sources_metadata(results: list[dict[str, Any]]) -> tuple[int, int, int | None, int | None]:
    """Merge sources metadata from multiple deep-research calls.

    Returns:
        Tuple of (unique_files, unique_chunks, total_files_indexed, total_chunks_indexed)
        where the totals are taken from the database stats if available.
    """
    files_seen: set[str] = set()
    chunk_keys: set[tuple[str, int | None, int | None]] = set()

    # Track best-effort database totals (we use the last non-zero stats)
    total_files_indexed: int | None = None
    total_chunks_indexed: int | None = None

    for result in results:
        metadata = result.get("metadata") or {}
        sources = metadata.get("sources") or {}

        for file_path in sources.get("files") or []:
            if file_path:
                files_seen.add(file_path)

        for chunk in sources.get("chunks") or []:
            file_path = chunk.get("file_path")
            if not file_path:
                continue
            start_line = chunk.get("start_line")
            end_line = chunk.get("end_line")
            key = (
                file_path,
                int(start_line) if isinstance(start_line, int) else None,
                int(end_line) if isinstance(end_line, int) else None,
            )
            chunk_keys.add(key)

        stats = metadata.get("aggregation_stats") or {}
        # Use stats totals if they look plausible; this is best-effort only.
        files_total = stats.get("files_total")
        chunks_total = stats.get("chunks_total")
        if isinstance(files_total, int) and files_total > 0:
            total_files_indexed = files_total
        if isinstance(chunks_total, int) and chunks_total > 0:
            total_chunks_indexed = chunks_total

    return len(files_seen), len(chunk_keys), total_files_indexed, total_chunks_indexed


async def autodoc_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the autodoc command using deep code research."""
    formatter = RichOutputFormatter(verbose=args.verbose)

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
    scope_path = Path(args.path).resolve()
    scope_label = _compute_scope_label(target_dir, scope_path)
    path_filter = _compute_path_filter(target_dir, scope_path)

    # Overview query: identify 5-10 points of interest for this scope.
    overview_query = (
        "You are preparing an architecture and operations guide for the scoped "
        f"folder '{scope_label}'. Based on partial evidence from the code and "
        "configuration (you will see only a subset of the repository), identify "
        "5–10 major points of interest that should be documented.\n\n"
        "Output ONLY a numbered markdown list. For each item, include:\n"
        "- A short heading (you may use bold or plain text), and\n"
        "- 1–2 sentences summarizing why this point is important.\n\n"
        "Do not write the full documentation yet; focus on listing the points "
        "of interest."
    )

    # Phase 1 + 2: run overview and per-point deep research with a shared TUI.
    overview_result: dict[str, Any]
    poi_results: list[dict[str, Any]] = []
    points_of_interest: list[str] = []

    with TreeProgressDisplay() as tree_progress:
        try:
            # Phase 1: overview / points of interest
            overview_result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=overview_query,
                progress=tree_progress,
                path=path_filter,
            )
        except Exception as e:
            formatter.error(f"Autodoc overview research failed: {e}")
            logger.exception("Full error details:")
            sys.exit(1)

        overview_answer = overview_result.get("answer", "")
        points_of_interest = _extract_points_of_interest(overview_answer, max_points=10)

        if not points_of_interest:
            formatter.error(
                "Autodoc could not extract any points of interest from the overview."
            )
            print("\n--- Overview answer ---\n")
            print(overview_answer)
            sys.exit(1)

        # Phase 2: deep dives per point of interest
        for idx, poi in enumerate(points_of_interest, start=1):
            section_query = (
                "Expand the following point of interest into a detailed, "
                "agent-facing documentation section for the scoped folder "
                f"'{scope_label}'. Explain how the relevant code and configuration "
                "implement this behavior, including responsibilities, key types, "
                "important flows, and operational constraints.\n\n"
                "Point of interest:\n"
                f"{poi}\n\n"
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
                    progress=tree_progress,
                    path=path_filter,
                )
                poi_results.append(result)
            except Exception as e:
                formatter.error(
                    f"Autodoc deep research failed for point {idx}: {e}"
                )
                logger.exception("Full error details:")
                # Continue with remaining points to salvage partial documentation

    # Phase 3: assemble final document.
    all_results: list[dict[str, Any]] = [overview_result] + poi_results
    referenced_files, referenced_chunks, total_files, total_chunks = _merge_sources_metadata(
        all_results
    )

    lines: list[str] = []
    lines.append(f"# AutoDoc for {scope_label}")
    lines.append("")

    # Simple coverage summary (database totals are best-effort global counts).
    lines.append("## Coverage Summary")
    lines.append("")
    if total_files and total_files > 0:
        file_cov = (referenced_files / total_files) * 100.0
        lines.append(
            f"- Referenced files: {referenced_files} / {total_files} "
            f"({file_cov:.2f}% of indexed files in this database)."
        )
    else:
        lines.append(f"- Referenced files: {referenced_files} (database totals unavailable).")

    if total_chunks and total_chunks > 0:
        chunk_cov = (referenced_chunks / total_chunks) * 100.0
        lines.append(
            f"- Referenced chunks: {referenced_chunks} / {total_chunks} "
            f"({chunk_cov:.2f}% of indexed chunks in this database)."
        )
    else:
        lines.append(
            f"- Referenced chunks: {referenced_chunks} (database totals unavailable)."
        )
    lines.append("")

    # Overview section with raw overview list for transparency.
    lines.append("## Points of Interest Overview")
    lines.append("")
    lines.append(overview_result.get("answer", "").strip())
    lines.append("")

    # Detailed sections per point of interest.
    for idx, (poi, result) in enumerate(
        zip(points_of_interest, poi_results, strict=False), start=1
    ):
        heading = _derive_heading_from_point(poi)
        lines.append(f"## {idx}. {heading}")
        lines.append("")
        section_body = result.get("answer", "").strip()
        lines.append(section_body)
        lines.append("")

    # Print a blank line after progress output then the assembled doc.
    print("\n")
    print("\n".join(lines))

