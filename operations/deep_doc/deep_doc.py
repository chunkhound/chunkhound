"""Generate a single, agent-optimized documentation file for a scoped folder.

This script uses ChunkHound's deep research and search pipeline to build an
LLM-friendly architecture and operations document for any folder under a
configured workspace. It assumes:

- A ChunkHound workspace database already exists for the target path
  (run `chunkhound index <workspace-root>`).
- Embedding and LLM providers are configured in `.chunkhound.json`.

On first run for a given output file, it records the current commit SHA as the
documentation baseline. On subsequent runs, it reads the embedded metadata,
computes diffs between the original baseline and the current HEAD (if Git is
available), and asks the research pipeline to generate an updated document that
reflects the current state and summarizes changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl
from chunkhound.services.research.citation_manager import CitationManager
from operations.deep_doc.split_deep_doc import split_agent_doc_text_fluid


DEFAULT_OUTPUT_RELATIVE = Path("operations/chunkhound_agent_doc.md")
PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class AgentDocMetadata:
    """Commit metadata used to drive incremental regeneration."""

    created_from_sha: str
    previous_target_sha: str
    target_sha: str
    generated_at: str
    llm_config: dict[str, str] = field(default_factory=dict)
    generation_stats: dict[str, str] = field(default_factory=dict)


def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout as text.

    This helper is intentionally forgiving: callers are expected to handle
    failures gracefully for workspaces that are not Git repositories.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout.strip()


def _get_head_sha(project_root: Path) -> str:
    """Return the current HEAD SHA for the project.

    For workspaces that are not Git repositories, this returns a stable
    placeholder instead of failing hard.
    """
    try:
        return _run_git(["rev-parse", "HEAD"], project_root)
    except subprocess.CalledProcessError:
        # Non-git workspace: downstream logic treats this as "no git metadata"
        return "NO_GIT_HEAD"


def _get_name_status_diff(base_sha: str, target_sha: str, project_root: Path) -> list[str]:
    """Return `git diff --name-status base..target` as a list of lines.

    For non-git workspaces, this returns an empty list.
    """
    if base_sha == target_sha:
        return []

    try:
        output = _run_git(
            ["diff", "--name-status", f"{base_sha}..{target_sha}"], project_root
        )
    except subprocess.CalledProcessError:
        return []

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines


def _parse_existing_metadata(doc_path: Path) -> AgentDocMetadata | None:
    """Parse the metadata comment block from an existing agent doc, if present.

    Expected format at the top of the file:

    <!--
    agent_doc_metadata:
      created_from_sha: <AAAA>
      previous_target_sha: <BBBB>
      target_sha: <CCCC>
      generated_at: 2025-11-22T...
    -->
    """
    if not doc_path.exists():
        return None

    created_from_sha: str | None = None
    previous_target_sha: str | None = None
    target_sha: str | None = None
    generated_at: str | None = None
    llm_cfg: dict[str, str] = {}
    gen_stats: dict[str, str] = {}

    in_metadata = False
    in_llm_block = False
    in_gen_stats_block = False
    lines_read = 0

    with doc_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            lines_read += 1

            # Stop scanning after a small header window
            if lines_read > 80:
                break

            if "agent_doc_metadata:" in line:
                in_metadata = True
                continue

            if in_metadata:
                if "-->" in line:
                    break
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "llm_config":
                    in_llm_block = True
                    in_gen_stats_block = False
                    continue
                if key == "generation_stats":
                    in_gen_stats_block = True
                    in_llm_block = False
                    continue
                if in_llm_block:
                    llm_cfg[key] = value
                elif in_gen_stats_block:
                    gen_stats[key] = value
                elif key == "created_from_sha":
                    created_from_sha = value
                elif key == "previous_target_sha":
                    previous_target_sha = value
                elif key == "target_sha":
                    target_sha = value
                elif key == "generated_at":
                    generated_at = value

    if not created_from_sha:
        return None

    # For newer docs that only record created_from_sha, synthesize target and
    # previous_target values so downstream logic continues to work. Older docs
    # that recorded explicit target/previous_target_sha will preserve them.
    if not target_sha:
        target_sha = created_from_sha
    if not previous_target_sha:
        previous_target_sha = target_sha

    if not generated_at:
        generated_at = ""

    return AgentDocMetadata(
        created_from_sha=created_from_sha,
        previous_target_sha=previous_target_sha,
        target_sha=target_sha,
        generated_at=generated_at,
        llm_config=llm_cfg,
        generation_stats=gen_stats,
    )


def _format_metadata_block(meta: AgentDocMetadata) -> str:
    """Render the metadata comment block."""
    lines = [
        "<!--",
        "agent_doc_metadata:",
    ]

    # For workspaces without Git, we avoid emitting placeholder SHAs such as
    # NO_GIT_HEAD in the document header. In that case we only record the
    # generated_at timestamp (and any LLM metadata), and skip commit fields
    # entirely. When real Git metadata is available, we keep a single baseline
    # SHA (created_from_sha) to anchor change reasoning without exposing
    # per-run target SHAs in the header.
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
            lines.append(f"    {key}: {value}")
    lines.append("-->")
    return "\n".join(lines) + "\n\n"


def _extract_hyde_bullets(hyde_plan: str | None, max_bullets: int) -> list[str]:
    """Extract bullet-shaped HyDE lines as generic queries."""
    if not hyde_plan:
        return []

    bullets: list[str] = []
    for raw in hyde_plan.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(("-", "*")):
            continue
        text = line.lstrip("-* ").strip()
        if len(text) < 20:
            continue
        bullets.append(text)
        if len(bullets) >= max_bullets:
            break
    return bullets


def _extract_hyde_sections(hyde_plan: str, max_sections: int) -> list[dict[str, str]]:
    """Extract grouped HyDE sections (title + body) for map passes.

    Sections are derived from the 'Global Hooks' and 'Subsystem Hooks' parts of
    the HyDE plan and are intentionally coarse-grained. This provides an
    alternative to per-bullet queries so we can run code_research on richer,
    paragraph-shaped contexts.
    """
    lines = hyde_plan.splitlines()
    sections: list[dict[str, str]] = []

    in_hooks = False
    current_title: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        nonlocal current_title, current_body
        if current_title and current_body:
            sections.append(
                {
                    "title": current_title.strip(),
                    "body": "\n".join(current_body).strip(),
                }
            )
        current_title = None
        current_body = []

    for raw in lines:
        stripped = raw.rstrip("\n")
        l = stripped.strip()

        # Enter/exit hooks region based on top-level headings
        if l.startswith("## "):
            heading = l[3:].strip()
            if heading in {"Global Hooks", "Subsystem Hooks"}:
                # Starting hooks region; flush any prior section
                flush()
                in_hooks = True
                continue
            else:
                # Leaving hooks region
                if in_hooks:
                    flush()
                    in_hooks = False
                continue

        if not in_hooks:
            continue

        # Subsection title within hooks
        if l.startswith("### "):
            # New section; flush previous one
            flush()
            current_title = l[4:].strip()
            current_body = []
            continue

        # Accumulate body lines (bullets or text) when inside hooks region
        if l:
            # If no explicit subsection title, synthesize one from first bullet
            if current_title is None and (l.startswith("-") or l.startswith("*")):
                current_title = l.lstrip("-* ").strip()[:80]
            current_body.append(stripped)

        # Stop if we've collected enough sections
        if len(sections) >= max_sections:
            break

    # Flush last section
    if len(sections) < max_sections:
        flush()

    return sections[:max_sections]


def _split_sources_footer(body: str) -> tuple[str, str | None]:
    """Split a deep-research answer into (main_body, sources_footer).

    Deep research synthesis appends a Sources footer that starts with a
    '## Sources' heading (usually preceded by a '---' separator). When we run
    a scope-trimming pass, the LLM may drop that footer. To keep behavior
    consistent, we extract it from the raw answer and re-attach it after
    trimming if necessary.
    """
    if not body:
        return body, None

    lines = body.splitlines()
    last_sources_idx: int | None = None

    for idx, line in enumerate(lines):
        if line.strip().startswith("## Sources"):
            last_sources_idx = idx

    if last_sources_idx is None:
        return body, None

    start_idx = last_sources_idx
    if (
        last_sources_idx >= 2
        and lines[last_sources_idx - 2].strip() == "---"
        and lines[last_sources_idx - 1].strip() == ""
    ):
        start_idx = last_sources_idx - 2

    footer_lines = lines[start_idx:]
    body_lines = lines[:start_idx]

    footer = "\n".join(footer_lines).strip()
    main_body = "\n".join(body_lines).rstrip()

    if not footer:
        return body, None

    return main_body, footer


def _load_prompt_template(filename: str) -> str:
    """Load a prompt template from the shared prompts directory.

    This keeps large CTA prompts in standalone markdown files for easier
    iteration without touching Python source. Callers pass the target
    filename (for example, 'overview_prompt.md').
    """
    template_path = PROMPTS_DIR / filename
    try:
        text = template_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise SystemExit(
            f"Deep-doc prompt file missing or unreadable: {template_path} ({exc})"
        ) from exc

    if not text:
        raise SystemExit(f"Deep-doc prompt file is empty: {template_path}")

    return text


def _build_research_prompt(
    meta: AgentDocMetadata,
    diff_since_created: list[str],
    diff_since_previous: list[str],
    scope_label: str,
    hyde_plan: str | None = None,
) -> str:
    """Construct the overview deep research query for the agent doc."""
    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    diff_created_block = (
        "No changes detected (initial build; baseline and target are identical)."
        if not diff_since_created
        else "\n".join(f"- {line}" for line in diff_since_created)
    )

    diff_previous_block = (
        "No changes since previous agent doc build."
        if not diff_since_previous
        else "\n".join(f"- {line}" for line in diff_since_previous)
    )

    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    plan_block = ""
    if hyde_plan and hyde_plan.strip():
        plan_block = f"""

Planning context (from a separate HyDE-style heuristic scan).
Use this ONLY as a rough outline. Do NOT copy phrases or sentences from it;
re-derive all explanations from the actual code, tests, and docs.

<<<PLAN>>>
{hyde_plan}
<<<ENDPLAN>>>
""".rstrip()

    template = _load_prompt_template("overview_prompt.md")
    return template.format(
        created=created,
        previous=previous,
        target=target,
        scope_display=scope_display,
        diff_created_block=diff_created_block,
        diff_previous_block=diff_previous_block,
        plan_block=plan_block,
    ).strip()



async def _run_research_query(
    services: Any,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager | None,
    prompt: str,
    scope_label: str | None = None,
) -> str:
    """Run a single deep_research_impl call and return only the answer text."""
    answer, _ = await _run_research_query_with_metadata(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        prompt=prompt,
        scope_label=scope_label,
    )
    return answer


async def _run_research_query_with_metadata(
    services: Any,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager | None,
    prompt: str,
    scope_label: str | None = None,
    path_override: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run deep_research_impl and return (answer, metadata)."""
    if path_override is not None and path_override.strip():
        path_filter = path_override.strip()
    else:
        path_filter = None if scope_label in (None, "/") else scope_label
    result: dict[str, Any] = await deep_research_impl(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        query=prompt,
        progress=None,
        path=path_filter,
    )

    answer = result.get(
        "answer",
        "Research incomplete: unable to generate this section for the current run.",
    )
    metadata = result.get("metadata", {}) or {}
    return answer, metadata


def _iter_scope_files(scope_path: Path, project_root: Path) -> list[str]:
    """Return normalized relative file paths within scope, skipping noise dirs.

    This helper centralizes the logic for walking the scoped tree so that
    coverage metrics, HyDE planning, and other path-based features all share a
    consistent view of which files are considered "in scope".
    """
    ignore_dirs = {".git", ".chunkhound", ".venv", "venv", "__pycache__", ".mypy_cache"}
    file_paths: list[str] = []
    for path in scope_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            # Scope should be under project_root, but guard just in case.
            continue
        file_paths.append(str(rel).replace("\\", "/"))
    return file_paths


def _infer_hyde_unit_path(
    scope_label: str,
    title: str,
    body: str,
) -> str | None:
    """Infer a scoped path_filter for a HyDE unit (section or bullet).

    This helper looks for explicit module hints in the HyDE plan, primarily
    backticked paths in section titles such as:

        ### Core Research Agent / Orchestration (`internal/agent`)

    When a relative path is found, it is resolved against the scope_label
    (e.g. 'arguseek') so that deep_research_impl receives a prefix such as
    'arguseek/internal/agent'. If no clear path can be inferred, this returns
    None and the caller falls back to the broader scope_label filter.
    """
    import re

    text = f"{title}\n{body}"
    match = re.search(r"\(`([^`]+)`\)", text)
    if not match:
        return None

    raw = match.group(1).strip()
    if not raw:
        return None

    # Normalize common relative path patterns
    candidate = raw.lstrip("./").replace("\\", "/")
    scope_clean = scope_label.strip().strip("/")

    if not scope_clean:
        # Scope is the workspace root; use the candidate as-is.
        return candidate

    # If the candidate already appears to be scoped (e.g. 'arguseek/internal/agent'),
    # respect it directly.
    if candidate == scope_clean or candidate.startswith(scope_clean + "/"):
        return candidate

    # Otherwise treat it as relative to the scoped folder.
    return f"{scope_clean}/{candidate}"


def _collect_scope_files(
    scope_path: Path,
    project_root: Path,
    hyde_cfg: HydeConfig,
) -> list[str]:
    """Collect a list of file paths within the scope, relative to project_root.

    This is intentionally lightweight: we only gather paths, not full contents,
    and we skip typical noise directories.
    """
    # Use the shared scope iterator to keep behavior consistent with coverage
    # metrics and semantic analysis. Preserve the depth-first walk order from
    # rglob so HyDE sees a natural tree traversal rather than a purely
    # lexicographic listing.
    file_paths = _iter_scope_files(scope_path, project_root)

    # Respect Git ignore rules when the scope or project root lives under a
    # Git repository so HyDE does not waste context budget on ignored binaries
    # or artifacts. We detect the effective git root first, then pass paths
    # relative to that root into `git check-ignore`.
    try:
        if file_paths:
            from subprocess import run

            git_root: Path | None = None
            for candidate in (scope_path, project_root):
                proc_root = run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=str(candidate),
                    text=True,
                    capture_output=True,
                )
                if proc_root.returncode == 0 and proc_root.stdout.strip():
                    git_root = Path(proc_root.stdout.strip()).resolve()
                    break

            if git_root is not None:
                abs_paths: list[Path] = []
                rel_for_git: list[str] = []
                rel_to_original: dict[str, str] = {}

                for rel in file_paths:
                    abs_path = (project_root / rel).resolve()
                    try:
                        git_rel = abs_path.relative_to(git_root).as_posix()
                    except ValueError:
                        # Path outside the git repo; keep it as-is.
                        continue
                    abs_paths.append(abs_path)
                    rel_for_git.append(git_rel)
                    rel_to_original[git_rel] = rel

                if rel_for_git:
                    proc = run(
                        ["git", "check-ignore", "--stdin"],
                        cwd=str(git_root),
                        input="\n".join(rel_for_git),
                        text=True,
                        capture_output=True,
                    )
                    # git check-ignore returns 0 when at least one path is
                    # ignored, 1 when no paths are ignored, and >1 for errors.
                    if proc.returncode in (0, 1):
                        ignored_git_rel = {
                            line.strip()
                            for line in proc.stdout.splitlines()
                            if line.strip()
                        }
                        if ignored_git_rel:
                            ignored_original = {
                                rel_to_original[p] for p in ignored_git_rel if p in rel_to_original
                            }
                            file_paths = [
                                p for p in file_paths if p not in ignored_original
                            ]
    except Exception:
        # If Git is unavailable or the call fails, fall back to the raw list.
        pass

    # Cap to a reasonable count based on HydeConfig so HyDE can see more of
    # very large scopes when desired without exploding prompt size.
    if len(file_paths) > hyde_cfg.max_scope_files:
        file_paths = file_paths[:hyde_cfg.max_scope_files]
    return file_paths


def _compute_db_scope_stats(
    services: Any, scope_label: str
) -> tuple[int, int, set[str]]:
    """Compute indexed file/chunk totals and scoped file set for the folder.

    This walks the underlying provider metadata and applies the same prefix
    filter used elsewhere so that coverage stats for different modes share a
    common denominator.
    """
    scope_total_files = 0
    scope_total_chunks = 0
    scoped_files: set[str] = set()
    try:
        provider = getattr(services, "provider", None)
        if provider is None:
            return 0, 0, scoped_files
        chunks_meta = provider.get_all_chunks_with_metadata()
        prefix = None if scope_label == "/" else scope_label.rstrip("/") + "/"
        for chunk in chunks_meta:
            path = (chunk.get("file_path") or "").replace("\\", "/")
            if not path:
                continue
            if prefix and not path.startswith(prefix):
                continue
            scoped_files.add(path)
            scope_total_chunks += 1
        scope_total_files = len(scoped_files)
    except Exception:
        scope_total_files = 0
        scope_total_chunks = 0
        scoped_files = set()
    return scope_total_files, scope_total_chunks, scoped_files


def _count_scope_files_for_coverage(scope_path: Path, project_root: Path) -> int:
    """Count the number of files within the scope, relative to project_root.

    This is used for coverage statistics in generation_stats and may traverse
    more than the capped list used for HyDE prompts. It remains lightweight by
    skipping typical noise directories.
    """
    return len(_iter_scope_files(scope_path, project_root))


@dataclass
class HydeConfig:
    max_scope_files: int
    max_snippet_files: int
    max_snippet_chars: int
    max_completion_tokens: int
    max_snippet_tokens: int

    @classmethod
    def from_env(cls) -> HydeConfig:
        # Defaults chosen to keep HyDE planning generous but bounded.
        max_scope = 200
        # Default to 0 (no per-file count cap); the global snippet token
        # budget will control total context size and distribute it across
        # files.
        max_snippet_files = 0
        # Default to 0 (no per-file cap); the global snippet token budget will
        # control total context size and distribute it across files.
        max_snippet_chars = 0
        max_tokens = 30_000
        # Global snippet token budget for HyDE source/context section.
        max_snippet_tokens = 50_000

        def _parse_positive_int(env_name: str, default: int) -> int:
            value = os.getenv(env_name)
            if not value:
                return default
            try:
                parsed = int(value)
            except ValueError:
                return default
            if parsed <= 0:
                return default
            return parsed

        max_scope = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_MAX_SCOPE_FILES",
            max_scope,
        )
        max_snippet_files = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_MAX_SNIPPET_FILES",
            max_snippet_files,
        )
        max_snippet_chars = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_MAX_SNIPPET_CHARS",
            max_snippet_chars,
        )

        max_snippet_tokens = _parse_positive_int(
            "CH_AGENT_DOC_HYDE_SNIPPET_TOKENS",
            max_snippet_tokens,
        )

        # Enforce a hard safety ceiling for completion tokens even when the
        # environment override is set very high.
        tokens_env = os.getenv("CH_AGENT_DOC_HYDE_COMPLETION_TOKENS")
        if tokens_env:
            try:
                parsed_tokens = int(tokens_env)
                if parsed_tokens > 0:
                    max_tokens = min(parsed_tokens, 30_000)
            except ValueError:
                pass

        return cls(
            max_scope_files=max_scope,
            max_snippet_files=max_snippet_files,
            max_snippet_chars=max_snippet_chars,
            max_completion_tokens=max_tokens,
            max_snippet_tokens=max_snippet_tokens,
        )


def _build_hyde_scope_prompt(
    meta: AgentDocMetadata,
    scope_label: str,
    file_paths: list[str],
    hyde_cfg: HydeConfig,
    project_root: Path | None = None,
) -> str:
    """Construct a HyDE-only prompt using only file layout within the scope.

    NOTE: This prompt is intentionally project-agnostic. It should not mention
    ChunkHound or any product-specific semantics; the goal is to hallucinate a
    generic architectural plan purely from file naming and layout.
    """
    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    if project_root is None:
        project_root = Path.cwd()

    files_block = "\n".join(f"- {p}" for p in file_paths) if file_paths else "- (no files discovered)"

    # Build a lightweight code context block from a subset of files in the
    # scope. This remains filesystem-only (no DB) and is intended purely to
    # give HyDE a better feel for the project while staying cheap. We
    # approximate token counts from file byte sizes to compute a global
    # overflow ratio, then cap each file to that ratio of its content so that
    # large files do not monopolize the context. When the total size is under
    # budget, files are included in full.
    snippet_token_budget = getattr(hyde_cfg, "max_snippet_tokens", 100_000)
    snippet_char_budget = max(0, snippet_token_budget * 4)
    max_chars_per_file = hyde_cfg.max_snippet_chars

    # Heuristic filters for binary/heavy files that should not contribute
    # code snippets (but still appear in the file listing so HyDE can see
    # the tree layout).
    binary_exts = {
        # Images
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".bmp",
        ".tiff",
        ".tif",
        # Fonts
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        # Archives / compressed
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        # Documents / media
        ".pdf",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".wav",
        ".flac",
        # Compiled artifacts
        ".so",
        ".dll",
        ".dylib",
        ".o",
        ".a",
        ".obj",
        ".exe",
        ".class",
        ".jar",
        # Raw binaries / blobs / DBs
        ".bin",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".duckdb",
        ".lance",
    }
    max_snippet_file_bytes = 1_000_000  # ~1 MiB

    file_infos: list[tuple[str, Path, int]] = []
    total_bytes = 0

    for rel_path in file_paths:
        path = project_root / rel_path
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except Exception:
            continue
        # Skip obviously binary or oversized files for snippet purposes.
        if path.suffix.lower() in binary_exts:
            continue
        if size > max_snippet_file_bytes:
            continue
        if size <= 0:
            continue
        file_infos.append((rel_path, path, size))
        total_bytes += size

    code_snippets: list[str] = []

    if snippet_char_budget <= 0 or total_bytes <= 0 or not file_infos:
        code_context_block = "(no sample code snippets available)"
    else:
        ratio = min(1.0, float(snippet_char_budget) / float(total_bytes))

        # Optional hard cap on per-file snippet length when configured; by
        # default we rely solely on the global ratio.
        per_file_cap = max_chars_per_file if max_chars_per_file > 0 else None
        max_files_for_snippets = hyde_cfg.max_snippet_files

        for idx, (rel_path, path, size) in enumerate(file_infos):
            if max_files_for_snippets > 0 and idx >= max_files_for_snippets:
                break

            target_chars = int(size * ratio)
            if per_file_cap is not None:
                target_chars = min(target_chars, per_file_cap)
            if target_chars <= 0:
                continue

            try:
                # Read a small prefix first to cheaply detect binary content.
                # If it looks binary, skip this file for snippets.
                with path.open("rb") as f:
                    raw_prefix = f.read(8192)
                if b"\x00" in raw_prefix:
                    continue
                # Decode the full file as text for sampling.
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue

            # If the file is smaller than the target budget, include it in full.
            if len(text) <= target_chars:
                snippet = text
            else:
                # Spread sampling across the file so HyDE sees context from
                # multiple regions instead of just the header. Use a small
                # fixed number of windows for simplicity.
                windows = 3
                if target_chars < windows:
                    windows = 1
                segment_len = max(1, target_chars // windows)

                pieces: list[str] = []
                text_len = len(text)
                if windows == 1:
                    start = max(0, (text_len - segment_len) // 2)
                    end = min(text_len, start + segment_len)
                    pieces.append(text[start:end])
                else:
                    for w in range(windows):
                        offset = (text_len - segment_len) * w // (windows - 1)
                        start = max(0, min(offset, max(0, text_len - segment_len)))
                        end = min(text_len, start + segment_len)
                        pieces.append(text[start:end])
                snippet = "\n...\n".join(pieces)

            # Infer language hint for fenced block from extension
            ext = path.suffix.lstrip(".").lower()
            lang = ext if ext in {"py", "rs", "ts", "tsx", "js", "jsx", "go", "rb", "java", "kt", "c", "h", "cpp", "md"} else ""
            fence = f"```{lang}" if lang else "```"

            code_snippets.append(
                f"File: {rel_path}\n{fence}\n{snippet}\n```"
            )

        if code_snippets:
            code_context_block = "\n\n".join(code_snippets)
        else:
            code_context_block = "(no sample code snippets available)"

    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    template = _load_prompt_template("hyde_scope_prompt.md")
    return template.format(
        created=created,
        previous=previous,
        target=target,
        scope_display=scope_display,
        files_block=files_block,
        code_context_block=code_context_block,
    ).strip()


async def _run_hyde_only_query(
    llm_manager: LLMManager | None,
    prompt: str,
    provider_override: LLMProvider | None = None,
    hyde_cfg: HydeConfig | None = None,
) -> str:
    """Run a HyDE-only query using the synthesis LLM provider.

    This bypasses deep_research_impl and does not hit the database or embeddings;
    it simply feeds the prompt directly to an LLM provider. When a provider
    override is supplied (for example, a specialized assembly model), it is
    used instead of the default synthesis provider from the LLM manager.
    """
    if provider_override is None and (not llm_manager or not llm_manager.is_configured()):
        return "LLM not configured for HyDE-only mode."

    try:
        if provider_override is not None:
            provider = provider_override
        else:
            assert llm_manager is not None
            provider = llm_manager.get_synthesis_provider()
    except Exception:
        return "Synthesis provider unavailable for HyDE-only mode."

    # Allow deep, expansive plans by default. The default max token budget is
    # generous (30k) so HyDE can write long, exploratory research plans. When
    # provided, HydeConfig controls the budget and still enforces the same
    # hard cap.
    if hyde_cfg is None:
        hyde_cfg = HydeConfig.from_env()
    max_tokens = hyde_cfg.max_completion_tokens

    try:
        response = await provider.complete(
            prompt=prompt,
            max_completion_tokens=max_tokens,
        )
        if not response or not getattr(response, "content", None):
            return "HyDE-only synthesis returned no content."
        return response.content
    except Exception as exc:
        return f"HyDE-only synthesis failed: {exc}"


async def _run_hyde_map_passes(
    services: Any,
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
    scope_label: str,
    hyde_plan: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Run HyDE-guided deep-research passes and return map-level findings.

    Each finding contains:
    - \"bullet\": original HyDE bullet text
    - \"analysis\": cleaned deep-research answer without any Sources footer
    - \"sources\": metadata[\"sources\"] from deep_research (if present)
    """
    if not hyde_plan or not hyde_plan.strip():
        return [], 0

    if services is None or embedding_manager is None:
        return [], 0

    # Respect workspace/full-scope semantics from the main run.
    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    # Decide whether to drive map passes from individual bullets or grouped
    # sections (title + body). This is controlled via an environment variable to
    # allow experimentation without breaking existing behavior.
    group_mode = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_GROUP_MODE", "bullet")

    if group_mode == "section":
        units = _extract_hyde_sections(hyde_plan, max_sections=10_000)
    else:
        bullets = _extract_hyde_bullets(hyde_plan, max_bullets=10_000)
        units = [{"title": b, "body": b} for b in bullets]

    limit_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT")
    if limit_env:
        try:
            n = int(limit_env)
            if n >= 0:
                units = units[:n]
        except ValueError:
            # Ignore invalid limits and fall back to default behavior.
            pass
    if not units:
        return [], 0

    findings: list[dict[str, Any]] = []
    calls = 0

    path_routing_enabled = os.getenv(
        "CH_AGENT_DOC_CODE_RESEARCH_HYDE_PATH_ROUTING", "0"
    ) == "1"

    for unit in units:
        title = unit["title"]
        body = unit["body"]

        # Build a focused, scoped query for this HyDE aspect or section using
        # an external prompt template.
        template = _load_prompt_template("hyde_map_prompt.md")
        query = template.format(
            scope_display=scope_display,
            section_title=title,
            section_body=body,
        ).strip()

        path_override: str | None = None
        if path_routing_enabled:
            path_override = _infer_hyde_unit_path(scope_label, title, body)

        answer, meta = await _run_research_query_with_metadata(
            services=services,
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            prompt=query,
            scope_label=scope_label,
            path_override=path_override,
        )
        calls += 1

        # Strip any per-section sources footer if the deep-research pipeline
        # decides to add one despite the prompt.
        cleaned, _ = _split_sources_footer(answer)
        cleaned = cleaned.strip()
        if not cleaned:
            continue

        findings.append(
            {
                "bullet": title,
                "analysis": cleaned,
                "sources": meta.get("sources") or {},
            }
        )

    return findings, calls


async def _run_hyde_map_deep_research(
    services: Any,
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
    scope_label: str,
    hyde_plan: str | None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Run optional HyDE-guided deep-research passes per bullet.

    This mirrors the semantic HyDE loop but uses deep_research_impl for each
    selected HyDE bullet. Calls are sequential to preserve single-threaded DB
    access and are always scoped via the same path_filter as the main run.
    """
    findings, calls = await _run_hyde_map_passes(
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        scope_label=scope_label,
        hyde_plan=hyde_plan,
    )
    if not findings:
        return "", [], 0

    sections: list[str] = []
    sections.append("## 7. HyDE-Guided Deep Dives")
    sources_metadata: list[dict[str, Any]] = []

    for idx, finding in enumerate(findings, 1):
        bullet = finding["bullet"]
        cleaned = finding["analysis"]
        sources_meta = finding.get("sources") or {}
        if sources_meta:
            sources_metadata.append(sources_meta)

        # Shorten the bullet for the heading to keep things compact.
        heading_text = bullet.strip()
        max_heading_len = 80
        if len(heading_text) > max_heading_len:
            heading_text = heading_text[: max_heading_len - 1].rstrip() + "…"

        sections.append("")
        sections.append(f"### {idx}. {heading_text}")
        sections.append("")
        sections.append(cleaned)

    return "\n".join(sections).strip(), sources_metadata, calls


async def _run_map_only_merge(
    llm_manager: LLMManager | None,
    project_root: Path,
    scope_label: str,
    meta: AgentDocMetadata,
    reference_table: str,
    findings: list[str],
    structure_mode: str = "canonical",
    assembly_provider: LLMProvider | None = None,
) -> str:
    """Merge multiple map-level findings into a single Agent Doc.

    This uses the synthesis provider directly and does not perform any
    additional database or embedding work. If the LLM manager is not
    configured, a simple concatenation fallback is returned.
    """
    # Degenerate but safe fallback: ensure we always return a doc body.
    cleaned_findings = [f.strip() for f in findings if f.strip()]
    if not cleaned_findings:
        return (
            "# Agent Doc\n\n"
            "_No map-level analyses were available for this agent-doc run._"
        )

    if assembly_provider is None and (not llm_manager or not llm_manager.is_configured()):
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    try:
        if assembly_provider is not None:
            provider = assembly_provider
        else:
            assert llm_manager is not None
            provider = llm_manager.get_synthesis_provider()
    except Exception:
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    # Always use the fluid structural template for the merged document.
    # The legacy canonical 1–5 heading skeleton has been removed.
    mode = "fluid"

    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    created = meta.created_from_sha
    previous = meta.previous_target_sha
    target = meta.target_sha

    findings_blocks: list[str] = []
    for idx, content in enumerate(cleaned_findings, 1):
        findings_blocks.append(f"## Finding {idx}\n\n{content}")
    findings_block = "\n\n".join(findings_blocks)

    reference_section = reference_table.strip()
    if not reference_section:
        reference_section = (
            "## Source References\n\n"
            "No structured source metadata was available for this run. "
            "If the findings contain [N] citations, treat them as best-effort hints."
        )

    # Load the map-only merge CTA prompt from an external template for easier
    # iteration.
    merge_template = _load_prompt_template("map_merge_fluid_prompt.md")
    prompt = merge_template.format(
        project_root=str(project_root),
        scope_display=scope_display,
        created=created,
        previous=previous,
        target=target,
        reference_section=reference_section,
        findings_block=findings_block,
    ).strip()

    try:
        response = await provider.complete(
            prompt=prompt,
            system=(
                "You are synthesizing multiple prior map-level analyses into a single, precise Agent Doc for another AI agent. "
                "Focus on clarity, architectural depth, and preserving existing citations without changing their [N] numbers."
            ),
            max_completion_tokens=8000,
        )
    except Exception:
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    if not response or not getattr(response, "content", None):
        merged = "\n\n".join(cleaned_findings)
        return f"# Agent Doc\n\n{merged}"

    merged_text = response.content.strip()
    if "# Agent Doc" not in merged_text:
        merged_text = "# Agent Doc\n\n" + merged_text

    return merged_text


def _resolve_scope(project_root: Path, scope: Optional[str]) -> tuple[Path, str]:
    """Resolve the scoped folder and its label relative to the project root."""
    if not scope or scope in {".", "/"}:
        return project_root, "/"

    candidate = (project_root / scope).resolve()
    try:
        # Ensure the scope stays inside the project root
        candidate.relative_to(project_root)
    except ValueError:
        raise SystemExit(
            f"Scope '{scope}' resolves outside project root: {candidate}"
        )

    if not candidate.exists() or not candidate.is_dir():
        raise SystemExit(f"Scope folder not found or not a directory: {candidate}")

    scope_label = str(candidate.relative_to(project_root))
    return candidate, scope_label


def _prepare_config_and_services(
    project_root: Path,
    hyde_only: bool,
) -> tuple[Config, LLMManager | None, Any | None, EmbeddingManager | None]:
    """Load ChunkHound config, LLM manager, and optional DB/services layer."""
    # Force Config to treat project_root as the target directory so that
    # .chunkhound.json and database paths are resolved relative to the
    # workspace we are documenting (not necessarily the current CWD).
    config_args = argparse.Namespace(path=str(project_root), config=None)
    config = Config(args=config_args)

    # LLM manager is required for all modes; DB/services are only required when
    # not running in HyDE-only mode.
    llm_manager: LLMManager | None = None
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
    except ValueError as exc:
        raise SystemExit(
            f"LLM provider setup failed: {exc}\n"
            "Configure an LLM provider in .chunkhound.json or via CHUNKHOUND_LLM_API_KEY."
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unexpected error setting up LLM provider: {exc}") from exc

    services: Any | None = None
    embedding_manager: EmbeddingManager | None = None

    if not hyde_only:
        db_path = config.database.path
        if not db_path:
            raise SystemExit("Database path not configured in ChunkHound config.")

        if not db_path.exists():
            raise SystemExit(
                f"Database not found at {db_path}. Run 'chunkhound index .' in the project root first."
            )

        embedding_manager = EmbeddingManager()

        try:
            if config.embedding:
                provider = EmbeddingProviderFactory.create_provider(config.embedding)
                embedding_manager.register_provider(provider, set_default=True)
        except ValueError as exc:
            raise SystemExit(
                f"Embedding provider setup failed: {exc}\n"
                "Configure embeddings in .chunkhound.json or via CHUNKHOUND_EMBEDDING__API_KEY."
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(
                f"Unexpected error setting up embedding provider: {exc}"
            ) from exc

        services = create_services(
            db_path=db_path,
            config=config,
            embedding_manager=embedding_manager,
        )

        # Ensure the database actually contains indexed code before invoking research.
        try:
            stats = services.provider.get_stats()
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Failed to read database stats: {exc}") from exc

        if not stats or stats.get("files", 0) == 0:
            raise SystemExit(
                "ChunkHound database contains no indexed files.\n"
                "Run 'chunkhound index .' in the project root before generating the agent doc."
            )

        if stats.get("embeddings", 0) == 0:
            raise SystemExit(
                "ChunkHound database contains no embeddings.\n"
                "Run 'chunkhound index .' without --no-embeddings so deep research can use semantic search."
            )

    return config, llm_manager, services, embedding_manager


def _compute_git_context(
    output_path: Path,
    project_root: Path,
    scope_path: Path,
) -> tuple[str, str, str, Path | None]:
    """Determine baseline/previous/current SHAs and the git root for diffs."""
    existing_meta = _parse_existing_metadata(output_path)

    git_root_for_metadata: Path | None = None
    current_sha = _get_head_sha(project_root)
    if current_sha != "NO_GIT_HEAD":
        git_root_for_metadata = project_root
    elif scope_path != project_root:
        alt_sha = _get_head_sha(scope_path)
        if alt_sha != "NO_GIT_HEAD":
            current_sha = alt_sha
            git_root_for_metadata = scope_path

    if existing_meta is None or existing_meta.created_from_sha == "NO_GIT_HEAD":
        # Either this is the first run, or a previous run took place in a
        # non-git workspace and recorded a placeholder. In both cases, treat
        # the current SHA (if available) as the new baseline.
        created_from_sha = current_sha
        previous_target_sha = current_sha
    else:
        created_from_sha = existing_meta.created_from_sha
        previous_target_sha = existing_meta.target_sha

    return created_from_sha, previous_target_sha, current_sha, git_root_for_metadata


def _build_llm_metadata_and_assembly(
    config: Config,
    llm_manager: LLMManager | None,
) -> tuple[dict[str, str], LLMProvider | None]:
    """Capture LLM configuration snapshot and optional assembly provider."""
    llm_meta: dict[str, str] = {}
    assembly_provider: LLMProvider | None = None

    if not config.llm:
        return llm_meta, assembly_provider

    llm = config.llm
    llm_meta["provider"] = llm.provider
    if llm.synthesis_provider:
        llm_meta["synthesis_provider"] = llm.synthesis_provider
    if llm.synthesis_model:
        llm_meta["synthesis_model"] = llm.synthesis_model
    if llm.utility_model:
        llm_meta["utility_model"] = llm.utility_model
    if llm.codex_reasoning_effort_synthesis:
        llm_meta["codex_reasoning_effort_synthesis"] = (
            llm.codex_reasoning_effort_synthesis
        )
    if llm.codex_reasoning_effort_utility:
        llm_meta["codex_reasoning_effort_utility"] = (
            llm.codex_reasoning_effort_utility
        )

    # Optional specialization for final assembly: allow env-driven overrides
    # for provider/model/reasoning_effort used by the map-only merge step.
    assembly_provider_name = os.getenv("CH_AGENT_DOC_ASSEMBLY_PROVIDER")
    assembly_model_name = os.getenv("CH_AGENT_DOC_ASSEMBLY_MODEL")
    assembly_effort = os.getenv("CH_AGENT_DOC_ASSEMBLY_REASONING_EFFORT")

    # If no per-run overrides are provided, fall back to the static LLM
    # config values loaded from .chunkhound.json or environment.
    if not assembly_provider_name and getattr(llm, "assembly_provider", None):
        assembly_provider_name = llm.assembly_provider
    if not assembly_model_name and getattr(llm, "assembly_model", None):
        assembly_model_name = llm.assembly_model
    if (
        not assembly_model_name
        and getattr(llm, "assembly_synthesis_model", None)  # type: ignore[attr-defined]
    ):
        assembly_model_name = llm.assembly_synthesis_model  # type: ignore[attr-defined]
    if not assembly_effort and getattr(llm, "assembly_reasoning_effort", None):
        assembly_effort = llm.assembly_reasoning_effort

    if llm_manager is not None and (
        assembly_provider_name or assembly_model_name or assembly_effort
    ):
        try:
            utility_cfg, synth_cfg = llm.get_provider_configs()
            assembly_cfg = synth_cfg.copy()
            if assembly_provider_name:
                assembly_cfg["provider"] = assembly_provider_name
            if assembly_model_name:
                assembly_cfg["model"] = assembly_model_name
            if assembly_effort:
                assembly_cfg["reasoning_effort"] = assembly_effort.strip().lower()

            # Optional HyDE-specific behavior: when enabled, request that the
            # Codex CLI overlay uses the full CODEX_HOME configuration instead
            # of the minimal, history-less overlay. This keeps code_research
            # providers on the default overlay while allowing HyDE + assembly
            # to mimic a manual `codex exec` run.
            if os.getenv("CH_AGENT_DOC_HYDE_USE_FULL_CODEX_CONFIG", "0") == "1":
                assembly_cfg["use_base_codex_config"] = True

            # Use the public factory on LLMManager to construct the provider.
            assembly_provider = llm_manager.create_provider_for_config(assembly_cfg)

            # Record effective assembly configuration in metadata for
            # inspection. Use resolved provider/model from the provider
            # instance as a source of truth.
            try:
                llm_meta["assembly_synthesis_provider"] = assembly_cfg.get(
                    "provider", assembly_provider.name
                )
                llm_meta["assembly_synthesis_model"] = assembly_cfg.get(
                    "model", assembly_provider.model
                )
            except Exception:
                # Fallback: best-effort metadata; assembly still works.
                pass
            if "reasoning_effort" in assembly_cfg:
                llm_meta["assembly_reasoning_effort"] = str(
                    assembly_cfg["reasoning_effort"]
                )
        except Exception:
            # If assembly specialization fails for any reason, fall back to
            # the standard synthesis provider and omit assembly-specific
            # metadata.
            assembly_provider = None

    return llm_meta, assembly_provider


def _compute_diffs_for_scope(
    created_from_sha: str,
    previous_target_sha: str,
    current_sha: str,
    scope_label: str,
    git_root_for_metadata: Path | None,
    project_root: Path,
) -> tuple[list[str], list[str]]:
    """Compute and scope-filter git diffs for prompt construction."""
    diff_root = git_root_for_metadata or project_root
    raw_diff_since_created = _get_name_status_diff(
        created_from_sha,
        current_sha,
        diff_root,
    )
    raw_diff_since_previous = _get_name_status_diff(
        previous_target_sha,
        current_sha,
        diff_root,
    )

    def _filter_diff_for_scope(diff_lines: list[str]) -> list[str]:
        if scope_label == "/":
            return diff_lines
        prefix = scope_label.rstrip("/") + "/"
        filtered: list[str] = []
        for line in diff_lines:
            parts = line.split("\t", 1)
            path = parts[1] if len(parts) == 2 else ""
            if path.startswith(prefix):
                filtered.append(line)
        return filtered

    return _filter_diff_for_scope(raw_diff_since_created), _filter_diff_for_scope(
        raw_diff_since_previous
    )


async def _run_hyde_bootstrap(
    *,
    hyde_only: bool,
    project_root: Path,
    scope_path: Path,
    scope_label: str,
    meta: AgentDocMetadata,
    hyde_cfg: HydeConfig,
    llm_manager: LLMManager | None,
    assembly_provider: LLMProvider | None,
    out_dir: Optional[Path],
) -> str | None:
    """Run a HyDE planning pass and persist the plan when applicable.

    HyDE planning is a core part of the deep-doc pipeline for code_research
    runs. For HyDE-only mode this is a no-op and returns None.
    """
    if hyde_only:
        return None

    file_paths = _collect_scope_files(scope_path, project_root, hyde_cfg=hyde_cfg)
    hyde_prompt = _build_hyde_scope_prompt(
        meta=meta,
        scope_label=scope_label,
        file_paths=file_paths,
        hyde_cfg=hyde_cfg,
        project_root=project_root,
    )

    # Optional debugging hooks: allow inspecting the exact HyDE scope prompt,
    # and optionally stop after writing it without calling the LLM.
    dump_prompt = os.getenv("CH_AGENT_DOC_HYDE_DUMP_PROMPT", "0") == "1"
    prompt_only = os.getenv("CH_AGENT_DOC_HYDE_PROMPT_ONLY", "0") == "1"
    if out_dir is not None and (dump_prompt or prompt_only):
        safe_scope = scope_label.replace("/", "_") or "root"
        prompt_path = out_dir / f"hyde_scope_prompt_{safe_scope}.md"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(hyde_prompt, encoding="utf-8")
    if prompt_only:
        # Stop after emitting the prompt file so it can be inspected manually.
        raise SystemExit(0)

    hyde_plan = await _run_hyde_only_query(
        llm_manager=llm_manager,
        prompt=hyde_prompt,
        provider_override=assembly_provider,
        hyde_cfg=hyde_cfg,
    )
    # Persist HyDE plan for debugging/inspection if output directory provided
    if out_dir is not None and hyde_plan and hyde_plan.strip():
        safe_scope = scope_label.replace("/", "_") or "root"
        hyde_path = out_dir / f"hyde_plan_{safe_scope}.md"
        hyde_path.parent.mkdir(parents=True, exist_ok=True)
        hyde_path.write_text(hyde_plan, encoding="utf-8")

    return hyde_plan


async def _run_deep_doc_body_pipeline(
    *,
    project_root: Path,
    scope_path: Path,
    scope_label: str,
    hyde_only: bool,
    generator_mode: str,
    structure_mode: str,
    services: Any | None,
    embedding_manager: EmbeddingManager | None,
    llm_manager: LLMManager | None,
    assembly_provider: LLMProvider | None,
    meta: AgentDocMetadata,
    hyde_cfg: HydeConfig,
    diff_since_created: list[str],
    diff_since_previous: list[str],
    hyde_plan: str | None,
    out_dir: Optional[Path],
) -> tuple[str, dict[str, str], list[dict[str, Any]], int, bool, bool]:
    """Run deep-research / HyDE pipelines and assemble the main body.

    Returns the final body (with any unified Sources footer appended),
    unified source files and chunks, total deep-research calls, whether HyDE
    map was enabled, and whether map-only mode was used.
    """
    # For code_research mode we track deep-research calls and sources across
    # all passes (overview, HyDE map, or map-only).
    unified_source_files: dict[str, str] = {}
    unified_source_chunks: list[dict[str, Any]] = []

    def _merge_sources_from_metadata(meta_dict: dict[str, Any]) -> None:
        sources = meta_dict.get("sources") or {}
        for fp in sources.get("files", []):
            if fp:
                unified_source_files.setdefault(fp, "")
        for c in sources.get("chunks", []):
            fp = c.get("file_path")
            if not fp:
                continue
            unified_source_chunks.append(
                {
                    "file_path": fp,
                    "start_line": c.get("start_line"),
                    "end_line": c.get("end_line"),
                }
            )

    # Map-only code_research is opt-in via environment to avoid breaking
    # existing flows.
    map_only_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_MAP_ONLY", "0")
    code_research_map_only = (not hyde_only and map_only_env == "1")

    body: str = ""
    total_research_calls = 0
    enable_hyde_map = False
    map_findings: list[dict[str, Any]] = []

    # Run overview research (or HyDE-only synthesis) for the main document
    # body, or collect map-only findings for later merge.
    if hyde_only:
        # HyDE-only mode: build prompt purely from scope file layout and LLM plan.
        file_paths = _collect_scope_files(scope_path, project_root, hyde_cfg=hyde_cfg)
        overview_prompt = _build_hyde_scope_prompt(
            meta=meta,
            scope_label=scope_label,
            file_paths=file_paths,
            hyde_cfg=hyde_cfg,
            project_root=project_root,
        )

        # Optional debugging hooks: allow inspecting the exact HyDE scope
        # prompt, and optionally stop after writing it without calling the LLM.
        dump_prompt = os.getenv("CH_AGENT_DOC_HYDE_DUMP_PROMPT", "0") == "1"
        prompt_only = os.getenv("CH_AGENT_DOC_HYDE_PROMPT_ONLY", "0") == "1"
        if out_dir is not None and (dump_prompt or prompt_only):
            safe_scope = scope_label.replace("/", "_") or "root"
            prompt_path = out_dir / f"hyde_scope_prompt_{safe_scope}.md"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(overview_prompt, encoding="utf-8")
        if prompt_only:
            # Stop after emitting the prompt file so it can be inspected
            # manually.
            raise SystemExit(0)

        overview_answer = await _run_hyde_only_query(
            llm_manager=llm_manager,
            prompt=overview_prompt,
            provider_override=assembly_provider,
            hyde_cfg=hyde_cfg,
        )
        body = overview_answer
    else:
        assert services is not None
        if code_research_map_only:
            assert embedding_manager is not None
            # Prefer HyDE bullets for map-only when a HyDE plan is available.
            if hyde_plan and hyde_plan.strip():
                map_findings, total_research_calls = await _run_hyde_map_passes(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    scope_label=scope_label,
                    hyde_plan=hyde_plan,
                )
            else:
                # Fallback: single overview-style map pass using the standard
                # research prompt, treated as one of the map findings.
                overview_prompt = _build_research_prompt(
                    meta,
                    diff_since_created,
                    diff_since_previous,
                    scope_label=scope_label,
                    hyde_plan=None,
                )
                answer, overview_meta = await _run_research_query_with_metadata(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    prompt=overview_prompt,
                    scope_label=scope_label,
                )
                total_research_calls = 1
                cleaned, _ = _split_sources_footer(answer)
                cleaned = cleaned.strip()
                if cleaned:
                    map_findings.append(
                        {
                            "bullet": "Overview",
                            "analysis": cleaned,
                            "sources": overview_meta.get("sources") or {},
                        }
                    )

            # Merge sources from all map-level passes.
            for finding in map_findings:
                sources_meta = finding.get("sources") or {}
                _merge_sources_from_metadata({"sources": sources_meta})
        else:
            # Standard code_research overview with optional HyDE-guided deep dives.
            overview_prompt = _build_research_prompt(
                meta,
                diff_since_created,
                diff_since_previous,
                scope_label=scope_label,
                hyde_plan=hyde_plan,
            )
            assert embedding_manager is not None
            overview_answer, overview_meta = await _run_research_query_with_metadata(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                prompt=overview_prompt,
                scope_label=scope_label,
            )

            # Optional debug hook: persist the raw overview code_research answer
            # before any post-processing or assembly. This mirrors the HyDE
            # prompt/plan dump behavior so engineers can inspect exactly what
            # deep_research_impl returned for the overview stage.
            dump_overview = os.getenv(
                "CH_AGENT_DOC_CODE_RESEARCH_DUMP_OVERVIEW", "0"
            ) == "1"
            if dump_overview and out_dir is not None:
                safe_scope = scope_label.replace("/", "_") or "root"
                overview_path = out_dir / f"overview_code_research_{safe_scope}.md"
                overview_path.parent.mkdir(parents=True, exist_ok=True)
                overview_path.write_text(overview_answer, encoding="utf-8")

            # Deep research synthesis already appends a Sources footer. Strip it
            # so we can build a unified footer later from merged sources.
            body, _ = _split_sources_footer(overview_answer)
            _merge_sources_from_metadata(overview_meta)
            total_research_calls = 1

            # Optional HyDE-guided map loop for code_research: run follow-up
            # deep research passes per HyDE bullet and append a dedicated
            # section with the resulting deep dives. Track iteration counts for
            # generation_stats.
            hyde_map_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP", "0")
            enable_hyde_map = (
                hyde_map_env == "1"
                and hyde_plan is not None
                and hyde_plan.strip() != ""
            )
            if enable_hyde_map:
                assert services is not None
                (
                    hyde_map_section,
                    hyde_map_sources_list,
                    hyde_calls,
                ) = await _run_hyde_map_deep_research(
                    services=services,
                    embedding_manager=embedding_manager,
                    llm_manager=llm_manager,
                    scope_label=scope_label,
                    hyde_plan=hyde_plan,
                )
                if hyde_map_section:
                    body = body.rstrip() + "\n\n" + hyde_map_section
                total_research_calls += hyde_calls
                for src_meta in hyde_map_sources_list:
                    _merge_sources_from_metadata(src_meta)

    # For any code_research mode, deduplicate the collected sources and build a
    # global file reference map that can be reused for both map-only merging
    # and the final Sources footer.
    unified_chunks_dedup: list[dict[str, Any]] = []
    ref_map: dict[str, int] = {}
    cm_for_sources: CitationManager | None = None

    if not hyde_only and generator_mode == "code_research":
        # Deduplicate chunks by (file_path, start_line, end_line)
        dedup: dict[tuple[str, int | None, int | None], dict[str, Any]] = {}
        for chunk in unified_source_chunks:
            key = (
                chunk.get("file_path"),
                chunk.get("start_line"),
                chunk.get("end_line"),
            )
            if not key[0]:
                continue
            if key not in dedup:
                dedup[key] = chunk

        unified_chunks_dedup = list(dedup.values())

        if unified_source_files and unified_chunks_dedup:
            cm_for_sources = CitationManager()
            ref_map = cm_for_sources.build_file_reference_map(
                unified_chunks_dedup, unified_source_files
            )

    # In map-only mode, perform a single agentic merge over the collected
    # map-level findings using only the synthesis provider (no DB access).
    if code_research_map_only:
        cm = cm_for_sources or CitationManager()
        reference_table = cm.format_reference_table(ref_map) if ref_map else ""
        remapped_findings: list[str] = []
        for finding in map_findings:
            analysis = finding["analysis"]
            sources_meta = finding.get("sources") or {}
            source_files = sources_meta.get("files") or []
            source_chunks = sources_meta.get("chunks") or []
            if ref_map and source_files:
                cluster_files_dict = {fp: "" for fp in source_files}
                cluster_ref_map = cm.build_file_reference_map(
                    source_chunks, cluster_files_dict
                )
                analysis = cm.remap_cluster_citations(
                    analysis, cluster_ref_map, ref_map
                )
            remapped_findings.append(analysis)

        body = await _run_map_only_merge(
            llm_manager=llm_manager,
            project_root=project_root,
            scope_label=scope_label,
            meta=meta,
            reference_table=reference_table,
            findings=remapped_findings,
            structure_mode=structure_mode,
            assembly_provider=assembly_provider,
        )

    body_to_use = body

    # Build a unified Sources footer for code_research mode using the merged
    # sources metadata from overview and/or map passes. This replaces any
    # internal deep-research footer so the final tree reflects all passes.
    if not hyde_only and generator_mode == "code_research":
        if unified_source_files and unified_chunks_dedup:
            try:
                cm = cm_for_sources or CitationManager()
                sources_footer = cm.build_sources_footer(
                    unified_chunks_dedup, unified_source_files, ref_map or None
                )
                # Remove any residual Sources footer that might have slipped in
                # from earlier flows, then append the unified one.
                if "## Sources" in body_to_use:
                    cleaned_body, _ = _split_sources_footer(body_to_use)
                    body_to_use = cleaned_body
                body_to_use = body_to_use.rstrip() + "\n\n" + sources_footer
            except Exception:
                # Footer is best-effort; never break main generation.
                pass

    return (
        body_to_use,
        unified_source_files,
        unified_chunks_dedup,
        total_research_calls,
        enable_hyde_map,
        code_research_map_only,
    )


def _build_generation_stats(
    *,
    generator_mode: str,
    hyde_map_enabled: bool,
    code_research_map_only: bool,
    structure_mode: str,
    hyde_only: bool,
    total_research_calls: int,
    unified_source_files: dict[str, str],
    unified_chunks_dedup: list[dict[str, Any]],
    services: Any | None,
    scope_label: str,
) -> dict[str, str]:
    """Populate generation_stats for visibility into the generation process."""
    gen_stats: dict[str, str] = {}
    gen_stats["generator_mode"] = generator_mode
    # HyDE planning is always enabled for code_research runs and skipped only
    # in HyDE-only mode.
    gen_stats["hyde_bootstrap"] = "0" if hyde_only else "1"
    gen_stats["hyde_map_enabled"] = "1" if hyde_map_enabled else "0"
    if code_research_map_only:
        gen_stats["code_research_map_only"] = "1"
    # Record which structural scheme was used for this run so downstream
    # analysis can distinguish canonical vs fluid layouts.
    if structure_mode:
        gen_stats["structure_mode"] = structure_mode

    if not hyde_only and generator_mode == "code_research" and services is not None:
        gen_stats["code_research_total_calls"] = str(total_research_calls)
        gen_stats["code_research_sources_files"] = str(len(unified_source_files))
        gen_stats["code_research_sources_chunks"] = str(len(unified_chunks_dedup))
        # Compute coverage stats for the scoped folder using the database as
        # ground truth for which files and chunks are actually indexed in ChunkHound.
        scope_total_files, scope_total_chunks, scoped_files = _compute_db_scope_stats(
            services, scope_label
        )

        if scope_total_files > 0:
            gen_stats["scope_total_files_indexed"] = str(scope_total_files)
            coverage_ratio = float(len(unified_source_files)) / float(scope_total_files)
            gen_stats["scope_coverage_percent_indexed"] = (
                f"{coverage_ratio * 100.0:.2f}"
            )
        if scope_total_chunks > 0:
            gen_stats["scope_total_chunks_indexed"] = str(scope_total_chunks)
            chunk_cov = float(len(unified_chunks_dedup)) / float(scope_total_chunks)
            gen_stats["scope_chunk_coverage_percent_indexed"] = (
                f"{chunk_cov * 100.0:.2f}"
            )
        # For debugging: track files that were indexed for the scoped folder
        # but never surfaced in unified sources.
        try:
            if scoped_files:
                referenced_files = set(unified_source_files.keys())
                missing_files = sorted(scoped_files - referenced_files)
                gen_stats["scope_unreferenced_files_count"] = str(len(missing_files))
                if missing_files:
                    # Store as a semicolon-separated list to keep the header
                    # reasonably compact while still debuggable.
                    gen_stats["scope_unreferenced_files"] = ";".join(missing_files)
        except Exception:
            # Debug-only; never break generation.
            pass

        limit_env = os.getenv("CH_AGENT_DOC_CODE_RESEARCH_HYDE_MAP_LIMIT")
        if limit_env:
            gen_stats["hyde_map_limit_env"] = limit_env

    return gen_stats


def _write_deep_doc_outputs(
    output_path: Path,
    out_dir: Optional[Path],
    body: str,
    meta: AgentDocMetadata,
) -> None:
    """Write the monolithic doc and per-subsystem split artifacts."""
    metadata_block = _format_metadata_block(meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_doc = metadata_block + body
    output_path.write_text(final_doc, encoding="utf-8")

    # Also emit split per-subsystem docs for agentic search
    meta_dict: dict[str, Any] = {
        "created_from_sha": meta.created_from_sha,
        "previous_target_sha": meta.previous_target_sha,
        "target_sha": meta.target_sha,
        "generated_at": meta.generated_at,
    }
    if meta.llm_config:
        meta_dict["llm_config"] = meta.llm_config

    if out_dir is not None:
        split_agent_doc_text_fluid(
            body,
            meta_dict,
            base_output_dir=out_dir,
        )


async def _generate_agent_doc(
    project_root: Path,
    output_path: Path,
    scope: Optional[str] = None,
    hyde_only: bool = False,
    out_dir: Optional[Path] = None,
) -> None:
    """Main implementation that coordinates config, research, and writing."""
    # Ensure we resolve paths early to avoid surprises
    project_root = project_root.resolve()
    output_path = output_path.resolve()

    # Resolve scope folder within project_root
    scope_path, scope_label = _resolve_scope(project_root, scope)

    # Force a single, code-research-based generation mode for the agent doc.
    generator_mode = "code_research"

    # Structural mode for the final Agent Doc. The legacy canonical 1–5
    # heading scheme has been removed; we keep this variable only so
    # generation_stats can record that the fluid layout is in use.
    structure_mode = "fluid"

    # Determine commit metadata (baseline, previous, current) and the git root
    # to use for diffs.
    created_from_sha, previous_target_sha, current_sha, git_root_for_metadata = (
        _compute_git_context(
            output_path=output_path,
            project_root=project_root,
            scope_path=scope_path,
        )
    )

    # Prepare ChunkHound config, LLM manager, and (optionally) services.
    config, llm_manager, services, embedding_manager = _prepare_config_and_services(
        project_root=project_root,
        hyde_only=hyde_only,
    )

    # Capture LLM configuration snapshot for metadata (excluding secrets) and
    # optionally build a specialized provider for the final Agent Doc assembly
    # step (map-only merge). When no assembly-specific overrides are provided,
    # all operations share the standard synthesis provider.
    llm_meta, assembly_provider = _build_llm_metadata_and_assembly(
        config=config,
        llm_manager=llm_manager,
    )

    meta = AgentDocMetadata(
        created_from_sha=created_from_sha,
        previous_target_sha=previous_target_sha,
        target_sha=current_sha,
        generated_at=datetime.now(timezone.utc).isoformat(),
        llm_config=llm_meta,
        generation_stats={},
    )

    # Build HyDE configuration once so all HyDE helpers share consistent
    # limits and environment-driven overrides.
    hyde_cfg = HydeConfig.from_env()

    # Compute diffs for the prompt, then scope-filter them.
    diff_since_created, diff_since_previous = _compute_diffs_for_scope(
        created_from_sha=meta.created_from_sha,
        previous_target_sha=meta.previous_target_sha,
        current_sha=meta.target_sha,
        scope_label=scope_label,
        git_root_for_metadata=git_root_for_metadata,
        project_root=project_root,
    )

    # Optional HyDE bootstrap: generate a heuristic plan from file layout and
    # feed it into deep research as planning context. This is disabled when
    # running in HyDE-only mode (which already uses HyDE output directly).
    hyde_plan = await _run_hyde_bootstrap(
        hyde_only=hyde_only,
        project_root=project_root,
        scope_path=scope_path,
        scope_label=scope_label,
        meta=meta,
        hyde_cfg=hyde_cfg,
        llm_manager=llm_manager,
        assembly_provider=assembly_provider,
        out_dir=out_dir,
    )

    (
        body_to_use,
        unified_source_files,
        unified_chunks_dedup,
        total_research_calls,
        enable_hyde_map,
        code_research_map_only,
    ) = await _run_deep_doc_body_pipeline(
        project_root=project_root,
        scope_path=scope_path,
        scope_label=scope_label,
        hyde_only=hyde_only,
        generator_mode=generator_mode,
        structure_mode=structure_mode,
        services=services,
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        assembly_provider=assembly_provider,
        meta=meta,
        hyde_cfg=hyde_cfg,
        diff_since_created=diff_since_created,
        diff_since_previous=diff_since_previous,
        hyde_plan=hyde_plan,
        out_dir=out_dir,
    )

    # Populate generation_stats for visibility into the generation process.
    gen_stats = _build_generation_stats(
        generator_mode=generator_mode,
        hyde_map_enabled=enable_hyde_map,
        code_research_map_only=code_research_map_only,
        structure_mode=structure_mode,
        hyde_only=hyde_only,
        total_research_calls=total_research_calls,
        unified_source_files=unified_source_files,
        unified_chunks_dedup=unified_chunks_dedup,
        services=services,
        scope_label=scope_label,
    )

    # Merge into metadata for header rendering (existing values win if set).
    meta.generation_stats.update({k: v for k, v in gen_stats.items() if v != ""})

    _write_deep_doc_outputs(
        output_path=output_path,
        out_dir=out_dir,
        body=body_to_use,
        meta=meta,
    )


def main() -> None:
    """Entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate or update a single agent-facing architecture/operations document "
            "for a scoped folder using ChunkHound's deep research pipeline."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("."),
        help="Project root path (defaults to current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_RELATIVE,
        help=(
            "Output markdown file path for the agent doc. "
            "Defaults to operations/chunkhound_agent_doc.md."
        ),
    )
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help=(
            "Relative folder within the project to document "
            "(e.g. 'chunkhound', '.' or '/' for entire workspace). "
            "If omitted, you will be prompted interactively."
        ),
    )
    parser.add_argument(
        "--hyde-only",
        action="store_true",
        help=(
            "Use HyDE-only mode: call the synthesis LLM directly with HyDE "
            "prompts, without deep_research or database/embedding access."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("operations"),
        help=(
            "Base directory for generated documentation and HyDE planning files "
            "(relative to workspace root unless absolute). The main doc will be "
            "written as '<out-dir>/chunkhound_agent_doc.md' unless --output is "
            "explicitly provided."
        ),
    )

    args = parser.parse_args()

    try:
        scope = args.scope
        if scope is None:
            try:
                user_input = input(
                    "Folder to document (relative to project root, '.' or '/' for entire workspace): "
                ).strip()
                scope = user_input or "."
            except EOFError:
                scope = "."

        # Resolve project root and output directory
        project_root = args.path.resolve()

        # Determine output directory relative to project root
        if args.output != DEFAULT_OUTPUT_RELATIVE:
            # Explicit output path wins; out-dir is derived from it
            output_path = (
                args.output
                if args.output.is_absolute()
                else (project_root / args.output).resolve()
            )
            out_dir = output_path.parent
        else:
            if args.out_dir.is_absolute():
                out_dir = args.out_dir.resolve()
            else:
                out_dir = (project_root / args.out_dir).resolve()
            output_path = out_dir / DEFAULT_OUTPUT_RELATIVE.name

        asyncio.run(
            _generate_agent_doc(
                project_root=project_root,
                output_path=output_path,
                scope=scope,
                hyde_only=args.hyde_only,
                out_dir=out_dir,
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
