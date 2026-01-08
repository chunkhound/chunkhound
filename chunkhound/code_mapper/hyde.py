from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Literal

from loguru import logger

from chunkhound.code_mapper.models import AgentDocMetadata, HydeConfig
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager


def load_hyde_scope_template(
    *, mode: Literal["architectural", "operational"] = "architectural"
) -> str:
    """Load the packaged HyDE scope prompt template."""
    package = "chunkhound.code_mapper.prompts"
    filename = (
        "hyde_scope_prompt.md"
        if mode == "architectural"
        else "hyde_scope_prompt_operational.md"
    )
    with (
        importlib.resources.files(package)
        .joinpath(filename)
        .open("r", encoding="utf-8") as f
    ):
        return f.read().strip()


def load_hyde_scope_context_template(
    *, mode: Literal["architectural", "operational"] = "architectural"
) -> str:
    """Load the packaged HyDE scope prompt template for --context mode.

    Note: The context-mode scope prompt is currently shared across both
    architectural and operational maps, so `mode` does not affect the loaded
    template.
    """
    package = "chunkhound.code_mapper.prompts"
    filename = "hyde_scope_prompt_context.md"
    with (
        importlib.resources.files(package)
        .joinpath(filename)
        .open("r", encoding="utf-8") as f
    ):
        return f.read().strip()


async def run_hyde_only_query(
    *,
    llm_manager: LLMManager | None,
    prompt: str,
    provider_override: LLMProvider | None = None,
    hyde_cfg: HydeConfig | None = None,
) -> tuple[str, bool]:
    """Run a HyDE-only query (no DB / embeddings).

    Returns (content, success). On failure, content contains a short diagnostic
    string suitable for CLI display.
    """
    if provider_override is None and (
        not llm_manager or not llm_manager.is_configured()
    ):
        return "LLM not configured for HyDE-only mode.", False

    try:
        provider = provider_override or (
            llm_manager.get_synthesis_provider() if llm_manager else None
        )
        if provider is None:
            return "Synthesis provider unavailable for HyDE-only mode.", False
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug(f"HyDE-only provider lookup failed: {exc}")
        return "Synthesis provider unavailable for HyDE-only mode.", False

    if hyde_cfg is None:
        hyde_cfg = HydeConfig.from_env()

    try:
        response = await provider.complete(
            prompt=prompt,
            max_completion_tokens=hyde_cfg.max_completion_tokens,
        )
        if not response or not getattr(response, "content", None):
            return "HyDE-only synthesis returned no content.", False
        return response.content, True
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug(f"HyDE-only synthesis failed: {exc}")
        return f"HyDE-only synthesis failed: {exc}", False


def build_hyde_scope_prompt(
    *,
    meta: AgentDocMetadata,
    scope_label: str,
    file_paths: list[str],
    hyde_cfg: HydeConfig,
    context: str | None = None,
    mode: Literal["architectural", "operational"] = "architectural",
    template: str | None = None,
    project_root: Path | None = None,
) -> str:
    """Build a HyDE-only prompt from file layout + sampled code snippets."""
    scope_display = "/" if scope_label == "/" else f"./{scope_label}"

    if project_root is None:
        project_root = Path.cwd()

    if template is None:
        template = (
            load_hyde_scope_context_template(mode=mode)
            if context is not None and context.strip()
            else load_hyde_scope_template(mode=mode)
        )

    if context is not None and context.strip():
        mode_label = "architectural" if mode == "architectural" else "operational"
        files_block = ""
        context_body = context.strip()
        code_context_block = (
            f"User-provided context (authoritative; {mode_label} planning):\n\n"
            "````markdown\n"
            f"{context_body}\n"
            "````"
        )
        return template.format(
            created=meta.created_from_sha,
            previous=meta.previous_target_sha,
            target=meta.target_sha,
            scope_display=scope_display,
            files_block=files_block,
            code_context_block=code_context_block,
        ).strip()

    files_block = (
        "\n".join(f"- {p}" for p in file_paths)
        if file_paths
        else ("- (no files discovered)")
    )

    snippet_char_budget = max(
        0, int(getattr(hyde_cfg, "max_snippet_tokens", 100_000)) * 4
    )
    max_chars_per_file = hyde_cfg.max_snippet_chars

    binary_exts = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".ico",
        ".bmp",
        ".tiff",
        ".tif",
        ".ttf",
        ".otf",
        ".woff",
        ".woff2",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".pdf",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".wav",
        ".flac",
        ".so",
        ".dll",
        ".dylib",
        ".o",
        ".a",
        ".obj",
        ".exe",
        ".class",
        ".jar",
        ".bin",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".duckdb",
        ".lance",
    }
    max_snippet_file_bytes = 1_000_000

    file_infos: list[tuple[str, Path, int]] = []
    total_bytes = 0

    for rel_path in file_paths:
        path = project_root / rel_path
        try:
            if not path.is_file():
                continue
            size = path.stat().st_size
        except OSError:
            continue
        if path.suffix.lower() in binary_exts:
            continue
        if size > max_snippet_file_bytes or size <= 0:
            continue
        file_infos.append((rel_path, path, size))
        total_bytes += size

    if snippet_char_budget <= 0 or total_bytes <= 0 or not file_infos:
        code_context_block = "(no sample code snippets available)"
    else:
        ratio = min(1.0, float(snippet_char_budget) / float(total_bytes))
        per_file_cap = max_chars_per_file if max_chars_per_file > 0 else None

        max_files_for_snippets = hyde_cfg.max_snippet_files
        code_snippets: list[str] = []

        for idx, (rel_path, path, size) in enumerate(file_infos):
            if max_files_for_snippets > 0 and idx >= max_files_for_snippets:
                break

            target_chars = int(size * ratio)
            if per_file_cap is not None:
                target_chars = min(target_chars, per_file_cap)
            if target_chars <= 0:
                continue

            try:
                with path.open("rb") as f:
                    raw_prefix = f.read(8192)
                if b"\x00" in raw_prefix:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue

            if len(text) <= target_chars:
                snippet = text
            else:
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

            from chunkhound.core.types.common import Language

            language = Language.from_file_extension(path)
            lang = "" if language == Language.UNKNOWN else language.value
            fence = f"```{lang}" if lang else "```"
            code_snippets.append(f"File: {rel_path}\n{fence}\n{snippet}\n```")

        code_context_block = (
            "\n\n".join(code_snippets)
            if code_snippets
            else "(no sample code snippets available)"
        )

    return template.format(
        created=meta.created_from_sha,
        previous=meta.previous_target_sha,
        target=meta.target_sha,
        scope_display=scope_display,
        files_block=files_block,
        code_context_block=code_context_block,
    ).strip()
