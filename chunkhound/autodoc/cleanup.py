from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from chunkhound.autodoc.markdown_utils import (
    _chunked,
    _ensure_overview_heading,
    _strip_first_heading,
)
from chunkhound.autodoc.models import CleanupConfig, CodeMapperTopic
from chunkhound.autodoc.taint import _normalize_taint
from chunkhound.interfaces.llm_provider import LLMProvider

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_CLEANUP_SYSTEM_PROMPT_FILE = "cleanup_system_v2.txt"
_CLEANUP_USER_PROMPT_FILE = "cleanup_user_v2.txt"
_CLEANUP_USER_PROMPT_FILE_END_USER = "cleanup_user_end_user_v1.txt"


def _taint_cleanup_system_guidance(taint: str) -> str:
    normalized = _normalize_taint(taint)
    if normalized == "technical":
        return "\n".join(
            [
                "Audience: technical (software engineers).",
                "- Prefer precise terminology and concrete implementation details present in the input.",
                "- When helpful, call out key modules/classes/functions and their responsibilities.",
                "- Avoid “product docs” tone; keep the writing crisp and technical.",
            ]
        )
    if normalized == "end-user":
        return "\n".join(
            [
                "Audience: end-user (less technical).",
                "- Prefer plain-language descriptions of how to set up, configure, and use the project when the input contains that information.",
                "- Keep code identifiers, but explain them in plain language and focus on user goals and workflows.",
                "- De-emphasize internal implementation details unless they are central in the input.",
            ]
        )
    return ""


def _build_cleanup_system_prompt(config: CleanupConfig) -> str:
    base = _read_prompt_file(
        _CLEANUP_SYSTEM_PROMPT_FILE,
        fallback=(
            "You are a senior technical writer polishing engineering documentation. "
            "Make the writing approachable without losing precision."
        ),
    )
    guidance = _taint_cleanup_system_guidance(config.taint)
    if not guidance:
        return base.strip()
    return (base.strip() + "\n\n" + guidance.strip()).strip()


async def _cleanup_with_llm(
    *,
    topics: list[CodeMapperTopic],
    provider: LLMProvider,
    config: CleanupConfig,
    log_info: Callable[[str], None] | None,
    log_warning: Callable[[str], None] | None,
) -> list[str]:
    system_prompt = _build_cleanup_system_prompt(config)

    prompts = [
        _build_cleanup_prompt(topic.title, topic.body_markdown, taint=config.taint)
        for topic in topics
    ]

    cleaned: list[str | None] = [None] * len(topics)

    indexed: list[tuple[int, CodeMapperTopic, str]] = list(
        zip(range(len(topics)), topics, prompts, strict=True)
    )

    for batch in _chunked(indexed, config.batch_size):
        batch_prompts = [prompt for _idx, _topic, prompt in batch]
        batch_topics = [topic for _idx, topic, _prompt in batch]
        batch_indices = [idx for idx, _topic, _prompt in batch]

        if log_info:
            log_info(f"Running cleanup batch with {len(batch_prompts)} topic(s).")

        batch_outputs: list[str | None] | None = None
        try:
            batch_responses = await provider.batch_complete(
                batch_prompts,
                system=system_prompt,
                max_completion_tokens=config.max_completion_tokens,
            )
            if len(batch_responses) != len(batch_prompts):
                raise ValueError(
                    "LLM cleanup batch response count mismatch: "
                    f"{len(batch_responses)} != {len(batch_prompts)}"
                )
            batch_outputs = [resp.content.strip() for resp in batch_responses]
        except Exception as exc:  # noqa: BLE001
            if log_warning:
                log_warning(
                    "LLM cleanup batch failed or returned unexpected results; "
                    "retrying with batch_size=1. "
                    f"Error: {exc}"
                )

        if batch_outputs is None:
            batch_outputs = []
            for prompt in batch_prompts:
                try:
                    single_responses = await provider.batch_complete(
                        [prompt],
                        system=system_prompt,
                        max_completion_tokens=config.max_completion_tokens,
                    )
                    if len(single_responses) != 1:
                        raise ValueError(
                            "LLM cleanup retry returned unexpected response count: "
                            f"{len(single_responses)}"
                        )
                    batch_outputs.append(single_responses[0].content.strip())
                except Exception as exc:  # noqa: BLE001
                    if log_warning:
                        log_warning(
                            "LLM cleanup retry failed for a topic; falling back to "
                            "minimal cleanup for that topic. "
                            f"Error: {exc}"
                        )
                    batch_outputs.append(None)

        for idx, topic, response in zip(
            batch_indices,
            batch_topics,
            batch_outputs,
            strict=True,
        ):
            if not response:
                cleaned[idx] = _minimal_cleanup(topic)
            else:
                cleaned[idx] = _normalize_llm_output(response)

    return [
        item if item is not None else _minimal_cleanup(topic)
        for item, topic in zip(cleaned, topics, strict=True)
    ]


def _build_cleanup_prompt(title: str, body: str, *, taint: str = "balanced") -> str:
    fallback = "\n".join(
        [
            "Rewrite the documentation section below as a polished, friendly doc page.",
            "Requirements:",
            "- Do NOT add new facts or speculation.",
            "- Keep citations like [1] exactly as-is.",
            "- Preserve code identifiers and inline code formatting.",
            '- Start with a short "Overview" section.',
            "- Use level-2 headings (##). Do NOT include a level-1 heading.",
            "- Do NOT include recommendations, follow-ups, or next steps.",
            "- Use visuals when clearly helpful: include a Mermaid diagram in a ```mermaid code fence",
            "  and/or a table/callout to clarify complex flows. Skip visuals if they would be redundant.",
            '- If the content includes a "Sources" section, rename it to "References".',
            "- Remove duplicate title lines or redundant bold title repeats.",
            "- Keep the length roughly similar to the input.",
            "- Mermaid rules: keep node labels on a single line (no raw newlines inside [brackets] or {diamonds}).",
            "  Prefer simple labels and avoid unusual punctuation.",
            "Mermaid examples (copy the style, not the content):",
            "```mermaid",
            "flowchart TD",
            "  A[CLI or Env Inputs] --> B[Config Resolution]",
            "  B --> C[Service Registry]",
            "  B --> D{Embeddings required?}",
            "  D -- yes --> E[Embedding Provider]",
            "  D -- no --> F[Search Only Mode]",
            "```",
            "```mermaid",
            "flowchart LR",
            "  A[Request] --> B[Validate]",
            "  B --> C{Valid?}",
            "  C -- yes --> D[Execute]",
            "  C -- no --> E[Error]",
            "```",
            "```mermaid",
            "flowchart TD",
            "  A[CLI entry] --> B[Config builder]",
            "  B --> C[validate_for_command]",
            "  C --> D{Config valid?}",
            "  D -- yes --> E[MCP server start]",
            "  D -- no --> F[CLI error]",
            "```",
            "",
            f"Title: {title}",
            "",
            "Input markdown:",
            body.strip(),
        ]
    )

    normalized = _normalize_taint(taint)
    template_file = (
        _CLEANUP_USER_PROMPT_FILE_END_USER
        if normalized == "end-user"
        else _CLEANUP_USER_PROMPT_FILE
    )
    template = _read_prompt_file(template_file, fallback=fallback)
    hydrated = (
        template.replace("<<TITLE>>", title)
        .replace("<<BODY>>", body.strip())
        .replace("{title}", title)
        .replace("{body}", body.strip())
    )
    return hydrated.strip()


def _read_prompt_file(filename: str, *, fallback: str) -> str:
    path = _PROMPTS_DIR / filename
    try:
        if path.exists():
            content = path.read_text(encoding="utf-8").strip()
            if content:
                return content
    except Exception:  # noqa: BLE001
        return fallback
    return fallback


def _normalize_llm_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = _strip_first_heading(cleaned)
    cleaned = _ensure_overview_heading(cleaned)
    return cleaned.strip()


def _minimal_cleanup(topic: CodeMapperTopic) -> str:
    return _ensure_overview_heading(topic.body_markdown.strip())

