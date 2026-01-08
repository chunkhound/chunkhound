from __future__ import annotations

import asyncio
import importlib.resources
import random
from collections.abc import Callable
from dataclasses import dataclass

from chunkhound.autodoc.markdown_utils import (
    _chunked,
    _ensure_overview_heading,
    _strip_first_heading,
)
from chunkhound.autodoc.models import CleanupConfig, CodeMapperTopic
from chunkhound.core.audience import normalize_audience
from chunkhound.interfaces.llm_provider import LLMProvider

_PROMPTS_PACKAGE = "chunkhound.autodoc"
_CLEANUP_SYSTEM_PROMPT_FILE = "cleanup_system_v2.txt"
_CLEANUP_USER_PROMPT_FILE = "cleanup_user_v2.txt"
_CLEANUP_USER_PROMPT_FILE_END_USER = "cleanup_user_end_user_v1.txt"

_CLEANUP_RETRY_MAX_ATTEMPTS = 5
_CLEANUP_RETRY_BASE_DELAY_SECONDS = 0.5
_CLEANUP_RETRY_MAX_DELAY_SECONDS = 8.0
_CLEANUP_RETRY_JITTER_RATIO = 0.2


@dataclass(frozen=True)
class _IndexedCleanupInput:
    idx: int
    topic: CodeMapperTopic
    prompt: str


def _audience_cleanup_system_guidance(audience: str) -> str:
    normalized = normalize_audience(audience)
    if normalized == "technical":
        return "\n".join(
            [
                "Audience: technical (software engineers).",
                "- Prefer precise terminology and concrete implementation details "
                "present in the input.",
                "- When helpful, call out key modules/classes/functions and their "
                "responsibilities.",
                "- Avoid “product docs” tone; keep the writing crisp and technical.",
            ]
        )
    if normalized == "end-user":
        return "\n".join(
            [
                "Audience: end-user (less technical).",
                "- Prefer plain-language descriptions of how to set up, configure, "
                "and use the project when the input contains that information.",
                "- Keep code identifiers, but explain them in plain language and "
                "focus on user goals and workflows.",
                "- De-emphasize internal implementation details unless they are "
                "central in the input.",
            ]
        )
    return ""


def _build_cleanup_system_prompt(config: CleanupConfig) -> str:
    base = _read_prompt_file(_CLEANUP_SYSTEM_PROMPT_FILE)
    guidance = _audience_cleanup_system_guidance(config.audience)
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
        _build_cleanup_prompt(
            topic.title,
            topic.body_markdown,
            audience=config.audience,
        )
        for topic in topics
    ]

    cleaned: list[str | None] = [None] * len(topics)

    indexed: list[_IndexedCleanupInput] = [
        _IndexedCleanupInput(idx=idx, topic=topic, prompt=prompt)
        for idx, topic, prompt in zip(range(len(topics)), topics, prompts, strict=True)
    ]

    for batch in _chunked(indexed, config.batch_size):
        batch_prompts = [item.prompt for item in batch]
        batch_topics = [item.topic for item in batch]
        batch_indices = [item.idx for item in batch]

        if log_info:
            log_info(f"Running cleanup batch with {len(batch_prompts)} topic(s).")

        batch_outputs: list[str] | None = None
        try:
            batch_outputs = await _batch_complete_with_backoff(
                provider=provider,
                prompts=batch_prompts,
                system=system_prompt,
                max_completion_tokens=config.max_completion_tokens,
                log_warning=log_warning,
                operation="cleanup batch",
            )
        except Exception as exc:  # noqa: BLE001
            if log_warning:
                log_warning(
                    "LLM cleanup batch failed or returned unexpected results; "
                    "retrying with batch_size=1. "
                    f"Error: {exc}"
                )

        if batch_outputs is None:
            single_outputs: list[str] = []
            for prompt, topic in zip(batch_prompts, batch_topics, strict=True):
                try:
                    outputs = await _batch_complete_with_backoff(
                        provider=provider,
                        prompts=[prompt],
                        system=system_prompt,
                        max_completion_tokens=config.max_completion_tokens,
                        log_warning=log_warning,
                        operation=f"cleanup retry for topic {topic.title!r}",
                    )
                    if len(outputs) != 1:
                        raise ValueError(
                            "LLM cleanup retry returned unexpected response count: "
                            f"{len(outputs)}"
                        )
                    single_outputs.append(outputs[0])
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"AutoDoc cleanup failed for topic {topic.title!r}: {exc}"
                    ) from exc
            batch_outputs = single_outputs

        for idx, response in zip(
            batch_indices,
            batch_outputs,
            strict=True,
        ):
            cleaned[idx] = _normalize_llm_output(response)

    if any(item is None for item in cleaned):
        raise RuntimeError("AutoDoc cleanup produced incomplete results.")

    return [item for item in cleaned if item is not None]


def _build_cleanup_prompt(title: str, body: str, *, audience: str = "balanced") -> str:
    normalized = normalize_audience(audience)
    template_file = (
        _CLEANUP_USER_PROMPT_FILE_END_USER
        if normalized == "end-user"
        else _CLEANUP_USER_PROMPT_FILE
    )
    template = _read_prompt_file(template_file)
    hydrated = (
        template.replace("<<TITLE>>", title)
        .replace("<<BODY>>", body.strip())
        .replace("{title}", title)
        .replace("{body}", body.strip())
    )
    return hydrated.strip()


def _read_prompt_file(filename: str) -> str:
    resource_path = (
        importlib.resources.files(_PROMPTS_PACKAGE)
        .joinpath("prompts")
        .joinpath(filename)
    )
    try:
        with resource_path.open("r", encoding="utf-8") as handle:
            content = handle.read().strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"AutoDoc prompt file missing: {_PROMPTS_PACKAGE}:prompts/{filename}"
        ) from exc
    except OSError as exc:
        raise OSError(
            f"AutoDoc prompt file unreadable: {_PROMPTS_PACKAGE}:prompts/{filename}"
        ) from exc

    if not content:
        raise ValueError(
            f"AutoDoc prompt file empty: {_PROMPTS_PACKAGE}:prompts/{filename}"
        )
    return content


def _normalize_llm_output(text: str) -> str:
    cleaned = text.strip()
    cleaned = _strip_first_heading(cleaned)
    cleaned = _ensure_overview_heading(cleaned)
    return cleaned.strip()


def _is_transient_llm_error(exc: BaseException) -> bool:
    error_type = type(exc).__name__
    if error_type in {
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "APIConnectionError",
        "APITimeoutError",
    }:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError, asyncio.TimeoutError)):
        return True
    error_str = str(exc).lower()
    return any(
        marker in error_str
        for marker in (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "overloaded",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
    )


async def _batch_complete_with_backoff(
    *,
    provider: LLMProvider,
    prompts: list[str],
    system: str,
    max_completion_tokens: int,
    log_warning: Callable[[str], None] | None,
    operation: str,
) -> list[str]:
    last_exc: Exception | None = None
    for attempt in range(_CLEANUP_RETRY_MAX_ATTEMPTS):
        try:
            responses = await provider.batch_complete(
                prompts,
                system=system,
                max_completion_tokens=max_completion_tokens,
            )
            if len(responses) != len(prompts):
                raise ValueError(
                    "LLM cleanup batch response count mismatch: "
                    f"{len(responses)} != {len(prompts)}"
                )
            outputs = [resp.content.strip() for resp in responses]
            if any(not out for out in outputs):
                raise ValueError("LLM cleanup returned empty output.")
            return outputs
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if (
                not _is_transient_llm_error(exc)
                or attempt >= _CLEANUP_RETRY_MAX_ATTEMPTS - 1
            ):
                raise
            base = _CLEANUP_RETRY_BASE_DELAY_SECONDS * (2**attempt)
            delay = min(_CLEANUP_RETRY_MAX_DELAY_SECONDS, base)
            jitter = delay * random.uniform(0.0, _CLEANUP_RETRY_JITTER_RATIO)
            sleep_seconds = delay + jitter
            if log_warning:
                log_warning(
                    f"Transient LLM error during {operation}; retrying in "
                    f"{sleep_seconds:.2f}s (attempt {attempt + 1}/"
                    f"{_CLEANUP_RETRY_MAX_ATTEMPTS}). Error: {exc}"
                )
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)
            continue
    raise RuntimeError(
        f"LLM operation failed after {_CLEANUP_RETRY_MAX_ATTEMPTS} attempts: {last_exc}"
    )
