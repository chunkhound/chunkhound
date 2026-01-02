from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.autodoc import cleanup as cleanup_module
from chunkhound.autodoc.cleanup import _cleanup_with_llm
from chunkhound.autodoc.models import CleanupConfig, CodeMapperTopic
from chunkhound.interfaces.llm_provider import LLMResponse


class _MismatchProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.systems: list[str | None] = []

    async def batch_complete(  # type: ignore[no-untyped-def]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        self.calls.append(list(prompts))
        self.systems.append(system)
        if len(prompts) == 2 and len(self.calls) == 1:
            # Return fewer responses than prompts.
            return [
                LLMResponse(
                    content="## Overview\nCleaned A",
                    tokens_used=0,
                    model="fake",
                    finish_reason="stop",
                )
            ]
        return [
            LLMResponse(
                content="## Overview\nCleaned OK",
                tokens_used=0,
                model="fake",
                finish_reason="stop",
            )
            for _ in prompts
        ]


@pytest.mark.asyncio
async def test_cleanup_with_llm_retries_batch_size_one_on_count_mismatch() -> None:
    topics = [
        CodeMapperTopic(
            order=1,
            title="Topic A",
            source_path=Path("a.md"),
            raw_markdown="## Overview\nA",
            body_markdown="## Overview\nA",
        ),
        CodeMapperTopic(
            order=2,
            title="Topic B",
            source_path=Path("b.md"),
            raw_markdown="## Overview\nB",
            body_markdown="## Overview\nB",
        ),
    ]
    provider = _MismatchProvider()
    warnings: list[str] = []

    cleaned = await _cleanup_with_llm(
        topics=topics,
        provider=provider,  # type: ignore[arg-type]
        config=CleanupConfig(
            mode="llm",
            batch_size=2,
            max_completion_tokens=512,
        ),
        log_info=None,
        log_warning=warnings.append,
    )

    assert len(cleaned) == 2
    assert provider.calls[0] and len(provider.calls[0]) == 2
    # After mismatch, retry with batch_size=1 for each prompt.
    assert len(provider.calls) == 3
    assert provider.systems and len(set(provider.systems)) == 1
    assert any("retrying with batch_size=1" in w for w in warnings)


class _ExceptionProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def batch_complete(  # type: ignore[no-untyped-def]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        self.calls.append(list(prompts))
        if len(prompts) > 1:
            raise RuntimeError("boom")
        return [
            LLMResponse(
                content="",
                tokens_used=0,
                model="fake",
                finish_reason="stop",
            )
        ]


@pytest.mark.asyncio
async def test_cleanup_with_llm_raises_when_single_topic_cleanup_invalid() -> None:
    topics = [
        CodeMapperTopic(
            order=1,
            title="Topic A",
            source_path=Path("a.md"),
            raw_markdown="## Overview\nA",
            body_markdown="## Overview\nA",
        ),
        CodeMapperTopic(
            order=2,
            title="Topic B",
            source_path=Path("b.md"),
            raw_markdown="## Overview\nB",
            body_markdown="## Overview\nB",
        ),
    ]
    provider = _ExceptionProvider()

    with pytest.raises(RuntimeError, match="cleanup"):
        await _cleanup_with_llm(
            topics=topics,
            provider=provider,  # type: ignore[arg-type]
            config=CleanupConfig(
                mode="llm",
                batch_size=2,
                max_completion_tokens=512,
            ),
            log_info=None,
            log_warning=None,
        )

    assert provider.calls


class _TransientProvider:
    def __init__(self, fail_times: int) -> None:
        self._fail_times = fail_times
        self.calls = 0

    async def batch_complete(  # type: ignore[no-untyped-def]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise TimeoutError("timeout")
        return [
            LLMResponse(
                content="## Overview\nCleaned OK",
                tokens_used=0,
                model="fake",
                finish_reason="stop",
            )
            for _ in prompts
        ]


@pytest.mark.asyncio
async def test_cleanup_with_llm_retries_transient_errors_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(cleanup_module.asyncio, "sleep", no_sleep)

    topics = [
        CodeMapperTopic(
            order=1,
            title="Topic A",
            source_path=Path("a.md"),
            raw_markdown="## Overview\nA",
            body_markdown="## Overview\nA",
        ),
    ]
    provider = _TransientProvider(fail_times=2)

    cleaned = await _cleanup_with_llm(
        topics=topics,
        provider=provider,  # type: ignore[arg-type]
        config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
        log_info=None,
        log_warning=None,
    )

    assert cleaned == ["## Overview\nCleaned OK"]
    assert provider.calls == 3
