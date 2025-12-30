from __future__ import annotations

import json
from hashlib import sha256

import pytest

from chunkhound.autodoc.ia import (
    _build_site_context,
    _build_site_ia_prompt,
    _synthesize_homepage_overview,
    _synthesize_site_ia,
)
from chunkhound.autodoc.models import DocsitePage
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


def _digest_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _digest_schema(schema: dict[str, object]) -> str:
    encoded = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return _digest_text(encoded)


def _pages_fixture() -> list[DocsitePage]:
    return [
        DocsitePage(
            order=1,
            title="Topic One",
            slug="01-topic-one",
            description="First topic.",
            body_markdown=(
                "## Overview\n\n"
                "Hello `world`.\n\n"
                "## Install\n\n"
                "Run `chunkhound`.\n\n"
                "### Step A\n\n"
                "Do A.\n\n"
                "## References\n\n"
                "- [1] x.py\n"
            ),
        ),
        DocsitePage(
            order=2,
            title="Topic Two",
            slug="02-topic-two",
            description="Second topic.",
            body_markdown="## Overview\n\nSecond body.\n\n## Usage\n\nUse it.\n",
        ),
    ]


def test_build_site_ia_prompt_is_byte_stable_across_taints() -> None:
    context = _build_site_context(_pages_fixture())

    prompt_balanced = _build_site_ia_prompt(context=context, taint="balanced")
    assert _digest_text(prompt_balanced) == (
        "520394ed1cdb84a33e834fd73d14d19cac23e8a6a22b1e03f5d3175aaa8ca4af"
    )

    prompt_technical = _build_site_ia_prompt(context=context, taint="technical")
    assert _digest_text(prompt_technical) == (
        "a341976d2aec43370707d2c5a1cd21b308f2c1eaf45af464b1dbfc80fbcd656d"
    )

    prompt_end_user = _build_site_ia_prompt(context=context, taint="end-user")
    assert _digest_text(prompt_end_user) == (
        "878216f85d42f4060c39d125adc41159f0c1259b964f320c915189a91009e14b"
    )


class _CapturingStructuredProvider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"
        self.last_prompt: str | None = None
        self.last_schema: dict[str, object] | None = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(  # type: ignore[override]
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        raise AssertionError("complete() not expected for structured IA synthesis")

    async def complete_structured(  # type: ignore[override]
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, object]:
        self.last_prompt = prompt
        self.last_schema = json_schema
        return {"nav": {"groups": []}, "glossary": []}

    async def batch_complete(  # type: ignore[override]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        raise AssertionError("batch_complete() not expected for IA synthesis")

    def estimate_tokens(self, text: str) -> int:
        return 0

    async def health_check(self) -> dict[str, object]:
        return {"ok": True}

    def get_usage_stats(self) -> dict[str, object]:
        return {}


@pytest.mark.asyncio
async def test_synthesize_site_ia_uses_byte_stable_schema_and_prompt() -> None:
    provider = _CapturingStructuredProvider()

    await _synthesize_site_ia(
        pages=_pages_fixture(),
        provider=provider,
        taint="end-user",
        log_info=None,
        log_warning=None,
    )

    assert provider.last_prompt is not None
    assert _digest_text(provider.last_prompt) == (
        "878216f85d42f4060c39d125adc41159f0c1259b964f320c915189a91009e14b"
    )

    assert provider.last_schema is not None
    assert _digest_schema(provider.last_schema) == (
        "7bbe91a52b3eace0243417f8a6978cb8cec244cac56edeb36e27efa695f363e0"
    )


class _CapturingCompleteProvider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"
        self.last_prompt: str | None = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(  # type: ignore[override]
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(
            content="## Overview\nHi\n\n## Topics\n- x",
            tokens_used=0,
            model=self._model,
            finish_reason="stop",
        )

    async def batch_complete(  # type: ignore[override]
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        raise AssertionError("batch_complete() not expected for homepage overview")

    def estimate_tokens(self, text: str) -> int:
        return 0

    async def health_check(self) -> dict[str, object]:
        return {"ok": True}

    def get_usage_stats(self) -> dict[str, object]:
        return {}


@pytest.mark.asyncio
async def test_synthesize_homepage_overview_prompt_and_normalization() -> None:
    provider = _CapturingCompleteProvider()

    overview = await _synthesize_homepage_overview(
        pages=_pages_fixture(),
        provider=provider,
        taint="end-user",
        log_info=None,
        log_warning=None,
    )

    assert provider.last_prompt is not None
    assert _digest_text(provider.last_prompt) == (
        "38a548cd574194b90838d0d398475aeab9e11aff1c36bed58bc0b87d14b36579"
    )

    assert overview == "Hi\n\n- x"
