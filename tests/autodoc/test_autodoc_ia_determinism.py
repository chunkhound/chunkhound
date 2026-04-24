from __future__ import annotations

import pytest

from chunkhound.autodoc.ia import (
    _synthesize_homepage_overview,
)
from chunkhound.autodoc.models import DocsitePage
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


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

    @property
    def timeout(self) -> int:
        return 0

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
        audience="end-user",
        log_info=None,
        log_warning=None,
    )

    assert provider.last_prompt is not None

    assert overview == "Hi\n\n- x"
