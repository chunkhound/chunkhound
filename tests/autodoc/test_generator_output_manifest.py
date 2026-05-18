from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from chunkhound.autodoc.generator import generate_docsite
from chunkhound.autodoc.models import CleanupConfig
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


class _FrozenDatetime:
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        return dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)


class _Provider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> int:
        return 0

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        return LLMResponse(
            content="## Overview\nEnd-user overview.\n\n- Use case 1\n- Use case 2",
            tokens_used=0,
            model=self._model,
            finish_reason="stop",
        )

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, object]:
        return {
            "nav": {"groups": [{"title": "Group", "slugs": ["01-topic-one"]}]},
            "glossary": [
                {"term": "Term", "definition": "Definition.", "pages": ["01-topic-one"]}
            ],
        }

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        return [
            LLMResponse(
                content="## Overview\nCleaned.\n\n## Details\nMore.",
                tokens_used=0,
                model=self._model,
                finish_reason="stop",
            )
            for _ in prompts
        ]

    def estimate_tokens(self, text: str) -> int:
        return 0

    async def health_check(self) -> dict[str, object]:
        return {"ok": True}

    def get_usage_stats(self) -> dict[str, object]:
        return {}


class _LLMManager:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def get_synthesis_provider(self) -> LLMProvider:
        return self._provider


def _write_minimal_input_dir(input_dir: Path) -> None:
    (input_dir / "scope_code_mapper_index.md").write_text(
        "\n".join(
            [
                "# AutoDoc Topics (/repo)",
                "",
                "1. [Topic One](topic_one.md)",
            ]
        ),
        encoding="utf-8",
    )

    (input_dir / "topic_one.md").write_text(
        "\n".join(
            [
                "# Topic One",
                "",
                "Overview body.",
                "",
                "## Sources",
                "",
                "└── repo/",
                "\t└── [1] x.py (1 chunks: L1-2)",
            ]
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_generate_docsite_requires_llm_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("chunkhound.autodoc.generator.datetime", _FrozenDatetime)

    input_dir = Path("input")
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    with pytest.raises(RuntimeError, match="LLM"):
        await generate_docsite(
            input_dir=input_dir,
            output_dir=Path("out"),
            llm_manager=None,
            cleanup_config=CleanupConfig(
                mode="llm",
                batch_size=1,
                max_completion_tokens=512,
                audience="end-user",
            ),
            site_title=None,
            site_tagline=None,
        )

    assert not Path("out").exists()


