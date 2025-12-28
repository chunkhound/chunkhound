import json
from pathlib import Path

import pytest

from chunkhound.autodoc import docsite
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


class _CapturingProvider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"
        self.last_batch_system: str | None = None
        self.last_structured_prompt: str | None = None
        self.last_complete_prompt: str | None = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        self.last_complete_prompt = prompt
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
        self.last_structured_prompt = prompt
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
        self.last_batch_system = system
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


class _FakeLLMManager:
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
async def test_taint_is_noop_in_minimal_cleanup_mode_even_with_llm_manager(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    output_dir = tmp_path / "out"
    provider = _CapturingProvider()
    llm_manager = _FakeLLMManager(provider)

    await docsite.generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=docsite.CleanupConfig(
            mode="minimal",
            batch_size=1,
            max_completion_tokens=512,
            taint="end-user",
        ),
        site_title=None,
        site_tagline=None,
    )

    assert provider.last_batch_system is None
    assert provider.last_structured_prompt is None
    assert provider.last_complete_prompt is None
    assert not (output_dir / "src" / "pages" / "glossary.md").exists()

    site_json = json.loads(
        (output_dir / "src" / "data" / "site.json").read_text(encoding="utf-8")
    )
    assert (
        site_json["tagline"]
        == "Approachable documentation generated from AutoDoc output."
    )


@pytest.mark.asyncio
async def test_taint_influences_llm_cleanup_and_global_ia_prompts(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    output_dir = tmp_path / "out"
    provider = _CapturingProvider()
    llm_manager = _FakeLLMManager(provider)

    await docsite.generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=docsite.CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
            taint="technical",
        ),
        site_title=None,
        site_tagline=None,
    )

    assert provider.last_batch_system is not None
    assert "Audience: technical" in provider.last_batch_system

    assert provider.last_structured_prompt is not None
    assert "Audience: technical" in provider.last_structured_prompt
    assert provider.last_complete_prompt is not None
    assert "Audience: technical" in provider.last_complete_prompt

    site_json = json.loads(
        (output_dir / "src" / "data" / "site.json").read_text(encoding="utf-8")
    )
    assert (
        site_json["tagline"]
        == "Engineering-focused documentation generated from AutoDoc output."
    )


@pytest.mark.asyncio
async def test_end_user_taint_synthesizes_homepage_overview(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    output_dir = tmp_path / "out"
    provider = _CapturingProvider()
    llm_manager = _FakeLLMManager(provider)

    await docsite.generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=docsite.CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
            taint="end-user",
        ),
        site_title=None,
        site_tagline=None,
    )

    assert provider.last_complete_prompt is not None
    assert "Audience: end-user" in provider.last_complete_prompt

    index_md = (output_dir / "src" / "pages" / "index.md").read_text(encoding="utf-8")
    assert "## Overview" in index_md
    assert "End-user overview." in index_md
