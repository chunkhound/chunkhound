import json
from pathlib import Path

import pytest

from chunkhound.autodoc.generator import generate_docsite
from chunkhound.autodoc.models import (
    CleanupConfig,
    CodeMapperIndex,
    DocsitePage,
    DocsiteSite,
    GlossaryTerm,
    NavGroup,
)
from chunkhound.autodoc.site_writer import write_astro_site
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


class _FakeLLMProvider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"

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
        return LLMResponse(
            content="## Overview\nok",
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
            "nav": {
                "groups": [
                    {"title": "Group", "slugs": ["01-topic-one"]},
                ]
            },
            "glossary": [
                {
                    "term": "Term",
                    "definition": "Definition.",
                    "pages": ["01-topic-one"],
                }
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


class _FakeLLMManager:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def get_synthesis_provider(self) -> LLMProvider:
        return self._provider


class _DuplicateNavProvider(_FakeLLMProvider):
    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, object]:
        return {
            "nav": {
                "groups": [
                    {"title": "Group A", "slugs": ["01-topic-one"]},
                    {"title": "Group B", "slugs": ["01-topic-one"]},
                    {"title": "Group C", "slugs": ["01-topic-one"]},
                    {"title": "Group D", "slugs": ["01-topic-one"]},
                ]
            },
            "glossary": [],
        }


class _FailStructuredProvider(_FakeLLMProvider):
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        return LLMResponse(
            content="## Overview\nHomepage overview survived.",
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
        raise RuntimeError("Structured output is unavailable")


def test_write_astro_site_writes_nav_and_glossary_when_present(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "site"
    site = DocsiteSite(
        title="Test",
        tagline="Tag",
        scope_label="/",
        generated_at="2025-12-22T00:00:00Z",
        source_dir=str(tmp_path),
        topic_count=1,
    )
    pages = [
        DocsitePage(
            order=1,
            title="Topic One",
            slug="topic-one",
            description="Desc",
            body_markdown=(
                "## Overview\nBody\n\n## References\n- [1] `x.py` (1 chunks: L1-2)"
            ),
        )
    ]
    index = CodeMapperIndex(
        title="Index",
        scope_label="/",
        metadata_block=None,
        topics=[],
    )

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=pages,
        index=index,
        allow_delete_topics_dir=False,
        nav_groups=[NavGroup(title="Group", slugs=["topic-one"])],
        glossary_terms=[
            GlossaryTerm(term="Term", definition="Definition.", pages=["topic-one"])
        ],
    )

    nav_path = output_dir / "src" / "data" / "nav.json"
    glossary_path = output_dir / "src" / "pages" / "glossary.md"

    assert nav_path.exists()
    payload = json.loads(nav_path.read_text(encoding="utf-8"))
    assert payload["groups"][0]["title"] == "Group"

    assert glossary_path.exists()
    glossary = glossary_path.read_text(encoding="utf-8")
    assert 'title: "Glossary"' in glossary
    assert "## Term" in glossary


def test_write_astro_site_removes_stale_nav_and_glossary(tmp_path: Path) -> None:
    output_dir = tmp_path / "site"
    (output_dir / "src" / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "src" / "pages").mkdir(parents=True, exist_ok=True)
    (output_dir / "src" / "data" / "nav.json").write_text("stale", encoding="utf-8")
    (output_dir / "src" / "pages" / "glossary.md").write_text("stale", encoding="utf-8")

    site = DocsiteSite(
        title="Test",
        tagline="Tag",
        scope_label="/",
        generated_at="2025-12-22T00:00:00Z",
        source_dir=str(tmp_path),
        topic_count=0,
    )
    index = CodeMapperIndex(
        title="Index",
        scope_label="/",
        metadata_block=None,
        topics=[],
    )

    write_astro_site(
        output_dir=output_dir,
        site=site,
        pages=[],
        index=index,
        allow_delete_topics_dir=False,
        nav_groups=None,
        glossary_terms=None,
    )

    nav_path = output_dir / "src" / "data" / "nav.json"
    assert nav_path.exists()
    json.loads(nav_path.read_text(encoding="utf-8"))
    assert not (output_dir / "src" / "pages" / "glossary.md").exists()


@pytest.mark.asyncio
async def test_generate_docsite_writes_nav_and_glossary_in_llm_mode(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)

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

    output_dir = tmp_path / "out"
    provider = _FakeLLMProvider()
    llm_manager = _FakeLLMManager(provider)

    await generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
        site_title=None,
        site_tagline=None,
    )

    assert (output_dir / "src" / "data" / "nav.json").exists()
    assert (output_dir / "src" / "pages" / "glossary.md").exists()


@pytest.mark.asyncio
async def test_generate_docsite_dedupes_duplicate_nav_slugs(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)

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

    output_dir = tmp_path / "out"
    provider = _DuplicateNavProvider()
    llm_manager = _FakeLLMManager(provider)

    await generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
        site_title=None,
        site_tagline=None,
    )

    nav_path = output_dir / "src" / "data" / "nav.json"
    payload = json.loads(nav_path.read_text(encoding="utf-8"))
    groups = payload["groups"]
    assert len(groups) == 1
    assert groups[0]["slugs"] == ["01-topic-one"]


@pytest.mark.asyncio
async def test_generate_docsite_keeps_homepage_overview_when_site_ia_fails(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True)

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

    output_dir = tmp_path / "out"
    provider = _FailStructuredProvider()
    llm_manager = _FakeLLMManager(provider)

    await generate_docsite(
        input_dir=input_dir,
        output_dir=output_dir,
        llm_manager=llm_manager,  # type: ignore[arg-type]
        cleanup_config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
        site_title=None,
        site_tagline=None,
    )

    index_path = output_dir / "src" / "pages" / "index.md"
    assert index_path.exists()
    index = index_path.read_text(encoding="utf-8")
    assert "Homepage overview survived." in index

    assert not (output_dir / "src" / "pages" / "glossary.md").exists()
