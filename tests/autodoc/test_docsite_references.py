from pathlib import Path

import pytest

from chunkhound.autodoc.generator import cleanup_topics
from chunkhound.autodoc.models import CleanupConfig, CodeMapperTopic
from chunkhound.autodoc.references import (
    _apply_reference_normalization,
    extract_sources_block,
    flatten_sources_block,
    strip_references_section,
)
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse


class _FixedCleanupProvider(LLMProvider):
    def __init__(self, cleaned_body: str) -> None:
        self._cleaned_body = cleaned_body
        self._model = "fake"

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(  # pragma: no cover
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        raise NotImplementedError

    async def complete_structured(  # pragma: no cover
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, object]:
        raise NotImplementedError

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        return [
            LLMResponse(
                content=self._cleaned_body,
                tokens_used=0,
                model=self._model,
                finish_reason="stop",
            )
            for _ in prompts
        ]

    def estimate_tokens(self, text: str) -> int:  # pragma: no cover
        return 0

    async def health_check(self) -> dict[str, object]:  # pragma: no cover
        return {"ok": True}

    def get_usage_stats(self) -> dict[str, object]:  # pragma: no cover
        return {}


class _FakeLLMManager:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def get_synthesis_provider(self) -> LLMProvider:
        return self._provider


def _sources_block() -> str:
    return "\n".join(
        [
            "## Sources",
            "",
            "**Files**: 2 | **Chunks**: 3",
            "",
            "└── repo/",
            "\t├── src/",
            "\t│\t└── [1] main.py (2 chunks: L1-10, L20-30)",
            "\t└── tests/",
            "\t\t└── [2] test_main.py (1 chunks: L5-8)",
        ]
    )


def _topic_body_with_sources() -> str:
    return "\n".join(
        [
            "## Overview",
            "",
            "Overview text.",
            "",
            _sources_block(),
            "",
            "## Details",
            "More info.",
        ]
    )

def _topic_body_with_sources_and_citations() -> str:
    return "\n".join(
        [
            "## Overview",
            "",
            "Overview text [1].",
            "",
            _sources_block(),
            "",
            "## Details",
            "More info.",
        ]
    )


def test_extract_sources_block_returns_section_body() -> None:
    markdown = _topic_body_with_sources()
    block = extract_sources_block(markdown)

    assert block is not None
    assert block.startswith("## Sources")
    assert "[1] main.py" in block
    assert "## Details" not in block


def test_strip_references_section_removes_sources_and_references() -> None:
    markdown = "\n".join(
        [
            "## Overview",
            "Overview text.",
            "",
            "## Sources",
            "- [1] src/main.py",
            "",
            "## Details",
            "Details text.",
            "",
            "## References",
            "- [2] src/other.py",
            "",
            "## Outro",
            "Outro text.",
        ]
    )

    stripped = strip_references_section(markdown)

    assert "## Sources" not in stripped
    assert "## References" not in stripped
    assert "## Overview" in stripped
    assert "## Details" in stripped
    assert "## Outro" in stripped


def test_strip_references_section_removes_separator_before_sources_footer() -> None:
    markdown = "\n".join(
        [
            "## Overview",
            "Overview text.",
            "",
            "---",
            "## Sources",
            "- [1] src/main.py",
            "",
            "## Details",
            "Details text.",
        ]
    )

    stripped = strip_references_section(markdown)

    assert "## Sources" not in stripped
    assert stripped.splitlines()[-1] == "Details text."
    assert "---" not in stripped


def test_flatten_sources_block_builds_list_with_paths_and_chunks() -> None:
    flat = flatten_sources_block(_sources_block())

    assert flat == [
        "- [1] `repo/src/main.py` (2 chunks: L1-10, L20-30)",
        "- [2] `repo/tests/test_main.py` (1 chunks: L5-8)",
    ]


def test_build_references_section_omits_when_missing_sources() -> None:
    body = "\n".join(["## Overview", "", "Overview text."])
    normalized = _apply_reference_normalization(body, None)

    assert "## References" not in normalized


@pytest.mark.asyncio
async def test_cleanup_topics_injects_flattened_references() -> None:
    body = _topic_body_with_sources()
    topic = CodeMapperTopic(
        order=1,
        title="Example",
        source_path=Path("example.md"),
        raw_markdown=body,
        body_markdown=body,
    )

    pages = await cleanup_topics(
        topics=[topic],
        llm_manager=_FakeLLMManager(_FixedCleanupProvider("## Overview\nOverview text.")),  # type: ignore[arg-type]
        config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
    )

    assert pages
    output = pages[0].body_markdown
    assert "## Sources" not in output
    assert "## References" in output
    assert '- <a id="ref-1"></a>[1] `repo/src/main.py` (2 chunks: L1-10, L20-30)' in output
    assert '- <a id="ref-2"></a>[2] `repo/tests/test_main.py` (1 chunks: L5-8)' in output


@pytest.mark.asyncio
async def test_cleanup_topics_filters_uncited_refs_with_citations() -> None:
    body = _topic_body_with_sources_and_citations()
    topic = CodeMapperTopic(
        order=1,
        title="Example",
        source_path=Path("example.md"),
        raw_markdown=body,
        body_markdown=body,
    )

    pages = await cleanup_topics(
        topics=[topic],
        llm_manager=_FakeLLMManager(
            _FixedCleanupProvider("## Overview\nOverview text [1].\n\n## Details\nMore info.")
        ),  # type: ignore[arg-type]
        config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
    )

    assert pages
    page = pages[0]
    output = page.body_markdown
    assert "## Sources" not in output
    assert "Overview text [[1]](#ref-1)." in output
    assert '- <a id="ref-1"></a>[1] `repo/src/main.py` (2 chunks: L1-10, L20-30)' in output
    assert "- [2] `repo/tests/test_main.py` (1 chunks: L5-8)" not in output
    assert page.references_count == 1


@pytest.mark.asyncio
async def test_cleanup_topics_does_not_filter_on_code_block_brackets() -> None:
    body = _topic_body_with_sources()
    topic = CodeMapperTopic(
        order=1,
        title="Example",
        source_path=Path("example.md"),
        raw_markdown=body,
        body_markdown=body,
    )

    cleaned = "\n".join(
        [
            "## Overview",
            "Example with brackets in code (not citations).",
            "",
            "```python",
            "arr[1]",
            "```",
            "",
            "Inline code: `arr[2]`.",
            "",
            "## Details",
            "More info.",
        ]
    )

    pages = await cleanup_topics(
        topics=[topic],
        llm_manager=_FakeLLMManager(_FixedCleanupProvider(cleaned)),  # type: ignore[arg-type]
        config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
        ),
    )

    assert pages
    output = pages[0].body_markdown
    assert "arr[1]" in output
    assert "`arr[2]`" in output
    assert '- <a id="ref-1"></a>[1] `repo/src/main.py` (2 chunks: L1-10, L20-30)' in output
    assert '- <a id="ref-2"></a>[2] `repo/tests/test_main.py` (1 chunks: L5-8)' in output


def test_reference_normalization_errors_when_sources_block_unparseable() -> None:
    markdown = "\n".join(
        [
            "## Overview",
            "",
            "Overview text.",
            "",
            "## Sources",
            "",
            "Files: 1",
            " - [1] src/main.py (L1-2)",
        ]
    )
    sources = extract_sources_block(markdown)
    assert sources is not None

    with pytest.raises(ValueError, match="Sources"):
        _apply_reference_normalization(markdown, sources)
