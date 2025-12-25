from pathlib import Path

import pytest

from chunkhound.autodoc import docsite


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


def test_extract_sources_block_returns_section_body() -> None:
    markdown = _topic_body_with_sources()
    block = docsite.extract_sources_block(markdown)

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

    stripped = docsite.strip_references_section(markdown)

    assert "## Sources" not in stripped
    assert "## References" not in stripped
    assert "## Overview" in stripped
    assert "## Details" in stripped
    assert "## Outro" in stripped


def test_flatten_sources_block_builds_list_with_paths_and_chunks() -> None:
    flat = docsite.flatten_sources_block(_sources_block())

    assert flat == [
        "- [1] `repo/src/main.py` (2 chunks: L1-10, L20-30)",
        "- [2] `repo/tests/test_main.py` (1 chunks: L5-8)",
    ]


def test_build_references_section_omits_when_missing_sources() -> None:
    body = "\n".join(["## Overview", "", "Overview text."])
    normalized = docsite._apply_reference_normalization(body, None)

    assert "## References" not in normalized


@pytest.mark.asyncio
async def test_cleanup_topics_injects_flattened_references() -> None:
    body = _topic_body_with_sources()
    topic = docsite.CodeMapperTopic(
        order=1,
        title="Example",
        source_path=Path("example.md"),
        raw_markdown=body,
        body_markdown=body,
    )

    pages = await docsite.cleanup_topics(
        topics=[topic],
        llm_manager=None,
        config=docsite.CleanupConfig(
            mode="minimal",
            batch_size=1,
            max_completion_tokens=512,
        ),
    )

    assert pages
    output = pages[0].body_markdown
    assert "## Sources" not in output
    assert "## References" in output
    assert "- [1] `repo/src/main.py` (2 chunks: L1-10, L20-30)" in output
