"""LLM-prompt body assembly in `option_chunk_rerank`.

The body-assembly loop protects one user-visible contract: the LLM prompt
must not waste tokens on redundant section headers, but must re-emit a
section header whenever the intervening context has drifted away from it.
Concretely:

- Consecutive chunks under the same `parent_header` emit the header once
  (no adjacent duplicates).
- A heading *chunk* (`parent_header=None`, `is_heading=True`) contributes
  its own content as the section header, so an immediately following
  chunk under that heading must not re-emit.
- A top-of-document non-heading chunk (`parent_header=None`,
  `is_heading=False`) must not invent a phantom header line, and must
  not prime the "current section" state.
- A chunk under a re-encountered header (after chunks under a different
  header interrupted) must re-emit the header, since the LLM otherwise
  loses the section context.

These are asserted structurally: parse the prompt body back into per-chunk
parts, extract each part's leading header line, and check adjacency and
sequence directly. This lets the tests survive cosmetic reformatting of
the loop (marker syntax, whitespace) while still failing loudly on the
real regressions. The join separator is mirrored as ``_SEPARATOR`` below
and must be updated in lockstep if ``fetchurl.py`` changes it.

Setup keeps rerank scores uniform so `filter_chunks_by_elbow` takes its
`no_elbow_detected` passthrough branch and every chunk reaches body
assembly in `start_line` order.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.core.config.config import Config
from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.interfaces.embedding_provider import EmbeddingProvider, RerankResult
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.fetchurl import option_chunk_rerank

# Mirrors the separator string joined in fetchurl.py::option_chunk_rerank.
# Kept as a single constant so a format change touches one line here.
_SEPARATOR = "\n\n---\n\n"


def _content_chunk(line: int, code: str, parent_header: str | None) -> Chunk:
    return Chunk(
        symbol=f"chunk_{line}",
        start_line=LineNumber(line),
        end_line=LineNumber(line),
        code=code,
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.MARKDOWN,
        parent_header=parent_header,
    )


def _heading_chunk(line: int, code: str) -> Chunk:
    return Chunk(
        symbol=f"heading_{line}",
        start_line=LineNumber(line),
        end_line=LineNumber(line),
        code=code,
        chunk_type=ChunkType.UNKNOWN,
        file_id=FileId(0),
        language=Language.MARKDOWN,
        parent_header=None,
        metadata={"node_type": "atx_heading"},
    )


async def _capture_body(chunks: list[Chunk], tmp_path) -> str:
    provider = MagicMock(spec=EmbeddingProvider)
    provider.get_max_rerank_batch_size.return_value = 100
    # Uniform scores => elbow filter passthrough => all chunks kept, sorted by
    # start_line in fetchurl.
    provider.rerank = AsyncMock(
        return_value=[
            RerankResult(index=i, score=0.5) for i in range(len(chunks))
        ]
    )

    utility_provider = MagicMock(spec=LLMProvider)
    complete = AsyncMock(return_value=SimpleNamespace(content="answer"))
    utility_provider.complete = complete
    llm_manager = MagicMock(spec=LLMManager)
    llm_manager.get_utility_provider.return_value = utility_provider

    await option_chunk_rerank(
        chunks,
        query="query-Q",
        url="http://example.test/doc",
        title="Doc-Title",
        embedding_provider=provider,
        llm_manager=llm_manager,
        config=Config(target_dir=tmp_path),
    )

    return complete.call_args.kwargs["prompt"]


def _parts(prompt: str) -> list[str]:
    """Split the assembled body into per-chunk parts.

    The prompt is wrapped by FOCUSED_USER_TEMPLATE; body sits between the
    ``<content>`` and ``</content>`` tags. Anchoring on those tags rather
    than on the locator prefix ``[L`` matters because the template itself
    contains a literal ``[L<start>-<end>]`` in its instructions.
    """
    start = prompt.index("<content>\n") + len("<content>\n")
    end = prompt.index("\n</content>")
    return prompt[start:end].split(_SEPARATOR)


def _header_line(part: str) -> str | None:
    """First non-locator line of a part, iff it starts with ``#``.

    Skips a leading locator line (``[L...]`` or ``[P...]``) since it
    changes format independently of the header-emit invariant. Returns
    the header for both surface shapes:
    - ``parent_header`` emitted:  ``[L5-5]\\n## Intro\\n\\nbody``
    - heading chunk:              ``[L5-5]\\n## Intro``
    Both yield ``"## Intro"`` — which is what we want, since a heading
    chunk and a re-emitted header contribute the same section context.
    """
    for line in part.splitlines():
        if line.startswith(("[L", "[P")):
            continue
        return line if line.startswith("#") else None
    return None


def _assert_no_adjacent_duplicate_headers(headers: list[str | None]) -> None:
    """The token-wasting failure mode expressed directly."""
    for a, b in zip(headers, headers[1:]):
        assert not (a is not None and a == b), (
            f"Adjacent chunk-parts share header line {a!r} — dedup regression."
        )


@pytest.mark.asyncio
async def test_parent_header_deduped_across_consecutive_chunks(tmp_path) -> None:
    chunks = [
        _content_chunk(5, "alpha body", parent_header="## Intro"),
        _content_chunk(10, "beta body", parent_header="## Intro"),
    ]
    prompt = await _capture_body(chunks, tmp_path)
    parts = _parts(prompt)
    headers = [_header_line(p) for p in parts]

    _assert_no_adjacent_duplicate_headers(headers)
    assert headers[0] == "## Intro"
    assert headers[1] is None
    assert "alpha body" in parts[0]
    assert "beta body" in parts[1]


@pytest.mark.asyncio
async def test_heading_chunk_primes_last_header_and_suppresses_next(
    tmp_path,
) -> None:
    chunks = [
        _heading_chunk(5, "## Intro"),
        _content_chunk(10, "under-intro body", parent_header="## Intro"),
    ]
    prompt = await _capture_body(chunks, tmp_path)
    parts = _parts(prompt)
    headers = [_header_line(p) for p in parts]

    _assert_no_adjacent_duplicate_headers(headers)
    # Heading chunk's own content is the section header; follower must not
    # re-emit.
    assert headers[0] == "## Intro"
    assert headers[1] is None
    assert "under-intro body" in parts[1]


@pytest.mark.asyncio
async def test_top_of_document_non_heading_chunk_emits_no_bogus_header(
    tmp_path,
) -> None:
    chunks = [
        _content_chunk(1, "preamble text", parent_header=None),
        _heading_chunk(5, "## Section"),
        _content_chunk(10, "under-section body", parent_header="## Section"),
    ]
    prompt = await _capture_body(chunks, tmp_path)
    parts = _parts(prompt)
    headers = [_header_line(p) for p in parts]

    _assert_no_adjacent_duplicate_headers(headers)
    # Preamble contributes no phantom header.
    assert headers[0] is None
    assert "preamble text" in parts[0]
    # Heading chunk emits, immediate follower dedups.
    assert headers[1] == "## Section"
    assert headers[2] is None
    assert "under-section body" in parts[2]


@pytest.mark.asyncio
async def test_re_encountered_header_re_emits(tmp_path) -> None:
    chunks = [
        _content_chunk(1, "x-body-1", parent_header="## X"),
        _content_chunk(5, "y-body", parent_header="## Y"),
        _content_chunk(10, "x-body-2", parent_header="## X"),
    ]
    prompt = await _capture_body(chunks, tmp_path)
    parts = _parts(prompt)
    headers = [_header_line(p) for p in parts]

    _assert_no_adjacent_duplicate_headers(headers)
    # X interrupted by Y — X must re-emit so the LLM re-anchors on the section.
    assert headers == ["## X", "## Y", "## X"]
