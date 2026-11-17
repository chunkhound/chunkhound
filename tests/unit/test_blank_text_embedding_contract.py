"""Contract: embedding N texts yields N vectors, even when some are blank."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import voyageai

from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.providers.embeddings.voyageai_provider import (
    VoyageAIEmbeddingProvider,
)
from chunkhound.services.web_research_service import _text_chunks


class _FakeVoyageClient:
    """Fake HTTP boundary: one vector per text actually received."""

    def __init__(self, dims: int = 1024) -> None:
        self.dims = dims
        self._params: dict[str, object] = {}
        self.received: list[list[str]] = []

    def embed(self, texts: list[str], **_: object) -> SimpleNamespace:
        self.received.append(list(texts))
        return SimpleNamespace(
            embeddings=[[1.0] + [0.0] * (self.dims - 1) for _ in texts], total_tokens=1
        )


def _provider(
    dims: int = 1024, **kwargs: object
) -> tuple[VoyageAIEmbeddingProvider, _FakeVoyageClient]:
    client = _FakeVoyageClient(dims)
    with patch.object(voyageai, "Client", return_value=client):
        return VoyageAIEmbeddingProvider(api_key="k", **kwargs), client


@pytest.mark.asyncio
async def test_voyage_returns_one_vector_per_text_with_blanks() -> None:
    provider, client = _provider()
    texts = ["a", "", "   \n", "b"]
    vectors = await provider.embed(texts)
    assert len(vectors) == len(texts)
    assert client.received == [["a", "b"]]
    assert vectors[0][0] == 1.0 and vectors[3][0] == 1.0
    assert not any(vectors[1]) and not any(vectors[2])
    assert len(vectors[1]) == len(vectors[0])


@pytest.mark.asyncio
async def test_voyage_blank_padding_matches_real_vector_length() -> None:
    # Custom model whose real dimension differs from provider.dims.
    provider, _ = _provider(dims=256, model="my-custom-model")
    vectors = await provider.embed(["a", " "])
    assert [len(v) for v in vectors] == [256, 256]


@pytest.mark.asyncio
async def test_voyage_batch_blank_padding_matches_real_vector_length() -> None:
    provider, client = _provider(dims=256, model="my-custom-model")
    texts = ["", "a", " \n", "b", "\t"]

    vectors = await provider.embed_batch(texts, batch_size=1)

    assert len(vectors) == len(texts)
    assert [len(vector) for vector in vectors] == [256] * len(texts)
    assert client.received == [["a"], ["b"]]
    assert vectors[1][0] == 1.0 and vectors[3][0] == 1.0
    assert all(not any(vectors[i]) for i in (0, 2, 4))


@pytest.mark.asyncio
async def test_voyage_all_blank_makes_no_request() -> None:
    provider, client = _provider()
    assert len(await provider.embed(["", " "])) == 2
    assert client.received == []


@pytest.mark.asyncio
async def test_blank_query_is_rejected_with_clear_error() -> None:
    provider, client = _provider()
    manager = EmbeddingManager()
    manager.register_provider(provider, set_default=True)
    with pytest.raises(ValueError, match="blank"):
        await manager.embed_query_texts(["  "])
    with pytest.raises(ValueError, match="blank"):
        await provider.embed_query([""])
    assert client.received == []


@pytest.mark.asyncio
async def test_whitespace_only_page_embeds_through_manager() -> None:
    provider, _ = _provider()
    manager = EmbeddingManager()
    manager.register_provider(provider, set_default=True)
    chunks = _text_chunks("https://example.test", "   \n\n\t\n")
    texts = [c.code for c in chunks] + ["real text"]
    result = await manager.embed_texts(texts)
    assert len(result.embeddings) == len(texts)


class _FakeOpenAIClient:
    """Fake HTTP boundary for OpenAI: one vector per input, optionally short."""

    def __init__(self, drop_last: bool = False) -> None:
        self.drop_last = drop_last
        self.inputs: list[int] = []
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: object) -> SimpleNamespace:
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        self.inputs.append(len(inputs))
        n = len(inputs) - 1 if self.drop_last and len(inputs) > 1 else len(inputs)
        data = [SimpleNamespace(index=i, embedding=[1.0] * 1536) for i in range(n)]
        return SimpleNamespace(data=data, usage=None)


def _openai_provider(client: _FakeOpenAIClient) -> OpenAIEmbeddingProvider:
    provider = OpenAIEmbeddingProvider(
        api_key="k", model="text-embedding-3-small", retry_attempts=1
    )
    provider._client = client  # type: ignore[assignment]
    provider._client_initialized = True
    return provider


@pytest.mark.asyncio
async def test_openai_embed_batch_split_oversized_text_is_not_a_mismatch() -> None:
    client = _FakeOpenAIClient()
    provider = _openai_provider(client)
    oversized = "word " * 20000  # far beyond the model token limit
    vectors = await provider.embed_batch(["short", oversized])
    # Splitting legitimately yields more vectors than inputs.
    assert len(vectors) > 2


@pytest.mark.asyncio
async def test_openai_embed_batch_short_response_fails_loudly() -> None:
    provider = _openai_provider(_FakeOpenAIClient(drop_last=True))
    with pytest.raises(Exception, match="(?i)vectors|incomplete|missing"):
        await provider.embed_batch(["a", "b", "word " * 20000])
