"""Backend URL contracts independent of the site and Node runtime."""

from inspect import signature

import pytest
from loguru import logger

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.providers.embeddings.voyageai_provider import VoyageAIEmbeddingProvider
from tests.fixtures.rerank_server_manager import RerankServerManager
from tests.rerank_server import MockRerankResult, MockRerankScenario


@pytest.fixture
def warning_records():
    """Capture rerank warnings without depending on the configured log format."""
    records: list[tuple[str, str]] = []

    def capture(message) -> None:
        text = message.record["message"]
        if "rerank_url=" in text:
            records.append((message.record["level"].name, text))

    sink_id = logger.add(capture, level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("scheme", ["http", "HTTP", "hTtPs"])
def test_active_rerank_url_normalizes_scheme_only(provider, scheme):
    endpoint = "Reranker.EXAMPLE/Case%2FPath?q=KeepCase#Fragment"
    config = EmbeddingConfig(
        provider=provider, rerank_format="tei", rerank_url=f"{scheme}://{endpoint}"
    )
    assert config.rerank_url == f"{scheme.lower()}://{endpoint}"


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/rerank",
        "http:/host",
        "https://",
        "//host/rerank",
        " https://host/rerank",
        # A backslash makes the effective host parser-dependent, so both layers
        # reject it instead of letting httpx target a host the UI never showed.
        r"http://host\@evil/rerank",
        r"http://example.com/a\b",
        r"https://host\evil",
    ],
)
@pytest.mark.parametrize("active", [False, True])
def test_only_active_rerank_endpoints_are_validated(provider, url, active):
    kwargs = dict(provider=provider, rerank_url=url, rerank_format="auto")
    if active:
        kwargs["rerank_model"] = "test-model"
        with pytest.raises(ValueError, match="complete HTTP"):
            EmbeddingConfig(**kwargs)
    else:
        assert EmbeddingConfig(**kwargs).rerank_url == url


@pytest.mark.parametrize("codepoint", ["\u0085", "\u001c", "\u001f", "\t", "\u00a0"])
def test_python_whitespace_class_is_rejected_in_active_urls(codepoint):
    # Python str.isspace() includes NEL and C0 separators that JS `\s` misses.
    with pytest.raises(ValueError, match="complete HTTP"):
        EmbeddingConfig(
            provider="openai",
            rerank_format="tei",
            rerank_model="test-model",
            rerank_url=f"http://host/rer{codepoint}ank",
        )


def test_bom_character_is_not_whitespace():
    # U+FEFF is not str.isspace(); the UI class must accept it too.
    config = EmbeddingConfig(
        provider="openai",
        rerank_format="tei",
        rerank_model="test-model",
        rerank_url="http://host/rer\ufeffank",
    )
    assert "\ufeff" in config.rerank_url


def test_full_python_whitespace_class_rejected_in_active_urls():
    # Derive expected set from Python so future Unicode additions stay pinned.
    rejected = [c for c in map(chr, range(0x3100)) if c.isspace()]
    assert len(rejected) > 20  # sanity: NEL, C0, NBSP, Ogham, em spaces, etc.
    for char in rejected:
        with pytest.raises(ValueError, match="complete HTTP"):
            EmbeddingConfig(
                provider="openai",
                rerank_format="tei",
                rerank_model="test-model",
                rerank_url=f"http://host/rer{char}ank",
            )
    # U+FEFF must stay accepted; it is not whitespace.
    assert not "\ufeff".isspace()


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
def test_unused_endpoint_does_not_prevent_provider_creation(provider):
    config = EmbeddingConfig(
        provider=provider,
        api_key="local-test-key",
        model="local-model",
        base_url="http://localhost:9999",
        rerank_url="ftp://unused/rerank",
        rerank_format="auto",
        rerank_model=None,
    )
    EmbeddingProviderFactory.create_provider(config)


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("base_url", [None, "https://embedding.example/v1"])
def test_active_relative_rerank_url_requires_base_url(provider, base_url):
    kwargs = dict(
        provider=provider, base_url=base_url, rerank_format="tei", rerank_url="/rerank"
    )
    if base_url:
        assert EmbeddingConfig(**kwargs).rerank_url == "/rerank"
    else:
        with pytest.raises(ValueError, match="requires base_url"):
            EmbeddingConfig(**kwargs)


def test_voyage_format_requires_voyageai_provider():
    """The voyage rerank format is only supported by the voyageai provider.

    Non-VoyageAI providers have no voyage request/response branch, so accepting
    this format would emit a config that silently degrades at rerank time.
    """
    # OpenAI provider with voyage format must fail
    with pytest.raises(ValueError, match="only supported by provider='voyageai'"):
        EmbeddingConfig(
            provider="openai",
            rerank_format="voyage",
            rerank_url="https://example.com/rerank",
            rerank_model="test-model",
        )

    # VoyageAI provider with voyage format must succeed (when model is provided)
    config = EmbeddingConfig(
        provider="voyageai",
        rerank_format="voyage",
        rerank_model="rerank-3-lite",
    )
    assert config.rerank_format == "voyage"
    assert config.rerank_model == "rerank-3-lite"


def _provider(provider, entry, base_url, rerank_url):
    config = dict(
        api_key="local-test-key",
        model="local-embedding-model",
        base_url=base_url,
        rerank_url=rerank_url,
        rerank_format="tei",
        rerank_model=None,
    )
    if entry == "factory":
        return EmbeddingProviderFactory.create_provider(
            EmbeddingConfig(provider=provider, **config)
        )
    cls = OpenAIEmbeddingProvider if provider == "openai" else VoyageAIEmbeddingProvider
    return cls(**config)


@pytest.fixture
async def reranker():
    scenario = MockRerankScenario(
        name="url-contract",
        query="query",
        documents=["document"],
        results=[MockRerankResult(index=0, score=0.9)],
        response_format="tei",
    )
    async with RerankServerManager(scenarios=[scenario]) as manager:
        yield manager


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("entry", ["direct", "factory"])
@pytest.mark.parametrize("url_kind", ["uppercase", "mixedcase", "relative"])
async def test_rerank_requests_reach_the_configured_endpoint(
    reranker, provider, entry, url_kind
):
    url = f"{reranker.base_url}/rerank"
    if url_kind == "relative":
        url = "/rerank"
    else:
        url = url.replace("http:", "HTTP:" if url_kind == "uppercase" else "hTtP:")
    client = _provider(provider, entry, reranker.base_url, url)
    results = await client.rerank("query", ["document"])
    assert [(result.index, result.score) for result in results] == [(0, 0.9)]
    assert len(reranker.requests) == 1


def _voyage_config(**overrides):
    config = dict(
        provider="voyageai",
        api_key="local-test-key",
        model="local-embedding-model",
        rerank_format="auto",
    )
    config.update(overrides)
    return EmbeddingConfig(**config)


def _default_rerank_model():
    """Read the provider default instead of hardcoding a model string."""
    return (
        signature(VoyageAIEmbeddingProvider.__init__).parameters["rerank_model"].default
    )


def _cohere_scenario(name):
    return MockRerankScenario(
        name=name,
        query="query",
        documents=["document"],
        results=[MockRerankResult(index=0, score=0.9)],
        response_format="cohere",
    )


@pytest.mark.asyncio
async def test_auto_default_model_relative_url_resolves_against_base_url(reranker):
    """auto + omitted model + relative rerank_url + base_url reaches the endpoint."""
    reranker.set_scenarios([_cohere_scenario("auto-default-relative")])
    client = EmbeddingProviderFactory.create_provider(
        _voyage_config(base_url=reranker.base_url, rerank_url="/rerank")
    )
    assert client._rerank_url is not None
    assert client._rerank_url.startswith("http")
    results = await client.rerank("query", ["document"])
    assert [(result.index, result.score) for result in results] == [(0, 0.9)]
    assert len(reranker.requests) == 1


def test_auto_default_model_relative_url_without_base_url_does_not_block_load():
    """auto + omitted model + relative rerank_url with no base_url still loads.

    The relative path cannot resolve, so it is treated as an unused endpoint: the
    frozen contract is that an inactive rerank_url never blocks config load.
    """
    config = _voyage_config(rerank_url="/rerank")
    assert config.rerank_url == "/rerank"
    EmbeddingProviderFactory.create_provider(config)


@pytest.mark.asyncio
async def test_auto_default_model_absolute_url_uses_default_and_cohere_body(reranker):
    """auto + omitted model + absolute rerank_url keeps the provider default model."""
    reranker.set_scenarios([_cohere_scenario("auto-default-absolute")])
    client = EmbeddingProviderFactory.create_provider(
        _voyage_config(rerank_url=f"{reranker.base_url}/rerank")
    )
    assert client._rerank_url is not None
    assert client._rerank_url.startswith("http")
    results = await client.rerank("query", ["document"])
    assert [(result.index, result.score) for result in results] == [(0, 0.9)]
    body = reranker.requests[0]
    assert body["documents"] == ["document"]
    assert body["model"] == _default_rerank_model()


def test_voyage_factory_drops_malformed_rerank_url():
    """A malformed rerank_url never blocks Voyage provider creation: the
    factory drops it and the provider keeps SDK reranking with no endpoint."""
    client = EmbeddingProviderFactory.create_provider(
        _voyage_config(rerank_url="ftp://host/rerank")
    )
    assert client._rerank_url is None


def test_voyage_direct_relative_url_without_base_leaves_no_endpoint():
    """Direct construction mirrors the factory: a relative rerank_url with
    no base_url resolves to no endpoint instead of a stored relative path."""
    client = VoyageAIEmbeddingProvider(
        api_key="local-test-key",
        model="local-embedding-model",
        rerank_url="/rerank",
    )
    assert client._rerank_url is None


@pytest.mark.parametrize("bad_url", ["ftp://host/rerank", "//host/rerank"])
def test_voyage_direct_malformed_url_leaves_no_endpoint(bad_url):
    """Direct construction drops malformed endpoints like the factory."""
    client = VoyageAIEmbeddingProvider(
        api_key="local-test-key",
        model="local-embedding-model",
        rerank_url=bad_url,
    )
    assert client._rerank_url is None


def test_voyage_creation_log_redacts_rerank_url_secrets():
    """Debug logs must not echo userinfo credentials from rerank_url."""
    records: list[str] = []

    def capture(message) -> None:
        text = message.record["message"]
        if "Creating VoyageAI provider" in text:
            records.append(text)

    sink_id = logger.add(capture, level="DEBUG")
    try:
        EmbeddingProviderFactory.create_provider(
            _voyage_config(
                rerank_url="https://REVIEW_USER:REVIEW_SECRET@host/rerank",
                base_url="https://host/v1",
            )
        )
    finally:
        logger.remove(sink_id)

    assert len(records) == 1
    message = records[0]
    assert "REVIEW_USER" not in message
    assert "REVIEW_SECRET" not in message
    assert "https://host/rerank" in message


def _voyage_for_warning(entry, url, base_url=None, rerank_model=None):
    config = dict(
        api_key="local-test-key",
        model="local-embedding-model",
        rerank_model=rerank_model,
        rerank_url=url,
        base_url=base_url,
    )
    if entry == "factory":
        return EmbeddingProviderFactory.create_provider(_voyage_config(**config))
    return VoyageAIEmbeddingProvider(**config)


@pytest.mark.parametrize("entry", ["direct", "factory"])
def test_voyage_empty_rerank_url_is_the_same_unset_shape(entry):
    """An empty endpoint must behave exactly like None, not as HTTP reranking.

    The direct entry must not report SDK reranking as HTTP capability when no
    model is configured; the factory applies the same provider default either
    way, so its capability only has to match the None case.
    """
    unset = _voyage_for_warning(entry, None)
    empty = _voyage_for_warning(entry, "")
    assert empty._rerank_url is None
    assert empty.supports_reranking() == unset.supports_reranking()
    if entry == "direct":
        assert empty.supports_reranking() is False


@pytest.mark.parametrize("entry", ["direct", "factory"])
@pytest.mark.parametrize(
    "url,endpoint,reason",
    [
        ("/rerank", "/rerank", "base_url is required"),
        (
            "/rerank?api_key=REVIEW_SECRET#REVIEW_FRAGMENT",
            "/rerank",
            "base_url is required",
        ),
        (
            "ftp://REVIEW_USER:REVIEW_SECRET@host/rerank?key=REVIEW_QUERY",
            "ftp://host/rerank",
            "complete HTTP(S) URL",
        ),
        (
            "//REVIEW_USER:REVIEW_SECRET@host/rerank#REVIEW_FRAGMENT",
            "//host/rerank",
            "complete HTTP(S) URL",
        ),
        (
            "https://REVIEW_USER:REVIEW_SECRET@host/rerank\nFORGED LOG LINE",
            "https://host/rerank\\nFORGED LOG LINE",
            "complete HTTP(S) URL",
        ),
        (
            "https:/REVIEW_USER:REVIEW_SECRET@host/rerank",
            "<invalid URL>",
            "complete HTTP(S) URL",
        ),
        (
            "https://REVIEW_USER:bad/REVIEW_SECRET@host/rerank",
            "<invalid URL>",
            "complete HTTP(S) URL",
        ),
        (
            "https://REVIEW_USER:REVIEW_SECRET?x@host/rerank",
            "<invalid URL>",
            "complete HTTP(S) URL",
        ),
        (
            "https://REVIEW_USER:REVIEW_SECRET#x@host/rerank",
            "<invalid URL>",
            "complete HTTP(S) URL",
        ),
        (
            "https://REVIEW_USER:pa?ss@host/rerank",
            "<invalid URL>",
            "complete HTTP(S) URL",
        ),
        (
            "http://[REVIEW_USER:REVIEW_SECRET@host/rerank",
            "http://host/rerank",
            "complete HTTP(S) URL",
        ),
        (
            "https://host/rerank\r\t\x1b\u0085\u2028",
            "https://host/rerank",
            "complete HTTP(S) URL",
        ),
    ],
)
def test_voyage_dropped_endpoint_warns_safely(
    entry, url, endpoint, reason, warning_records
):
    _voyage_for_warning(entry, url)
    assert len(warning_records) == 1
    level, message = warning_records[0]
    assert level == "WARNING"
    assert "Dropping" in message and "rerank_url=" in message
    assert endpoint in message
    assert "VoyageAI provider" in message
    assert reason in message
    assert all(char.isprintable() for char in message)
    assert not any(char in message for char in "\r\n\t\x1b\u0085\u2028")
    for secret in ("USER", "SECRET", "QUERY", "FRAGMENT"):
        assert f"REVIEW_{secret}" not in message


@pytest.mark.parametrize("entry", ["direct", "factory"])
@pytest.mark.parametrize(
    "url,base_url",
    [
        (None, None),
        ("", None),
        ("https://host/rerank", None),
        ("/rerank", "https://host/v1"),
    ],
)
def test_voyage_usable_or_omitted_endpoint_does_not_warn(
    entry, url, base_url, warning_records
):
    _voyage_for_warning(entry, url, base_url)
    assert warning_records == []
