"""Tests for token estimation utilities."""

import pytest

from chunkhound.core.utils import token_utils
from chunkhound.core.utils.token_utils import (
    EMBEDDING_CHARS_PER_TOKEN,
    LLM_CHARS_PER_TOKEN,
    _cl100k_base_safe,
    _encoding_for_model_safe,
    estimate_tokens,
    estimate_tokens_chunking,
    estimate_tokens_llm,
)


class TestEstimateTokensLLM:
    """Tests for LLM token estimation."""

    def test_basic_ratio(self):
        """4 chars per token."""
        assert estimate_tokens_llm("a" * 400) == 100
        assert estimate_tokens_llm("a" * 4) == 1

    def test_empty_string(self):
        """Empty returns 0."""
        assert estimate_tokens_llm("") == 0

    def test_minimum_one_token(self):
        """Short strings return at least 1."""
        assert estimate_tokens_llm("a") == 1
        assert estimate_tokens_llm("abc") == 1

    def test_constants_defined(self):
        """Verify constants are exposed."""
        assert LLM_CHARS_PER_TOKEN == 4
        assert EMBEDDING_CHARS_PER_TOKEN == 3


class TestEstimateTokens:
    """Tests for the provider-dispatching estimate_tokens function."""

    def test_azure_openai_returns_positive(self):
        """estimate_tokens with azure_openai provider returns a positive count."""
        result = estimate_tokens("hello world", provider="azure_openai")
        assert result > 0

    def test_special_token_literal_does_not_raise(self):
        """A tiktoken special-token literal must be counted, not crash. #315."""
        result = estimate_tokens(
            "def f():  # <|endoftext|>\n    return 1", provider="azure_openai"
        )
        assert result > 0

    def test_empty_string_returns_zero(self):
        """Empty input short-circuits to 0, bypassing tiktoken entirely."""
        assert (
            estimate_tokens("", provider="openai", model="text-embedding-3-small")
            == 0
        )


class TestEstimateTokensChunking:
    """Tests for chunking token estimation."""

    def test_basic_ratio(self):
        """3 chars per token."""
        assert estimate_tokens_chunking("a" * 300) == 100
        assert estimate_tokens_chunking("a" * 3) == 1

    def test_empty_string(self):
        """Empty returns 0."""
        assert estimate_tokens_chunking("") == 0

    def test_minimum_one_token(self):
        """Short strings return at least 1."""
        assert estimate_tokens_chunking("a") == 1
        assert estimate_tokens_chunking("ab") == 1


class _FakeEnc:
    """Fake tiktoken encoding — `// 2` distinguishes it from the `// 3` heuristic."""

    def encode(self, text, disallowed_special=()):
        return [0] * (len(text) // 2)


class TestEstimateTokensFallback:
    """Contract tests for tiktoken gating and heuristic fallback in estimate_tokens()."""

    @pytest.fixture(autouse=True)
    def _clear_tiktoken_caches(self):
        _encoding_for_model_safe.cache_clear()
        _cl100k_base_safe.cache_clear()
        yield
        _encoding_for_model_safe.cache_clear()
        _cl100k_base_safe.cache_clear()

    @pytest.fixture
    def tiktoken_unknown_model(self, monkeypatch):
        """encoding_for_model raises KeyError; get_encoding returns the // 2 stub."""
        def _raise_keyerror(model):
            raise KeyError(f"unknown model: {model}")

        monkeypatch.setattr(
            token_utils.tiktoken, "encoding_for_model", _raise_keyerror
        )
        monkeypatch.setattr(
            token_utils.tiktoken, "get_encoding", lambda name: _FakeEnc()
        )

    def test_proxy_unknown_model_uses_char_heuristic(self, tiktoken_unknown_model):
        """Proxy endpoint + unknown model → char heuristic, NOT cl100k_base."""
        text = "a" * 60
        result = estimate_tokens(
            text,
            provider="openai",
            model="qwen-7b",
            base_url="http://localhost:8080/v1",
        )
        assert result == len(text) // EMBEDDING_CHARS_PER_TOKEN
        assert result != len(text) // 2

    def test_official_openai_unknown_model_uses_cl100k(self, tiktoken_unknown_model):
        """Official OpenAI + unknown model → cl100k_base fallback runs."""
        text = "a" * 60
        result = estimate_tokens(
            text,
            provider="openai",
            model="weird-model",
            base_url="https://api.openai.com/v1",
        )
        assert result == len(text) // 2

    def test_azure_unknown_model_uses_cl100k(self, tiktoken_unknown_model):
        """Azure trusts cl100k_base regardless of base_url."""
        text = "a" * 60
        result = estimate_tokens(
            text,
            provider="azure_openai",
            model="my-deployment",
            base_url="https://my-resource.openai.azure.com",
        )
        assert result == len(text) // 2

    def test_network_failure_falls_back_to_heuristic(self, monkeypatch):
        """Simulated blocked BPE download → char heuristic (no stall, no raise)."""
        def _raise_connerror(*args, **kwargs):
            raise ConnectionError("network blocked")

        monkeypatch.setattr(
            token_utils.tiktoken, "encoding_for_model", _raise_connerror
        )
        monkeypatch.setattr(
            token_utils.tiktoken, "get_encoding", _raise_connerror
        )

        text = "a" * 60
        result = estimate_tokens(
            text, provider="openai", model="text-embedding-3-small"
        )
        assert result == len(text) // EMBEDDING_CHARS_PER_TOKEN

    def test_voyageai_never_uses_tiktoken(self, monkeypatch):
        """provider=voyageai must bypass the OpenAI tokenizer path entirely."""
        # Tripwire: voyageai shouldn't reach tiktoken today, but if a future
        # refactor routes it through _estimate_tokens_openai the // 2 stub
        # would surface via the `!= len(text) // 2` assertion below.
        monkeypatch.setattr(
            token_utils.tiktoken, "encoding_for_model", lambda model: _FakeEnc()
        )
        monkeypatch.setattr(
            token_utils.tiktoken, "get_encoding", lambda name: _FakeEnc()
        )

        text = "a" * 60
        result = estimate_tokens(text, provider="voyageai", model="voyage-3")
        assert result == len(text) // EMBEDDING_CHARS_PER_TOKEN
        assert result != len(text) // 2
