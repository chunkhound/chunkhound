"""Tests for token estimation utilities."""

from chunkhound.core.utils.token_utils import (
    EMBEDDING_CHARS_PER_TOKEN,
    LLM_CHARS_PER_TOKEN,
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
