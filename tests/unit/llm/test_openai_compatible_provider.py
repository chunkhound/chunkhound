"""Contract tests for the OpenAI-compatible provider pipeline.

All OpenAI-compatible providers (openai, deepseek, grok) share the same
``OpenAICompatibleProvider`` class.  They differ only in spec data stored
in ``OPENAI_COMPATIBLE_PROVIDERS`` (base URL, default model, feature flags).

This file tests two things:
1.  The **factory contract**: config → registry → correctly configured
    ``OpenAICompatibleProvider`` instance.
2.  The **behavioral contract** of ``OpenAICompatibleProvider`` itself.

Every OpenAI-compatible provider is exercised through parametrized spec
entries.  Adding a new provider = one dict in ``SPECS`` + one dict in the
production registry.  No per-provider subclass or per-provider test file.

Per-provider subclass tests are an anti-pattern — they all test the same
base class with different constructor defaults.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from chunkhound.llm_manager import LLMManager
from chunkhound.providers.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
)

# =============================================================================
# Spec data — kept in sync with OPENAI_COMPATIBLE_PROVIDERS in production.
# Each entry is a subset of the config dict that would come from LLMConfig.
# Only registry-based providers (deepseek, grok) are included.
# =============================================================================

SPECS = [
    pytest.param(
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "expected_name": "deepseek",
            "expected_base_url": "https://api.deepseek.com",
            "expected_sso": False,
        },
        id="deepseek",
    ),
    pytest.param(
        {
            "provider": "grok",
            "model": "grok-4-1-fast-reasoning",
            "expected_name": "grok",
            "expected_base_url": "https://api.x.ai/v1",
            "expected_sso": True,
        },
        id="grok",
    ),

]


@pytest.fixture
def mock_openai():
    """Prevent real HTTP calls by patching AsyncOpenAI at the boundary.

    The mock client's ``base_url`` is set to a real string matching the kwargs
    passed to ``AsyncOpenAI()``.  All clients created by the factory share a
    single ``chat.completions.create`` mock so tests can set return values
    globally.
    """
    shared_complete = AsyncMock()

    def _factory(**kwargs):
        client = AsyncMock()
        client.base_url = kwargs.get("base_url", "https://api.openai.com/v1/")
        client.chat.completions.create = shared_complete
        return client

    with patch(
        "chunkhound.providers.llm.openai_compatible_provider.AsyncOpenAI",
        side_effect=_factory,
    ):
        yield shared_complete


# =============================================================================
# Factory contract  —  does LLMManager._create_provider produce a correctly
# configured OpenAICompatibleProvider for each spec?
# =============================================================================

class TestFactoryPipeline:
    """Config dict → registry lookup → provider instance with correct fields."""

    @pytest.mark.parametrize("spec", SPECS)
    def test_creates_openai_compatible_provider(self, spec, mock_openai):
        """Returns the right class with the right name and model."""
        manager = _bare_manager()
        provider = manager._create_provider(spec)

        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.name == spec["expected_name"]
        assert provider.model == spec["model"]

    @pytest.mark.parametrize("spec", SPECS)
    def test_default_base_url(self, spec, mock_openai):
        """Provider gets the spec's default base_url when none is in the config."""
        manager = _bare_manager()
        provider = manager._create_provider(spec)

        expected = spec["expected_base_url"]
        if expected is not None:
            assert str(provider._client.base_url).rstrip("/") == expected.rstrip("/")

    @pytest.mark.parametrize("spec", SPECS)
    def test_explicit_base_url_overrides_spec(self, spec, mock_openai):
        """Config-level base_url takes precedence over the spec default."""
        cfg = {**spec, "base_url": "http://localhost:11434/v1"}
        manager = _bare_manager()
        provider = manager._create_provider(cfg)

        assert str(provider._client.base_url).rstrip("/") == "http://localhost:11434/v1"

    @pytest.mark.parametrize("spec", SPECS)
    def test_explicit_model_overrides_spec(self, spec, mock_openai):
        """Config-level model takes precedence over the spec default."""
        cfg = {**spec, "model": "my-custom-model"}
        manager = _bare_manager()
        provider = manager._create_provider(cfg)

        assert provider.model == "my-custom-model"

    @pytest.mark.parametrize("spec", SPECS)
    def test_structured_outputs_flag(self, spec, mock_openai):
        """supports_structured_outputs matches the spec unless config overrides."""
        manager = _bare_manager()
        provider = manager._create_provider(spec)

        assert provider._supports_structured_outputs == spec["expected_sso"]

    @pytest.mark.parametrize("spec", SPECS)
    def test_can_override_structured_outputs_flag(self, spec, mock_openai):
        """Explicit supports_structured_outputs in config beats spec default."""
        inverted = not spec["expected_sso"]
        cfg = {**spec, "supports_structured_outputs": inverted}
        manager = _bare_manager()
        provider = manager._create_provider(cfg)

        assert provider._supports_structured_outputs == inverted

    @pytest.mark.parametrize("spec", SPECS)
    def test_ssl_verify_forwarded_without_crash(self, spec, mock_openai):
        """ssl_verify=True must not cause TypeError. (THE BUG)"""
        cfg = {**spec, "ssl_verify": True, "base_url": "http://localhost:11434/v1"}
        manager = _bare_manager()
        manager._create_provider(cfg)

    @pytest.mark.parametrize("spec", SPECS)
    def test_custom_endpoint_requires_explicit_model(self, spec, mock_openai):
        """Custom base_url without explicit model → clear ValueError."""
        cfg = {"provider": spec["provider"], "base_url": "http://localhost:11434/v1"}
        with pytest.raises(ValueError) as exc:
            manager = _bare_manager()
            manager._create_provider(cfg)
        assert "Model is required" in str(exc.value)

    @pytest.mark.parametrize("spec", SPECS)
    def test_timeout_default_applied(self, spec, mock_openai):
        """Default timeout is used when not in config."""
        from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT

        manager = _bare_manager()
        provider = manager._create_provider(spec)

        assert provider.timeout == DEFAULT_LLM_TIMEOUT

    @pytest.mark.parametrize("spec", SPECS)
    def test_explicit_timeout_used(self, spec, mock_openai):
        """Config-level timeout overrides the default."""
        cfg = {**spec, "timeout": 300}
        manager = _bare_manager()
        provider = manager._create_provider(cfg)

        assert provider.timeout == 300

    @pytest.mark.parametrize("spec", SPECS)
    def test_synthesis_concurrency_matches_spec(self, spec, mock_openai):
        """Synthesis concurrency matches the spec value."""
        manager = _bare_manager()
        provider = manager._create_provider(spec)
        expected = {"deepseek": 10, "grok": 5}[spec["provider"]]
        assert provider.get_synthesis_concurrency() == expected


# =============================================================================
# Behavioral contract  —  core provider behavior exercised once (not per spec),
# since all OpenAI-compatible providers share the same class.
# =============================================================================

class TestCompletionContract:
    """Core completion behavior shared by every OpenAI-compatible provider."""

    def _make_resp(self, content="ok", finish="stop", p=10, c=5):
        """Build a fake ChatCompletion response.

        Returns a minimal object with the attributes ``complete()`` reads.
        Not an AsyncMock — avoids auto-generated child mocks for ``content``.
        """
        from types import SimpleNamespace

        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish,
        )
        usage = SimpleNamespace(
            prompt_tokens=p,
            completion_tokens=c,
            total_tokens=p + c,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    @pytest.mark.asyncio
    async def test_complete_returns_llmresponse(self, mock_openai):
        """Core contract: the public complete() returns an LLMResponse."""
        from chunkhound.interfaces.llm_provider import LLMResponse

        mock_openai.return_value = self._make_resp("hello")

        provider = _provider(provider_name="deepseek", model="deepseek-v4-flash")
        response = await provider.complete("Say hello")

        assert isinstance(response, LLMResponse)
        assert response.content == "hello"
        assert response.tokens_used == 15
        assert response.model == "deepseek-v4-flash"

    @pytest.mark.asyncio
    async def test_complete_sends_model_in_api_call(self, mock_openai):
        """The configured model is forwarded to the API."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(provider_name="grok", model="grok-beta")
        await provider.complete("hi")

        kwargs = mock_openai.call_args[1]
        assert kwargs["model"] == "grok-beta"

    @pytest.mark.asyncio
    async def test_api_error_propagates(self, mock_openai):
        """API exceptions are wrapped in RuntimeError, never swallowed."""
        mock_openai.side_effect = Exception("API error")

        provider = _provider()
        with pytest.raises(RuntimeError, match="LLM completion failed"):
            await provider.complete("fail")

    @pytest.mark.asyncio
    async def test_truncation_error_before_empty_check(self, mock_openai):
        """finish_reason='length' → token-limit error, not generic empty error."""
        mock_openai.return_value = self._make_resp(
            content="  ", finish="length", p=500, c=0
        )

        provider = _provider()
        with pytest.raises(RuntimeError, match="token limit exceeded"):
            await provider.complete("hi")

    @pytest.mark.asyncio
    async def test_max_tokens_param_used_for_deepseek_style_providers(
        self, mock_openai
    ):
        """Providers with max_tokens_param_name='max_tokens' use that in the API."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(max_tokens_param_name="max_tokens")
        await provider.complete("hi", max_completion_tokens=200)

        kwargs = mock_openai.call_args[1]
        assert "max_tokens" in kwargs
        assert kwargs["max_tokens"] == 200
        assert "max_completion_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_default_max_completion_tokens_param(self, mock_openai):
        """Default max_tokens_param_name='max_completion_tokens' is used in API call."""
        mock_openai.return_value = self._make_resp()

        # No max_tokens_param_name override → default "max_completion_tokens"
        provider = _provider(provider_name="grok")
        await provider.complete("hi", max_completion_tokens=200)

        kwargs = mock_openai.call_args[1]
        assert kwargs["max_completion_tokens"] == 200
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_reasoning_effort_forwarded_when_set(self, mock_openai):
        """reasoning_effort is included in API call when configured."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(reasoning_effort="high")
        await provider.complete("hi")

        kwargs = mock_openai.call_args[1]
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_reasoning_effort_omitted_when_unset(self, mock_openai):
        """reasoning_effort is absent from API call when not configured."""
        mock_openai.return_value = self._make_resp()

        provider = _provider()  # no reasoning_effort
        await provider.complete("hi")

        kwargs = mock_openai.call_args[1]
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_timeout_parameter_forwarded(self, mock_openai):
        """Per-call timeout is sent to the API."""
        mock_openai.return_value = self._make_resp()

        provider = _provider()
        await provider.complete("hi", timeout=30)

        kwargs = mock_openai.call_args[1]
        assert kwargs["timeout"] == 30

    def test_usage_stats_tracking(self, mock_openai):
        """get_usage_stats() tracks requests and token counts."""
        provider = _provider(model="test-model")
        stats = provider.get_usage_stats()
        assert stats["requests_made"] == 0
        assert stats["total_tokens"] == 0
        assert stats["prompt_tokens"] == 0
        assert stats["completion_tokens"] == 0


# =============================================================================
# Structured-output contract  —  behavioral difference between providers with
# supports_structured_outputs=True vs False (parametrized over both paths).
# =============================================================================

class TestStructuredOutputContract:
    """Two code paths depending on the supports_structured_outputs flag."""

    SCHEMA = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def _make_resp(self, content='{"answer": "42"}'):
        r = AsyncMock()
        r.choices = [AsyncMock(message=AsyncMock(content=content))]
        r.choices[0].finish_reason = "stop"
        r.usage = AsyncMock(total_tokens=10)
        return r

    @pytest.mark.asyncio
    async def test_native_json_schema_path(self, mock_openai):
        """sso=True → native json_schema response_format."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(supports_structured_outputs=True)
        result = await provider.complete_structured("?", json_schema=self.SCHEMA)

        assert result == {"answer": "42"}
        call = mock_openai.call_args[1]
        assert call["response_format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_prompt_based_fallback_path(self, mock_openai):
        """sso=False → json_object format + schema injected into system prompt."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(supports_structured_outputs=False)
        result = await provider.complete_structured("?", json_schema=self.SCHEMA)

        assert result == {"answer": "42"}
        call = mock_openai.call_args[1]
        assert call["response_format"] == {"type": "json_object"}
        assert '"answer"' in call["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_fallback_preserves_existing_system_prompt(self, mock_openai):
        """Existing system prompt is preserved, schema is appended."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(supports_structured_outputs=False)
        await provider.complete_structured(
            "?", json_schema=self.SCHEMA, system="You are helpful.",
        )

        call = mock_openai.call_args[1]
        content = call["messages"][0]["content"]
        assert "You are helpful." in content
        assert '"answer"' in content

    @pytest.mark.asyncio
    async def test_native_json_schema_with_reasoning_effort(
        self, mock_openai
    ):
        """sso=True includes both response_format and reasoning_effort."""
        mock_openai.return_value = self._make_resp()

        provider = _provider(
            supports_structured_outputs=True,
            reasoning_effort="medium",
        )
        await provider.complete_structured("?", json_schema=self.SCHEMA)

        call = mock_openai.call_args[1]
        assert call["response_format"]["type"] == "json_schema"
        assert call["reasoning_effort"] == "medium"


# =============================================================================
# Helpers
# =============================================================================

def _bare_manager() -> LLMManager:
    """Return an LLMManager instance without running __init__ side effects."""
    manager = object.__new__(LLMManager)
    manager._providers = LLMManager._providers
    return manager


def _provider(**overrides: object) -> OpenAICompatibleProvider:
    """Construct an OpenAICompatibleProvider with minimal defaults."""
    kwargs = {
        "provider_name": "deepseek",
        "api_key": "sk-test",
        **overrides,
    }
    return OpenAICompatibleProvider(**kwargs)
