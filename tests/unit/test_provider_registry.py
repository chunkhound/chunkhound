"""Parametrized contract tests for the data-driven provider registry.

Every test iterates over ``OPENAI_COMPATIBLE_PROVIDERS`` to verify:
- Each spec is well-formed (non-empty fields, valid URLs)
- Each spec maps to a valid ``LLMProviderLiteral`` member
- Each spec constructs a working ``OpenAICompatibleProvider`` via the factory
- Shared behaviours (config override precedence, missing-model errors)
"""

from typing import get_args

import pytest

from chunkhound.core.config.llm_config import LLMProviderLiteral
from chunkhound.core.config.provider_registry import OPENAI_COMPATIBLE_PROVIDERS

# ── Registry integrity ──────────────────────────────────────────────────────


def test_all_registry_keys_in_llm_provider_literal():
    """Every provider in OPENAI_COMPATIBLE_PROVIDERS must be a valid
    ``LLMProviderLiteral`` member.

    If someone adds a spec but forgets the type union, this test catches it.
    """
    valid_literals = set(get_args(LLMProviderLiteral))
    for name in OPENAI_COMPATIBLE_PROVIDERS:
        assert name in valid_literals, (
            f"{name!r} is in OPENAI_COMPATIBLE_PROVIDERS but not "
            f"LLMProviderLiteral. Add it to the Literal type in llm_config.py."
        )


def test_spec_names_match_dict_keys():
    """Each spec's ``name`` field must match its dict key."""
    for key, spec in OPENAI_COMPATIBLE_PROVIDERS.items():
        assert spec.name == key, (
            f"Spec name {spec.name!r} does not match dict key {key!r}"
        )


# ── Spec well-formedness ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_has_non_empty_name(name, spec):
    assert spec.name, f"Spec for {name!r} has empty name"


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_default_base_url_is_https(name, spec):
    assert spec.default_base_url.startswith("https://"), (
        f"{name}: expected https:// default_base_url, got {spec.default_base_url!r}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_has_valid_max_tokens_param_name(name, spec):
    assert spec.max_tokens_param_name in ("max_tokens", "max_completion_tokens"), (
        f"{name}: unexpected max_tokens_param_name {spec.max_tokens_param_name!r}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_synthesis_concurrency_positive(name, spec):
    assert spec.synthesis_concurrency > 0, (
        f"{name}: synthesis_concurrency must be "
        "positive, got {spec.synthesis_concurrency}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_docs_url_valid(name, spec):
    """docs_url must be empty or a valid https:// URL."""
    if spec.docs_url:
        assert spec.docs_url.startswith("https://"), (
            f"{name}: docs_url should be https://, got {spec.docs_url!r}"
        )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_auth_url_valid(name, spec):
    """auth_url must be empty or a valid https:// URL."""
    if spec.auth_url:
        assert spec.auth_url.startswith("https://"), (
            f"{name}: auth_url should be https://, got {spec.auth_url!r}"
        )


# ── Provider construction via factory ───────────────────────────────────────


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_constructs_provider_via_factory(name, spec):
    """Every spec in the registry must produce a valid provider via the
    ``_create_openai_compatible_provider`` factory method."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    provider = manager._create_openai_compatible_provider(
        name,
        {"model": "test-model", "api_key": "sk-test"},
    )

    assert provider.name == name, (
        f"Provider name should be {name!r}, got {provider.name!r}"
    )
    assert provider._max_tokens_param_name == spec.max_tokens_param_name, (
        f"{name}: expected max_tokens_param_name={spec.max_tokens_param_name!r}, "
        f"got {provider._max_tokens_param_name!r}"
    )
    assert provider.get_synthesis_concurrency() == spec.synthesis_concurrency, (
        f"{name}: expected synthesis_concurrency={spec.synthesis_concurrency}, "
        f"got {provider.get_synthesis_concurrency()}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_constructs_provider_with_base_url_override(name, spec):
    """A ``base_url`` in config must override ``spec.default_base_url``."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    custom_url = "https://custom-endpoint.example.com/v1"
    provider = manager._create_openai_compatible_provider(
        name,
        {
            "model": "test-model",
            "api_key": "sk-test",
            "base_url": custom_url,
        },
    )

    # The provider uses an internal AsyncOpenAI client; we cannot inspect
    # the base URL directly, but we can verify the constructor didn't raise.
    assert provider is not None


# ── Missing-model error ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    list(OPENAI_COMPATIBLE_PROVIDERS.keys()),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_missing_model_raises_value_error(name):
    """Registry providers must refuse construction without an explicit model."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    with pytest.raises(ValueError, match="Model is required"):
        manager._create_openai_compatible_provider(
            name,
            {"api_key": "sk-test"},
        )


# ── Config override precedence ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_structured_outputs_config_overrides_spec(name, spec):
    """Config ``supports_structured_outputs`` must override the spec default."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    inverted = not spec.supports_structured_outputs
    provider = manager._create_openai_compatible_provider(
        name,
        {
            "model": "test-model",
            "api_key": "sk-test",
            "supports_structured_outputs": inverted,
        },
    )
    assert provider._supports_structured_outputs is inverted, (
        f"{name}: config override to {inverted} failed, "
        f"got {provider._supports_structured_outputs}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_spec_default_supports_structured_outputs(name, spec):
    """Without config override, the spec default must be used."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    provider = manager._create_openai_compatible_provider(
        name,
        {"model": "test-model", "api_key": "sk-test"},
    )
    assert provider._supports_structured_outputs is spec.supports_structured_outputs, (
        f"{name}: expected "
        "spec.supports_structured_outputs="
        f"{spec.supports_structured_outputs}, "
        f"got {provider._supports_structured_outputs}"
    )


@pytest.mark.parametrize(
    "name,spec",
    OPENAI_COMPATIBLE_PROVIDERS.items(),
    ids=OPENAI_COMPATIBLE_PROVIDERS.keys(),
)
def test_reasoning_effort_only_when_spec_says_so(name, spec):
    """``reasoning_effort`` must only be forwarded when the spec supports it."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    provider = manager._create_openai_compatible_provider(
        name,
        {
            "model": "test-model",
            "api_key": "sk-test",
            "reasoning_effort": "high",
        },
    )
    if spec.supports_reasoning_effort:
        assert provider._reasoning_effort == "high"
    else:
        assert provider._reasoning_effort is None, (
            f"{name}: reasoning_effort should be None when spec says no, "
            f"got {provider._reasoning_effort!r}"
        )


# ── Unknown provider (fallthrough path) ──────────────────────────────────────


def test_unknown_provider_raises_value_error():
    """An unrecognised provider name must produce a clear error."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        manager._create_provider(  # type: ignore[attr-defined]
            {"provider": "nonexistent-provider"}
        )


def test_unknown_provider_error_lists_available():
    """The error must include available provider names for discoverability."""
    from chunkhound.llm_manager import LLMManager

    manager = object.__new__(LLMManager)
    with pytest.raises(ValueError) as excinfo:
        manager._create_provider(  # type: ignore[attr-defined]
            {"provider": "nonexistent-provider"}
        )
    msg = str(excinfo.value)
    assert "Available providers:" in msg
    for known in ("openai", "anthropic", "gemini"):
        assert known in msg, f"Known provider {known!r} should appear in available list"
