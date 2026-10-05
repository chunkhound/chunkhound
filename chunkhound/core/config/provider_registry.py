"""Data-driven registry for OpenAI-compatible LLM providers.

Each entry describes what's stable about a provider's API protocol.
Model names are NEVER baked in — they come from user configuration.

Lives in ``core/config/`` (not ``providers/llm/``) as it defines
config-domain data (API spec entries) — both ``llm_config.py``
and ``llm_manager.py`` import here.

To add a new OpenAI-compatible provider you must also touch:
  - ``LLMProviderLiteral`` in ``llm_config.py``
  - ``CLI_PROVIDER_CHOICES`` in ``llm_config.py``
  - test ``SPECS`` in ``test_openai_compatible_provider.py``
  - ``REASONING_EFFORT_PROVIDERS`` in ``llm_config.py`` (if applicable)
  - site configurator presets in ``site/src/components/configurator/``
    (``providers.ts``, ``requirements.ts``, ``constants.ts``, ``icons.ts``)
  - the configurator tests and highlight goldens in ``tests/site/``
  - ``site/src/pages/docs/configuration.md``
  - ``CHANGELOG.md``
"""

from __future__ import annotations

from dataclasses import dataclass

from chunkhound.interfaces.llm_provider import OutputLimitCapability


@dataclass(frozen=True)
class OpenAICompatibleSpec:
    """Stable API properties of an OpenAI-compatible provider.

    Attributes:
        name: Provider identifier string (matches config ``provider`` value)
        default_base_url: API endpoint base URL
        supports_structured_outputs: Whether native ``json_schema``
            response_format is supported
        supports_reasoning_effort: Whether ``reasoning_effort`` API parameter
            is accepted
        max_tokens_param_name: API parameter name for output token limit
            (``"max_completion_tokens"`` for newer APIs, ``"max_tokens"`` for older)
        synthesis_concurrency: Recommended parallel synthesis operations count
        output_limit_omission: Whether the canonical endpoint authoritatively
            supports omitting its output-token cap
        docs_url: External API documentation URL
        auth_url: Authentication portal URL
    """

    name: str
    default_base_url: str
    supports_structured_outputs: bool = True
    supports_reasoning_effort: bool = False
    max_tokens_param_name: str = "max_completion_tokens"
    synthesis_concurrency: int = 3
    output_limit_omission: OutputLimitCapability = OutputLimitCapability.UNKNOWN
    docs_url: str = ""
    auth_url: str = ""


# ── Provider specs ─────────────────────────────────────────────────────────
# Append one entry here, then update LLMProviderLiteral, CLI_PROVIDER_CHOICES,
# test SPECS, and the site configurator/docs/changelog touch points — see the
# module docstring above.
OPENAI_COMPATIBLE_PROVIDERS: dict[str, OpenAICompatibleSpec] = {
    "deepseek": OpenAICompatibleSpec(
        name="deepseek",
        default_base_url="https://api.deepseek.com",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://platform.deepseek.com/api-docs",
        auth_url="https://platform.deepseek.com",
    ),
    "grok": OpenAICompatibleSpec(
        name="grok",
        default_base_url="https://api.x.ai/v1",
        supports_reasoning_effort=True,
        synthesis_concurrency=5,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://docs.x.ai/docs/models",
        auth_url="https://console.x.ai",
    ),
    "openrouter": OpenAICompatibleSpec(
        name="openrouter",
        default_base_url="https://openrouter.ai/api/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://openrouter.ai/docs",
        auth_url="https://openrouter.ai",
    ),
    "orcarouter": OpenAICompatibleSpec(
        name="orcarouter",
        default_base_url="https://api.orcarouter.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://docs.orcarouter.ai",
        auth_url="https://www.orcarouter.ai",
    ),
    "vercel": OpenAICompatibleSpec(
        name="vercel",
        default_base_url="https://ai-gateway.vercel.sh/v1",
        # Router: structured-output and reasoning support is per upstream
        # model, so stay on the prompt-injection fallback like OpenRouter.
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        # Omission is not yet documented as authoritative for the gateway;
        # UNKNOWN keeps the conservative fallback cap (see output-limit docs).
        output_limit_omission=OutputLimitCapability.UNKNOWN,
        docs_url="https://vercel.com/docs/ai-gateway",
        auth_url="https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai-gateway%2Fapi-keys&title=AI+Gateway+API+Keys",
    ),
    "requesty": OpenAICompatibleSpec(
        name="requesty",
        default_base_url="https://router.requesty.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://docs.requesty.ai",
        auth_url="https://app.requesty.ai/api-keys",
    ),
}
