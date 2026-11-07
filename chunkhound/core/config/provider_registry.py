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
  - the preset itself (model pin, copy) in
    ``site/src/components/configurator/providers.ts``
  - its SVG in ``site/src/components/configurator/icons.ts``
  - then regenerate the derived site catalog and run the drift guard:
    ``uv run python scripts/generate_site_provider_catalog.py``
    ``uv run pytest tests/site/test_configurator_catalog_drift.py``
  - ``site/src/pages/docs/configuration.md``
  - ``CHANGELOG.md``
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
        structured_reasoning_disable_extra_body: Extra request body sent only
            for structured calls when the canonical endpoint accepts it
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
    structured_reasoning_disable_extra_body: Mapping[str, Any] | None = None


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
        auth_url="https://platform.deepseek.com/api_keys",
    ),
    "grok": OpenAICompatibleSpec(
        name="grok",
        default_base_url="https://api.x.ai/v1",
        supports_reasoning_effort=True,
        synthesis_concurrency=5,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        docs_url="https://docs.x.ai/docs/models",
        auth_url="https://console.x.ai/team/default/api-keys",
    ),
    "openrouter": OpenAICompatibleSpec(
        name="openrouter",
        default_base_url="https://openrouter.ai/api/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        structured_reasoning_disable_extra_body={"reasoning": {"enabled": False}},
        docs_url="https://openrouter.ai/docs",
        auth_url="https://openrouter.ai/keys",
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
        # Gateway docs list max_tokens as optional, so omission is authoritative
        # like OpenRouter. Omitting avoids a single fallback cap that smaller
        # upstream models would reject.
        output_limit_omission=OutputLimitCapability.SUPPORTED,
        # Gateway extension: reasoning.enabled=false disables reasoning for
        # structured calls; probed and cached per model like OpenRouter.
        structured_reasoning_disable_extra_body={"reasoning": {"enabled": False}},
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
    "api-route": OpenAICompatibleSpec(
        name="api-route",
        default_base_url="https://global.api-route.com/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://www.api-route.com/en/docs/quickstart",
        auth_url="https://www.api-route.com/api-keys",
    ),
    "opper": OpenAICompatibleSpec(
        name="opper",
        default_base_url="https://api.opper.ai/v3/compat",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://docs.opper.ai",
        auth_url="https://platform.opper.ai",
    ),
    "obscurify-ai": OpenAICompatibleSpec(
        name="obscurify-ai",
        default_base_url="https://obscurify.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://obscurify.ai/api-docs",
        auth_url="https://obscurify.ai/account",
    ),
    "atlascloud": OpenAICompatibleSpec(
        name="atlascloud",
        default_base_url="https://api.atlascloud.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://www.atlascloud.ai/docs",
        auth_url="https://console.atlascloud.ai/api-keys",
    ),
    "runcrate": OpenAICompatibleSpec(
        name="runcrate",
        default_base_url="https://api.runcrate.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://www.runcrate.ai/inference/pricing",
        auth_url="https://www.runcrate.ai/dashboard/api-keys",
    ),
    "cheapaimodel": OpenAICompatibleSpec(
        name="cheapaimodel",
        default_base_url="https://api.cheapaimodel.ai/v1",
        supports_structured_outputs=False,
        max_tokens_param_name="max_tokens",
        synthesis_concurrency=10,
        docs_url="https://cheapaimodel.ai/docs",
        auth_url="https://console.cheapaimodel.ai/keys",
    ),
}
