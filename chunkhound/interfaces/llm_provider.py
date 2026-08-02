"""LLM Provider Interface for ChunkHound deep research."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _normalize_schema_for_structured_outputs(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively normalize a JSON schema for structured outputs.

    Forces 'additionalProperties: false' on all object types in the schema,
    including nested objects and $defs. This is required by Anthropic's
    structured outputs API - any other value will cause a 400 error.

    Args:
        schema: JSON schema dictionary to normalize

    Returns:
        Normalized schema with additionalProperties: false on all objects
    """
    if not isinstance(schema, dict):
        return schema  # type: ignore[unreachable]  # Defensive guard for runtime

    result = schema.copy()

    # Force additionalProperties: false on all object types
    # Anthropic's structured outputs API does not support any other value
    if result.get("type") == "object":
        existing = result.get("additionalProperties")
        if existing is not None and existing is not False:
            logger.warning(
                "additionalProperties=%r not supported by structured outputs API, "
                "forcing to false",
                existing,
            )
        result["additionalProperties"] = False

    # Recursively process $defs (Pydantic's way of defining nested models)
    if "$defs" in result:
        result["$defs"] = {
            name: _normalize_schema_for_structured_outputs(def_schema)
            for name, def_schema in result["$defs"].items()
        }

    # Recursively process properties
    if "properties" in result:
        result["properties"] = {
            name: _normalize_schema_for_structured_outputs(prop_schema)
            for name, prop_schema in result["properties"].items()
        }

    # Process array items
    if "items" in result:
        result["items"] = _normalize_schema_for_structured_outputs(result["items"])

    # Process prefixItems (Pydantic tuples generate this)
    if "prefixItems" in result:
        result["prefixItems"] = [
            _normalize_schema_for_structured_outputs(item_schema)
            for item_schema in result["prefixItems"]
        ]

    # Process anyOf/oneOf/allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in result:
            result[key] = [
                _normalize_schema_for_structured_outputs(sub_schema)
                for sub_schema in result[key]
            ]

    return result


@dataclass
class LLMResponse:
    """Response from LLM completion."""

    content: str
    tokens_used: int
    model: str
    finish_reason: str | None = None


class OutputLimitCapability(str, Enum):
    """Whether a provider authoritatively supports omitting its output cap."""

    SUPPORTED = "supported"
    REQUIRED = "required"
    UNKNOWN = "unknown"


class OutputLimitIntent(str, Enum):
    """Named non-numeric output-limit requests."""

    PROVIDER_MANAGED = "provider_managed"


PROVIDER_MANAGED_OUTPUT = OutputLimitIntent.PROVIDER_MANAGED


class OutputLimitDecisionKind(str, Enum):
    """Observable source of the output limit selected for one request."""

    OMIT = "omit"
    DECLARATION = "declaration"
    FALLBACK = "fallback"
    EXPLICIT = "explicit"


@dataclass(frozen=True)
class OutputLimitMetadata:
    """Provider-owned capability and raw declared-limit provenance."""

    omission: OutputLimitCapability = OutputLimitCapability.UNKNOWN
    declared_max_tokens: object = None
    declared_max_source: object = None

    def eligible_declaration(self) -> tuple[int, str] | None:
        """Return a declaration only when both value and provenance are valid."""
        if type(self.declared_max_tokens) is not int:
            return None
        if self.declared_max_tokens <= 0:
            return None
        if not isinstance(self.declared_max_source, str):
            return None
        if not self.declared_max_source.strip():
            return None
        return self.declared_max_tokens, self.declared_max_source


@dataclass(frozen=True)
class OutputLimitDecision:
    """Resolved wire-cap decision with provenance when one exists."""

    kind: OutputLimitDecisionKind
    max_tokens: int | None
    source: str | None = None


@dataclass(frozen=True)
class OutputLimitPolicy:
    """Resolve explicit or provider-managed synthesis output allowances."""

    output_limits_enabled: bool
    fallback_tokens: int
    metadata: OutputLimitMetadata

    def __post_init__(self) -> None:
        if type(self.output_limits_enabled) is not bool:
            raise ValueError("output_limits_enabled must be a boolean")
        if type(self.fallback_tokens) is not int or self.fallback_tokens <= 0:
            raise ValueError("fallback_tokens must be a positive integer")

    def resolve(self, requested: int | OutputLimitIntent) -> OutputLimitDecision:
        """Resolve a request without guessing provider/model output limits."""
        if requested is PROVIDER_MANAGED_OUTPUT:
            if self.output_limits_enabled:
                raise ValueError(
                    "provider-managed output was requested while legacy output "
                    "limits are enabled"
                )
            if self.metadata.omission is OutputLimitCapability.SUPPORTED:
                return OutputLimitDecision(OutputLimitDecisionKind.OMIT, None)
            if declaration := self.metadata.eligible_declaration():
                max_tokens, source = declaration
                return OutputLimitDecision(
                    OutputLimitDecisionKind.DECLARATION, max_tokens, source
                )
            return OutputLimitDecision(
                OutputLimitDecisionKind.FALLBACK, self.fallback_tokens
            )

        if type(requested) is not int or requested <= 0:
            raise ValueError("output allowance must be a positive integer")
        return OutputLimitDecision(OutputLimitDecisionKind.EXPLICIT, requested)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    def output_limit_metadata(self) -> OutputLimitMetadata:
        """Return conservative provider-owned output-limit capabilities."""
        return OutputLimitMetadata()

    @property
    def synthesis_output_limit_policy(self) -> OutputLimitPolicy:
        """Return the immutable policy attached to this synthesis instance."""
        policy = getattr(self, "_synthesis_output_limit_policy", None)
        if policy is None:
            return OutputLimitPolicy(True, 64_000, self.output_limit_metadata)
        return policy

    def configure_synthesis_output_limit_policy(
        self, *, output_limits_enabled: bool, fallback_tokens: int
    ) -> None:
        """Attach synthesis policy using this selected provider's metadata."""
        self._synthesis_output_limit_policy = OutputLimitPolicy(
            output_limits_enabled=output_limits_enabled,
            fallback_tokens=fallback_tokens,
            metadata=self.output_limit_metadata,
        )

    def resolve_synthesis_output_limit(
        self, requested: int | OutputLimitIntent
    ) -> OutputLimitDecision:
        """Resolve one synthesis output allowance through the attached policy."""
        return self.synthesis_output_limit_policy.resolve(requested)

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        ...

    @property
    @abstractmethod
    def timeout(self) -> int:
        """Request timeout in seconds."""
        ...

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: User prompt
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate
            timeout: Optional timeout in seconds for the request

        Returns:
            LLMResponse with content and metadata

        Raises:
            RuntimeError: If the API call fails for any reason.
        """
        ...

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON completion for the given schema.

        Implementations may use native constrained decoding or a fallback that
        injects the schema into the prompt and validates the parsed response.

        Args:
            prompt: User prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate
            timeout: Optional timeout in seconds for the request

        Returns:
            Parsed JSON object accepted by the provider's structured-output path

        Raises:
            NotImplementedError: If provider doesn't support structured outputs
            RuntimeError: If the API call fails for any reason.
        """
        raise NotImplementedError(
            f"{self.name} provider does not support structured outputs"
        )

    async def complete_structured_typed(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> T:
        """
        Generate a typed structured completion using a Pydantic model.

        This is the recommended way to use structured outputs - provides type
        safety, IDE autocomplete, automatic schema generation, and final
        Pydantic validation.

        Automatically normalizes the schema by adding 'additionalProperties: false'
        to all object types, as required by Anthropic's structured outputs API.

        Args:
            prompt: User prompt
            response_model: Pydantic model class defining the response schema
            system: Optional system message
            max_completion_tokens: Maximum completion tokens to generate

        Returns:
            Validated Pydantic model instance

        Raises:
            NotImplementedError: If provider doesn't support structured outputs
            ValidationError: If response doesn't match the schema
        """
        # Generate JSON schema from Pydantic model
        schema = response_model.model_json_schema()

        # Normalize schema: recursively add additionalProperties: false to all objects
        # This is required by Anthropic's structured outputs API for strict validation
        schema = _normalize_schema_for_structured_outputs(schema)

        # Call the dict-based method
        result = await self.complete_structured(
            prompt=prompt,
            json_schema=schema,
            system=system,
            max_completion_tokens=max_completion_tokens,
        )

        # Validate and return typed model
        return response_model.model_validate(result)

    @abstractmethod
    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """
        Generate completions for multiple prompts concurrently.

        Args:
            prompts: List of user prompts
            system: Optional system message (same for all)
            max_completion_tokens: Maximum completion tokens to generate per completion

        Returns:
            List of LLMResponse objects
        """
        ...

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        ...

    @abstractmethod
    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Usage stats dictionary
        """
        ...

    def get_synthesis_concurrency(self) -> int:
        """
        Get recommended concurrency for parallel synthesis operations.

        Returns:
            Number of concurrent synthesis tasks this provider can handle.
            Used for map-reduce synthesis to execute cluster summaries in
            parallel. Default implementations return provider-specific values
            based on rate limits.
        """
        return 3  # Conservative default
