"""OpenAI-compatible LLM provider base class.

Provides common functionality for providers that use OpenAI-compatible APIs
(Chat Completions API with JSON Schema structured outputs).

Directly instantiable via constructor kwargs (provider_name, default_base_url, etc.).
Subclass overrides of _get_provider_name() / _get_default_base_url() are optional
(both return sensible defaults).
"""

import asyncio
import copy
import json
import time
from collections.abc import Mapping
from typing import Any

import httpx
from loguru import logger

from chunkhound.core import analytics as ch_analytics
from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.core.utils.openai_utils import is_official_openai_endpoint
from chunkhound.core.utils.structured_reasoning_diagnostics import (
    record_empty_failure,
    structured_reasoning_failure_key,
)
from chunkhound.core.utils.token_utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import (
    LLMProvider,
    LLMResponse,
    OutputLimitCapability,
    OutputLimitIntent,
    OutputLimitMetadata,
)
from chunkhound.providers.llm.capability_cache import (
    ACCEPTED,
    REJECTED,
    UNKNOWN,
    CapabilityState,
    LLMCapabilityStore,
    deadline_for,
)
from chunkhound.utils.json_extraction import (
    build_schema_system_instruction,
    parse_and_validate_structured_json,
)

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available - install with: uv pip install openai")


# Process-wide latch so a provider/model warns about rejected-capability empty
# content once per process rather than once per provider instance.
_EMPTY_STRUCTURED_WARNING_KEY: set[str] = set()
# Report only empty failures observed within this window so a single historical
# failure cannot keep the health signal alive forever.
_EMPTY_STRUCTURED_HEALTH_WINDOW_SECONDS = 60 * 60


class OpenAICompatibleProvider(LLMProvider):
    """Base class for OpenAI-compatible LLM providers.

    Handles Chat Completions API with structured outputs.
    Subclasses can override base URL and provider name.

    Set ``_supports_structured_outputs=False`` (class or instance) for
    providers whose APIs do not support OpenAI's native ``response_format``
    with ``type: "json_schema"`` (e.g. DeepSeek). The fallback injects the
    JSON schema into the system prompt instead.
    """

    _supports_structured_outputs: bool = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
        ssl_verify: bool = True,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
        supports_structured_outputs: bool | None = None,
        *,
        provider_name: str | None = None,
        default_base_url: str | None = None,
        max_tokens_param_name: str = "max_completion_tokens",
        reasoning_effort: str | None = None,
        synthesis_concurrency: int = 3,
        output_limit_omission: OutputLimitCapability = OutputLimitCapability.UNKNOWN,
        structured_reasoning_disable_extra_body: Mapping[str, Any] | None = None,
        capability_store: LLMCapabilityStore | None = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            api_key: API key (defaults to environment variable)
            model: Model name
            base_url: Base URL (defaults to subclass implementation)
            ssl_verify: Verify TLS certificates for HTTP requests. When False, disables
                TLS verification for the resolved base URL. Ignored when no base URL is
                set (explicit or provider-resolved). Only disable for self-signed /
                local endpoints.
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            supports_structured_outputs: Override class-level flag.
                When explicitly set, takes precedence over the class default.
            provider_name: Provider identifier. When set, ``name`` returns this
                instead of calling ``_get_provider_name()``.
            default_base_url: Fallback base URL when ``base_url`` is None.
                Applied before ``_get_default_base_url()``.
            max_tokens_param_name: API parameter name for output token limit.
                Use ``"max_tokens"`` for APIs that don't support
                ``"max_completion_tokens"`` (e.g. DeepSeek).
            reasoning_effort: Reasoning effort for compatible providers
                (e.g. Grok). Omitted from API calls when None.
            synthesis_concurrency: Recommended parallel synthesis operations count.
            output_limit_omission: Authoritative omission capability for the
                configured endpoint. Generic and custom endpoints default to unknown.
            structured_reasoning_disable_extra_body: Extra request body used only for
                structured calls when the endpoint accepts it.
            capability_store: Persistent structured-request capability store.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI not available - install with: uv pip install openai"
            )

        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries
        if supports_structured_outputs is not None:
            self._supports_structured_outputs = supports_structured_outputs
        self._ssl_verify = ssl_verify
        self._provider_name = provider_name
        self._max_tokens_param_name = max_tokens_param_name
        self._reasoning_effort = reasoning_effort
        self._synthesis_concurrency = synthesis_concurrency
        self._output_limit_metadata = OutputLimitMetadata(
            omission=output_limit_omission
        )
        self._structured_reasoning_disable_extra_body = (
            structured_reasoning_disable_extra_body
        )
        self._capability_store = capability_store or LLMCapabilityStore()
        self._structured_reasoning_capability: CapabilityState = UNKNOWN
        self._structured_reasoning_capability_deadline: float | None = None
        if self._structured_reasoning_disable_extra_body is not None:
            (
                self._structured_reasoning_capability,
                self._structured_reasoning_capability_deadline,
            ) = self._capability_store.get_with_expiry(self.name, self._model)
        self._structured_reasoning_empty_failure_times: list[float] = []
        self._structured_reasoning_last_empty_ts: float | None = None
        self._structured_reasoning_capability_lock: asyncio.Lock | None = None
        self._structured_reasoning_capability_loop: asyncio.AbstractEventLoop | None = (
            None
        )

        # Use provided base_url, or default_base_url, or subclass override
        effective_base_url = (
            base_url or default_base_url or self._get_default_base_url()
        )

        # Initialize OpenAI-compatible client
        api_key_value = api_key
        if not is_official_openai_endpoint(effective_base_url) and not api_key_value:
            # The OpenAI SDK still expects a string even for local/custom backends
            api_key_value = "not-required"

        client_kwargs: dict[str, Any] = {
            "api_key": api_key_value,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
            if not ssl_verify:
                client_kwargs["http_client"] = httpx.AsyncClient(
                    timeout=httpx.Timeout(timeout=timeout),
                    verify=False,
                )
        self._client = AsyncOpenAI(**client_kwargs)

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def output_limit_metadata(self) -> OutputLimitMetadata:
        """Return omission capability for the configured compatible endpoint."""
        return self._output_limit_metadata

    @property
    def base_url(self) -> str | None:
        """Resolved API base URL for this provider, or None if unset."""
        url = self._client.base_url
        return str(url) if url else None

    def _get_default_base_url(self) -> str | None:
        """Get the default base URL for this provider.

        Subclasses can override this to provide their API endpoint.
        Returns None to allow AsyncOpenAI to fall back to environment variables.
        """
        return None

    def _get_provider_name(self) -> str:
        """Get the provider name for this implementation.

        Subclasses can override this.  The base returns a generic identifier;
        use the ``provider_name`` constructor parameter to set a specific name.
        """
        return "openai-compatible"

    def _get_max_completion_tokens_param_name(self) -> str:
        """Get the output-token parameter name for chat completions requests."""
        return self._max_tokens_param_name

    def _parse_structured_response(
        self, content: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse and validate a structured response against the JSON schema."""
        return parse_and_validate_structured_json(content, json_schema)

    def _build_chat_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        max_completion_tokens: int | None,
        timeout: int,
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build chat-completions kwargs for subclasses to extend safely."""
        self._expire_capability_if_due()
        max_tokens_param = self._get_max_completion_tokens_param_name()
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "timeout": timeout,
        }
        if max_completion_tokens is not None:
            kwargs[max_tokens_param] = max_completion_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
            if (
                self._structured_reasoning_disable_extra_body is not None
                and self._structured_reasoning_capability != REJECTED
            ):
                kwargs["extra_body"] = copy.deepcopy(
                    self._structured_reasoning_disable_extra_body
                )
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        return kwargs

    @property
    def name(self) -> str:
        """Provider name."""
        if self._provider_name is not None:
            return self._provider_name
        return self._get_provider_name()

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    async def _create_chat_completion(self, **kwargs: Any) -> Any:
        """Single instrumented chokepoint for chat.completions.create(),
        used by every caller. Retries are SDK-internal (max_retries= on the
        client): one call here can trigger several real HTTP attempts that
        never surface individually, so record_provider_call counts the
        outcome after the SDK's retry budget is exhausted, not
        per-HTTP-attempt. A transient failure the SDK silently retries past
        never appears in analytics. This differs from the embedding/rerank
        providers, which wrap their own manual retry loop and record one
        call per real attempt -- an accepted fidelity tradeoff for the LLM
        path, not a bug."""
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            ch_analytics.record_provider_call(
                "llm", self.name, self._model, False, error_type=type(exc).__name__
            )
            raise
        ch_analytics.record_provider_call(
            "llm",
            self.name,
            self._model,
            True,
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )
        return response

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)
        """
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Resolve provider-managed intent before constructing SDK wire kwargs.
        resolved_max_tokens = self.resolve_synthesis_output_limit(
            max_completion_tokens
        ).max_tokens

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            response = await self._create_chat_completion(
                **self._build_chat_completion_kwargs(
                    messages,
                    resolved_max_tokens,
                    request_timeout,
                )
            )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            # Extract response content
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason

            # Check for truncation FIRST (fix C2) so token-limit diagnostics
            # are not masked by a generic empty-content error.
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM response truncated - token limit exceeded{usage_info}. "
                    f"For reasoning models, this indicates the query requires "
                    f"extensive reasoning that exhausted the output budget. "
                    f"Try breaking your query into smaller, more focused questions."
                )

            # Validate content (reached only if answer is complete)
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} returned empty content "
                    f"(finish_reason={finish_reason}, tokens={tokens})"
                )
                raise RuntimeError(
                    f"LLM returned empty response (finish_reason={finish_reason}). "
                    "This may indicate a content filter, API error, or model refusal."
                )

            # Warn on unexpected finish_reason
            if finish_reason not in ("stop",):
                logger.warning(
                    f"Unexpected finish_reason: {finish_reason} "
                    f"(content_length={len(content)})"
                )
                if finish_reason == "content_filter":
                    raise RuntimeError(
                        "LLM response blocked by content filter. "
                        "Try rephrasing your query or adjusting the prompt."
                    )

            return LLMResponse(
                content=content,
                tokens_used=tokens,
                model=self._model,
                finish_reason=finish_reason,
            )
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"{self.name} completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def _create_structured_completion(self, kwargs: dict[str, Any]) -> Any:
        """Create a structured completion with sticky payload negotiation."""
        self._expire_capability_if_due()
        if self._structured_reasoning_disable_extra_body is None:
            return await self._create_chat_completion(**kwargs)

        state: CapabilityState = self._structured_reasoning_capability
        kwargs = self._set_structured_payload(kwargs, state != REJECTED)
        if state == UNKNOWN:
            async with self._capability_lock():
                self._expire_capability_if_due()
                state = self._structured_reasoning_capability
                if state == UNKNOWN:
                    return await self._probe_structured_reasoning_payload(kwargs)
                kwargs = self._set_structured_payload(kwargs, state == ACCEPTED)
            return await self._send_flagged(kwargs)

        return await self._send_flagged(kwargs)

    async def _send_flagged(self, kwargs: dict[str, Any]) -> Any:
        try:
            return await self._create_chat_completion(**kwargs)
        except Exception as error:
            if "extra_body" not in kwargs or not self._is_payload_rejection(error):
                raise
            return await self._replay_without_payload(kwargs)

    async def _probe_structured_reasoning_payload(self, kwargs: dict[str, Any]) -> Any:
        try:
            response = await self._create_chat_completion(**kwargs)
        except Exception as error:
            if not self._is_payload_rejection(error):
                raise
            return await self._replay_without_payload(kwargs)

        self._set_structured_reasoning_capability(ACCEPTED)
        return response

    async def _replay_without_payload(self, kwargs: dict[str, Any]) -> Any:
        """Retry an unflagged request and record the capability rejection."""
        response = await self._create_chat_completion(
            **self._set_structured_payload(kwargs, False)
        )
        self._set_structured_reasoning_capability(REJECTED)
        return response

    def _set_structured_payload(
        self, kwargs: dict[str, Any], enabled: bool
    ) -> dict[str, Any]:
        updated = dict(kwargs)
        if enabled:
            updated["extra_body"] = copy.deepcopy(
                self._structured_reasoning_disable_extra_body or {}
            )
        else:
            updated.pop("extra_body", None)
        return updated

    @staticmethod
    def _is_payload_rejection(error: Exception) -> bool:
        response = getattr(error, "response", None)
        return getattr(response, "status_code", None) in {400, 422}

    def _expire_capability_if_due(self) -> None:
        """Reset expired in-memory capability decisions without writing."""
        if (
            self._structured_reasoning_disable_extra_body is None
            or self._structured_reasoning_capability == UNKNOWN
        ):
            return
        deadline = self._structured_reasoning_capability_deadline
        if deadline is None or time.time() < deadline:
            return
        self._structured_reasoning_capability = UNKNOWN
        self._structured_reasoning_capability_deadline = None

    def _set_structured_reasoning_capability(self, state: CapabilityState) -> None:
        if self._structured_reasoning_capability == state:
            return
        self._structured_reasoning_capability = state
        try:
            deadline = self._capability_store.set(self.name, self._model, state)
            if deadline is None:
                deadline = deadline_for(state, time.time())
        except OSError as error:
            logger.warning(
                f"Failed to persist {self.name} structured reasoning "
                f"capability: {error}"
            )
            deadline = deadline_for(state, time.time())
        self._structured_reasoning_capability_deadline = deadline

    def _capability_lock(self) -> asyncio.Lock:
        """Return a capability lock bound to the current event loop."""
        loop = asyncio.get_running_loop()
        if (
            self._structured_reasoning_capability_lock is None
            or self._structured_reasoning_capability_loop is not loop
        ):
            self._structured_reasoning_capability_lock = asyncio.Lock()
            self._structured_reasoning_capability_loop = loop
        return self._structured_reasoning_capability_lock

    def _record_empty_structured_content(self) -> None:
        """Count an empty structured response only under a rejected capability."""
        if self._structured_reasoning_capability != REJECTED:
            return
        now = time.time()
        self._structured_reasoning_empty_failure_times.append(now)
        self._structured_reasoning_last_empty_ts = now
        self._prune_empty_failure_times(now)
        record_empty_failure(self)
        warning_key = structured_reasoning_failure_key(self)
        if warning_key not in _EMPTY_STRUCTURED_WARNING_KEY:
            _EMPTY_STRUCTURED_WARNING_KEY.add(warning_key)
            logger.warning(
                f"{self.name}:{self._model} returned empty structured content "
                "while reasoning-disable is known-rejected; results may be incomplete."
            )

    def _prune_empty_failure_times(self, now: float) -> None:
        cutoff = now - _EMPTY_STRUCTURED_HEALTH_WINDOW_SECONDS
        self._structured_reasoning_empty_failure_times = [
            observed
            for observed in self._structured_reasoning_empty_failure_times
            if observed >= cutoff
        ]

    def get_structured_reasoning_health(self) -> dict[str, Any] | None:
        """Read-only health for recent empty structured failures under rejection."""
        cutoff = time.time() - _EMPTY_STRUCTURED_HEALTH_WINDOW_SECONDS
        empty_failures = sum(
            1
            for observed in self._structured_reasoning_empty_failure_times
            if observed >= cutoff
        )
        if empty_failures == 0:
            return None
        return {
            "provider": self.name,
            "model": self._model,
            "capability": self._structured_reasoning_capability,
            "empty_failures": empty_failures,
            "last_empty_ts": self._structured_reasoning_last_empty_ts,
        }

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Uses native ``response_format`` with ``type: "json_schema"`` when the
        provider supports it (``_supports_structured_outputs = True``).
        Otherwise injects the schema into the system prompt and parses the
        free-form text response — ``extract_json_from_response`` handles
        markdown code blocks and raw JSON alike.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object conforming to schema
        """
        resolved_max_tokens = self.resolve_synthesis_output_limit(
            max_completion_tokens
        ).max_tokens
        request_timeout = timeout if timeout is not None else self._timeout

        try:
            if self._supports_structured_outputs:
                # Native structured outputs path
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})

                response = await self._create_structured_completion(
                    self._build_chat_completion_kwargs(
                        messages,
                        resolved_max_tokens,
                        request_timeout,
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_response",
                                "strict": True,
                                "schema": json_schema,
                            },
                        },
                    )
                )
            else:
                logger.debug(
                    f"{self.name}: using prompt-based structured output fallback "
                    "(provider does not support native json_schema response_format)"
                )
                # Prompt-based fallback: inject schema into system message
                schema_instruction = build_schema_system_instruction(json_schema)
                if system:
                    effective_system = f"{system}\n\n{schema_instruction}"
                else:
                    effective_system = schema_instruction

                messages = [
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": prompt},
                ]

                response = await self._create_structured_completion(
                    self._build_chat_completion_kwargs(
                        messages,
                        resolved_max_tokens,
                        request_timeout,
                        response_format={"type": "json_object"},
                    )
                )

            self._requests_made += 1
            if response.usage:
                self._prompt_tokens += response.usage.prompt_tokens
                self._completion_tokens += response.usage.completion_tokens
                self._tokens_used += response.usage.total_tokens

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Check for truncation FIRST (more actionable — includes token counts)
            # Must precede the empty-content check: when finish_reason="length" and
            # content is empty/whitespace, the token-count info points at the fix.
            if finish_reason == "length":
                usage_info = ""
                if response.usage:
                    usage_info = (
                        f" (prompt={response.usage.prompt_tokens:,}, "
                        f"completion={response.usage.completion_tokens:,})"
                    )

                raise RuntimeError(
                    f"LLM structured completion truncated - token limit "
                    f"exceeded{usage_info}. "
                    "This indicates insufficient max_completion_tokens for the "
                    "structured output. "
                    "Consider increasing the token limit or reducing input context."
                )

            # Validate content (only reached if finish_reason != "length")
            if content is None or not content.strip():
                logger.error(
                    f"{self.name} structured completion returned empty content "
                    f"(finish_reason={finish_reason})"
                )
                self._record_empty_structured_content()
                diagnostic = (
                    "LLM structured completion returned empty response "
                    f"(finish_reason={finish_reason}, provider={self.name}, "
                    f"model={self._model}"
                )
                if (
                    self._structured_reasoning_disable_extra_body is not None
                    and self._structured_reasoning_capability == REJECTED
                ):
                    raise RuntimeError(
                        f"{diagnostic}, capability=rejected). "
                        "This provider/model may be unable to disable reasoning for "
                        "structured calls, so structured output cannot be relied on. "
                        "Configure a different model for structured calls "
                        "(`llm.utility_model` for utility stages, or the synthesis "
                        "model for synthesis stages)."
                    )
                raise RuntimeError(
                    f"{diagnostic}). This may indicate a content filter, API error, "
                    "or model refusal."
                )

            # Parse JSON
            try:
                return self._parse_structured_response(content, json_schema)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse structured output as JSON: {e}")
                raise RuntimeError(f"Invalid JSON in structured output: {e}") from e

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"{self.name} structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        return estimate_tokens_llm(text)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            Provider-specific concurrency (default 3, overridable via
            ``_synthesis_concurrency`` instance attribute set by the factory).
        """
        return self._synthesis_concurrency
