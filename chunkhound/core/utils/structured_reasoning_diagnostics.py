"""Request-local structured reasoning failures for research caller diagnostics."""

from contextvars import ContextVar

# Mutable only within a deep_research run; child asyncio tasks inherit this binding.
# Each run creates its own dict, so unrelated requests sharing a provider stay isolated.
# Keys are provider/model identities rather than object identities so attribution
# survives a provider instance being rebuilt between requests.
current_empty_failures: ContextVar[dict[str, int] | None] = ContextVar(
    "chunkhound_structured_reasoning_empty_failures", default=None
)


def structured_reasoning_failure_key(provider: object) -> str:
    """Return a stable ``provider:model`` identity for failure attribution."""
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name:
        name = getattr(provider, "_provider_name", None)
    if not isinstance(name, str) or not name:
        name = type(provider).__name__

    model = getattr(provider, "model", None)
    if not isinstance(model, str) or not model:
        model = getattr(provider, "_model", None)
    if not isinstance(model, str) or not model:
        model = type(provider).__name__

    return f"{name}:{model}"


def record_empty_failure(provider: object) -> None:
    """Attribute a rejected-capability empty response to the active run, if any."""
    failures = current_empty_failures.get()
    if failures is not None:
        key = structured_reasoning_failure_key(provider)
        failures[key] = failures.get(key, 0) + 1
