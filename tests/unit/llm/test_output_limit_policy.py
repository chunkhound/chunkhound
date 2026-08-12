"""Behavior contracts for provider-owned synthesis output-limit policy."""

from dataclasses import FrozenInstanceError

import pytest

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    OutputLimitCapability,
    OutputLimitDecisionKind,
    OutputLimitMetadata,
    OutputLimitPolicy,
)


@pytest.mark.parametrize(
    ("capability", "expected_kind", "expected_tokens"),
    [
        (OutputLimitCapability.SUPPORTED, OutputLimitDecisionKind.OMIT, None),
        (OutputLimitCapability.REQUIRED, OutputLimitDecisionKind.FALLBACK, 71_111),
        (OutputLimitCapability.UNKNOWN, OutputLimitDecisionKind.FALLBACK, 71_111),
    ],
)
def test_provider_managed_policy_is_conservative_without_declaration(
    capability: OutputLimitCapability,
    expected_kind: OutputLimitDecisionKind,
    expected_tokens: int | None,
) -> None:
    policy = OutputLimitPolicy(
        output_limits_enabled=False,
        fallback_tokens=71_111,
        metadata=OutputLimitMetadata(omission=capability),
    )

    decision = policy.resolve(PROVIDER_MANAGED_OUTPUT)

    assert decision.kind is expected_kind
    assert decision.max_tokens == expected_tokens
    assert decision.source is None


@pytest.mark.parametrize(
    "capability", [OutputLimitCapability.REQUIRED, OutputLimitCapability.UNKNOWN]
)
def test_authoritative_declaration_precedes_fallback(
    capability: OutputLimitCapability,
) -> None:
    policy = OutputLimitPolicy(
        output_limits_enabled=False,
        fallback_tokens=71_111,
        metadata=OutputLimitMetadata(
            omission=capability,
            declared_max_tokens=98_765,
            declared_max_source="https://provider.example/docs/output-limits",
        ),
    )

    decision = policy.resolve(PROVIDER_MANAGED_OUTPUT)

    assert decision.kind is OutputLimitDecisionKind.DECLARATION
    assert decision.max_tokens == 98_765
    assert decision.source == "https://provider.example/docs/output-limits"


@pytest.mark.parametrize(
    ("value", "source"),
    [
        (None, None),
        (True, "https://provider.example/docs"),
        (0, "https://provider.example/docs"),
        (-1, "https://provider.example/docs"),
        (1.5, "https://provider.example/docs"),
        ("100000", "https://provider.example/docs"),
        (100_000, None),
        (100_000, ""),
        (100_000, "   "),
    ],
)
def test_malformed_or_unproven_declarations_use_fallback(
    value: object, source: object
) -> None:
    metadata = OutputLimitMetadata(
        omission=OutputLimitCapability.UNKNOWN,
        declared_max_tokens=value,
        declared_max_source=source,
    )
    policy = OutputLimitPolicy(
        output_limits_enabled=False,
        fallback_tokens=71_111,
        metadata=metadata,
    )

    decision = policy.resolve(PROVIDER_MANAGED_OUTPUT)

    assert metadata.declared_max_tokens is value
    assert metadata.declared_max_source is source
    assert decision.kind is OutputLimitDecisionKind.FALLBACK
    assert decision.max_tokens == 71_111
    assert decision.source is None


def test_supported_omission_wins_even_when_declaration_exists() -> None:
    policy = OutputLimitPolicy(
        output_limits_enabled=False,
        fallback_tokens=71_111,
        metadata=OutputLimitMetadata(
            omission=OutputLimitCapability.SUPPORTED,
            declared_max_tokens=98_765,
            declared_max_source="https://provider.example/docs/output-limits",
        ),
    )

    decision = policy.resolve(PROVIDER_MANAGED_OUTPUT)

    assert decision.kind is OutputLimitDecisionKind.OMIT
    assert decision.max_tokens is None


def test_explicit_allowance_is_preserved_in_every_mode() -> None:
    metadata = OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED)

    for enabled in (False, True):
        decision = OutputLimitPolicy(
            output_limits_enabled=enabled,
            fallback_tokens=71_111,
            metadata=metadata,
        ).resolve(12_345)

        assert decision.kind is OutputLimitDecisionKind.EXPLICIT
        assert decision.max_tokens == 12_345


@pytest.mark.parametrize("invalid", [None, False, True, 0, -1, 1.5, "4096"])
def test_invalid_explicit_allowances_are_rejected(invalid: object) -> None:
    policy = OutputLimitPolicy(
        output_limits_enabled=False,
        fallback_tokens=71_111,
        metadata=OutputLimitMetadata(),
    )

    with pytest.raises(ValueError, match="positive integer"):
        policy.resolve(invalid)  # type: ignore[arg-type]


def test_policy_and_metadata_are_immutable() -> None:
    metadata = OutputLimitMetadata()
    policy = OutputLimitPolicy(False, 71_111, metadata)

    with pytest.raises(FrozenInstanceError):
        policy.fallback_tokens = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        metadata.declared_max_tokens = 1  # type: ignore[misc]
