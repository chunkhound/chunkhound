"""Unit tests for ResearchConfig validation logic.

Tests validation of ResearchConfig fields and cross-field constraints.
Current implementation uses Pydantic field validators (ge/le constraints).
Tests document expected validation behavior vs actual implementation gaps.
"""

import pytest
from pydantic import ValidationError

from chunkhound.core.config.research_config import ResearchConfig


class TestNegativeValueValidation:
    """Test validation of negative values that should be rejected."""

    def test_negative_multi_hop_time_limit(self):
        """Should raise ValidationError for negative time limit.

        Expected: ValueError/ValidationError with message about positive values.
        Current: Pydantic validates ge=1.0 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(multi_hop_time_limit=-1.0)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("multi_hop_time_limit",)
        assert "greater than or equal to 1" in str(error["msg"]).lower()

    def test_negative_target_tokens(self):
        """Should raise ValidationError for negative token budget.

        Expected: ValueError/ValidationError with message about positive values.
        Current: Pydantic validates ge=10_000 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(target_tokens=-1000)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("target_tokens",)
        assert "greater than or equal to" in str(error["msg"]).lower()

    def test_negative_exhaustive_time_limit(self):
        """Should raise ValidationError for negative exhaustive time limit.

        Expected: ValueError/ValidationError with message about positive values.
        Current: Pydantic validates ge=60.0 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(exhaustive_time_limit=-10.0)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("exhaustive_time_limit",)
        assert "greater than or equal to 60" in str(error["msg"]).lower()

    def test_negative_num_expanded_queries(self):
        """Should raise ValidationError for negative query count.

        Expected: ValueError/ValidationError with message about positive values.
        Current: Pydantic validates ge=1 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(num_expanded_queries=-5)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("num_expanded_queries",)
        assert "greater than or equal to 1" in str(error["msg"]).lower()


class TestZeroValueValidation:
    """Test validation of zero values for fields requiring positive values."""

    def test_zero_max_compression_iterations(self):
        """Should raise ValidationError for zero compression iterations.

        Expected: ValueError with message 'max_compression_iterations must be >= 1'.
        Current: Pydantic validates ge=1 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(max_compression_iterations=0)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("max_compression_iterations",)
        assert "greater than or equal to 1" in str(error["msg"]).lower()

    def test_zero_target_tokens(self):
        """Should raise ValidationError for zero target tokens.

        Expected: ValueError with message 'target_tokens must be >= 1000'.
        Current: Pydantic validates ge=10_000 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(target_tokens=0)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("target_tokens",)
        assert "greater than or equal to 10000" in str(error["msg"]).lower()

    def test_zero_multi_hop_time_limit(self):
        """Should raise ValidationError for zero time limit.

        Expected: ValueError with message 'multi_hop_time_limit must be positive'.
        Current: Pydantic validates ge=1.0 constraint, should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(multi_hop_time_limit=0.0)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("multi_hop_time_limit",)
        assert "greater than or equal to 1" in str(error["msg"]).lower()


class TestConflictingConstraints:
    """Test validation of conflicting field values (cross-field validation).

    NOTE: These tests document MISSING validation that should be added.
    Current implementation has NO cross-field validation between min_gaps/max_gaps.
    """

    @pytest.mark.xfail(
        reason=(
            "Cross-field validation not implemented - "
            "min_gaps > max_gaps check needed"
        )
    )
    def test_min_gaps_greater_than_max_gaps(self):
        """Should raise ValueError when min_gaps > max_gaps.

        Expected: ValueError with message 'min_gaps (5) must be <= max_gaps (3)'.
        Current: NO VALIDATION - Test will FAIL until model_validator is added.

        Required fix in ResearchConfig:
        ```python
        from pydantic import model_validator

        @model_validator(mode='after')
        def validate_gap_constraints(self) -> 'ResearchConfig':
            if self.min_gaps > self.max_gaps:
                raise ValueError(
                    f"min_gaps ({self.min_gaps}) must be <= max_gaps ({self.max_gaps})"
                )
            return self
        ```
        """
        # This test documents expected behavior - it will FAIL without validator
        with pytest.raises(ValueError, match=r"min_gaps.*must be <= max_gaps"):
            ResearchConfig(min_gaps=5, max_gaps=3)

    def test_min_gaps_equal_to_max_gaps_allowed(self):
        """Should allow min_gaps == max_gaps (boundary condition)."""
        config = ResearchConfig(min_gaps=5, max_gaps=5)
        assert config.min_gaps == 5
        assert config.max_gaps == 5

    def test_min_gaps_less_than_max_gaps_allowed(self):
        """Should allow min_gaps < max_gaps (normal case)."""
        config = ResearchConfig(min_gaps=2, max_gaps=8)
        assert config.min_gaps == 2
        assert config.max_gaps == 8


class TestBoundaryValues:
    """Test validation at field boundaries (min/max constraints)."""

    def test_min_target_tokens_boundary(self):
        """Should accept minimum valid target_tokens (10,000)."""
        config = ResearchConfig(target_tokens=10_000)
        assert config.target_tokens == 10_000

    def test_below_min_target_tokens_boundary(self):
        """Should reject target_tokens below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(target_tokens=9_999)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("target_tokens",)
        assert "greater than or equal to 10000" in str(error["msg"]).lower()

    def test_max_target_tokens_boundary(self):
        """Should accept maximum valid target_tokens (100,000)."""
        config = ResearchConfig(target_tokens=100_000)
        assert config.target_tokens == 100_000

    def test_above_max_target_tokens_boundary(self):
        """Should reject target_tokens above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(target_tokens=100_001)

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("target_tokens",)
        assert "less than or equal to 100000" in str(error["msg"]).lower()

    def test_min_multi_hop_time_limit_boundary(self):
        """Should accept minimum valid multi_hop_time_limit (1.0s)."""
        config = ResearchConfig(multi_hop_time_limit=1.0)
        assert config.multi_hop_time_limit == 1.0

    def test_max_multi_hop_time_limit_boundary(self):
        """Should accept maximum valid multi_hop_time_limit (15.0s)."""
        config = ResearchConfig(multi_hop_time_limit=15.0)
        assert config.multi_hop_time_limit == 15.0

    def test_min_max_compression_iterations_boundary(self):
        """Should accept minimum valid max_compression_iterations (1)."""
        config = ResearchConfig(max_compression_iterations=1)
        assert config.max_compression_iterations == 1

    def test_max_max_compression_iterations_boundary(self):
        """Should accept maximum valid max_compression_iterations (10)."""
        config = ResearchConfig(max_compression_iterations=10)
        assert config.max_compression_iterations == 10


class TestExtremeValues:
    """Test handling of extreme but technically valid values."""

    def test_extreme_target_tokens_at_max(self):
        """Should accept extreme target_tokens at maximum boundary.

        Note: 100,000 tokens is large but within field constraint.
        No warning/clamping behavior expected - just validation.
        """
        config = ResearchConfig(target_tokens=100_000)
        assert config.target_tokens == 100_000

    def test_extreme_exhaustive_time_limit_at_max(self):
        """Should accept extreme exhaustive_time_limit at maximum (30 minutes).

        Note: 1800 seconds (30 min) is the configured maximum.
        """
        config = ResearchConfig(exhaustive_time_limit=1800.0)
        assert config.exhaustive_time_limit == 1800.0

    def test_extreme_multi_hop_result_limit_at_max(self):
        """Should accept extreme result limit at maximum (2000 chunks)."""
        config = ResearchConfig(multi_hop_result_limit=2000)
        assert config.multi_hop_result_limit == 2000

    def test_extreme_initial_page_size_at_max(self):
        """Should accept extreme page size at maximum (100 results)."""
        config = ResearchConfig(initial_page_size=100)
        assert config.initial_page_size == 100

    def test_extreme_num_expanded_queries_at_max(self):
        """Should accept maximum expanded queries (5)."""
        config = ResearchConfig(num_expanded_queries=5)
        assert config.num_expanded_queries == 5


class TestValidConfigurations:
    """Test that valid configurations are accepted without errors."""

    def test_default_configuration(self):
        """Should accept default configuration values."""
        config = ResearchConfig()

        # Verify critical defaults
        assert config.algorithm == "v1"
        assert config.target_tokens == 20_000
        assert config.multi_hop_time_limit == 5.0
        assert config.max_compression_iterations == 5
        assert config.min_gaps == 1
        assert config.max_gaps == 10
        assert config.exhaustive_mode is False

    def test_valid_custom_configuration(self):
        """Should accept valid custom configuration."""
        config = ResearchConfig(
            algorithm="v2",
            target_tokens=50_000,
            multi_hop_time_limit=10.0,
            max_compression_iterations=7,
            min_gaps=2,
            max_gaps=15,
            exhaustive_mode=True,
        )

        assert config.algorithm == "v2"
        assert config.target_tokens == 50_000
        assert config.multi_hop_time_limit == 10.0
        assert config.max_compression_iterations == 7
        assert config.min_gaps == 2
        assert config.max_gaps == 15
        assert config.exhaustive_mode is True

    def test_boundary_values_equal_gaps(self):
        """Should accept min_gaps == max_gaps (valid boundary condition)."""
        # Use values within valid ranges: min_gaps <= 5, max_gaps >= 5
        config = ResearchConfig(min_gaps=5, max_gaps=5)

        assert config.min_gaps == 5
        assert config.max_gaps == 5

    def test_minimal_valid_configuration(self):
        """Should accept minimal valid configuration at all lower bounds."""
        config = ResearchConfig(
            num_expanded_queries=1,
            initial_page_size=10,
            relevance_threshold=0.3,
            max_symbols=1,
            regex_augmentation_ratio=0.1,
            regex_min_results=10,
            multi_hop_time_limit=1.0,
            multi_hop_result_limit=100,
            min_gaps=0,
            max_gaps=5,
            target_tokens=10_000,
            max_compression_iterations=1,
        )

        # All values should be at minimum boundaries
        assert config.num_expanded_queries == 1
        assert config.initial_page_size == 10
        assert config.min_gaps == 0
        assert config.max_gaps == 5
        assert config.target_tokens == 10_000
        assert config.max_compression_iterations == 1

    def test_maximal_valid_configuration(self):
        """Should accept maximal valid configuration at all upper bounds."""
        config = ResearchConfig(
            num_expanded_queries=5,
            initial_page_size=100,
            relevance_threshold=0.8,
            max_symbols=20,
            regex_augmentation_ratio=1.0,
            regex_min_results=100,
            multi_hop_time_limit=15.0,
            multi_hop_result_limit=2000,
            min_gaps=5,
            max_gaps=30,
            target_tokens=100_000,
            max_compression_iterations=10,
            exhaustive_time_limit=1800.0,
        )

        # All values should be at maximum boundaries
        assert config.num_expanded_queries == 5
        assert config.initial_page_size == 100
        assert config.min_gaps == 5
        assert config.max_gaps == 30
        assert config.target_tokens == 100_000
        assert config.max_compression_iterations == 10


class TestGetEffectiveMethods:
    """Test helper methods for exhaustive mode behavior."""

    def test_get_effective_time_limit_normal_mode(self):
        """Should return multi_hop_time_limit when exhaustive_mode=False."""
        config = ResearchConfig(
            exhaustive_mode=False,
            multi_hop_time_limit=5.0,
            exhaustive_time_limit=600.0,
        )

        assert config.get_effective_time_limit() == 5.0

    def test_get_effective_time_limit_exhaustive_mode(self):
        """Should return exhaustive_time_limit when exhaustive_mode=True."""
        config = ResearchConfig(
            exhaustive_mode=True,
            multi_hop_time_limit=5.0,
            exhaustive_time_limit=600.0,
        )

        assert config.get_effective_time_limit() == 600.0

    def test_get_effective_result_limit_normal_mode(self):
        """Should return multi_hop_result_limit when exhaustive_mode=False."""
        config = ResearchConfig(
            exhaustive_mode=False,
            multi_hop_result_limit=500,
        )

        assert config.get_effective_result_limit() == 500

    def test_get_effective_result_limit_exhaustive_mode(self):
        """Should return None when exhaustive_mode=True (no limit)."""
        config = ResearchConfig(
            exhaustive_mode=True,
            multi_hop_result_limit=500,
        )

        assert config.get_effective_result_limit() is None


class TestFloatPrecisionEdgeCases:
    """Test floating-point precision edge cases."""

    def test_float_precision_time_limit(self):
        """Should handle floating-point precision correctly."""
        config = ResearchConfig(multi_hop_time_limit=5.123456789)

        # Python floats have limited precision
        assert abs(config.multi_hop_time_limit - 5.123456789) < 1e-10

    def test_float_boundary_just_above_minimum(self):
        """Should accept float just above minimum boundary."""
        config = ResearchConfig(multi_hop_time_limit=1.0001)
        assert config.multi_hop_time_limit > 1.0

    def test_float_boundary_just_below_maximum(self):
        """Should accept float just below maximum boundary."""
        config = ResearchConfig(multi_hop_time_limit=14.9999)
        assert config.multi_hop_time_limit < 15.0

    def test_relevance_threshold_precision(self):
        """Should handle relevance threshold precision correctly."""
        config = ResearchConfig(relevance_threshold=0.55555)
        assert abs(config.relevance_threshold - 0.55555) < 1e-10


class TestAlgorithmVersionValidation:
    """Test algorithm version field validation."""

    def test_valid_algorithm_v1(self):
        """Should accept 'v1' as valid algorithm version."""
        config = ResearchConfig(algorithm="v1")
        assert config.algorithm == "v1"

    def test_valid_algorithm_v2(self):
        """Should accept 'v2' as valid algorithm version."""
        config = ResearchConfig(algorithm="v2")
        assert config.algorithm == "v2"

    def test_invalid_algorithm_version(self):
        """Should reject invalid algorithm version.

        Expected: ValidationError with message about valid choices.
        Current: Pydantic Literal validation should catch this.
        """
        with pytest.raises(ValidationError) as exc_info:
            ResearchConfig(algorithm="v3")

        error = exc_info.value.errors()[0]
        assert error["loc"] == ("algorithm",)
        # Pydantic Literal validation error message mentions valid values
        error_msg = str(error["msg"]).lower()
        assert "v1" in error_msg and "v2" in error_msg
