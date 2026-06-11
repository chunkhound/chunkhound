"""Tests for matryoshka embedding support.

Covers:
- output_dims configuration validation
- Provider protocol properties (native_dims, supported_dimensions, supports_matryoshka)
- Model whitelist validation
- Dimension boundary testing
- Dimension discovery for unknown models
"""

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.interfaces.embedding_provider import EmbeddingConfigurationError
from chunkhound.providers.embeddings.voyageai_provider import VOYAGE_MODEL_CONFIG
from tests.unit.provider_test_helpers import _bare_provider, _ok_response


class TestOutputDimsConfig:
    """Test output_dims configuration field."""

    def test_output_dims_default_none(self):
        """output_dims defaults to None."""
        config = EmbeddingConfig()
        assert config.output_dims is None

    def test_output_dims_positive_valid(self):
        """Positive output_dims values are accepted."""
        config = EmbeddingConfig(output_dims=512)
        assert config.output_dims == 512

    def test_output_dims_zero_invalid(self):
        """output_dims=0 raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            EmbeddingConfig(output_dims=0)

    def test_output_dims_negative_invalid(self):
        """Negative output_dims raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            EmbeddingConfig(output_dims=-100)

    def test_output_dims_in_provider_config(self):
        """output_dims is included in provider config dict."""
        config = EmbeddingConfig(output_dims=256, base_url="http://localhost:8000")
        provider_config = config.get_provider_config()
        assert provider_config["output_dims"] == 256

    def test_output_dims_none_not_in_provider_config(self):
        """output_dims=None is not included in provider config dict."""
        config = EmbeddingConfig(base_url="http://localhost:8000")
        provider_config = config.get_provider_config()
        assert "output_dims" not in provider_config


class TestEmbeddingCliAliases:
    """Test embedding CLI alias extraction."""

    def test_extracts_azure_deployment_standard_alias(self):
        """--azure-deployment populates azure_deployment."""
        parser = argparse.ArgumentParser()
        EmbeddingConfig.add_cli_arguments(parser)
        args = parser.parse_args(["--azure-deployment", "deploy-a"])

        overrides = EmbeddingConfig.extract_cli_overrides(args)

        assert overrides["azure_deployment"] == "deploy-a"

    def test_extracts_azure_deployment_embedding_alias(self):
        """--embedding-azure-deployment populates azure_deployment."""
        parser = argparse.ArgumentParser()
        EmbeddingConfig.add_cli_arguments(parser)
        args = parser.parse_args(["--embedding-azure-deployment", "deploy-b"])

        overrides = EmbeddingConfig.extract_cli_overrides(args)

        assert overrides["azure_deployment"] == "deploy-b"



class TestOpenAIProviderMatryoshka:
    """Test OpenAI provider matryoshka support."""

    def test_matryoshka_model_supports(self):
        """text-embedding-3-* models support matryoshka."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small"
        )
        assert provider.supports_matryoshka() is True
        assert provider.native_dims == 1536
        assert 512 in provider.supported_dimensions

    def test_ada_model_no_matryoshka(self):
        """text-embedding-ada-002 does not support matryoshka."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-ada-002"
        )
        assert provider.supports_matryoshka() is False
        assert provider.supported_dimensions == [1536]

    def test_output_dims_changes_dims_property(self):
        """Setting output_dims changes dims property."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small", output_dims=512
        )
        assert provider.dims == 512
        assert provider.native_dims == 1536

    def test_embedding_3_large_dimensions(self):
        """text-embedding-3-large has correct dimension properties."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-large"
        )
        assert provider.supports_matryoshka() is True
        assert provider.native_dims == 3072
        assert provider.dims == 3072
        # Continuous range support
        assert 1024 in provider.supported_dimensions
        assert 2048 in provider.supported_dimensions


class TestVoyageModelConfigProperties:
    """Test VoyageAI model config properties without requiring SDK.

    Validates VOYAGE_MODEL_CONFIG entries produce the correct matryoshka
    properties. These are the same invariants the provider derives at runtime
    via trivial dict lookups.
    """

    def test_voyage_35_supports_matryoshka(self):
        """voyage-3.5 config has multiple dimensions (matryoshka)."""

        dims = VOYAGE_MODEL_CONFIG["voyage-3.5"]["dimensions"]
        assert len(dims) > 1
        assert max(dims) == 2048
        assert all(isinstance(d, int) for d in dims)

    def test_voyage_35_output_dims_valid(self):
        """voyage-3.5 config accepts 512 as valid output_dims."""

        config = VOYAGE_MODEL_CONFIG["voyage-3.5"]
        assert 512 in config["dimensions"]
        assert max(config["dimensions"]) == 2048

    def test_voyage_35_output_dims_invalid(self):
        """768 is not a valid output_dims for voyage-3.5."""

        assert 768 not in VOYAGE_MODEL_CONFIG["voyage-3.5"]["dimensions"]


class TestDimensionBoundaries:
    """Test dimension boundary conditions."""

    def test_openai_min_dims_one(self):
        """OpenAI matryoshka models accept dims=1."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small", output_dims=1
        )
        assert provider.dims == 1

    def test_openai_max_dims_native(self):
        """OpenAI matryoshka models accept dims=native_dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small", output_dims=1536
        )
        assert provider.dims == 1536
        assert provider.native_dims == 1536



class TestConfigIntegration:
    """Test configuration integration between EmbeddingConfig and providers."""

    def test_config_output_dims_flows_to_provider_config(self):
        """output_dims from EmbeddingConfig flows to get_provider_config."""
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            output_dims=512,
            base_url="http://localhost:8000",  # Custom endpoint to bypass whitelist
        )
        provider_config = config.get_provider_config()
        assert provider_config["output_dims"] == 512
        assert provider_config["model"] == "text-embedding-3-small"

    def test_config_default_model_resolution(self):
        """EmbeddingConfig resolves default model correctly."""
        openai_config = EmbeddingConfig(provider="openai")
        assert openai_config.get_default_model() == "text-embedding-3-small"

        voyageai_config = EmbeddingConfig(provider="voyageai")
        # Default voyage model
        assert voyageai_config.get_default_model() is not None

    def test_default_openai_model_has_correct_dimensions(self):
        """Default OpenAI model (text-embedding-3-small) maps to 1536 native dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OPENAI_MODEL_CONFIG,
            OpenAIEmbeddingProvider,
        )

        # Verify config table agrees with constant
        default_model = EmbeddingConfig(provider="openai").get_default_model()
        assert default_model in OPENAI_MODEL_CONFIG
        assert OPENAI_MODEL_CONFIG[default_model]["native_dims"] == 1536  # type: ignore[index]

        # Verify provider resolves correctly
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.model == "text-embedding-3-small"
        assert provider.native_dims == 1536
        assert provider.dims == 1536


class TestVoyageSingleDimModels:
    """Test single-dimension VoyageAI models (no matryoshka)."""

    def test_voyage_finance_no_matryoshka(self):
        """voyage-finance-2 has single dimension (no matryoshka)."""

        config = VOYAGE_MODEL_CONFIG["voyage-finance-2"]
        dims = config["dimensions"]
        assert len(dims) == 1
        assert dims == [1024]
        assert max(dims) == 1024
        assert config["default_dimension"] == 1024

    def test_voyage_law_no_matryoshka(self):
        """voyage-law-2 has single dimension (no matryoshka)."""

        config = VOYAGE_MODEL_CONFIG["voyage-law-2"]
        dims = config["dimensions"]
        assert len(dims) == 1
        assert dims == [1024]


class TestClientSideTruncation:
    """Test client-side truncation for APIs that don't support dimensions parameter."""

    def test_requires_output_dims(self):
        """client_side_truncation without output_dims raises ValueError."""
        with pytest.raises(ValueError, match="requires output_dims"):
            EmbeddingConfig(
                base_url="http://localhost:8000", client_side_truncation=True
            )

    def test_valid_with_output_dims(self):
        """client_side_truncation with output_dims is valid."""
        config = EmbeddingConfig(
            base_url="http://localhost:8000",
            output_dims=512,
            client_side_truncation=True,
        )
        assert config.client_side_truncation is True
        assert config.output_dims == 512

    def test_default_false(self):
        """client_side_truncation defaults to False."""
        config = EmbeddingConfig(base_url="http://localhost:8000")
        assert config.client_side_truncation is False

    def test_in_provider_config_when_true(self):
        """client_side_truncation is included in provider config when True."""
        config = EmbeddingConfig(
            base_url="http://localhost:8000",
            output_dims=512,
            client_side_truncation=True,
        )
        provider_config = config.get_provider_config()
        assert provider_config["client_side_truncation"] is True

    def test_not_in_provider_config_when_false(self):
        """client_side_truncation is not included in provider config when False."""
        config = EmbeddingConfig(
            base_url="http://localhost:8000",
            output_dims=512,
            client_side_truncation=False,
        )
        provider_config = config.get_provider_config()
        assert "client_side_truncation" not in provider_config

    def test_l2_normalize_unit_length(self):
        """L2 normalize produces unit-length vectors."""
        from chunkhound.providers.embeddings.shared_utils import l2_normalize

        # 3-4-5 right triangle: magnitude = 5
        normalized = l2_normalize([3.0, 4.0])
        # Check it's unit length
        magnitude = sum(x * x for x in normalized) ** 0.5
        assert abs(magnitude - 1.0) < 1e-9

    def test_l2_normalize_zero_vector(self):
        """L2 normalize handles zero vector gracefully."""
        from chunkhound.providers.embeddings.shared_utils import l2_normalize

        normalized = l2_normalize([0.0, 0.0, 0.0])
        assert normalized == [0.0, 0.0, 0.0]

    def test_l2_normalize_already_unit(self):
        """L2 normalize preserves already-normalized vectors."""
        from chunkhound.providers.embeddings.shared_utils import l2_normalize

        # Already unit length
        normalized = l2_normalize([1.0, 0.0])
        assert abs(normalized[0] - 1.0) < 1e-9
        assert abs(normalized[1] - 0.0) < 1e-9

    def test_client_side_truncation_config_roundtrip(self):
        """Client-side truncation config flows through to provider config dict."""
        config = EmbeddingConfig(
            base_url="http://localhost:8000",
            output_dims=512,
            client_side_truncation=True,
        )
        provider_config = config.get_provider_config()
        assert provider_config["output_dims"] == 512
        assert provider_config["client_side_truncation"] is True

    def test_client_side_truncation_false_not_in_config(self):
        """client_side_truncation=False is excluded from provider config dict."""
        config = EmbeddingConfig(
            base_url="http://localhost:8000",
            output_dims=512,
            client_side_truncation=False,
        )
        provider_config = config.get_provider_config()
        assert provider_config["output_dims"] == 512
        assert "client_side_truncation" not in provider_config


class TestVoyageOutputDimsValidation:
    """Test VoyageAI output_dims validation at provider construction."""

    def test_output_dims_not_supported_raises_config_error(self):
        """output_dims not in model's supported dimensions raises EmbeddingConfigurationError."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError,
            match="not in supported dimensions",
        ):
            VoyageAIEmbeddingProvider(
                api_key="test-key",
                model="voyage-2",
                output_dims=512,
            )

    def test_output_dims_supported_ok(self):
        """Valid output_dims does not raise."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key",
            model="voyage-2",
            output_dims=1024,  # voyage-2 only supports 1024
        )
        assert provider.dims == 1024


class TestFactoryRouting:
    """Test EmbeddingProviderFactory routing for all providers."""

    def test_openai_provider_routes_correctly(self):
        """openai provider routes through factory without error."""
        config = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
        )
        provider = EmbeddingProviderFactory.create_provider(config)
        assert provider.name == "openai"
        assert provider.model == "text-embedding-3-small"

    def test_factory_supported_providers_list(self):
        """EmbeddingProviderFactory lists openai as supported."""
        supported = EmbeddingProviderFactory.get_supported_providers()
        assert "openai" in supported

    def test_validate_provider_dependencies_supports_openai(self):
        """openai should have its dependency path available."""
        available, error = EmbeddingProviderFactory.validate_provider_dependencies(
            "openai"
        )
        assert available is True
        assert error is None


class TestOpenAIProviderRuntimeBehavior:
    """Test OpenAI matryoshka request and validation behavior."""

    def test_known_non_matryoshka_model_rejects_invalid_output_dims(self):
        """Known non-matryoshka models must reject incompatible output_dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(EmbeddingConfigurationError, match="does not support"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-ada-002",
                output_dims=512,
            )

    def test_known_matryoshka_model_rejects_output_dims_above_native(self):
        """Known matryoshka models must reject dimensions above native size."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(EmbeddingConfigurationError, match="range 1-1536"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
                output_dims=1537,
            )



class TestMeanPoolEmbeddings:
    """Tests for mean_pool_embeddings in shared_utils."""

    def test_two_orthogonal_vectors(self):
        """Mean-pool two orthogonal vectors produces an L2-normalized result."""
        from chunkhound.providers.embeddings.shared_utils import mean_pool_embeddings

        result = mean_pool_embeddings([[1.0, 0.0], [0.0, 1.0]])
        # Mean is [0.5, 0.5]; L2-normalized: each component = 0.5 / sqrt(0.5)
        expected = 0.5 / (0.5**0.5)
        assert abs(result[0] - expected) < 1e-9
        assert abs(result[1] - expected) < 1e-9
        magnitude = sum(x * x for x in result) ** 0.5
        assert abs(magnitude - 1.0) < 1e-9

    def test_empty_list_raises_value_error(self):
        """Empty list raises ValueError."""
        from chunkhound.providers.embeddings.shared_utils import mean_pool_embeddings

        with pytest.raises(ValueError, match="empty"):
            mean_pool_embeddings([])

    def test_single_item_passthrough(self):
        """Single embedding is returned as-is (value equality)."""
        from chunkhound.providers.embeddings.shared_utils import mean_pool_embeddings

        vec = [3.0, 4.0]
        result = mean_pool_embeddings([vec])
        assert result == vec

    def test_multiple_embeddings(self):
        """Three vectors produce a correctly averaged and L2-normalized result."""
        from chunkhound.providers.embeddings.shared_utils import mean_pool_embeddings

        result = mean_pool_embeddings([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
        # Mean: [2.0, 2.0]; L2-norm = sqrt(8)
        expected = 2.0 / (8.0**0.5)
        assert abs(result[0] - expected) < 1e-9
        assert abs(result[1] - expected) < 1e-9
        magnitude = sum(x * x for x in result) ** 0.5
        assert abs(magnitude - 1.0) < 1e-9


class TestValidateEmbeddingDims:
    """Tests for validate_embedding_dims invariant (INV-1)."""

    def test_match_does_not_raise(self):
        """Matching dimensions do not raise."""
        from chunkhound.providers.embeddings.shared_utils import (
            validate_embedding_dims,
        )

        validate_embedding_dims(1536, 1536)  # should not raise

    def test_match_does_not_raise_with_model(self):
        """Matching dimensions do not raise even with model param."""
        from chunkhound.providers.embeddings.shared_utils import (
            validate_embedding_dims,
        )

        validate_embedding_dims(
            512, 512, model="text-embedding-3-small"
        )  # should not raise

    def test_mismatch_raises_embedding_dimension_error(self):
        """Mismatched dimensions raise EmbeddingDimensionError."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError
        from chunkhound.providers.embeddings.shared_utils import (
            validate_embedding_dims,
        )

        with pytest.raises(
            EmbeddingDimensionError,
            match="got 768, expected 512",
        ):
            validate_embedding_dims(actual_dims=768, expected_dims=512)

    def test_mismatch_includes_model_in_message(self):
        """Mismatch error message includes model name when provided."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError
        from chunkhound.providers.embeddings.shared_utils import (
            validate_embedding_dims,
        )

        with pytest.raises(EmbeddingDimensionError, match="model=test-model"):
            validate_embedding_dims(
                actual_dims=256, expected_dims=1536, model="test-model"
            )


class TestFactoryMatryoshkaForwarding:
    """Test factory forwards output_dims and client_side_truncation to providers."""

    def test_factory_forwards_output_dims_to_openai(self):
        """EmbeddingProviderFactory passes output_dims to OpenAI provider."""
        from chunkhound.core.config.embedding_config import EmbeddingConfig
        from chunkhound.core.config.embedding_factory import (
            EmbeddingProviderFactory,
        )

        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            output_dims=256,
            base_url="http://localhost:8000",
        )
        provider = EmbeddingProviderFactory.create_provider(config)
        assert provider.output_dims == 256
        assert provider.dims == 256

    def test_factory_forwards_output_dims_to_voyageai(self):
        """EmbeddingProviderFactory passes output_dims to VoyageAI provider."""
        from chunkhound.core.config.embedding_config import EmbeddingConfig
        from chunkhound.core.config.embedding_factory import (
            EmbeddingProviderFactory,
        )

        config = EmbeddingConfig(
            provider="voyageai",
            model="voyage-2",
            output_dims=1024,
            api_key="test-key",
        )
        provider = EmbeddingProviderFactory.create_provider(config)
        assert provider.output_dims == 1024
        assert provider.dims == 1024

    def test_factory_forwards_client_side_truncation_to_openai(self):
        """EmbeddingProviderFactory passes client_side_truncation to OpenAI provider."""
        from chunkhound.core.config.embedding_config import EmbeddingConfig
        from chunkhound.core.config.embedding_factory import (
            EmbeddingProviderFactory,
        )

        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            output_dims=256,
            client_side_truncation=True,
            base_url="http://localhost:8000",
        )
        provider = EmbeddingProviderFactory.create_provider(config)
        assert provider.client_side_truncation is True
        assert provider.output_dims == 256
        assert provider.dims == 256


class TestSharedMatryoshkaUtils:
    """Tests for shared matryoshka utility functions."""

    def test_apply_client_side_truncation_truncates_and_normalizes(self):
        """apply_client_side_truncation truncates and L2-normalizes."""
        from chunkhound.providers.embeddings.shared_utils import (
            apply_client_side_truncation,
        )

        # 4-dim vectors → truncate to 2
        vectors = [[3.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        result = apply_client_side_truncation(vectors, 2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        # First vector truncated to [3, 4], L2-norm = 5, so unit = [0.6, 0.8]
        assert abs(result[0][0] - 0.6) < 1e-9
        assert abs(result[0][1] - 0.8) < 1e-9
        # Second vector truncated to [1, 0], already unit
        assert abs(result[1][0] - 1.0) < 1e-9
        assert abs(result[1][1] - 0.0) < 1e-9

    def test_apply_client_side_truncation_preserves_unit_length(self):
        """Truncated vectors are L2-normalized (unit length)."""
        from chunkhound.providers.embeddings.shared_utils import (
            apply_client_side_truncation,
        )

        vectors = [[3.0, 4.0, 5.0]]
        result = apply_client_side_truncation(vectors, 2)
        magnitude = sum(x * x for x in result[0]) ** 0.5
        assert abs(magnitude - 1.0) < 1e-9

    def test_build_dimension_request_param_server_side(self):
        """Returns output_dims when not client-side."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_dimension_request_param,
        )

        result = build_dimension_request_param(output_dims=512, client_side_truncation=False)
        assert result == 512

    def test_build_dimension_request_param_client_side(self):
        """Returns None when client-side truncation."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_dimension_request_param,
        )

        result = build_dimension_request_param(output_dims=512, client_side_truncation=True)
        assert result is None

    def test_build_dimension_request_param_no_output_dims(self):
        """Returns None when output_dims is None."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_dimension_request_param,
        )

        result = build_dimension_request_param(output_dims=None, client_side_truncation=False)
        assert result is None

        result = build_dimension_request_param(output_dims=None, client_side_truncation=True)
        assert result is None


class TestVoyageAIClientSideTruncation:
    """Tests for VoyageAI client_side_truncation support."""

    def test_client_side_truncation_property(self):
        """VoyageAI provider respects client_side_truncation constructor arg."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        p = VoyageAIEmbeddingProvider(
            api_key="test-key",
            model="voyage-2",
            output_dims=1024,
            client_side_truncation=True,
        )
        assert p.client_side_truncation is True
        assert p.output_dims == 1024
        assert p.dims == 1024

    def test_client_side_truncation_default_false(self):
        """VoyageAI client_side_truncation defaults to False."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        p = VoyageAIEmbeddingProvider(api_key="test-key", model="voyage-2")
        assert p.client_side_truncation is False

    def test_server_side_truncation_with_output_dims(self):
        """VoyageAI with output_dims and no client_side_truncation uses server-side."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        p = VoyageAIEmbeddingProvider(
            api_key="test-key", model="voyage-2", output_dims=1024
        )
        assert p.client_side_truncation is False
        assert p.output_dims == 1024
        assert p.dims == 1024


class TestFakeProviderMatryoshka:
    """Tests for FakeEmbeddingProvider matryoshka simulation."""

    def test_server_side_truncation_returns_output_dim_vectors(self):
        """FakeEmbeddingProvider with output_dims returns output_dims-sized vectors."""
        import asyncio

        from tests.fixtures.fake_providers import FakeEmbeddingProvider

        p = FakeEmbeddingProvider(dims=1536, output_dims=512)
        assert p.dims == 512
        assert p.native_dims == 1536
        assert p.supports_matryoshka() is True

        vecs = asyncio.run(p.embed(["test text"]))
        assert len(vecs) == 1
        assert len(vecs[0]) == 512

    def test_client_side_truncation_returns_normalized_vectors(self):
        """FakeEmbeddingProvider with client_side_truncation returns truncated L2-normalized vectors."""
        import asyncio

        from tests.fixtures.fake_providers import FakeEmbeddingProvider

        p = FakeEmbeddingProvider(dims=1536, output_dims=512, client_side_truncation=True)
        assert p.dims == 512
        assert p.native_dims == 1536
        assert p.client_side_truncation is True

        vecs = asyncio.run(p.embed(["test text"]))
        assert len(vecs) == 1
        assert len(vecs[0]) == 512
        # Verify L2-normalized
        magnitude = sum(x * x for x in vecs[0]) ** 0.5
        assert abs(magnitude - 1.0) < 1e-9

    def test_no_output_dims_uses_native_dims(self):
        """FakeEmbeddingProvider without output_dims uses native dims."""
        import asyncio

        from tests.fixtures.fake_providers import FakeEmbeddingProvider

        p = FakeEmbeddingProvider(dims=1536)
        assert p.dims == 1536
        assert p.native_dims == 1536
        assert p.supports_matryoshka() is False

        vecs = asyncio.run(p.embed(["test text"]))
        assert len(vecs[0]) == 1536


class TestApplyClientSideTruncationShortVector:
    """Test the short-vector guard in apply_client_side_truncation."""

    def test_short_vector_raises_embedding_dimension_error(self):
        """Vector shorter than output_dims raises EmbeddingDimensionError."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError
        from chunkhound.providers.embeddings.shared_utils import (
            apply_client_side_truncation,
        )

        with pytest.raises(EmbeddingDimensionError, match="< requested"):
            apply_client_side_truncation([[1.0, 2.0, 3.0]], output_dims=4)

    def test_short_vector_in_multi_element_batch(self):
        """One short vector in a batch raises EmbeddingDimensionError."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError
        from chunkhound.providers.embeddings.shared_utils import (
            apply_client_side_truncation,
        )

        with pytest.raises(EmbeddingDimensionError, match="< requested"):
            apply_client_side_truncation(
                [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], output_dims=4
            )


# ---------------------------------------------------------------------------
# Dimension discovery for unknown models
# ---------------------------------------------------------------------------


class TestOpenAIDimensionDiscovery:
    """Test unknown-model fallback behavior in OpenAIEmbeddingProvider."""

    @pytest.mark.asyncio
    async def test_unknown_model_discovers_dims_after_first_embed(self):
        """Unknown model learns native dims from an untruncated API response."""
        provider, _, _ = _bare_provider(model="my-custom-embed-v1")
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 768
        assert provider.dims == 768
        assert provider.native_dims == 768

    @pytest.mark.asyncio
    async def test_unknown_model_server_side_truncation_keeps_output_contract(self):
        """Unknown models still return requested output dims with API truncation."""
        provider, _, _ = _bare_provider(
            model="my-custom-embed-v1",
            output_dims=256,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=256)
        )

        result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 256
        assert provider.dims == 256

    @pytest.mark.asyncio
    async def test_unknown_model_client_side_truncation_discovers_native_dims(
        self,
    ):
        """Client-side truncation keeps output dims and discovers native dims."""
        provider, _, _ = _bare_provider(
            model="my-custom-embed-v1",
            output_dims=256,
            client_side_truncation=True,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 256
        assert provider.dims == 256
        assert provider.native_dims == 768

    @pytest.mark.asyncio
    async def test_unknown_model_warns_default_dims(self):
        """First access of dims for an unknown model logs a warning."""
        provider, _, mod = _bare_provider(model="my-custom-embed-v1")

        with patch.object(mod.logger, "warning") as mock_warn:
            dims = provider.dims

        assert dims == 1536
        mock_warn.assert_called_once()
        assert "Unknown model" in mock_warn.call_args[0][0]
