"""Tests for matryoshka embedding support.

Covers:
- output_dims configuration validation
- Provider protocol properties (native_dims, supported_dimensions)
- Model whitelist validation
- Dimension boundary testing
- Public embed() truncation/runtime contracts
- Selected _embed_batch_internal invariants
"""

import argparse
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.exceptions.embedding import (
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
)
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

    @pytest.mark.parametrize("output_dims", [True, False, 1.0, "256"])
    def test_output_dims_rejects_coerced_non_int_values(self, output_dims):
        """Config must reject implicit coercion for output_dims."""
        with pytest.raises(ValueError, match="positive integer"):
            EmbeddingConfig(output_dims=output_dims)

    def test_load_from_env_parses_output_dims_string(self, monkeypatch):
        """Env loader parses numeric strings into explicit ints."""
        monkeypatch.setenv("CHUNKHOUND_EMBEDDING__OUTPUT_DIMS", "256")

        config = EmbeddingConfig.load_from_env()

        assert config["output_dims"] == 256

    @pytest.mark.parametrize("output_dims", ["abc", "12.5", "", "0xff", "0", "-1"])
    def test_load_from_env_rejects_invalid_output_dims(self, monkeypatch, output_dims):
        """Invalid env values must fail explicitly instead of being ignored."""
        monkeypatch.setenv("CHUNKHOUND_EMBEDDING__OUTPUT_DIMS", output_dims)

        with pytest.raises(ValueError, match="CHUNKHOUND_EMBEDDING__OUTPUT_DIMS"):
            EmbeddingConfig.load_from_env()

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
    """Test OpenAI provider dimension support."""

    def test_matryoshka_model_supports(self):
        """text-embedding-3-* models have multiple supported dimensions."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-3-small"
        )
        assert provider.native_dims == 1536
        assert 512 in provider.supported_dimensions

    def test_ada_model_no_matryoshka(self):
        """text-embedding-ada-002 has a single supported dimension."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key", model="text-embedding-ada-002"
        )
        assert provider.supported_dimensions == [1536]

    def test_unknown_model_supported_dimensions_empty_until_discovery(self):
        """Unknown models must not fabricate supported dimensions pre-discovery."""
        provider = TestOpenAIProviderRuntimeBehavior._unknown_model_provider()

        assert provider.supported_dimensions == []

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

    def test_voyage_35_has_multiple_dimensions(self):
        """voyage-3.5 config has multiple supported dimensions."""

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
        assert (
            OPENAI_MODEL_CONFIG[default_model]["native_dims"] == 1536
        )  # type: ignore[index]

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

    @pytest.mark.parametrize("raw_value", ["", "maybe", "2", "truthy"])
    def test_load_from_env_rejects_invalid_bool(self, monkeypatch, raw_value):
        """Invalid env bools must fail explicitly instead of being ignored."""
        monkeypatch.setenv(
            "CHUNKHOUND_EMBEDDING__CLIENT_SIDE_TRUNCATION", raw_value
        )

        with pytest.raises(
            ValueError, match="CHUNKHOUND_EMBEDDING__CLIENT_SIDE_TRUNCATION"
        ):
            EmbeddingConfig.load_from_env()

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
        """Unsupported output_dims raises EmbeddingConfigurationError."""
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
    """Test OpenAI output_dims validation across init and runtime paths."""

    @staticmethod
    def _unknown_model_provider(**kwargs):
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="my-custom-embed-v1",
            **kwargs,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=256)
        )
        provider._ensure_client = AsyncMock()
        return provider

    @staticmethod
    def _custom_endpoint_provider(**kwargs):
        """Create a provider for a non-official trust-runtime endpoint."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="my-custom-embed-v1",
            **kwargs,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=512)
        )
        provider._ensure_client = AsyncMock()
        return provider

    def test_known_non_matryoshka_model_rejects_invalid_output_dims(self):
        """Official OpenAI endpoints reject incompatible output_dims for ada-002."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError, match="does not support"
        ):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-ada-002",
                output_dims=512,
            )

    def test_known_model_on_custom_endpoint_defers_output_dims_semantics(self):
        """Custom endpoints may reuse known model names with different dims support."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-ada-002",
            output_dims=512,
        )

        assert provider.output_dims == 512
        assert provider.dims == 512

    def test_known_matryoshka_model_rejects_output_dims_above_native(self):
        """Known matryoshka models must reject dimensions above native size."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError, match="range 1-1536"
        ):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
                output_dims=1537,
            )

    @pytest.mark.parametrize("output_dims", [0, "256", True])
    def test_known_model_rejects_invalid_output_dims_types_and_values(
        self, output_dims
    ):
        """Known models fail fast on invalid output_dims input."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError,
            match="positive integer",
        ):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
                output_dims=output_dims,
            )

    def test_known_model_rejects_client_truncation_without_output_dims(self):
        """Known models fail fast when client truncation has no target dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError,
            match="output_dims is not set",
        ):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
                client_side_truncation=True,
            )

    def test_unknown_model_init_allows_valid_custom_truncation_config(self):
        """Unknown/custom models allow valid configs while deferring checks."""
        provider = self._unknown_model_provider(
            output_dims=256,
            client_side_truncation=True,
        )

        assert provider.model == "my-custom-embed-v1"
        assert provider.output_dims == 256
        assert provider.client_side_truncation is True

    @pytest.mark.asyncio
    async def test_unknown_model_embed_uses_dimensions_param_for_server_side(
        self,
    ):
        """Server-side-truncated responses must not be cached as native dims."""
        provider = self._unknown_model_provider(output_dims=256)

        result = await provider.embed(["hello"])

        assert len(result[0]) == 256
        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 256
        assert provider._discovered_native_dims is None
        assert provider.native_dims == 1536

    @pytest.mark.asyncio
    async def test_known_model_on_custom_endpoint_uses_dimensions_param_at_runtime(
        self,
    ):
        """Custom endpoints stay permissive even under official OpenAI model names."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-ada-002",
            output_dims=512,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=512)
        )
        provider._ensure_client = AsyncMock()

        result = await provider.embed(["hello"])

        assert len(result[0]) == 512
        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert call_kwargs["dimensions"] == 512

    @pytest.mark.asyncio
    async def test_known_model_on_custom_endpoint_does_not_cache_truncated_probe_dims(
        self,
    ):
        """Server-side truncation on custom endpoints must not overwrite native dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
            output_dims=512,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=512)
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is True
        result = await provider.embed(["hello"])

        assert len(result[0]) == 512
        assert provider.dims == 512
        assert provider.native_dims == 1536
        assert provider.supported_dimensions == []

    @pytest.mark.asyncio
    async def test_known_model_on_custom_endpoint_discovers_runtime_native_dims(self):
        """Client truncation should expose runtime-supported dims after discovery."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
            client_side_truncation=True,
            output_dims=2,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=3)
        )
        provider._ensure_client = AsyncMock()

        assert provider.supported_dimensions == []
        assert await provider.validate_api_key() is True
        assert provider.native_dims == 3
        assert list(provider.supported_dimensions) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_known_model_on_custom_endpoint_without_output_dims_trusts_runtime(self):
        """Custom endpoints without truncation must accept and cache runtime dims."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is True
        assert provider.native_dims == 768
        assert provider.dims == 768
        assert provider.supported_dimensions == [768]

    def test_known_model_on_custom_endpoint_pre_discovery_warning_is_accurate(self):
        """Pre-discovery fallback messaging must describe endpoint runtime discovery."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
        )

        with patch(
            "chunkhound.providers.embeddings.openai_provider.logger.warning"
        ) as warn:
            assert provider.dims == 1536

        assert warn.call_count == 1
        warning_message = warn.call_args.args[0]
        assert "Unknown model" not in warning_message
        assert "runtime dimension for this endpoint" in warning_message

    @pytest.mark.asyncio
    async def test_known_ada_official_endpoint_omits_dimensions_at_native_dims(
        self,
    ):
        """Official OpenAI no-op native dims must not send unsupported dimensions."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-ada-002",
            output_dims=1536,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=1536)
        )
        provider._ensure_client = AsyncMock()

        result = await provider.embed(["hello"])

        assert len(result[0]) == 1536
        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert "dimensions" not in call_kwargs

    @pytest.mark.asyncio
    async def test_unknown_model_embed_omits_dimensions_param_for_client_side(
        self,
    ):
        """Unknown models expose client-side truncation through the public embed API."""
        provider = self._unknown_model_provider(
            output_dims=256,
            client_side_truncation=True,
        )
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        result = await provider.embed(["hello"])

        assert len(result[0]) == 256
        call_kwargs = provider._client.embeddings.create.call_args.kwargs
        assert "dimensions" not in call_kwargs
        assert provider.native_dims == 768

    @pytest.mark.asyncio
    async def test_unknown_model_embed_rejects_output_dims_exceeding_api_dims(
        self,
    ):
        """Unknown model + client truncation rejects output_dims > API dims."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError

        provider = self._unknown_model_provider(
            output_dims=1024,
            client_side_truncation=True,
        )
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        with pytest.raises(
            EmbeddingDimensionError,
            match="exceeds API response dimension",
        ):
            await provider.embed(["hello"])

    @pytest.mark.asyncio
    async def test_validate_api_key_returns_false_without_api_key(self):
        """Official OpenAI validation without a key is a silent False."""
        provider, _, _ = _bare_provider()
        provider._api_key = None
        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_custom_endpoint_validate_api_key_allows_missing_api_key(self):
        """Custom OpenAI-compatible endpoints validate connectivity without a key."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key=None,
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
            output_dims=2,
            client_side_truncation=True,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=3)
        )
        provider._ensure_client = AsyncMock()

        assert provider.supported_dimensions == []
        assert await provider.validate_api_key() is True
        assert list(provider.supported_dimensions) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_azure_endpoint_validate_api_key_rejects_missing_key(self):
        """Azure OpenAI validation without a key is a silent False."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key=None,
            azure_endpoint="https://my-resource.openai.azure.com",
            model="text-embedding-3-small",
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_azure_endpoint_validate_api_key_rejects_missing_api_version(self):
        """Azure OpenAI validation without an API version is a silent False."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="az-key",
            azure_endpoint="https://my-resource.openai.azure.com",
            api_version=None,
            model="text-embedding-3-small",
        )

        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_custom_endpoint_validate_api_key_without_truncation_settings(self):
        """Custom endpoint validates connectivity with no truncation config."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key=None,
            base_url="http://localhost:8000",
            model="text-embedding-3-small",
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )
        provider._ensure_client = AsyncMock()

        assert provider.supported_dimensions == []
        assert await provider.validate_api_key() is True
        assert provider.native_dims == 768
        assert list(provider.supported_dimensions) == [768]

    @pytest.mark.asyncio
    async def test_unknown_model_validate_api_key_discovers_runtime_dims(self):
        """Probe should discover native dims for unknown models without truncation."""
        provider = self._unknown_model_provider()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        assert await provider.validate_api_key() is True
        assert provider.native_dims == 768

    @pytest.mark.asyncio
    async def test_unknown_model_validate_api_key_accepts_client_side_runtime_dims(
        self,
    ):
        """Validation should accept custom runtime native dims for client truncation."""
        provider = self._unknown_model_provider(
            output_dims=256,
            client_side_truncation=True,
        )
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        assert await provider.validate_api_key() is True
        assert provider.native_dims == 768

    @pytest.mark.asyncio
    async def test_unknown_model_validate_api_key_rejects_oversized_client_dims(
        self,
    ):
        """Validation should fail when runtime native dims are too small."""
        provider = self._unknown_model_provider(
            output_dims=1024,
            client_side_truncation=True,
        )
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_known_model_validate_api_key_success(self):
        """Known model with native dims passes validation."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=1536)
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is True

    @pytest.mark.asyncio
    async def test_known_model_validate_api_key_server_side_truncation(self):
        """Known model with output_dims passes validation when API truncates."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            output_dims=512,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=512)
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is True

    @pytest.mark.asyncio
    async def test_validate_api_key_handles_api_error_gracefully(self):
        """API errors during probe return False, not propagate."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_custom_endpoint_validate_api_key_trusts_runtime_dims(self):
        """Custom endpoint + client_side_truncation exercises trust-runtime path."""
        provider = self._custom_endpoint_provider(
            output_dims=256,
            client_side_truncation=True,
        )

        assert await provider.validate_api_key() is True
        assert provider.native_dims == 512

    @pytest.mark.asyncio
    async def test_validate_api_key_rejects_malformed_probe_response(self):
        """Probe must reject responses that do not contain exactly one embedding."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
        )
        malformed_response = MagicMock()
        malformed_response.data = [
            MagicMock(index=0, embedding=[0.1] * 1536),
            MagicMock(index=1, embedding=[0.1] * 1536),
        ]
        malformed_response.usage = MagicMock(total_tokens=10)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=malformed_response
        )
        provider._ensure_client = AsyncMock()

        assert await provider.validate_api_key() is False

    @pytest.mark.asyncio
    async def test_unknown_model_embed_rejects_ignored_server_truncation(
        self,
    ):
        """Unknown model server-side truncation still enforces returned dims."""
        from chunkhound.core.exceptions.embedding import EmbeddingDimensionError

        provider = self._unknown_model_provider(output_dims=256)
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        with pytest.raises(
            EmbeddingDimensionError,
            match="got 768, expected 256",
        ):
            await provider.embed(["hello"])

    @pytest.mark.parametrize("output_dims", [0, -1])
    def test_unknown_model_rejects_non_positive_output_dims_at_init(
        self, output_dims
    ):
        """Generic invalid output_dims still fail fast before any runtime probing."""
        with pytest.raises(
            EmbeddingConfigurationError,
            match="positive integer",
        ):
            self._unknown_model_provider(output_dims=output_dims)

    def test_unknown_model_rejects_non_int_output_dims_at_init(self):
        """String output_dims is a local config error, not a runtime probe."""
        with pytest.raises(
            EmbeddingConfigurationError,
            match="positive integer",
        ):
            self._unknown_model_provider(output_dims="256")

    def test_unknown_model_rejects_bool_output_dims_at_init(self):
        """Bool output_dims is rejected explicitly even on permissive custom paths."""
        with pytest.raises(
            EmbeddingConfigurationError,
            match="positive integer",
        ):
            self._unknown_model_provider(output_dims=True)

    def test_unknown_model_rejects_missing_output_dims_for_client_truncation_at_init(
        self,
    ):
        """Missing client-side target dims is a local config error, so fail fast."""
        with pytest.raises(
            EmbeddingConfigurationError,
            match="output_dims is not set",
        ):
            self._unknown_model_provider(client_side_truncation=True)


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


class TestProviderConfigMutation:
    """Contracts for provider update_config matryoshka semantics."""

    def test_openai_update_config_revalidates_model_changes_before_mutation(self):
        """Model changes must fail fast instead of leaving invalid output_dims behind."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            output_dims=256,
        )

        with pytest.raises(EmbeddingConfigurationError, match="does not support"):
            provider.update_config(model="text-embedding-ada-002")

        assert provider.model == "text-embedding-3-small"
        assert provider.output_dims == 256
        assert provider.dims == 256

    def test_openai_update_config_revalidates_base_url_changes_before_mutation(self):
        """Switching back to the official endpoint must re-apply OpenAI dims rules."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="text-embedding-ada-002",
            output_dims=512,
        )

        with pytest.raises(EmbeddingConfigurationError, match="does not support"):
            provider.update_config(base_url="https://api.openai.com/v1")

        assert provider.base_url == "http://localhost:8000"
        assert provider.output_dims == 512
        assert provider.dims == 512

    def test_openai_update_config_resets_runtime_dimension_introspection(self):
        """Runtime-discovered dims and fallback warnings must not leak across models."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="my-custom-embed-v1",
        )
        provider._discovered_native_dims = 768
        provider._warned_default_dims = True

        provider.update_config(model="my-custom-embed-v2")

        assert provider._discovered_native_dims is None
        assert provider._warned_default_dims is False
        assert provider.native_dims == 1536
        assert provider.supported_dimensions == []
        with patch(
            "chunkhound.providers.embeddings.openai_provider.logger.warning"
        ) as warn:
            assert provider.dims == 1536
        assert warn.call_count == 1

    def test_voyageai_update_config_revalidates_model_changes_before_mutation(self):
        """Voyage model changes must reject stale output_dims and keep prior config."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key",
            model="voyage-3.5",
            output_dims=512,
        )

        with pytest.raises(
            EmbeddingConfigurationError,
            match="not in supported dimensions",
        ):
            provider.update_config(model="voyage-2")

        assert provider.model == "voyage-3.5"
        assert provider.output_dims == 512
        assert provider.dims == 512

    def test_voyageai_update_config_resets_runtime_dimension_introspection(self):
        """Unknown-model native dim discovery must reset when the model changes."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key",
            model="acme-voyage-compatible-a",
            output_dims=256,
            client_side_truncation=True,
        )
        provider._discovered_native_dims = 777

        provider.update_config(model="acme-voyage-compatible-b")

        assert provider._discovered_native_dims is None
        assert provider.native_dims == 1024
        assert provider.supported_dimensions == []


class TestProviderConfigSnapshots:
    """Tests for provider config snapshots preserving matryoshka state."""

    def test_openai_config_snapshot_preserves_server_side_truncation(self):
        """OpenAI config snapshots keep output_dims in server-side mode."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            output_dims=256,
            base_url="http://localhost:8000",
        )

        snapshot = provider.config

        assert snapshot.provider == "openai"
        assert snapshot.dims == 256
        assert snapshot.output_dims == 256
        assert snapshot.client_side_truncation is False

    def test_openai_config_snapshot_preserves_client_side_truncation(self):
        """OpenAI config snapshots keep client-side truncation state."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            output_dims=256,
            client_side_truncation=True,
            base_url="http://localhost:8000",
        )

        snapshot = provider.config

        assert snapshot.dims == 256
        assert snapshot.output_dims == 256
        assert snapshot.client_side_truncation is True

    def test_voyageai_config_snapshot_preserves_client_side_truncation(self):
        """VoyageAI config snapshots keep matryoshka fields."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key",
            model="voyage-3.5",
            output_dims=512,
            client_side_truncation=True,
        )

        snapshot = provider.config

        assert snapshot.provider == "voyageai"
        assert snapshot.dims == 512
        assert snapshot.output_dims == 512
        assert snapshot.client_side_truncation is True

    def test_fake_provider_config_snapshot_round_trips_matryoshka_state(self):
        """Fake provider snapshots model the real provider contract."""
        from tests.fixtures.fake_providers import FakeEmbeddingProvider

        provider = FakeEmbeddingProvider(
            dims=1536,
            output_dims=512,
            client_side_truncation=True,
        )

        snapshot = provider.config

        assert snapshot.dims == 512
        assert snapshot.output_dims == 512
        assert snapshot.client_side_truncation is True


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

        result = build_dimension_request_param(
            output_dims=512,
            client_side_truncation=False,
        )
        assert result == 512

    def test_build_dimension_request_param_client_side(self):
        """Returns None when client-side truncation."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_dimension_request_param,
        )

        result = build_dimension_request_param(
            output_dims=512,
            client_side_truncation=True,
        )
        assert result is None

    def test_build_dimension_request_param_no_output_dims(self):
        """Returns None when output_dims is None."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_dimension_request_param,
        )

        result = build_dimension_request_param(
            output_dims=None,
            client_side_truncation=False,
        )
        assert result is None

        result = build_dimension_request_param(
            output_dims=None,
            client_side_truncation=True,
        )
        assert result is None

    @pytest.mark.parametrize(
        "native_dims,client_side_truncation,expected",
        [
            (None, False, []),
            (None, True, []),
            (768, False, [768]),
            (768, True, list(range(1, 769))),
            (1, True, [1]),
            (3, True, [1, 2, 3]),
        ],
    )
    def test_build_runtime_supported_dimensions(
        self, native_dims, client_side_truncation, expected
    ):
        """Contract: client truncation → full range, server truncation → native only."""
        from chunkhound.providers.embeddings.shared_utils import (
            build_runtime_supported_dimensions,
        )

        result = build_runtime_supported_dimensions(native_dims, client_side_truncation)
        assert list(result) == expected


class TestValidateRuntimeOutputDimsConfig:
    """Tests for validate_runtime_output_dims_config."""

    def test_none_without_client_truncation_returns_none(self):
        """Unset output_dims is valid when server-side truncation is disabled."""
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        assert validate_runtime_output_dims_config(None, False) is None

    def test_positive_int_returns_value(self):
        """A valid positive output_dims passes through unchanged."""
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        assert validate_runtime_output_dims_config(256, False) == 256

    def test_positive_int_with_client_truncation_returns_value(self):
        """Valid output_dims with client-side truncation passes through."""
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        assert validate_runtime_output_dims_config(256, True) == 256

    @pytest.mark.parametrize("output_dims", [0, -1, True, "256"])
    def test_invalid_values_raise_configuration_error(self, output_dims):
        """Non-positive or non-int output_dims fail explicitly."""
        from chunkhound.core.exceptions.embedding import EmbeddingConfigurationError
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        with pytest.raises(EmbeddingConfigurationError, match="positive integer"):
            validate_runtime_output_dims_config(output_dims, False)

    def test_client_side_truncation_requires_output_dims(self):
        """Client-side truncation without output_dims must fail."""
        from chunkhound.core.exceptions.embedding import EmbeddingConfigurationError
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        with pytest.raises(EmbeddingConfigurationError, match="output_dims is not set"):
            validate_runtime_output_dims_config(None, True, model="test-model")

    def test_context_appears_in_error_message(self):
        """The context parameter is included in the error message."""
        from chunkhound.core.exceptions.embedding import EmbeddingConfigurationError
        from chunkhound.providers.embeddings.shared_utils import (
            validate_runtime_output_dims_config,
        )

        with pytest.raises(EmbeddingConfigurationError, match="Set output_dims before using runtime truncation"):
            validate_runtime_output_dims_config(
                None, True, model="test-model", context="runtime truncation"
            )


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

    def test_client_side_truncation_requires_output_dims_at_init(self):
        """VoyageAI direct construction must fail fast without output_dims."""
        from chunkhound.core.exceptions.embedding import EmbeddingConfigurationError
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        with pytest.raises(
            EmbeddingConfigurationError,
            match="output_dims is not set",
        ):
            VoyageAIEmbeddingProvider(
                api_key="test-key",
                model="voyage-2",
                client_side_truncation=True,
            )

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

        vecs = asyncio.run(p.embed(["test text"]))
        assert len(vecs) == 1
        assert len(vecs[0]) == 512

    def test_client_side_truncation_returns_normalized_vectors(self):
        """FakeEmbeddingProvider returns truncated, L2-normalized vectors."""
        import asyncio

        from tests.fixtures.fake_providers import FakeEmbeddingProvider

        p = FakeEmbeddingProvider(
            dims=1536,
            output_dims=512,
            client_side_truncation=True,
        )
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
    async def test_unknown_model_warns_default_dims(self):
        """First access of dims for an unknown model logs a warning."""
        provider, _, mod = _bare_provider(model="my-custom-embed-v1")

        with patch.object(mod.logger, "warning") as mock_warn:
            dims = provider.dims

        assert dims == 1536
        mock_warn.assert_called_once()
        assert "default dims=1536" in mock_warn.call_args[0][0]


# ---------------------------------------------------------------------------
# Mocked embed() with matryoshka settings
# ---------------------------------------------------------------------------


class TestOpenAIMatryoshkaEmbed:
    """Test low-level _embed_batch_internal invariants kept outside public embed()."""

    @pytest.mark.asyncio
    async def test_server_side_truncation_embed(self):
        """Server-side truncation sends dimensions param to API."""
        provider, _, _ = _bare_provider(
            output_dims=512,
        )
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=512)
        )

        result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 512
        # Verify dimensions param was sent
        call_kwargs = provider._client.embeddings.create.call_args
        assert call_kwargs.kwargs.get("dimensions") == 512

    @pytest.mark.asyncio
    async def test_client_side_truncation_embed(self):
        """Client-side truncation truncates + L2-normalizes locally."""
        provider, _, _ = _bare_provider(
            output_dims=512,
            client_side_truncation=True,
        )
        provider._client = MagicMock()
        # API returns full-dimension vectors
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=1536)
        )

        result = await provider._embed_batch_internal(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 512
        # Verify no dimensions param was sent
        call_kwargs = provider._client.embeddings.create.call_args
        assert "dimensions" not in call_kwargs.kwargs
        # Verify L2-normalization
        magnitude = math.sqrt(sum(x * x for x in result[0]))
        assert abs(magnitude - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_inv1_violation_raises(self):
        """API returning wrong dims raises EmbeddingDimensionError."""

        provider, _, _ = _bare_provider(
            output_dims=512,
        )
        provider._client = MagicMock()
        # API returns wrong dims despite output_dims=512
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=768)
        )

        with pytest.raises(EmbeddingDimensionError, match="got 768, expected 512"):
            await provider._embed_batch_internal(["hello"])


class TestOpenAINonRetryableErrors:
    """Contract: EmbeddingProviderError subclasses propagate without retry."""

    @pytest.mark.asyncio
    async def test_embedding_dimension_error_propagates_without_retry(self):
        """EmbeddingDimensionError must propagate immediately, not retry."""

        provider, _, _ = _bare_provider(retry_attempts=3)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=EmbeddingDimensionError(
                "dimension mismatch: got 768, expected 512"
            )
        )

        with pytest.raises(EmbeddingDimensionError, match="dimension mismatch"):
            await provider._embed_batch_internal(["hello"])

        # Must fail on first attempt, not retry 3 times
        assert provider._client.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_embedding_configuration_error_propagates_without_retry(self):
        """EmbeddingConfigurationError must propagate immediately, not retry."""
        provider, _, _ = _bare_provider(retry_attempts=3)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=EmbeddingConfigurationError(
                "output_dims 999 not in supported dimensions"
            )
        )

        with pytest.raises(
            EmbeddingConfigurationError, match="not in supported dimensions"
        ):
            await provider._embed_batch_internal(["hello"])

        assert provider._client.embeddings.create.call_count == 1



class TestOpenAIEmbedErrorPropagation:
    """Contract: domain errors from embed() skip error stats and verbose logging."""

    @pytest.mark.asyncio
    async def test_dimension_error_does_not_increment_error_stat(self):
        """EmbeddingDimensionError must propagate without incrementing usage stats errors."""
        provider, _, _ = _bare_provider(retry_attempts=1)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=EmbeddingDimensionError(
                "dimension mismatch: got 768, expected 512"
            )
        )

        with pytest.raises(EmbeddingDimensionError, match="dimension mismatch"):
            await provider.embed(["hello"])

        assert provider.get_usage_stats()["errors"] == 0

    @pytest.mark.asyncio
    async def test_configuration_error_does_not_increment_error_stat(self):
        """EmbeddingConfigurationError must propagate without incrementing usage stats errors."""
        provider, _, _ = _bare_provider(retry_attempts=1)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=EmbeddingConfigurationError(
                "output_dims 999 not in supported dimensions"
            )
        )

        with pytest.raises(
            EmbeddingConfigurationError, match="not in supported dimensions"
        ):
            await provider.embed(["hello"])

        assert provider.get_usage_stats()["errors"] == 0

    @pytest.mark.asyncio
    async def test_transient_error_increments_error_stat(self):
        """Non-domain errors must increment usage stats errors."""
        provider, _, _ = _bare_provider(retry_attempts=1)
        provider._client = MagicMock()
        provider._client.embeddings.create = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with pytest.raises(RuntimeError, match="connection refused"):
            await provider.embed(["hello"])

        assert provider.get_usage_stats()["errors"] == 1


class TestConcurrentDimensionDiscovery:
    """Contract: concurrent embed() calls discover native dims without corruption."""

    @pytest.mark.asyncio
    async def test_concurrent_embed_calls_discover_same_native_dims(self):
        """Multiple concurrent embed() calls on an unknown model converge on the same native dims."""
        import asyncio

        provider, _, _ = _bare_provider(model="my-custom-embed-v1")
        provider._client = MagicMock()
        # All calls return 256-dim vectors
        provider._client.embeddings.create = AsyncMock(
            return_value=_ok_response(dim=256)
        )
        provider._ensure_client = AsyncMock()

        # Launch 5 concurrent embed calls
        results = await asyncio.gather(
            *[provider.embed([f"text {i}"]) for i in range(5)]
        )

        # All should return 256-dim vectors
        assert all(len(r[0]) == 256 for r in results)
        # Native dims should be discovered once, consistently
        assert provider.native_dims == 256
