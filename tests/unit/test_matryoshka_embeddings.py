"""Tests for matryoshka embedding support.

Covers:
- output_dims configuration validation
- Provider protocol properties (native_dims, supported_dimensions, supports_matryoshka)
- Model whitelist validation
- Dimension boundary testing
"""

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.interfaces.embedding_provider import EmbeddingConfigurationError
from tests.fixtures.fake_providers import FakeEmbeddingProvider


def _voyageai_available() -> bool:
    """Check if VoyageAI SDK is available."""
    import importlib.util

    return importlib.util.find_spec("voyageai") is not None


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


class TestFakeProviderMatryoshkaProtocol:
    """Test FakeEmbeddingProvider implements matryoshka protocol."""

    def test_dims_default(self):
        """dims returns default value."""
        provider = FakeEmbeddingProvider(dims=1536)
        assert provider.dims == 1536

    def test_dims_with_output_dims(self):
        """dims returns output_dims when set."""
        provider = FakeEmbeddingProvider(dims=1536, output_dims=512)
        assert provider.dims == 512

    def test_native_dims(self):
        """native_dims returns the native dimension."""
        provider = FakeEmbeddingProvider(dims=1536, output_dims=512)
        assert provider.native_dims == 1536

    def test_supported_dimensions(self):
        """supported_dimensions returns valid dimensions."""
        provider = FakeEmbeddingProvider(dims=1536)
        assert provider.supported_dimensions == [1536]

    def test_supports_matryoshka(self):
        """supports_matryoshka returns False for fake provider."""
        provider = FakeEmbeddingProvider()
        assert provider.supports_matryoshka() is False


class TestModelWhitelist:
    """Test model whitelist validation."""

    def test_valid_openai_model(self):
        """Valid OpenAI model passes validation."""
        config = EmbeddingConfig(provider="openai", model="text-embedding-3-small")
        assert config.model == "text-embedding-3-small"

    def test_valid_voyageai_model(self):
        """Valid VoyageAI model passes validation."""
        config = EmbeddingConfig(provider="voyageai", model="voyage-3.5")
        assert config.model == "voyage-3.5"

    def test_unknown_model_with_custom_base_url_allowed(self):
        """Unknown model with custom base_url is allowed."""
        config = EmbeddingConfig(
            provider="openai",
            model="custom-model",
            base_url="http://localhost:8000/v1",
        )
        assert config.model == "custom-model"

    def test_unknown_openai_model_raises_error(self):
        """Unknown OpenAI model on official API raises EmbeddingConfigurationError."""
        with pytest.raises(EmbeddingConfigurationError, match="Unknown model"):
            EmbeddingConfig(provider="openai", model="invalid-model")

    def test_unknown_voyageai_model_raises_error(self):
        """Unknown VoyageAI model on official API raises EmbeddingConfigurationError."""
        with pytest.raises(EmbeddingConfigurationError, match="Unknown model"):
            EmbeddingConfig(provider="voyageai", model="invalid-model")


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

    def test_invalid_output_dims_raises_error(self):
        """Invalid output_dims for non-matryoshka model raises error."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(EmbeddingConfigurationError, match="does not support"):
            OpenAIEmbeddingProvider(
                api_key="test-key", model="text-embedding-ada-002", output_dims=512
            )

    def test_output_dims_out_of_range_raises_error(self):
        """output_dims outside valid range raises EmbeddingConfigurationError."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        with pytest.raises(EmbeddingConfigurationError, match="out of range"):
            OpenAIEmbeddingProvider(
                api_key="test-key",
                model="text-embedding-3-small",
                output_dims=2000,  # Exceeds 1536 native dims
            )

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


class TestVoyageAIProviderMatryoshka:
    """Test VoyageAI provider matryoshka support."""

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_35_supports_matryoshka(self):
        """voyage-3.5 supports matryoshka with discrete dimensions."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(api_key="test-key", model="voyage-3.5")
        assert provider.supports_matryoshka() is True
        assert provider.native_dims == 2048
        assert provider.supported_dimensions == [256, 512, 1024, 2048]

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_35_output_dims_valid(self):
        """voyage-3.5 accepts valid output_dims from discrete set."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key", model="voyage-3.5", output_dims=512
        )
        assert provider.dims == 512
        assert provider.native_dims == 2048

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_35_output_dims_invalid_raises_error(self):
        """voyage-3.5 rejects output_dims not in discrete set."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        with pytest.raises(EmbeddingConfigurationError, match="not in supported dimensions"):
            VoyageAIEmbeddingProvider(
                api_key="test-key",
                model="voyage-3.5",
                output_dims=768,  # Not in [256, 512, 1024, 2048]
            )

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_finance_no_matryoshka(self):
        """voyage-finance-2 does not support matryoshka (single dimension)."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(
            api_key="test-key", model="voyage-finance-2"
        )
        assert provider.supports_matryoshka() is False
        assert provider.supported_dimensions == [1024]


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

    def test_fake_provider_vector_length_matches_dims(self):
        """FakeEmbeddingProvider generates vectors matching dims property."""
        import asyncio

        provider = FakeEmbeddingProvider(dims=1536)
        embeddings = asyncio.run(provider.embed(["test text"]))
        assert len(embeddings[0]) == 1536

    def test_fake_provider_vector_length_matches_output_dims(self):
        """FakeEmbeddingProvider generates vectors matching output_dims."""
        import asyncio

        provider = FakeEmbeddingProvider(dims=1536, output_dims=512)
        embeddings = asyncio.run(provider.embed(["test text"]))
        assert len(embeddings[0]) == 512


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


class TestVoyage3ModelConfig:
    """Test voyage-3 model configuration (Gap 1 fix)."""

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_3_in_model_config(self):
        """voyage-3 exists in VOYAGE_MODEL_CONFIG."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VOYAGE_MODEL_CONFIG,
        )

        assert "voyage-3" in VOYAGE_MODEL_CONFIG

    @pytest.mark.skipif(
        not _voyageai_available(),
        reason="Requires VoyageAI SDK",
    )
    def test_voyage_3_no_matryoshka(self):
        """voyage-3 does not support matryoshka (single dimension)."""
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        provider = VoyageAIEmbeddingProvider(api_key="test-key", model="voyage-3")
        assert provider.supports_matryoshka() is False
        assert provider.supported_dimensions == [1024]
        assert provider.native_dims == 1024
        assert provider.dims == 1024


class TestRerankerWhitelist:
    """Test reranker model whitelist validation (Gap 2 fix)."""

    def test_valid_voyageai_reranker_passes(self):
        """Valid VoyageAI reranker model passes validation."""
        config = EmbeddingConfig(
            provider="voyageai", model="voyage-3.5", rerank_model="rerank-2.5"
        )
        assert config.rerank_model == "rerank-2.5"

    def test_valid_voyageai_reranker_lite_passes(self):
        """Valid VoyageAI reranker-lite model passes validation."""
        config = EmbeddingConfig(
            provider="voyageai", model="voyage-3.5", rerank_model="rerank-2.5-lite"
        )
        assert config.rerank_model == "rerank-2.5-lite"

    def test_unknown_voyageai_reranker_raises_error(self):
        """Unknown VoyageAI reranker raises EmbeddingConfigurationError."""
        with pytest.raises(EmbeddingConfigurationError, match="Unknown reranker model"):
            EmbeddingConfig(
                provider="voyageai", model="voyage-3.5", rerank_model="invalid-reranker"
            )

    def test_unknown_reranker_with_custom_base_url_allowed(self):
        """Unknown reranker with custom base_url is allowed."""
        config = EmbeddingConfig(
            provider="voyageai",
            model="voyage-3.5",
            rerank_model="custom-reranker",
            base_url="http://localhost:8000",  # Custom endpoint bypasses whitelist
        )
        assert config.rerank_model == "custom-reranker"

    def test_openai_provider_accepts_any_reranker(self):
        """OpenAI provider accepts any reranker (no whitelist for OpenAI rerankers)."""
        # OpenAI doesn't have a reranker whitelist, so any model is accepted
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            rerank_model="any-reranker",
            base_url="http://localhost:8000",  # Custom endpoint to bypass embed whitelist
        )
        assert config.rerank_model == "any-reranker"


class TestDimensionValidation:
    """Test post-embed dimension validation (Gap 3 fix)."""

    def test_fake_provider_dimension_mismatch_detected(self):
        """Dimension mismatch in returned embeddings should be detectable."""
        # This test verifies the FakeEmbeddingProvider generates correct dims
        # Real dimension validation testing requires mocking the API response
        import asyncio

        provider = FakeEmbeddingProvider(dims=1536, output_dims=512)
        embeddings = asyncio.run(provider.embed(["test text"]))
        # FakeEmbeddingProvider respects output_dims
        assert len(embeddings[0]) == 512

    def test_embedding_dimension_error_exists(self):
        """EmbeddingDimensionError is importable and usable."""
        from chunkhound.interfaces.embedding_provider import EmbeddingDimensionError

        error = EmbeddingDimensionError("API returned wrong dimension")
        assert "wrong dimension" in str(error)


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

    def test_provider_stores_flag(self):
        """OpenAIEmbeddingProvider stores client_side_truncation flag."""
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            output_dims=512,
            client_side_truncation=True,
        )
        assert provider._client_side_truncation is True
        assert provider._output_dims == 512
