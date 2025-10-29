"""Tests for Qwen model detection and batch size configuration."""

import pytest

from chunkhound.providers.embeddings.openai_provider import (
    QWEN_MODEL_CONFIG,
    OpenAIEmbeddingProvider,
)


class TestQwenModelDetection:
    """Test automatic detection and configuration of Qwen models."""

    def test_qwen_embedding_model_detected(self):
        """Test that Qwen embedding models are detected and configured."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="dengcao/Qwen3-Embedding-8B:Q5_K_M",
            batch_size=512,  # Request large batch
        )

        # Should limit to model max (128 for 8B model)
        assert provider._batch_size == 128
        assert provider._qwen_model_config is not None
        assert provider._qwen_model_config["max_texts_per_batch"] == 128

    def test_qwen_reranker_model_detected(self):
        """Test that Qwen reranker models are detected."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-8b",
        )

        # Embedding model is not Qwen
        assert provider._qwen_model_config is None

        # But rerank model is Qwen
        assert provider._qwen_rerank_config is not None
        assert provider._qwen_rerank_config["max_rerank_batch"] == 64

    def test_qwen_0_6b_high_batch_size(self):
        """Test that Qwen 0.6B model allows larger batch sizes."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="qwen3-embedding-0.6b",
            batch_size=1000,
        )

        # Should limit to model max (512 for 0.6B model)
        assert provider._batch_size == 512

    def test_qwen_4b_medium_batch_size(self):
        """Test that Qwen 4B model uses medium batch sizes."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="dengcao/Qwen3-Embedding-4B:Q5_K_M",
            batch_size=512,
        )

        # Should limit to model max (256 for 4B model)
        assert provider._batch_size == 256

    def test_non_qwen_model_uses_user_batch_size(self):
        """Test that non-Qwen models use the user-requested batch size."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            batch_size=512,
        )

        # Should use user-requested batch size
        assert provider._batch_size == 512
        assert provider._qwen_model_config is None

    def test_qwen_batch_size_respects_user_lower_limit(self):
        """Test that user batch size is used if lower than model limit."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="dengcao/Qwen3-Embedding-8B:Q5_K_M",
            batch_size=64,  # Lower than model max of 128
        )

        # Should use user's lower value
        assert provider._batch_size == 64

    def test_get_max_rerank_batch_size_qwen_reranker(self):
        """Test get_max_rerank_batch_size() returns Qwen-specific limits."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-4b",
        )

        max_batch = provider.get_max_rerank_batch_size()
        assert max_batch == 96  # Qwen 4B reranker limit

    def test_get_max_rerank_batch_size_qwen_embedding_fallback(self):
        """Test that rerank batch size falls back to embedding model config."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="dengcao/Qwen3-Embedding-8B:Q5_K_M",
            batch_size=100,
        )

        max_batch = provider.get_max_rerank_batch_size()
        assert max_batch == 64  # Qwen 8B embedding model's rerank limit

    def test_get_max_rerank_batch_size_default(self):
        """Test default rerank batch size for non-Qwen models."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            batch_size=200,
        )

        max_batch = provider.get_max_rerank_batch_size()
        # Should use min(batch_size, 128) as default
        assert max_batch == 128


class TestQwenModelConfig:
    """Test the QWEN_MODEL_CONFIG dictionary structure."""

    def test_all_qwen_models_have_required_fields(self):
        """Verify all Qwen model configs have required fields."""
        required_fields = {
            "max_tokens_per_batch",
            "max_texts_per_batch",
            "context_length",
            "max_rerank_batch",
        }

        for model_name, config in QWEN_MODEL_CONFIG.items():
            assert all(
                field in config for field in required_fields
            ), f"Model {model_name} missing required fields"

    def test_qwen_reranker_models_exist(self):
        """Verify Qwen reranker models are in config."""
        reranker_models = [
            "qwen3-reranker-0.6b",
            "qwen3-reranker-4b",
            "qwen3-reranker-8b",
        ]

        for model in reranker_models:
            assert (
                model in QWEN_MODEL_CONFIG or f"fireworks/{model}" in QWEN_MODEL_CONFIG
            ), f"Reranker model {model} not found in config"

    def test_qwen_embedding_models_exist(self):
        """Verify Qwen embedding models are in config."""
        embedding_models = [
            "qwen3-embedding-0.6b",
            "qwen3-embedding-4b",
            "qwen3-embedding-8b",
        ]

        for model in embedding_models:
            assert (
                model in QWEN_MODEL_CONFIG
                or f"dengcao/Qwen3-Embedding-{model.split('-')[-1].upper()}:Q5_K_M"
                in QWEN_MODEL_CONFIG
            ), f"Embedding model {model} not found in config"

    def test_batch_sizes_inversely_proportional_to_model_size(self):
        """Verify larger models have smaller batch sizes."""
        # Embedding models
        qwen_0_6b = QWEN_MODEL_CONFIG.get("qwen3-embedding-0.6b")
        qwen_8b = QWEN_MODEL_CONFIG.get("qwen3-embedding-8b")

        if qwen_0_6b and qwen_8b:
            assert (
                qwen_0_6b["max_texts_per_batch"] > qwen_8b["max_texts_per_batch"]
            ), "Smaller model should have larger batch size"

        # Reranker models
        rerank_0_6b = QWEN_MODEL_CONFIG.get("qwen3-reranker-0.6b")
        rerank_8b = QWEN_MODEL_CONFIG.get("qwen3-reranker-8b")

        if rerank_0_6b and rerank_8b:
            assert (
                rerank_0_6b["max_rerank_batch"] > rerank_8b["max_rerank_batch"]
            ), "Smaller reranker should have larger batch size"


@pytest.mark.asyncio
class TestQwenBatchSplitting:
    """Test batch splitting behavior with Qwen models."""

    async def test_rerank_single_batch_no_splitting(self):
        """Test that small document sets don't trigger batch splitting."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-8b",
        )

        # Mock the _rerank_single_batch method to track calls
        call_count = 0
        original_rerank = provider._rerank_single_batch

        async def mock_rerank(query, docs, top_k):
            nonlocal call_count
            call_count += 1
            # Return mock results
            from chunkhound.interfaces.embedding_provider import RerankResult

            results = [RerankResult(index=i, score=1.0 - i * 0.1) for i in range(len(docs))]
            # Respect top_k if provided
            if top_k is not None:
                return results[:top_k]
            return results

        provider._rerank_single_batch = mock_rerank

        # Small document set (< max_batch_size of 64 for qwen3-reranker-8b)
        documents = [f"Document {i}" for i in range(30)]
        query = "test query"

        results = await provider.rerank(query, documents, top_k=10)

        # Should only call _rerank_single_batch once (no splitting)
        assert call_count == 1
        assert len(results) == 10  # Respects top_k

    async def test_rerank_multiple_batches_with_splitting(self):
        """Test that large document sets trigger batch splitting."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-8b",  # max_batch = 64
        )

        # Mock the _rerank_single_batch method
        call_count = 0
        batch_sizes = []

        async def mock_rerank(query, docs, top_k):
            nonlocal call_count
            call_count += 1
            batch_sizes.append(len(docs))

            from chunkhound.interfaces.embedding_provider import RerankResult

            return [RerankResult(index=i, score=1.0 - i * 0.01) for i in range(len(docs))]

        provider._rerank_single_batch = mock_rerank

        # Large document set (> max_batch_size of 64)
        documents = [f"Document {i}" for i in range(150)]
        query = "test query"

        results = await provider.rerank(query, documents)

        # Should split into 3 batches: 64, 64, 22
        assert call_count == 3
        assert batch_sizes == [64, 64, 22]
        assert len(results) == 150  # All documents ranked

    async def test_rerank_index_adjustment_across_batches(self):
        """Test that indices are correctly adjusted across batches."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-8b",
        )

        async def mock_rerank(query, docs, top_k):
            from chunkhound.interfaces.embedding_provider import RerankResult

            # Return results with indices 0, 1, 2... relative to batch
            return [RerankResult(index=i, score=1.0) for i in range(len(docs))]

        provider._rerank_single_batch = mock_rerank

        # 100 documents: will split into batches of 64 and 36
        documents = [f"Document {i}" for i in range(100)]
        query = "test query"

        results = await provider.rerank(query, documents)

        # Check that indices span full range
        indices = [r.index for r in results]
        assert min(indices) == 0
        assert max(indices) == 99
        assert len(set(indices)) == 100  # All unique indices

    async def test_rerank_empty_documents_list(self):
        """Test that empty document list returns empty results."""
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:11434/v1",
            model="text-embedding-3-small",
            rerank_model="qwen3-reranker-8b",
        )

        results = await provider.rerank("test query", [], top_k=10)
        assert results == []
