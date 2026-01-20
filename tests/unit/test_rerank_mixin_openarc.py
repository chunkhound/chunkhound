"""Unit tests for OpenArc format support in RerankMixin."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from chunkhound.interfaces.embedding_provider import RerankResult
from chunkhound.providers.embeddings.rerank_mixin import RerankMixin


class MockProvider(RerankMixin):
    """Mock provider implementing RerankMixin for testing."""

    def __init__(
        self,
        rerank_model: str | None = None,
        rerank_format: str = "auto",
        rerank_url: str = "http://localhost:8080/v1/rerank",
        base_url: str | None = None,
    ):
        self.name = "mock"
        self._rerank_model = rerank_model
        self._rerank_format = rerank_format
        self._rerank_url = rerank_url
        self._base_url = base_url
        self._rerank_batch_size = None
        self._api_key = None
        self._timeout = 30
        self._retry_attempts = 3
        self._retry_delay = 1.0
        self._detected_rerank_format = None
        self._format_detection_lock = asyncio.Lock()

    def get_max_rerank_batch_size(self) -> int:
        """Return test batch size."""
        return 1000

    def _get_rerank_client_kwargs(self) -> dict:
        """Return empty kwargs for testing."""
        return {"timeout": self._timeout}


class TestOpenArcFormatParsing:
    """Tests for OpenArc response format parsing."""

    @pytest.mark.asyncio
    async def test_parse_openarc_response_basic(self):
        """Test parsing a basic OpenArc response."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "doc text 1", "score": 0.95, "index": 2},
                        {"document": "doc text 2", "score": 0.82, "index": 0},
                        {"document": "doc text 3", "score": 0.71, "index": 1},
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=3
        )

        assert len(results) == 3
        assert results[0].index == 2
        assert results[0].score == 0.95
        assert results[1].index == 0
        assert results[1].score == 0.82
        assert results[2].index == 1
        assert results[2].score == 0.71

    @pytest.mark.asyncio
    async def test_parse_openarc_response_auto_detect(self):
        """Test that OpenArc format is auto-detected and cached."""
        provider = MockProvider(rerank_format="auto")

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "test", "score": 0.9, "index": 0},
                    ]
                }
            ]
        }

        # Initially no format detected
        assert provider._detected_rerank_format is None

        await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=1
        )

        # Should now be detected as openarc
        assert provider._detected_rerank_format == "openarc"

    @pytest.mark.asyncio
    async def test_parse_openarc_response_validates_index_bounds(self):
        """Test that out-of-bounds indices are rejected."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "valid", "score": 0.9, "index": 0},
                        {
                            "document": "invalid",
                            "score": 0.8,
                            "index": 5,
                        },  # Out of bounds
                        {"document": "negative", "score": 0.7, "index": -1},  # Negative
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=3
        )

        # Should only have the valid result
        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_parse_openarc_response_missing_fields(self):
        """Test handling of results with missing required fields."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "valid", "score": 0.9, "index": 0},
                        {"document": "no score", "index": 1},  # Missing score
                        {"document": "no index", "score": 0.7},  # Missing index
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=3
        )

        # Should only have the valid result
        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_parse_openarc_response_invalid_types(self):
        """Test handling of results with invalid data types."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "valid", "score": 0.9, "index": 0},
                        {
                            "document": "bad index",
                            "score": 0.8,
                            "index": "not_a_number",
                        },
                        {"document": "bad score", "score": "not_a_float", "index": 2},
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=3
        )

        # Should only have the valid result
        assert len(results) == 1
        assert results[0].index == 0
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_parse_openarc_response_empty_ranked_documents(self):
        """Test handling of empty ranked_documents list."""
        provider = MockProvider()

        response_data = {"data": [{"ranked_documents": []}]}

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=5
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_parse_openarc_response_not_a_dict(self):
        """Test handling of non-dict items in ranked_documents."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "valid", "score": 0.9, "index": 0},
                        "not a dict",  # Invalid item
                        {"document": "also valid", "score": 0.7, "index": 1},
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=2
        )

        # Should skip the invalid item
        assert len(results) == 2
        assert results[0].index == 0
        assert results[1].index == 1


class TestOpenArcPayloadBuilding:
    """Tests for OpenArc request payload building."""

    def test_build_openarc_payload_basic(self):
        """Test building a basic OpenArc request payload."""
        provider = MockProvider(rerank_model="Qwen3-Reranker", rerank_format="openarc")

        payload = provider._build_rerank_payload(
            query="test query",
            documents=["doc1", "doc2", "doc3"],
            top_k=None,
            format_to_use="openarc",
        )

        assert payload == {
            "model": "Qwen3-Reranker",
            "query": "test query",
            "documents": ["doc1", "doc2", "doc3"],
        }

    def test_build_openarc_payload_ignores_top_k(self):
        """Test that OpenArc format doesn't include top_k in payload."""
        provider = MockProvider(rerank_model="Qwen3-Reranker", rerank_format="openarc")

        payload = provider._build_rerank_payload(
            query="test query",
            documents=["doc1", "doc2"],
            top_k=10,  # Should be ignored for OpenArc
            format_to_use="openarc",
        )

        # OpenArc doesn't support top_n in request (uses client-side filtering)
        assert "top_n" not in payload
        assert payload["model"] == "Qwen3-Reranker"

    def test_build_openarc_payload_requires_model(self):
        """Test that OpenArc format uses rerank_model."""
        provider = MockProvider(rerank_model="test-model", rerank_format="openarc")

        payload = provider._build_rerank_payload(
            query="query", documents=["doc"], top_k=None, format_to_use="openarc"
        )

        assert payload["model"] == "test-model"


class TestOpenArcFormatDetection:
    """Tests for OpenArc format auto-detection priority."""

    @pytest.mark.asyncio
    async def test_openarc_detected_before_cohere(self):
        """Test that OpenArc format is detected before trying Cohere format."""
        provider = MockProvider(rerank_format="auto", rerank_model="test-model")

        # OpenArc response structure
        openarc_response = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "test", "score": 0.9, "index": 0},
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            openarc_response, format_hint="auto", num_documents=1
        )

        # Should detect OpenArc format
        assert provider._detected_rerank_format == "openarc"
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_cohere_format_still_works(self):
        """Test that Cohere format still works when OpenArc is not detected."""
        provider = MockProvider(rerank_format="auto", rerank_model="test-model")

        # Cohere response structure (no "data" field)
        cohere_response = {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.82},
            ]
        }

        results = await provider._parse_rerank_response(
            cohere_response, format_hint="auto", num_documents=3
        )

        # Should detect Cohere format
        assert provider._detected_rerank_format == "cohere"
        assert len(results) == 2
        assert results[0].index == 2
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_tei_format_still_works(self):
        """Test that TEI format still works when OpenArc is not detected."""
        provider = MockProvider(rerank_format="auto")

        # TEI response structure (no "data" field, uses "score" not "relevance_score")
        tei_response = {
            "results": [
                {"index": 1, "score": 0.88},
                {"index": 0, "score": 0.75},
            ]
        }

        results = await provider._parse_rerank_response(
            tei_response, format_hint="auto", num_documents=2
        )

        # Should detect TEI format
        assert provider._detected_rerank_format == "tei"
        assert len(results) == 2
        assert results[0].index == 1
        assert results[0].score == 0.88


class TestOpenArcEdgeCases:
    """Tests for edge cases in OpenArc format handling."""

    @pytest.mark.asyncio
    async def test_openarc_invalid_ranked_documents_type(self):
        """Test handling when ranked_documents is not a list."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": "not a list"  # Invalid type
                }
            ]
        }

        with pytest.raises(ValueError, match="'ranked_documents' must be a list"):
            await provider._parse_rerank_response(
                response_data, format_hint="auto", num_documents=3
            )

    @pytest.mark.asyncio
    async def test_openarc_no_ranked_documents_key(self):
        """Test handling when data list item doesn't have ranked_documents."""
        provider = MockProvider()

        # No ranked_documents key - should fall through to standard validation
        response_data = {"data": [{"some_other_key": "value"}]}

        with pytest.raises(ValueError, match="missing 'results' field"):
            await provider._parse_rerank_response(
                response_data, format_hint="auto", num_documents=3
            )

    @pytest.mark.asyncio
    async def test_openarc_empty_data_list(self):
        """Test handling when data list is empty."""
        provider = MockProvider()

        # Empty data list - should fall through to standard validation
        response_data = {"data": []}

        with pytest.raises(ValueError, match="missing 'results' field"):
            await provider._parse_rerank_response(
                response_data, format_hint="auto", num_documents=3
            )

    @pytest.mark.asyncio
    async def test_openarc_data_not_list(self):
        """Test handling when data field is not a list."""
        provider = MockProvider()

        # data is not a list - should fall through to standard validation
        response_data = {"data": "not a list"}

        with pytest.raises(ValueError, match="missing 'results' field"):
            await provider._parse_rerank_response(
                response_data, format_hint="auto", num_documents=3
            )

    @pytest.mark.asyncio
    async def test_openarc_zero_documents(self):
        """Test handling when num_documents is zero."""
        provider = MockProvider()

        response_data = {
            "data": [
                {
                    "ranked_documents": [
                        {"document": "test", "score": 0.9, "index": 0},
                    ]
                }
            ]
        }

        results = await provider._parse_rerank_response(
            response_data, format_hint="auto", num_documents=0
        )

        # Should return empty list with warning
        assert len(results) == 0
