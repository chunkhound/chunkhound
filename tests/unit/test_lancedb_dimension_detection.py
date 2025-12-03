"""Unit tests for LanceDB dimension detection edge cases."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from chunkhound.embeddings import EmbeddingManager


class TestLanceDBDimensionDetection:
    """Test dimension detection from embedding manager."""

    def test_get_dimensions_no_manager(self, tmp_path):
        """Edge case: No embedding_manager"""
        db_path = tmp_path / "test.lancedb"
        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=None
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_no_provider_registered(self, tmp_path):
        """Edge case: embedding_manager with no providers registered"""
        db_path = tmp_path / "test.lancedb"
        em = EmbeddingManager()
        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_provider_no_dims_attribute(self, tmp_path):
        """Edge case: Provider missing .dims attribute"""
        db_path = tmp_path / "test.lancedb"

        # Create mock provider without dims attribute
        mock_provider = MagicMock(spec=['name', 'model'])  # No dims in spec
        mock_provider.name = "test"
        mock_provider.model = "test-model"

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_provider_dims_raises_exception(self, tmp_path):
        """Edge case: Provider.dims raises exception"""
        db_path = tmp_path / "test.lancedb"

        # Create mock provider where accessing dims raises exception
        mock_provider = MagicMock(spec=['name', 'model', 'dims'])
        mock_provider.name = "test"
        mock_provider.model = "test-model"
        type(mock_provider).dims = PropertyMock(side_effect=RuntimeError("API error"))

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_invalid_type_string(self, tmp_path):
        """Edge case: Provider returns dims as string instead of int"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.model = "test-model"
        mock_provider.dims = "1536"  # String instead of int

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_invalid_value_zero(self, tmp_path):
        """Edge case: Provider returns dims = 0"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.model = "test-model"
        mock_provider.dims = 0  # Invalid

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_invalid_value_negative(self, tmp_path):
        """Edge case: Provider returns negative dims"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.model = "test-model"
        mock_provider.dims = -1  # Invalid

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims is None

    def test_get_dimensions_valid_common_dimension(self, tmp_path):
        """Happy path: Provider with valid common dimension (1536)"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.model = "text-embedding-3-small"
        mock_provider.dims = 1536

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims == 1536

    def test_get_dimensions_valid_alternative_dimension(self, tmp_path):
        """Happy path: Provider with valid alternative dimension (768)"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "ollama"
        mock_provider.model = "nomic-embed-text"
        mock_provider.dims = 768

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims == 768

    def test_get_dimensions_valid_large_dimension(self, tmp_path):
        """Happy path: Provider with valid large dimension (4096)"""
        db_path = tmp_path / "test.lancedb"

        mock_provider = MagicMock()
        mock_provider.name = "custom"
        mock_provider.model = "large-model"
        mock_provider.dims = 4096

        em = EmbeddingManager()
        em.register_provider(mock_provider, set_default=True)

        provider = LanceDBProvider(
            db_path,
            tmp_path,
            embedding_manager=em
        )
        dims = provider._get_embedding_dimensions_safe()
        assert dims == 4096
