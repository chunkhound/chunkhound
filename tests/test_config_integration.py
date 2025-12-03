"""
Test configuration integration with registry system.

This module tests that configuration loading from .chunkhound.json files
integrates correctly with the registry system, ensuring embedding providers
are properly registered and available to services without producing warnings.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import gc
import time
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY
from chunkhound.registry import configure_registry, get_registry
from tests.utils.windows_compat import database_cleanup_context, cleanup_database_resources, windows_safe_tempdir


def _cleanup_registry_and_connections():
    """Clean up registry and database connections for Windows compatibility."""
    try:
        registry = get_registry()
        
        # Try to close database provider if it has a close method
        try:
            db_provider = registry.get_provider("database")
            if hasattr(db_provider, 'close'):
                db_provider.close()
            elif hasattr(db_provider, 'cleanup'):
                db_provider.cleanup()
            # For serial providers, try to close the underlying executor connection
            elif hasattr(db_provider, '_executor') and hasattr(db_provider._executor, '_connection'):
                if hasattr(db_provider._executor._connection, 'close'):
                    db_provider._executor._connection.close()
        except (ValueError, AttributeError):
            # No database provider or connection to clean up
            pass
            
        # Clear registry providers
        registry._providers.clear()
        registry._language_parsers.clear()
        registry._config = None
        
    except Exception:
        # Best effort cleanup - don't fail the test if cleanup fails
        pass
    
    # Force garbage collection to help with Windows file locking
    gc.collect()
    
    # On Windows, give a brief moment for file handles to be released
    if IS_WINDOWS:
        time.sleep(WINDOWS_FILE_HANDLE_DELAY)


def test_embedding_config_initializes_cleanly(clean_environment):
    """
    Test that valid embedding configuration from .chunkhound.json initializes without warnings.
    
    This test validates the integration between configuration loading and registry
    initialization, ensuring that:
    - Valid embedding configs are loaded correctly from JSON files
    - Registry initialization processes the config without emitting warnings
    - Embedding providers are properly registered and available to services
    
    This is a regression test for initialization order issues where services
    were created before embedding providers were registered.
    """
    with windows_safe_tempdir() as temp_path:
        
        # Create .chunkhound.json with valid embedding provider config
        config_path = temp_path / ".chunkhound.json" 
        db_path = temp_path / ".chunkhound" / "test.db"
        db_path.parent.mkdir(exist_ok=True)
        
        config_data = {
            "database": {
                "path": str(db_path),
                "provider": "duckdb"
            },
            "embedding": {
                "provider": "openai", 
                "base_url": "https://test-api-endpoint/v1",
                "api_key": "test-key-for-validation",
                "model": "test-embedding-model"
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        # Change to temp directory to simulate normal usage
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_path)
            
            # Load config using ChunkHound's configuration system
            config = Config()
            
            # Verify config loaded correctly
            assert config.embedding is not None
            assert config.embedding.provider == "openai"
            assert config.embedding.api_key.get_secret_value() == "test-key-for-validation"
            assert config.embedding.base_url == "https://test-api-endpoint/v1"
            assert config.embedding.model == "test-embedding-model"
            
            # Mock the provider to avoid network calls during testing
            with patch('chunkhound.providers.embeddings.openai_provider.OpenAIEmbeddingProvider') as mock_provider_class:
                mock_provider = MagicMock()
                mock_provider_class.return_value = mock_provider
                
                # Use database cleanup context for proper resource management
                with database_cleanup_context():
                    # Capture registry logger to check for warnings
                    with patch('chunkhound.registry.logger') as mock_logger:
                        # Configure registry - this should complete without warnings
                        configure_registry(config)
                        
                        # Check for any warning calls
                        warning_calls = [call for call in mock_logger.warning.call_args_list]
                        
                        # Look for provider-related warnings that indicate initialization issues
                        provider_warnings = [
                            call for call in warning_calls 
                            if call[0] and "No embedding provider configured" in str(call[0][0])
                        ]
                        
                        # Assert no provider warnings were emitted
                        assert len(provider_warnings) == 0, (
                            f"Valid embedding config should initialize without warnings, but got: "
                            f"{[str(call[0][0]) for call in provider_warnings]}"
                        )
                        
        finally:
            # Clean up database connections and registry before directory cleanup
            _cleanup_registry_and_connections()
            os.chdir(original_cwd)


def test_config_loading_from_json_file(clean_environment):
    """
    Test that .chunkhound.json files are properly loaded and parsed.
    
    This test validates the basic configuration loading mechanism to ensure
    JSON files are correctly processed and converted to Config objects.
    """
    with windows_safe_tempdir() as temp_path:
        
        # Create minimal valid config
        config_path = temp_path / ".chunkhound.json"
        config_data = {
            "embedding": {
                "provider": "openai",
                "api_key": "test-key",
                "model": "text-embedding-3-small"
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config_data, f)
        
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(temp_path)
            
            # Use database cleanup context for proper resource management
            with database_cleanup_context():
                # Load and verify config
                config = Config()
                
                assert config.embedding is not None
                assert config.embedding.provider == "openai"
                assert config.embedding.api_key.get_secret_value() == "test-key"
                assert config.embedding.model == "text-embedding-3-small"
                
        finally:
            # Clean up any registry state
            _cleanup_registry_and_connections()
            os.chdir(original_cwd)


def test_embedding_config_rerank_env_vars(monkeypatch, clean_environment):
    """
    Test that reranking environment variables are loaded correctly by load_from_env().

    This is a regression test for the bug where CHUNKHOUND_EMBEDDING__RERANK_*
    env vars were not loaded despite being documented in tests/RERANKING_TEST_SETUP.md.
    """
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_MODEL", "test-rerank-model")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_URL", "http://localhost:8080/rerank")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_FORMAT", "tei")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE", "64")

    config = EmbeddingConfig.load_from_env()

    assert config["rerank_model"] == "test-rerank-model"
    assert config["rerank_url"] == "http://localhost:8080/rerank"
    assert config["rerank_format"] == "tei"
    assert config["rerank_batch_size"] == 64


def test_embedding_config_rerank_batch_size_invalid_silently_ignored(monkeypatch, clean_environment):
    """
    Test that invalid rerank_batch_size env var is silently ignored.

    ChunkHound uses a silent-failure pattern for numeric env vars: invalid values
    are ignored rather than crashing config loading, allowing defaults to apply.
    This matches the pattern in indexing_config.py:386-389.
    """
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE", "not-a-number")

    config = EmbeddingConfig.load_from_env()

    # Invalid value should be silently ignored (not in config dict)
    assert "rerank_batch_size" not in config