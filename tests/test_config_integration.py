"""
Test configuration integration with registry system.

This module tests that configuration loading from .chunkhound.json files
integrates correctly with the registry system, ensuring embedding providers
are properly registered and available to services without producing warnings.
"""

import json
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import gc
import time
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.llm_config import LLMConfig
from chunkhound.utils.windows_constants import IS_WINDOWS, WINDOWS_FILE_HANDLE_DELAY
from chunkhound.registry import configure_registry, get_registry
from tests.utils.windows_compat import (
    cleanup_database_resources,
    database_cleanup_context,
    paths_equal,
    windows_safe_tempdir,
)


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
    env vars were not loaded despite being part of the supported test/runtime config surface.
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


def test_embedding_config_ssl_env_vars(monkeypatch, clean_environment):
    """SSL verification env vars should parse into booleans."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__SSL_VERIFY", "false")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_SSL_VERIFY", "true")

    config = EmbeddingConfig.load_from_env()

    assert config["ssl_verify"] is False
    assert config["rerank_ssl_verify"] is True


@pytest.mark.parametrize("raw_value", ["", "maybe", "2", "truthy"])
def test_embedding_config_ssl_verify_rejects_invalid_env_bool(
    monkeypatch, clean_environment, raw_value
):
    """Invalid embedding SSL env bools must fail explicitly."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__SSL_VERIFY", raw_value)

    with pytest.raises(ValueError, match="CHUNKHOUND_EMBEDDING__SSL_VERIFY"):
        EmbeddingConfig.load_from_env()


@pytest.mark.parametrize("raw_value", ["", "maybe", "2", "truthy"])
def test_embedding_config_rerank_ssl_verify_rejects_invalid_env_bool(
    monkeypatch, clean_environment, raw_value
):
    """Invalid rerank SSL env bools must fail explicitly."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__RERANK_SSL_VERIFY", raw_value)

    with pytest.raises(
        ValueError, match="CHUNKHOUND_EMBEDDING__RERANK_SSL_VERIFY"
    ):
        EmbeddingConfig.load_from_env()


def test_llm_config_ssl_env_var(monkeypatch, clean_environment):
    """LLM ssl_verify env var should parse into a boolean."""
    monkeypatch.setenv("CHUNKHOUND_LLM_SSL_VERIFY", "false")

    config = LLMConfig.load_from_env()

    assert config["ssl_verify"] is False


def test_gemini_env_model_satisfies_runtime_config_validation(
    monkeypatch, clean_environment
) -> None:
    """CHUNKHOUND_LLM_MODEL must satisfy Gemini's explicit-model contract."""
    monkeypatch.setenv("CHUNKHOUND_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("CHUNKHOUND_LLM_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("CHUNKHOUND_LLM_API_KEY", "sk-test")

    with windows_safe_tempdir() as temp_path:
        config = Config(target_dir=temp_path)

    assert config.llm is not None
    assert config.llm.model == "gemini-3.5-flash"
    assert config.validate_for_command("research") == []

    utility_config, synthesis_config = config.llm.get_provider_configs()
    assert utility_config["model"] == "gemini-3.5-flash"
    assert synthesis_config["model"] == "gemini-3.5-flash"


def test_gemini_legacy_env_model_does_not_satisfy_runtime_config_validation(
    monkeypatch, clean_environment
) -> None:
    """CHUNKHOUND_GEMINI_MODEL is removed; only CHUNKHOUND_LLM_MODEL is valid."""
    monkeypatch.setenv("CHUNKHOUND_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("CHUNKHOUND_GEMINI_MODEL", "gemini-3.5-flash")
    monkeypatch.setenv("CHUNKHOUND_LLM_API_KEY", "sk-test")

    with windows_safe_tempdir() as temp_path:
        config = Config(target_dir=temp_path)

    assert config.llm is not None
    assert config.llm.model is None
    assert config.validate_for_command("research") == [
        "Missing required configuration: llm.explicit model selection required "
        "for gemini roles: utility, synthesis"
    ]


def test_embedding_cli_ssl_flags_parse() -> None:
    """Embedding CLI should expose explicit positive/negative SSL flags."""
    parser = argparse.ArgumentParser()
    EmbeddingConfig.add_cli_arguments(parser)

    args = parser.parse_args(["--no-ssl-verify", "--no-rerank-ssl-verify"])

    assert args.ssl_verify is False
    assert args.rerank_ssl_verify is False


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


def test_embedding_config_legacy_env_vars(monkeypatch, clean_environment):
    """Test that single-underscore legacy env vars are read as fallbacks."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_API_KEY", "legacy-key")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_BASE_URL", "http://legacy-url")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_PROVIDER", "legacy-provider")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_MODEL", "legacy-model")

    config = EmbeddingConfig.load_from_env()

    assert config["api_key"] == "legacy-key"
    assert config["base_url"] == "http://legacy-url"
    assert config["provider"] == "legacy-provider"
    assert config["model"] == "legacy-model"


def test_embedding_config_new_env_vars_take_precedence(monkeypatch, clean_environment):
    """Test that canonical double-underscore env vars override legacy single-underscore vars."""
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__API_KEY", "canonical-key")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_API_KEY", "legacy-key")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING__MODEL", "canonical-model")
    monkeypatch.setenv("CHUNKHOUND_EMBEDDING_MODEL", "legacy-model")

    config = EmbeddingConfig.load_from_env()

    assert config["api_key"] == "canonical-key"
    assert config["model"] == "canonical-model"


def test_azure_api_version_rejects_malformed() -> None:
    """Malformed api_version like 'latest' must fail at config validation time."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        EmbeddingConfig(
            azure_endpoint="https://myresource.openai.azure.com",
            api_key="sk-test",
            api_version="latest",
        )


def test_azure_api_version_accepts_valid_formats() -> None:
    """Standard and preview api_version formats must pass validation."""
    cfg = EmbeddingConfig(
        azure_endpoint="https://myresource.openai.azure.com",
        api_key="sk-test",
        api_version="2024-02-01",
    )
    assert cfg.api_version == "2024-02-01"

    cfg2 = EmbeddingConfig(
        azure_endpoint="https://myresource.openai.azure.com",
        api_key="sk-test",
        api_version="2024-02-01-preview",
    )
    assert cfg2.api_version == "2024-02-01-preview"

    cfg3 = EmbeddingConfig(
        azure_endpoint="https://myresource.openai.azure.com",
        api_key="sk-test",
        api_version="2024-10-01-preview2",
    )
    assert cfg3.api_version == "2024-10-01-preview2"


def test_local_config_overrides_env_vars(monkeypatch, clean_environment):
    """Local .chunkhound.json must override environment variables.

    Config precedence (high to low): CLI > explicit --config > local > global > env > defaults.
    This is a regression test for the reorder in config.py where local config
    was moved after env vars so it wins.
    """
    # Env var sets provider to grok
    monkeypatch.setenv("CHUNKHOUND_LLM_PROVIDER", "grok")

    with windows_safe_tempdir() as temp_path:
        # Local config sets provider to openai
        config_path = temp_path / ".chunkhound.json"
        config_data = {
            "llm": {
                "provider": "openai",
                "api_key": "test-key",
            },
            "indexing": {"exclude": ["from-local/**"]},
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = Config(target_dir=temp_path)

        assert config.llm is not None
        assert config.llm.provider == "openai", (
            "Local .chunkhound.json should override env vars"
        )
        assert "from-local/**" in config.indexing.exclude


def test_cli_overrides_local_config(clean_environment):
    """CLI arguments must override local .chunkhound.json.

    Config precedence (high to low): CLI > explicit --config > local > global > env > defaults.
    Complementary to test_local_config_overrides_env_vars.
    """
    from types import SimpleNamespace

    with windows_safe_tempdir() as temp_path:
        config_path = temp_path / ".chunkhound.json"
        config_data = {
            "llm": {
                "provider": "openai",
                "api_key": "test-key",
            },
            "indexing": {"exclude": ["from-local/**"]},
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        args = SimpleNamespace(
            llm_provider="grok",
            path=None,
            config=None,
            exclude=["from-cli/**"],
        )
        config = Config(args, target_dir=temp_path)

        assert config.llm is not None
        assert config.llm.provider == "grok", (
            "CLI --llm-provider should override local .chunkhound.json"
        )
        assert "from-cli/**" in config.indexing.exclude


def test_explicit_config_file_overrides_local_config(clean_environment):
    """Explicit config files must win over auto-discovered local config."""
    from types import SimpleNamespace

    with windows_safe_tempdir() as temp_path:
        local_config_path = temp_path / ".chunkhound.json"
        explicit_config_path = temp_path / "alt-config.json"

        with open(local_config_path, "w") as f:
            json.dump(
                {
                    "llm": {"provider": "openai", "api_key": "local-key"},
                    "indexing": {"exclude": ["from-local/**"]},
                },
                f,
            )

        with open(explicit_config_path, "w") as f:
            json.dump(
                {
                    "llm": {"provider": "grok", "api_key": "explicit-key"},
                    "indexing": {"exclude": ["from-explicit/**"]},
                },
                f,
            )

        args = SimpleNamespace(
            command="index",
            config=str(explicit_config_path),
            path=str(temp_path),
        )
        config = Config(args)

        assert config.llm is not None
        assert config.llm.provider == "grok", (
            "Explicit --config should override local .chunkhound.json"
        )
        assert "from-explicit/**" in config.indexing.exclude


def test_env_config_file_overrides_local_config(monkeypatch, clean_environment):
    """CHUNKHOUND_CONFIG_FILE must behave like an explicit config file."""
    with windows_safe_tempdir() as temp_path:
        local_config_path = temp_path / ".chunkhound.json"
        explicit_config_path = temp_path / "env-config.json"

        with open(local_config_path, "w") as f:
            json.dump(
                {
                    "llm": {"provider": "openai", "api_key": "local-key"},
                    "indexing": {"exclude": ["from-local/**"]},
                },
                f,
            )

        with open(explicit_config_path, "w") as f:
            json.dump(
                {
                    "llm": {"provider": "grok", "api_key": "env-key"},
                    "indexing": {"exclude": ["from-env-file/**"]},
                },
                f,
            )

        monkeypatch.setenv("CHUNKHOUND_CONFIG_FILE", str(explicit_config_path))

        config = Config(target_dir=temp_path)

        assert config.llm is not None
        assert config.llm.provider == "grok", (
            "CHUNKHOUND_CONFIG_FILE should override local .chunkhound.json"
        )
        assert "from-env-file/**" in config.indexing.exclude


def test_map_command_uses_config_parent_for_workspace_root(clean_environment):
    """`map` must treat explicit config files as the workspace root, not args.path."""
    from types import SimpleNamespace

    with windows_safe_tempdir() as temp_path:
        workspace_root = temp_path / "workspace"
        docs_scope = temp_path / "docs-scope"
        workspace_root.mkdir()
        docs_scope.mkdir()

        with open(workspace_root / "map-config.json", "w") as f:
            json.dump({"llm": {"provider": "grok", "api_key": "workspace-key"}}, f)

        with open(docs_scope / ".chunkhound.json", "w") as f:
            json.dump({"llm": {"provider": "openai", "api_key": "scope-key"}}, f)

        args = SimpleNamespace(
            command="map",
            config=str(workspace_root / "map-config.json"),
            path=str(docs_scope),
        )

        config = Config(args)

        assert config.target_dir == workspace_root.resolve()
        assert config.llm is not None
        assert config.llm.provider == "grok"
        assert config.database.path == workspace_root.resolve() / ".chunkhound" / "db"


def test_global_config_provides_defaults_and_is_overridden_by_local(monkeypatch, clean_environment):
    """Global config (via CHUNKHOUND_GLOBAL_CONFIG_FILE) supplies cross-project defaults.

    Local .chunkhound.json and explicit config override globals (deep merge).
    """
    with windows_safe_tempdir() as td:
        td = Path(td)
        global_cfg = td / "global.json"
        global_cfg.write_text(
            json.dumps(
                {
                    "embedding": {"provider": "openai", "api_key": "gkey", "model": "gmodel"},
                    "indexing": {"batch_size": 42},
                }
            )
        )
        local_cfg = td / ".chunkhound.json"
        local_cfg.write_text(json.dumps({"embedding": {"api_key": "lkey"}}))

        monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(global_cfg))

        c = Config(target_dir=td)
        assert paths_equal(c.global_config_file, global_cfg)
        assert paths_equal(c.local_config_file, local_cfg)
        assert c.embedding is not None
        assert c.embedding.api_key.get_secret_value() == "lkey"  # local overrides
        assert c.embedding.model == "gmodel"  # from global
        assert c.indexing.batch_size == 42  # global not overridden by local


def test_global_local_deep_merge_partial_overrides_and_list_replacement(
    monkeypatch, clean_environment
):
    """Global + local demonstrate deep merge for objects and list replacement.

    Covers the user-facing contract for global config merging:
    - Partial overrides inside embedding/llm/research (specific keys only;
      siblings from global survive).
    - Full provider/model switch possible by supplying a sub-dict.
    - Project exclude list replaces global's (raw level).
    - Built-in defaults still appear in effective excludes.
    - Untouched global values survive.
    """

    with windows_safe_tempdir() as td:
        td = Path(td)
        global_cfg = td / "global.json"
        global_cfg.write_text(
            json.dumps(
                {
                    "embedding": {
                        "provider": "voyageai",
                        "model": "voyage-3.5",
                        "api_key": "g-voyage-key",
                    },
                    "llm": {
                        "provider": "anthropic",
                        "model": "claude-3-5-sonnet-20241022",
                        "api_key": "g-anthropic-key",
                    },
                    "research": {
                        "algorithm": "v2",
                        "query_expansion_enabled": True,
                        "num_expanded_queries": 3,
                    },
                    "indexing": {
                        "batch_size": 77,
                        "exclude": ["**/global-exclude-pattern/**"],
                    },
                }
            )
        )
        local_cfg = td / ".chunkhound.json"
        local_cfg.write_text(
            json.dumps(
                {
                    # Partial: only the secret (provider + model from global survive)
                    "embedding": {"api_key": "project-voyage-key"},
                    # Different model for this project (provider from global survives)
                    "llm": {"model": "claude-3-opus-20240229"},
                    # Research tuning override (other research fields from global survive)
                    "research": {"algorithm": "v3"},
                    # Project-specific excludes fully replace global's list
                    "indexing": {
                        "exclude": ["**/my-project-vendor/**", "**/generated/**"]
                    },
                }
            )
        )

        monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(global_cfg))

        c = Config(target_dir=td)

        # embedding: partial override works, global siblings preserved
        assert c.embedding is not None
        assert c.embedding.provider == "voyageai"  # from global
        assert c.embedding.model == "voyage-3.5"  # from global
        assert c.embedding.api_key.get_secret_value() == "project-voyage-key"  # local

        # llm: partial model override
        assert c.llm is not None
        assert c.llm.provider == "anthropic"  # from global
        assert c.llm.model == "claude-3-opus-20240229"  # local override

        # research: partial algorithm override, other fields from global
        assert c.research.algorithm == "v3"  # local
        assert c.research.query_expansion_enabled is True  # global
        assert c.research.num_expanded_queries == 3  # global

        # indexing scalar from global survives when local only touches exclude
        assert c.indexing.batch_size == 77  # global

        # exclude list: local fully replaces global's
        assert c.indexing.exclude == ["**/my-project-vendor/**", "**/generated/**"]
        # but effective excludes still include built-in defaults
        effective = c.indexing.get_effective_config_excludes()
        assert any("my-project-vendor" in p for p in effective)
        assert any("node_modules" in p for p in effective)  # built-in default present
        # global's list is gone from the raw exclude (replaced)
        assert not any(
            "global-exclude-pattern" in p for p in c.indexing.exclude
        )

        # global_config_file / local_config_file still recorded
        assert paths_equal(c.global_config_file, global_cfg)
        assert paths_equal(c.local_config_file, local_cfg)
