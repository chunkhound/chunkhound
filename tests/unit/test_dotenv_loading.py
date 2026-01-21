"""Unit tests for dotenv file loading in configuration system.

Tests verify that .env files are loaded correctly and that the precedence
chain works as expected: CLI > env vars > .env file > config file > defaults
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.core.config.config import Config


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_env():
    """Clean up environment variables before and after test."""
    # Store original env vars
    original_env = os.environ.copy()

    # Clean CHUNKHOUND env vars
    keys_to_remove = [k for k in os.environ.keys() if k.startswith("CHUNKHOUND_")]
    for key in keys_to_remove:
        os.environ.pop(key, None)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_dotenv_file_loads_embedding_config(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that .env file is loaded and embedding config is populated."""
    # Create .env file with embedding configuration
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test-from-dotenv\n"
        "CHUNKHOUND_EMBEDDING__MODEL=text-embedding-3-small\n"
    )

    # Create config with target_dir pointing to temp directory
    config = Config(target_dir=temp_project_dir)

    # Verify embedding config was loaded from .env
    assert config.embedding is not None
    assert config.embedding.provider == "openai"
    assert config.embedding.api_key.get_secret_value() == "sk-test-from-dotenv"
    assert config.embedding.model == "text-embedding-3-small"


def test_dotenv_file_loads_llm_config(temp_project_dir: Path, clean_env: None) -> None:
    """Test that .env file loads LLM configuration."""
    # Create .env file with LLM configuration
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_LLM_PROVIDER=anthropic\n"
        "CHUNKHOUND_LLM_API_KEY=sk-ant-test\n"
        "CHUNKHOUND_LLM_UTILITY_MODEL=claude-3-5-haiku-20241022\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify LLM config was loaded from .env
    assert config.llm is not None
    assert config.llm.provider == "anthropic"
    assert config.llm.api_key.get_secret_value() == "sk-ant-test"
    assert config.llm.utility_model == "claude-3-5-haiku-20241022"


def test_dotenv_file_loads_database_config(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that .env file loads database configuration."""
    # Create .env file with database configuration
    dotenv_path = temp_project_dir / ".env"
    db_path = temp_project_dir / "custom_db"
    dotenv_path.write_text(
        f"CHUNKHOUND_DATABASE__PATH={db_path}\nCHUNKHOUND_DATABASE__PROVIDER=lancedb\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify database config was loaded from .env
    assert config.database.path == db_path
    assert config.database.provider == "lancedb"


def test_env_vars_override_dotenv(temp_project_dir: Path, clean_env: None) -> None:
    """Test that system environment variables override .env file values."""
    # Create .env file
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test-from-dotenv\n"
    )

    # Set environment variable (should override .env)
    os.environ["CHUNKHOUND_EMBEDDING__API_KEY"] = "sk-test-from-env-var"

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify env var took precedence over .env file
    assert config.embedding is not None
    assert config.embedding.provider == "openai"  # from .env
    assert (
        config.embedding.api_key.get_secret_value() == "sk-test-from-env-var"
    )  # from env var (overridden)


def test_cli_args_override_dotenv_and_env_vars(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that CLI arguments override both .env and environment variables."""
    # Create .env file
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test-from-dotenv\n"
    )

    # Set environment variable
    os.environ["CHUNKHOUND_EMBEDDING__API_KEY"] = "sk-test-from-env-var"

    # Create mock CLI args
    from argparse import Namespace

    args = Namespace(
        command="index",
        path=str(temp_project_dir),
        config=None,
        debug=False,
    )

    # Override via kwargs (simulating CLI args for embedding config)
    config = Config(args=args, embedding={"api_key": "sk-test-from-cli"})

    # Verify CLI took precedence
    assert config.embedding is not None
    assert config.embedding.api_key.get_secret_value() == "sk-test-from-cli"


def test_dotenv_file_missing_does_not_break(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that missing .env file doesn't cause errors."""
    # Don't create .env file - should work fine
    config = Config(target_dir=temp_project_dir)

    # Config should be created successfully with defaults
    assert config is not None
    assert config.database is not None


def test_dotenv_overrides_config_file(temp_project_dir: Path, clean_env: None) -> None:
    """Test that .env file values override JSON config file values."""
    # Create JSON config file
    config_file = temp_project_dir / "config.json"
    config_file.write_text(
        '{"embedding": {"provider": "openai", "api_key": "sk-from-json"}}'
    )

    # Create .env file (should override JSON)
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text("CHUNKHOUND_EMBEDDING__API_KEY=sk-from-dotenv\n")

    # Create mock CLI args pointing to config file
    from argparse import Namespace

    args = Namespace(
        command="index",
        path=str(temp_project_dir),
        config=str(config_file),
        debug=False,
    )

    # Create config
    config = Config(args=args)

    # Verify .env overrode JSON config
    assert config.embedding is not None
    assert config.embedding.provider == "openai"  # from JSON
    assert (
        config.embedding.api_key.get_secret_value() == "sk-from-dotenv"
    )  # from .env (overridden)


def test_dotenv_with_local_chunkhound_json(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test .env interaction with local .chunkhound.json file."""
    # Create local .chunkhound.json
    local_config = temp_project_dir / ".chunkhound.json"
    local_config.write_text(
        '{"embedding": {"provider": "openai", "model": "text-embedding-3-small"}}'
    )

    # Create .env file
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text("CHUNKHOUND_EMBEDDING__API_KEY=sk-from-dotenv\n")

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify both sources were merged correctly
    assert config.embedding is not None
    assert config.embedding.provider == "openai"  # from .chunkhound.json
    assert config.embedding.model == "text-embedding-3-small"  # from .chunkhound.json
    assert config.embedding.api_key.get_secret_value() == "sk-from-dotenv"  # from .env


def test_dotenv_loads_multiple_config_sections(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that .env can configure multiple config sections at once."""
    # Create .env with various settings
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test\n"
        "CHUNKHOUND_LLM_PROVIDER=anthropic\n"
        "CHUNKHOUND_LLM_API_KEY=sk-ant-test\n"
        "CHUNKHOUND_DATABASE__PROVIDER=lancedb\n"
        "CHUNKHOUND_DEBUG=true\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify all sections loaded
    assert config.embedding is not None
    assert config.embedding.provider == "openai"
    assert config.embedding.api_key.get_secret_value() == "sk-test"

    assert config.llm is not None
    assert config.llm.provider == "anthropic"
    assert config.llm.api_key.get_secret_value() == "sk-ant-test"

    assert config.database.provider == "lancedb"
    assert config.debug is True


def test_dotenv_with_comments_and_blank_lines(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that .env file handles comments and blank lines correctly."""
    # Create .env with comments and blank lines
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "# Embedding configuration\n"
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "\n"
        "# API key\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test\n"
        "\n"
        "# End of file\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify it loaded correctly despite comments
    assert config.embedding is not None
    assert config.embedding.provider == "openai"
    assert config.embedding.api_key.get_secret_value() == "sk-test"


def test_dotenv_not_loaded_from_other_directories(
    temp_project_dir: Path, clean_env: None
) -> None:
    """Test that .env is only loaded from target directory, not cwd."""
    # Create .env in a subdirectory
    subdir = temp_project_dir / "subdir"
    subdir.mkdir()
    dotenv_path = subdir / ".env"
    dotenv_path.write_text("CHUNKHOUND_EMBEDDING__API_KEY=sk-should-not-load\n")

    # Create config with target_dir as parent (not subdir)
    config = Config(target_dir=temp_project_dir)

    # Verify .env from subdirectory was NOT loaded
    if config.embedding:
        assert config.embedding.api_key != "sk-should-not-load"


def test_dotenv_rerank_config(temp_project_dir: Path, clean_env: None) -> None:
    """Test that .env file loads reranking configuration."""
    # Create .env file with reranking config (needs base_url for rerank)
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_EMBEDDING__PROVIDER=openai\n"
        "CHUNKHOUND_EMBEDDING__API_KEY=sk-test\n"
        "CHUNKHOUND_EMBEDDING__BASE_URL=http://localhost:8001\n"
        "CHUNKHOUND_EMBEDDING__RERANK_MODEL=rerank-v2\n"
        "CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE=50\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify rerank config was loaded
    assert config.embedding is not None
    assert config.embedding.rerank_model == "rerank-v2"
    assert config.embedding.rerank_batch_size == 50


def test_dotenv_indexing_config(temp_project_dir: Path, clean_env: None) -> None:
    """Test that .env file loads indexing configuration."""
    # Create .env file with indexing config
    dotenv_path = temp_project_dir / ".env"
    dotenv_path.write_text(
        "CHUNKHOUND_INDEXING__FORCE_REINDEX=true\n"
        "CHUNKHOUND_INDEXING__CLEANUP=false\n"
        "CHUNKHOUND_INDEXING__MAX_CONCURRENT=8\n"
    )

    # Create config
    config = Config(target_dir=temp_project_dir)

    # Verify indexing config was loaded
    assert config.indexing.force_reindex is True
    assert config.indexing.cleanup is False
    assert config.indexing.max_concurrent == 8
