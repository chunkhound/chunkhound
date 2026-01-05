"""Test utilities shared across test files."""

import json
import os
from pathlib import Path


def _find_config_file() -> Path | None:
    """Find .chunkhound.json in current or parent directory."""
    search_paths = [
        Path(".chunkhound.json"),  # Current directory
        Path("../.chunkhound.json"),  # Parent directory
    ]

    for config_file in search_paths:
        if config_file.exists():
            return config_file
    return None


def get_api_key_for_tests() -> tuple[str | None, str | None]:
    """
    Intelligently discover API key and provider for testing.

    Priority:
    1. CHUNKHOUND_EMBEDDING__API_KEY environment variable
    2. .chunkhound.json in current or parent directory
    3. Return (None, None) if not found

    Returns:
        Tuple of (api_key, provider) or (None, None) if not found
    """
    # Priority 1: Environment variable (only if non-empty)
    api_key = os.environ.get("CHUNKHOUND_EMBEDDING__API_KEY")
    if api_key and api_key.strip():
        # Also check for provider from config
        config_file = _find_config_file()
        if config_file:
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                embedding_config = config_data.get("embedding", {})
                provider = embedding_config.get("provider")
                return api_key.strip(), provider
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                pass
        return api_key.strip(), None

    # Priority 2: Local .chunkhound.json file
    config_file = _find_config_file()
    if config_file:
        try:
            with open(config_file) as f:
                config_data = json.load(f)

            embedding_config = config_data.get("embedding", {})
            api_key = embedding_config.get("api_key")
            provider = embedding_config.get("provider")

            if api_key and api_key.strip():
                return api_key.strip(), provider

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    return None, None


def should_run_live_api_tests(expected_provider: str | None = None) -> bool:
    """Check if live API tests should run for the specified provider."""
    api_key, provider = get_api_key_for_tests()
    if not api_key:
        return False

    if expected_provider and provider != expected_provider:
        return False

    return True
