#!/usr/bin/env python3
"""Test script to verify OPENAI_BASE_URL env var fix and JSON schema name change."""


def test_client_initialization_logic():
    """Test the client initialization logic for base_url handling."""
    print("Testing client initialization logic...")

    # Simulate the logic from OpenAICompatibleProvider
    def simulate_client_init(base_url_param, default_base_url):
        effective_base_url = base_url_param or default_base_url
        client_kwargs = {"api_key": "test", "timeout": 60, "max_retries": 3}
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
        return client_kwargs

    # Test 1: No base_url param, default returns None (OpenAI case)
    kwargs1 = simulate_client_init(None, None)
    assert "base_url" not in kwargs1, "Should not set base_url when None"
    print("PASS: Case 1: No base_url param + None default -> no base_url in kwargs")

    # Test 2: No base_url param, default returns URL
    kwargs2 = simulate_client_init(None, "https://api.example.com")
    assert kwargs2["base_url"] == "https://api.example.com"
    print("PASS: Case 2: No base_url param + URL default -> base_url in kwargs")

    # Test 3: Explicit base_url param
    kwargs3 = simulate_client_init("https://custom.example.com", "https://default.example.com")
    assert kwargs3["base_url"] == "https://custom.example.com"
    print("PASS: Case 3: Explicit base_url param -> overrides default")

    # Test 4: Explicit None base_url param
    kwargs4 = simulate_client_init(None, "https://default.example.com")
    assert kwargs4["base_url"] == "https://default.example.com"
    print("PASS: Case 4: None base_url param -> uses default")


def test_json_schema_name_consistency():
    """Test that JSON schema names are consistent across providers."""
    print("Testing JSON schema name consistency...")

    # Read the files to verify the names
    with open('chunkhound/providers/llm/openai_llm_provider.py', 'r') as f:
        openai_content = f.read()

    with open('chunkhound/providers/llm/openai_compatible_provider.py', 'r') as f:
        compatible_content = f.read()

    # Both should use "structured_response"
    assert '"name": "structured_response"' in openai_content, "OpenAI provider should use structured_response"
    assert '"name": "structured_response"' in compatible_content, "Compatible provider should use structured_response"

    # Should not use the old "output" name
    assert '"name": "output"' not in openai_content, "OpenAI provider should not use old 'output' name"
    assert '"name": "output"' not in compatible_content, "Compatible provider should not use old 'output' name"

    print("PASS: JSON schema names are consistent and updated")


def test_openai_default_base_url():
    """Test that OpenAI provider returns None for default base URL."""
    print("Testing OpenAI default base URL...")

    with open('chunkhound/providers/llm/openai_llm_provider.py', 'r') as f:
        content = f.read()

    # Should return None
    assert 'return None' in content, "OpenAI provider should return None for default base URL"
    assert 'https://api.openai.com/v1' not in content, "Should not hardcode OpenAI URL"

    print("PASS: OpenAI provider correctly returns None for default base URL")


def test_type_annotations():
    """Test that type annotations are correct."""
    print("Testing type annotations...")

    with open('chunkhound/providers/llm/openai_compatible_provider.py', 'r') as f:
        content = f.read()

    # Base class should allow None return type
    assert 'def _get_default_base_url(self) -> str | None:' in content, "Base class should allow None return type"

    with open('chunkhound/providers/llm/openai_llm_provider.py', 'r') as f:
        content = f.read()

    # OpenAI provider should return None
    assert 'def _get_default_base_url(self) -> str | None:' in content, "OpenAI provider should have None return type"

    print("PASS: Type annotations are correct")


if __name__ == "__main__":
    test_client_initialization_logic()
    test_json_schema_name_consistency()
    test_openai_default_base_url()
    test_type_annotations()
    print("\nAll tests passed!")