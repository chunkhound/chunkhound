"""Utilities for OpenAI API endpoint detection."""


def is_azure_openai_endpoint(azure_endpoint: str | None) -> bool:
    """
    Check if this is an Azure OpenAI endpoint.

    Azure OpenAI endpoints follow the pattern:
    https://<resource-name>.openai.azure.com

    Args:
        azure_endpoint: The Azure endpoint URL to check, or None

    Returns:
        True if this is a valid Azure OpenAI endpoint
    """
    if not azure_endpoint:
        return False

    # Normalize URL for comparison
    endpoint = azure_endpoint.lower().rstrip("/")

    # Azure OpenAI endpoints must:
    # 1. Start with https://
    # 2. End with .openai.azure.com (to prevent URLs like https://evil.com/openai.azure.com/)
    return endpoint.startswith("https://") and endpoint.endswith(".openai.azure.com")


def is_official_openai_endpoint(base_url: str | None) -> bool:
    """
    Determine if a base URL points to the official OpenAI API.

    Args:
        base_url: The base URL to check, or None for default OpenAI endpoint

    Returns:
        True if this is an official OpenAI endpoint requiring API key authentication
    """
    if not base_url:
        # No base_url means default OpenAI endpoint
        return True

    # Check if URL starts with official OpenAI domain
    return base_url.startswith("https://api.openai.com") and (
        base_url == "https://api.openai.com"
        or base_url.startswith("https://api.openai.com/")
    )
