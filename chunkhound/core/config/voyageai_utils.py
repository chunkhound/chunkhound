"""Utilities for VoyageAI API endpoint detection."""

OFFICIAL_VOYAGEAI_BASE = "https://api.voyageai.com"


def is_official_voyageai_endpoint(base_url: str | None) -> bool:
    """
    Determine if a base URL points to the official VoyageAI API.

    Args:
        base_url: The base URL to check, or None for default VoyageAI endpoint

    Returns:
        True if this is an official VoyageAI endpoint requiring API key authentication
    """
    if not base_url:
        # No base_url means default VoyageAI endpoint
        return True

    return base_url.startswith(OFFICIAL_VOYAGEAI_BASE)
