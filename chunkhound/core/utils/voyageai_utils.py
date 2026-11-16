"""Utilities for VoyageAI API endpoint detection."""

OFFICIAL_VOYAGEAI_BASE = "https://api.voyageai.com"
OFFICIAL_VOYAGEAI_BASE_V1 = f"{OFFICIAL_VOYAGEAI_BASE}/v1"


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

    # Require an exact host match or a path boundary ("/") after the base so a
    # look-alike host such as "https://api.voyageai.com.evil.com" is not treated
    # as official (mirrors is_official_openai_endpoint's spoofing guard).
    return base_url == OFFICIAL_VOYAGEAI_BASE or base_url.startswith(
        OFFICIAL_VOYAGEAI_BASE + "/"
    )
