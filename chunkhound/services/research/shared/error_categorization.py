"""Error categorization utilities for research operations.

Categorizes exceptions into standard types for structured failure tracking.
"""

import asyncio


def categorize_error(exception: Exception) -> str:
    """Categorize exception into standard error type.

    Args:
        exception: Exception to categorize

    Returns:
        Error category: timeout, api_error, validation, network, or unknown
    """
    # Timeout errors
    if isinstance(exception, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exception, TimeoutError):
        return "timeout"

    # Network errors
    if "ConnectionError" in type(exception).__name__:
        return "network"
    if "HTTPError" in type(exception).__name__:
        return "network"
    if "ConnectTimeout" in type(exception).__name__:
        return "network"
    if "ReadTimeout" in type(exception).__name__:
        return "network"

    # API errors (rate limits, auth, etc.)
    exc_str = str(exception).lower()
    if "rate limit" in exc_str or "429" in exc_str:
        return "api_error"
    if "unauthorized" in exc_str or "401" in exc_str or "403" in exc_str:
        return "api_error"
    if "api key" in exc_str:
        return "api_error"
    if "quota" in exc_str or "exceeded" in exc_str:
        return "api_error"

    # Validation errors
    if isinstance(exception, ValueError):
        return "validation"
    if isinstance(exception, TypeError):
        return "validation"
    if isinstance(exception, KeyError):
        return "validation"
    if "ValidationError" in type(exception).__name__:
        return "validation"
    if "json" in exc_str and ("parse" in exc_str or "decode" in exc_str):
        return "validation"

    # Unknown/other
    return "unknown"
