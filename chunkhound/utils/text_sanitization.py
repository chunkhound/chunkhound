"""Text sanitization utilities for error messages and logs."""

import re

# Default patterns to redact (secrets, tokens, cookies)
_REDACT_PATTERNS = [
    r"(?i)(authorization\s*:\s*bearer\s+)[A-Za-z0-9._-]+",
    r"(?i)(api[_-]?key\s*[=:]\s*)([A-Za-z0-9-_]{10,})",
    r"(?i)(secret|token)[\s=:]+([A-Za-z0-9._-]{10,})",
    r"(?i)(set-cookie\s*:\s*)([^;\n]+)",
]


def sanitize_error_text(text: str, max_length: int = 800) -> str:
    """Truncate and redact secrets from error text.

    Args:
        text: Raw error text
        max_length: Maximum output length (default 800)

    Returns:
        Sanitized text with secrets redacted and length limited
    """
    if not text:
        return ""

    # Truncate
    result = text if len(text) <= max_length else (text[:max_length] + "...[truncated]")

    # Redact secrets
    for pattern in _REDACT_PATTERNS:
        result = re.sub(pattern, r"\1[REDACTED]", result)

    return result
