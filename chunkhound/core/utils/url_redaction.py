"""URL redaction for log messages.

Credentials, query strings, and fragments must not reach logs. A rejected URL
can break a URL parser, so this helper uses a regex and a safe fallback.
"""

import re


def has_url_ambiguous_characters(value: str) -> bool:
    """Backslash or whitespace makes the effective host parser-dependent."""
    return "\\" in value or any(char.isspace() for char in value)


def url_for_log(value: str) -> str:
    """Redact before escaping: rejected URLs cannot safely use a URL parser."""
    endpoint = re.split(r"[?#]", value, maxsplit=1)[0]
    # A delimiter can hide the only userinfo @. The prefix before it may still
    # hold credentials, so withhold it rather than trust its shape.
    if "@" in value and "@" not in endpoint:
        return ascii("<invalid URL>")
    authority = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*:)?//([^/]*)(.*)$", endpoint, re.S)
    if authority and "@" not in authority[3]:
        scheme, host, path = authority.groups()
        endpoint = f"{scheme or ''}//{host.rsplit('@', 1)[-1]}{path}"
    elif "@" in endpoint:
        # Malformed authorities cannot distinguish credentials from a path.
        endpoint = "<invalid URL>"
    return ascii(endpoint)
