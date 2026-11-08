"""URL redaction for log messages.

Credentials, query strings, and fragments must not reach logs. A rejected URL
can break a URL parser, so this helper uses a regex and a safe fallback.
"""

import re


def url_for_log(value: str) -> str:
    """Redact before escaping: rejected URLs cannot safely use a URL parser."""
    endpoint = re.split(r"[?#]", value, maxsplit=1)[0]
    # A delimiter can hide userinfo from endpoint. If the raw URL has @, do
    # not echo the pre-delimiter text because it can hold credentials.
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
