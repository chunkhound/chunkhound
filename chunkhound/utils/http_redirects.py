"""Manual redirect mechanics, independent of response body consumption."""

from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_DEFAULT_PORTS = {"http": 80, "https": 443}


class RedirectError(ValueError):
    """A redirect refused with URL-free diagnostic text."""


def url_origin(url: str) -> tuple[str, str | None, int | None]:
    parts = urlsplit(url)
    port = parts.port if parts.port is not None else _DEFAULT_PORTS.get(parts.scheme)
    return parts.scheme, parts.hostname, port


class RedirectChain:
    """Track validated destinations without owning HTTP requests or responses."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str],
        *,
        validate: Callable[[str, str | None], None],
        max_redirects: int,
        missing_location_is_terminal: bool = False,
        strip_cross_origin_authorization: bool = False,
    ) -> None:
        validate(url, None)
        self.current_url = url
        self.headers = dict(headers)
        self._validate = validate
        self._remaining = max_redirects
        self._missing_location_is_terminal = missing_location_is_terminal
        self._strip_authorization = strip_cross_origin_authorization

    def advance(self, status_code: int, location: str | None) -> bool:
        """True means request the next hop; False means consume this response."""
        if status_code not in _REDIRECT_STATUSES:
            return False
        if not location:
            if self._missing_location_is_terminal:
                return False
            raise RedirectError("Redirect response has no location")
        if self._remaining == 0:
            raise RedirectError("Redirect limit exceeded")
        try:
            # Joining can repair a malformed absolute URL using the current host.
            # Validate explicit schemes first; relative references still resolve.
            if urlsplit(location).scheme:
                self._validate(location, self.current_url)
            target = urljoin(self.current_url, location)
            self._validate(target, self.current_url)
            if self._strip_authorization and url_origin(target) != url_origin(
                self.current_url
            ):
                self.headers = {
                    key: value
                    for key, value in self.headers.items()
                    if key.lower() != "authorization"
                }
        except RedirectError:
            raise
        except ValueError:
            raise RedirectError("Redirect destination is invalid") from None
        self._remaining -= 1
        self.current_url = target
        return True
