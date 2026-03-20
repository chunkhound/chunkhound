"""Environment variable detection for ChunkHound setup wizard"""

import os
from typing import Any

from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def detect_provider_config() -> dict[str, dict[str, Any] | None]:
    """
    Detect provider configurations from environment variables.

    Returns:
        Dictionary mapping provider names to their detected configuration
    """
    configs: dict[str, dict[str, Any] | None] = {}

    # Check VoyageAI
    voyage_config = _detect_voyageai()
    if voyage_config:
        configs["voyageai"] = voyage_config

    # Check OpenAI
    openai_config = _detect_openai()
    if openai_config:
        configs["openai"] = openai_config

    # Check common local endpoints
    local_config = _detect_local_endpoints()
    if local_config:
        configs["local"] = local_config

    return configs


def _detect_voyageai() -> dict[str, Any] | None:
    """Detect VoyageAI configuration from environment."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if api_key:
        return {
            "provider": "voyageai",
            "api_key": api_key,
            "model": VOYAGE_DEFAULT_MODEL,
        }
    return None


def _detect_openai() -> dict[str, Any] | None:
    """Detect OpenAI configuration from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        config = {
            "provider": "openai",
            "api_key": api_key,
            "model": os.getenv("OPENAI_MODEL", "text-embedding-3-small"),
        }

        # Add base URL if specified
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        if base_url:
            config["base_url"] = base_url

        # Add organization if specified
        organization = os.getenv("OPENAI_ORGANIZATION") or os.getenv("OPENAI_ORG_ID")
        if organization:
            config["organization"] = organization

        return config
    return None


def _detect_local_endpoints() -> dict[str, Any] | None:
    """Check for running local LLM servers."""
    # Common local endpoints to check
    endpoints = [
        ("http://localhost:11434/v1", "Ollama"),
        ("http://localhost:1234/v1", "LM Studio"),
        ("http://localhost:8000/v1", "vLLM"),
        ("http://localhost:5000/v1", "Local API"),
        ("http://127.0.0.1:11434/v1", "Ollama"),
        ("http://127.0.0.1:1234/v1", "LM Studio"),
    ]

    # Check endpoints from common environment variables
    env_endpoints = []

    # Check Ollama-specific variables
    ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_API_BASE")
    if ollama_host:
        ollama_url = _normalize_endpoint_url(ollama_host)
        if ollama_url:
            env_endpoints.append((ollama_url, "Ollama (from env)"))

    # Check generic OpenAI base URL for local endpoints
    openai_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if openai_base and _is_local_url(openai_base):
        provider_name = _guess_provider_from_url(openai_base)
        env_endpoints.append((openai_base, f"{provider_name} (from OPENAI_BASE_URL)"))

    # Test environment variables first (higher priority)
    for url, name in env_endpoints:
        if _check_endpoint_alive(url):
            return {
                "base_url": url,
                "provider_name": name,
                "detected_from": "environment",
            }

    # Then test common endpoints
    for url, name in endpoints:
        if _check_endpoint_alive(url):
            return {"base_url": url, "provider_name": name, "detected_from": "scan"}

    return None


def _normalize_endpoint_url(url: str) -> str | None:
    """Normalize an endpoint URL to include proper scheme and path."""
    if not url:
        return None

    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    # Add /v1 suffix if not present and doesn't already end with /v1 or /api
    if not url.endswith(("/v1", "/v1/", "/api", "/api/")):
        if url.endswith("/"):
            url = f"{url}v1"
        else:
            url = f"{url}/v1"

    return url


def _is_local_url(url: str) -> bool:
    """Check if URL points to a local endpoint."""
    local_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"]
    return any(host in url.lower() for host in local_hosts)


def _guess_provider_from_url(url: str) -> str:
    """Guess the provider name from URL patterns."""
    url_lower = url.lower()
    if "11434" in url_lower:
        return "Ollama"
    elif "1234" in url_lower:
        return "LM Studio"
    elif "8000" in url_lower:
        return "vLLM"
    elif "5000" in url_lower:
        return "Local API"
    else:
        return "Local Provider"


def _check_endpoint_alive(url: str) -> bool:
    """Quick check if endpoint is responding."""
    if not HTTPX_AVAILABLE:
        # If httpx is not available, we can't check endpoints
        return False

    try:
        with httpx.Client(timeout=2.0) as client:
            # Try a simple GET to see if anything is listening
            response = client.get(url)
            # Accept any response that's not a connection error
            # Even 404 means something is listening
            return response.status_code < 500
    except (httpx.ConnectError, httpx.TimeoutException):
        return False
    except Exception:
        # Other errors (like invalid URLs) mean endpoint is not alive
        return False


def format_detected_config_summary(configs: dict[str, dict[str, Any] | None]) -> str:
    """Format detected configurations for display."""
    lines = []

    for provider, config in configs.items():
        if not config:
            continue

        if provider == "voyageai":
            lines.append("  - VoyageAI API key found (VOYAGE_API_KEY)")

        elif provider == "openai":
            lines.append("  - OpenAI API key found (OPENAI_API_KEY)")
            if config.get("base_url"):
                if _is_local_url(config["base_url"]):
                    lines.append(f"    Local endpoint: {config['base_url']}")
                else:
                    lines.append(f"    Custom endpoint: {config['base_url']}")
            if config.get("organization"):
                lines.append(f"    Organization: {config['organization']}")

        elif provider == "local":
            detection_source = config.get("detected_from", "scan")
            if detection_source == "environment":
                lines.append(
                    f"  - {config['provider_name']} configured via environment"
                )
            else:
                lines.append(
                    f"  - {config['provider_name']} server at {config['base_url']}"
                )

    return "\n".join(lines)


def get_priority_config(
    configs: dict[str, dict[str, Any] | None],
) -> dict[str, Any] | None:
    """
    Get the highest priority detected configuration.

    Priority order:
    1. VoyageAI (recommended)
    2. OpenAI with official endpoint
    3. OpenAI with custom endpoint
    4. Local endpoints
    """
    if configs.get("voyageai"):
        return configs["voyageai"]

    if configs.get("openai"):
        openai_config = configs["openai"]
        # Prefer official OpenAI over local endpoints configured via OPENAI_BASE_URL
        if not openai_config.get("base_url") or not _is_local_url(
            openai_config["base_url"]
        ):
            return openai_config

    # Check for local endpoints
    if configs.get("local"):
        local_config = configs["local"]
        return {
            "provider": "openai",  # Use OpenAI provider for compatibility
            "base_url": local_config["base_url"],
            "model": None,  # Will be prompted for
            "provider_name": local_config["provider_name"],
        }

    # Finally, OpenAI with local endpoint
    if configs.get("openai"):
        return configs["openai"]

    return None
