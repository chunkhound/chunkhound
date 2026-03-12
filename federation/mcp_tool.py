"""MCP integration for Cross-Repository Federation.

Enhances existing ``semantic_search`` and ``regex_search`` tools with
an optional ``repos`` parameter.  When federation is configured, searches
fan out across all connected repositories automatically.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from chunkhound.federation.config import FederationConfig
from chunkhound.federation.service import FederatedSearchService


def create_federated_service(
    config: FederationConfig,
    embedding_manager: Any | None = None,
) -> FederatedSearchService | None:
    """Create and connect a FederatedSearchService if federation is configured.

    Returns None if federation is not configured (fewer than 2 repos).
    """
    if not config.is_configured():
        return None

    service = FederatedSearchService(
        federation_config=config,
        embedding_manager=embedding_manager,
    )
    service.connect_all()

    enabled = config.get_enabled_repos()
    logger.info(
        f"Federation: active with {len(enabled)} repos: "
        f"{[r.name for r in enabled]}"
    )
    return service


# ------------------------------------------------------------------
# Enhanced tool parameter schema (extends existing tools)
# ------------------------------------------------------------------

FEDERATION_PARAM_SCHEMA: dict[str, Any] = {
    "repos": {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Optional list of repository names to search. "
            "Omit to search all federated repos."
        ),
    },
}


async def federated_semantic_search(
    service: FederatedSearchService,
    query_embedding: list[float],
    provider: str,
    model: str,
    arguments: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Wrapper for MCP semantic_search with federation support."""
    return await service.search_semantic(
        query_embedding=query_embedding,
        provider=provider,
        model=model,
        page_size=int(arguments.get("page_size", 20)),
        path_filter=arguments.get("path"),
        repos=arguments.get("repos"),
    )


async def federated_regex_search(
    service: FederatedSearchService,
    arguments: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Wrapper for MCP regex_search with federation support."""
    return await service.search_regex(
        pattern=arguments.get("pattern", ""),
        page_size=int(arguments.get("page_size", 20)),
        path_filter=arguments.get("path"),
        repos=arguments.get("repos"),
    )