"""FederatedSearchService – concurrent multi-repo search with merge/rerank.

Wraps multiple ``DatabaseServices`` instances (one per repo) and exposes
the same search interface as a single-repo service.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.federation.config import FederationConfig, RepoConfig


class FederatedSearchService:
    """Fan-out search across multiple ChunkHound indexes."""

    def __init__(
        self,
        federation_config: FederationConfig,
        embedding_manager: EmbeddingManager | None = None,
    ) -> None:
        self._config = federation_config
        self._embedding_manager = embedding_manager
        self._services: dict[str, DatabaseServices] = {}
        self._weights: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect_all(self) -> None:
        """Initialise DatabaseServices for every enabled repo."""
        for repo in self._config.get_enabled_repos():
            try:
                repo_config = Config.from_environment()
                # Override database path to point at this repo's index
                db_root = repo.path / ".chunkhound" / "db"
                repo_config.database.path = db_root

                services = create_services(
                    db_path=repo_config.database.get_db_path(),
                    config=repo_config,
                    embedding_manager=self._embedding_manager,
                )
                services.provider.connect()
                self._services[repo.name] = services
                self._weights[repo.name] = repo.weight
                logger.info(f"Federation: connected to repo '{repo.name}' at {repo.path}")
            except Exception as exc:
                logger.warning(
                    f"Federation: failed to connect repo '{repo.name}': {exc}"
                )

    def disconnect_all(self) -> None:
        """Disconnect all repo databases."""
        for name, services in self._services.items():
            try:
                services.provider.disconnect()
            except Exception as exc:
                logger.debug(f"Federation: disconnect error for '{name}': {exc}")
        self._services.clear()

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    async def search_semantic(
        self,
        query_embedding: list[float],
        provider: str,
        model: str,
        page_size: int = 20,
        path_filter: str | None = None,
        repos: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fan-out semantic search and merge results.

        Args:
            query_embedding: Query vector.
            provider / model: Embedding provider metadata.
            page_size: Final number of results after merging.
            path_filter: Optional path prefix (applied per-repo).
            repos: Optional list of repo names to restrict to.

        Returns:
            ``(results, metadata)`` following the standard ChunkHound
            search return signature.
        """
        target_services = self._resolve_repos(repos)

        async def _search_one(
            name: str, svc: DatabaseServices
        ) -> list[dict[str, Any]]:
            results, _meta = svc.provider.search_semantic(
                query_embedding=query_embedding,
                provider=provider,
                model=model,
                page_size=self._config.max_results_per_repo,
                path_filter=path_filter,
            )
            # Tag each result with its source repo
            for r in results:
                r["_federation_repo"] = name
                r["_federation_weight"] = self._weights.get(name, 1.0)
            return results

        tasks = [_search_one(n, s) for n, s in target_services.items()]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        flat: list[dict[str, Any]] = []
        for batch in all_results:
            if isinstance(batch, Exception):
                logger.warning(f"Federation: search error: {batch}")
                continue
            flat.extend(batch)

        merged = self._merge_results(flat, page_size)

        metadata = {
            "federation": True,
            "repos_searched": list(target_services.keys()),
            "total_candidates": len(flat),
            "merge_strategy": self._config.merge_strategy,
        }
        return merged, metadata

    # ------------------------------------------------------------------
    # Regex search
    # ------------------------------------------------------------------

    async def search_regex(
        self,
        pattern: str,
        page_size: int = 20,
        path_filter: str | None = None,
        repos: list[str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fan-out regex search across repos."""
        target_services = self._resolve_repos(repos)

        async def _search_one(
            name: str, svc: DatabaseServices
        ) -> list[dict[str, Any]]:
            results, _meta = svc.provider.search_regex(
                pattern=pattern,
                page_size=self._config.max_results_per_repo,
                path_filter=path_filter,
            )
            for r in results:
                r["_federation_repo"] = name
            return results

        tasks = [_search_one(n, s) for n, s in target_services.items()]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        flat: list[dict[str, Any]] = []
        for batch in all_results:
            if isinstance(batch, Exception):
                logger.warning(f"Federation: regex search error: {batch}")
                continue
            flat.extend(batch)

        merged = self._merge_results(flat, page_size)
        metadata = {
            "federation": True,
            "repos_searched": list(target_services.keys()),
            "total_candidates": len(flat),
        }
        return merged, metadata

    # ------------------------------------------------------------------
    # Merge strategies
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        strategy = self._config.merge_strategy

        if strategy == "interleave_rerank":
            return self._merge_interleave_rerank(results, top_k)
        if strategy == "round_robin":
            return self._merge_round_robin(results, top_k)
        if strategy == "repo_priority":
            return self._merge_repo_priority(results, top_k)

        return results[:top_k]

    def _merge_interleave_rerank(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Score-weighted merge: multiply each result's score by its repo weight."""
        for r in results:
            weight = r.get("_federation_weight", 1.0)
            raw_score = r.get("score", r.get("distance", 0.0))
            r["_weighted_score"] = float(raw_score) * float(weight)

        results.sort(key=lambda r: r.get("_weighted_score", 0.0), reverse=True)
        return results[:top_k]

    def _merge_round_robin(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Alternate results from each repo by rank."""
        by_repo: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            repo = r.get("_federation_repo", "unknown")
            by_repo.setdefault(repo, []).append(r)

        merged: list[dict[str, Any]] = []
        repo_iters = {name: iter(items) for name, items in by_repo.items()}
        while len(merged) < top_k and repo_iters:
            exhausted: list[str] = []
            for name, it in repo_iters.items():
                try:
                    merged.append(next(it))
                    if len(merged) >= top_k:
                        break
                except StopIteration:
                    exhausted.append(name)
            for name in exhausted:
                del repo_iters[name]

        return merged[:top_k]

    def _merge_repo_priority(
        self,
        results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return highest-weight repo results first, then fill from others."""
        results.sort(
            key=lambda r: (-r.get("_federation_weight", 1.0), -r.get("score", 0.0))
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_repos(
        self, repos: list[str] | None
    ) -> dict[str, DatabaseServices]:
        """Resolve which repos to search."""
        if repos:
            return {n: s for n, s in self._services.items() if n in repos}
        return dict(self._services)