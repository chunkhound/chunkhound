"""Configuration models for Cross-Repository Federation.

Follows ChunkHound conventions:
- Pydantic BaseModel for validated, serialisable config
- Environment variable loading via class methods
- CLI override extraction
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class RepoConfig(BaseModel):
    """Configuration for a single federated repository."""

    name: str = Field(description="Human-readable repo identifier")
    path: Path = Field(description="Absolute or relative path to repo root")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Relevance weight (1.0 = normal, 0.5 = deprioritised)",
    )
    enabled: bool = Field(default=True, description="Include in federated queries")

    @field_validator("path")
    def validate_path(cls, v: Path) -> Path:
        return Path(v)


class FederationConfig(BaseModel):
    """Top-level federation configuration.

    Added to ``.chunkhound.json`` under a ``federation`` key::

        {
            "federation": {
                "repositories": [
                    {"name": "backend", "path": "/workspace/backend"},
                    {"name": "frontend", "path": "/workspace/frontend"}
                ],
                "merge_strategy": "interleave_rerank",
                "max_results_per_repo": 50
            }
        }
    """

    repositories: list[RepoConfig] = Field(
        default_factory=list,
        description="List of repositories to federate",
    )
    merge_strategy: Literal[
        "interleave_rerank", "round_robin", "repo_priority"
    ] = Field(
        default="interleave_rerank",
        description="How to merge results from multiple repos",
    )
    max_results_per_repo: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Maximum results to fetch per repository before merging",
    )
    rerank_top_k: int = Field(
        default=20,
        ge=5,
        le=200,
        description="Final top-K after cross-repo reranking",
    )

    def get_enabled_repos(self) -> list[RepoConfig]:
        """Return only enabled repositories."""
        return [r for r in self.repositories if r.enabled]

    def is_configured(self) -> bool:
        """Return True if at least two repos are configured."""
        return len(self.get_enabled_repos()) >= 2

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load federation config from environment variables."""
        config: dict[str, Any] = {}
        if strategy := os.getenv("CHUNKHOUND_FEDERATION__MERGE_STRATEGY"):
            config["merge_strategy"] = strategy
        if max_results := os.getenv("CHUNKHOUND_FEDERATION__MAX_RESULTS_PER_REPO"):
            try:
                config["max_results_per_repo"] = int(max_results)
            except ValueError:
                pass
        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract federation config from CLI arguments."""
        overrides: dict[str, Any] = {}
        if hasattr(args, "merge_strategy") and args.merge_strategy:
            overrides["merge_strategy"] = args.merge_strategy
        if hasattr(args, "repos") and args.repos:
            # CLI --repos flag: comma-separated repo names to include
            overrides["_cli_repo_filter"] = args.repos
        return overrides