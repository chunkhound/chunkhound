from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from chunkhound.code_mapper.llm import build_llm_metadata_and_map_hyde
from chunkhound.code_mapper.models import AgentDocMetadata
from chunkhound.core.config.config import Config
from chunkhound.interfaces.llm_provider import LLMProvider
from chunkhound.llm_manager import LLMManager
from chunkhound.utils.git_safe import GitCommandError, run_git


@dataclass
class CodeMapperScope:
    target_dir: Path
    scope_path: Path
    scope_label: str
    path_filter: str | None


@dataclass
class CodeMapperRunContext:
    comprehensiveness: str
    max_points: int


@dataclass
class CodeMapperMetadataBundle:
    meta: AgentDocMetadata
    map_hyde_provider: LLMProvider | None


def _get_head_sha(project_root: Path) -> str:
    """Return the current HEAD SHA for the project or a stable placeholder."""
    try:
        result = run_git(["rev-parse", "HEAD"], cwd=project_root, timeout_s=5.0)
        stdout = (result.stdout or "").strip()
        if result.returncode == 0 and stdout:
            return stdout
    except (GitCommandError, OSError, ValueError):
        pass
    return "NO_GIT_HEAD"


def _compute_scope_label(target_dir: Path, scope_path: Path) -> str:
    """Compute a human-readable scope label relative to the target directory."""
    try:
        rel = scope_path.resolve().relative_to(target_dir.resolve())
        label = str(rel).replace(os.sep, "/")
        if not label or label == ".":
            return "/"
        return label
    except ValueError:
        return scope_path.name or "/"


def _compute_path_filter(target_dir: Path, scope_path: Path) -> str | None:
    """Compute a database path filter relative to the target directory."""
    try:
        rel = scope_path.resolve().relative_to(target_dir.resolve())
    except ValueError:
        return None

    rel_str = str(rel).replace(os.sep, "/")
    if not rel_str or rel_str == ".":
        return None
    return rel_str


def _max_points_for_comprehensiveness(comprehensiveness: str) -> int:
    if comprehensiveness == "minimal":
        return 1
    if comprehensiveness == "low":
        return 5
    if comprehensiveness == "medium":
        return 10
    if comprehensiveness == "high":
        return 15
    if comprehensiveness == "ultra":
        return 20
    return 10


def _resolve_scope_path(target_dir: Path, raw_scope: Path) -> Path:
    if raw_scope.is_absolute():
        return raw_scope.resolve()
    if raw_scope == Path("."):
        try:
            cwd = Path.cwd().resolve()
            target = target_dir.resolve()
            if cwd == target or cwd.is_relative_to(target):
                return cwd
        except (OSError, RuntimeError, ValueError):
            pass
    return (target_dir / raw_scope).resolve()


class CodeMapperOrchestrator:
    """Prepare scope, metadata, and run context for Code Mapper CLI flows."""

    def __init__(
        self,
        *,
        config: Config,
        args: argparse.Namespace,
        llm_manager: LLMManager | None,
    ) -> None:
        self._config = config
        self._args = args
        self._llm_manager = llm_manager

    def resolve_scope(self) -> CodeMapperScope:
        target_dir = self._config.target_dir or Path(".").resolve()
        raw_scope = Path(self._args.path)
        scope_path = _resolve_scope_path(target_dir, raw_scope)
        scope_label = _compute_scope_label(target_dir, scope_path)
        path_filter = _compute_path_filter(target_dir, scope_path)
        return CodeMapperScope(
            target_dir=target_dir,
            scope_path=scope_path,
            scope_label=scope_label,
            path_filter=path_filter,
        )

    def run_context(self) -> CodeMapperRunContext:
        comprehensiveness = getattr(self._args, "comprehensiveness", "medium")
        max_points = _max_points_for_comprehensiveness(comprehensiveness)
        return CodeMapperRunContext(
            comprehensiveness=comprehensiveness,
            max_points=max_points,
        )

    def metadata_bundle(
        self,
        *,
        scope_path: Path,
        target_dir: Path,
        overview_only: bool,
    ) -> CodeMapperMetadataBundle:
        llm_meta, map_hyde_provider = build_llm_metadata_and_map_hyde(
            config=self._config,
            llm_manager=self._llm_manager,
        )

        created_from_sha = _get_head_sha(scope_path)
        if (
            not overview_only
            and created_from_sha == "NO_GIT_HEAD"
            and scope_path != target_dir
        ):
            created_from_sha = _get_head_sha(target_dir)

        meta = AgentDocMetadata(
            created_from_sha=created_from_sha,
            previous_target_sha=created_from_sha,
            target_sha=created_from_sha,
            generated_at=datetime.now(timezone.utc).isoformat(),
            llm_config=llm_meta,
            generation_stats={"overview_only": "true"} if overview_only else {},
        )
        return CodeMapperMetadataBundle(meta=meta, map_hyde_provider=map_hyde_provider)
