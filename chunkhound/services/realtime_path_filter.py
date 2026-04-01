"""Backend-neutral path filtering for realtime indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config


@dataclass(frozen=True, slots=True)
class RealtimePathFilterSettings:
    """Resolved scope settings shared by realtime admission and reconciliation."""

    include_patterns: tuple[str, ...] | None = None
    ignore_sources: tuple[str, ...] = ()
    config_excludes: tuple[str, ...] = ()
    chignore_file: str = ".chignore"
    gitignore_backend: str = "python"
    workspace_root_only_gitignore: bool = False

    @classmethod
    def from_config(
        cls,
        config: Config | None,
        *,
        include_patterns: list[str] | tuple[str, ...] | None = None,
    ) -> RealtimePathFilterSettings | None:
        if config is None or getattr(config, "indexing", None) is None:
            return None

        indexing = config.indexing
        resolved_includes = include_patterns
        if resolved_includes is None:
            resolved_includes = list(indexing.include)

        return cls(
            include_patterns=(
                tuple(resolved_includes) if resolved_includes is not None else None
            ),
            ignore_sources=tuple(indexing.resolve_ignore_sources()),
            config_excludes=tuple(indexing.get_effective_config_excludes()),
            chignore_file=indexing.chignore_file,
            gitignore_backend=str(getattr(indexing, "gitignore_backend", "python")),
            workspace_root_only_gitignore=bool(
                getattr(indexing, "workspace_gitignore_nonrepo", False)
            ),
        )


class RealtimePathFilter:
    """Apply ChunkHound's realtime path filtering independently of the backend."""

    def __init__(
        self,
        *,
        config: Config | None,
        root_path: Path | None = None,
        settings: RealtimePathFilterSettings | None = None,
    ) -> None:
        self._engine: Any | None = None
        self._engine_initialized = False
        self._engine_initialization_failed = False
        self._ignore_engine_degraded = False
        self._ignore_engine_degraded_warned = False
        self._include_patterns: list[str] | None = None
        self._include_degraded = False
        self._include_degraded_warned = False
        self._pattern_cache: dict[str, Any] = {}
        self._settings = settings or RealtimePathFilterSettings.from_config(config)
        self._root = self._resolve_root(config=config, root_path=root_path)

    @staticmethod
    def _resolve_root(*, config: Config | None, root_path: Path | None) -> Path:
        if root_path is not None:
            return root_path.resolve()
        try:
            target_dir = (
                config.target_dir if config and config.target_dir else Path.cwd()
            )
            return target_dir.resolve()
        except Exception:
            return Path.cwd().resolve()

    def should_index(self, file_path: Path) -> bool:
        """Return whether a path should enter the realtime indexing pipeline."""
        settings = self._settings
        if settings is None:
            return self._language_fallback(file_path)

        if not self._engine_initialized and not self._engine_initialization_failed:
            try:
                from chunkhound.utils.ignore_engine import (
                    build_repo_aware_ignore_engine,
                )

                self._engine = build_repo_aware_ignore_engine(
                    self._root,
                    list(settings.ignore_sources),
                    settings.chignore_file,
                    list(settings.config_excludes),
                    backend=settings.gitignore_backend,
                    workspace_root_only_gitignore=(
                        settings.workspace_root_only_gitignore
                    ),
                )
                self._engine_initialized = True
            except Exception as error:
                self._engine_initialization_failed = True
                self._engine = None
                logger.warning(
                    "RealtimePathFilter failed to build repo-aware ignore engine "
                    f"for {self._root}: {error}; rejecting realtime events because "
                    "ignore policy could not be evaluated"
                )
                return False

        if self._engine_initialization_failed:
            return False

        if not self._ignore_engine_degraded:
            try:
                if self._engine is not None and self._engine.matches(
                    file_path, is_dir=False
                ):
                    return False
            except Exception as error:
                self._ignore_engine_degraded = True
                self._engine = None
                self._warn_ignore_engine_degraded_once(error)
                return False

        if self._ignore_engine_degraded:
            return False

        if self._include_degraded:
            return False

        try:
            if self._include_patterns is None:
                from chunkhound.utils.file_patterns import normalize_include_patterns

                if settings.include_patterns is None:
                    return self._language_fallback(file_path)

                includes = list(settings.include_patterns)
                self._include_patterns = normalize_include_patterns(includes)

            if not self._include_patterns:
                return False

            from chunkhound.utils.file_patterns import should_include_file

            return should_include_file(
                file_path,
                self._root,
                self._include_patterns,
                self._pattern_cache,
            )
        except Exception as error:
            self._include_degraded = True
            self._warn_include_degraded_once(error)
            return False

    def _warn_ignore_engine_degraded_once(self, error: Exception) -> None:
        if self._ignore_engine_degraded_warned:
            return
        self._ignore_engine_degraded_warned = True
        logger.warning(
            "RealtimePathFilter ignore-engine evaluation failed "
            f"for {self._root}: {error}; ignore-based exclusion could not be "
            "applied, rejecting affected realtime events"
        )

    def _warn_include_degraded_once(self, error: Exception) -> None:
        if self._include_degraded_warned:
            return
        self._include_degraded_warned = True
        logger.warning(
            "RealtimePathFilter include-pattern evaluation failed "
            f"for {self._root}: {error}; include filtering could not be applied, "
            "rejecting affected realtime events"
        )

    @staticmethod
    def _language_fallback(file_path: Path) -> bool:
        from chunkhound.core.types.common import Language

        if file_path.suffix.lower() in Language.get_all_extensions():
            return True
        if file_path.name.lower() in Language.get_all_filename_patterns():
            return True
        return False
