"""Backend-neutral path filtering for realtime indexing."""

from __future__ import annotations

import os
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
    index_unknown_files: bool = False

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
            index_unknown_files=bool(getattr(indexing, "index_unknown_files", False)),
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
        self._extension_filter: tuple[frozenset[str], frozenset[str]] | None = None
        self._include_degraded = False
        self._include_degraded_warned = False
        self._pattern_cache: dict[str, Any] = {}
        self._settings = settings or RealtimePathFilterSettings.from_config(config)
        self._root = self._resolve_root(config=config, root_path=root_path)
        # The root as given (normalized, links not resolved) and as the ignore
        # engine sees it (resolved); see _engine_path.
        self._in_tree_root = Path(os.path.abspath(self._root))
        self._engine_root = self._root.resolve()

    @staticmethod
    def _resolve_root(*, config: Config | None, root_path: Path | None) -> Path:
        if root_path is not None:
            return root_path.expanduser().absolute()
        try:
            target_dir = (
                config.target_dir if config and config.target_dir else Path.cwd()
            )
            return target_dir.resolve()
        except Exception:
            return Path.cwd().resolve()

    @property
    def is_degraded(self) -> bool:
        """Return True when ignore/include evaluation cannot be trusted.

        Realtime admission still fails closed when the filter is degraded,
        but cleanup paths use this flag to preserve rows we cannot prove are
        excluded — losing rows when the exclusion oracle is broken would
        be user-visible data loss.
        """
        return (
            self._engine_initialization_failed
            or self._ignore_engine_degraded
            or self._include_degraded
        )

    def excludes_directory(self, dir_path: Path) -> bool:
        """Return whether an ignore rule excludes this directory where it sits.

        Pass the directory's path in the tree, before resolving links: a link
        under ``node_modules/`` is excluded by its place there, not by its
        target. Returns False when the ignore policy cannot be evaluated; each
        file under the directory is then still checked, and rejected, by
        ``should_index``. An evaluation error marks the filter degraded and
        warns once, as it does for a file.
        """
        return bool(self._ignore_rule_matches(dir_path, is_dir=True))

    def _ignore_rule_matches(self, path: Path, *, is_dir: bool) -> bool | None:
        """Whether an ignore rule matches ``path`` where it sits in the tree.

        None means the ignore policy cannot be evaluated: the engine could not
        be built, or it has failed. The first evaluation error marks the engine
        degraded and warns once, whichever caller hit it, so ``is_degraded``
        reports it.
        """
        if not self._ensure_engine() or self._ignore_engine_degraded:
            return None
        engine = self._engine
        if engine is None:  # No engine and no failure: no ignore rule applies.
            return False
        try:
            return engine.matches(self._engine_path(path), is_dir=is_dir) is not None
        except Exception as error:
            self._ignore_engine_degraded = True
            self._engine = None
            self._warn_ignore_engine_degraded_once(error)
            return None

    def _engine_path(self, path: Path) -> Path:
        """``path`` spelled under the resolved root the ignore engine was built on.

        The engine matches a path where it sits only when the path is lexically
        under its resolved root. Watchman reports paths under the root as it
        was requested. When that spelling goes through a link (``/var`` is
        ``/private/var`` on macOS), only the root part is respelled here, so a
        link further down is still matched where it sits, not at its target.
        """
        try:
            relative = Path(os.path.abspath(path)).relative_to(self._in_tree_root)
        except ValueError:
            return path
        return self._engine_root / relative

    def _ensure_engine(self) -> bool:
        """Build the ignore engine once; False if it could not be built."""
        settings = self._settings
        if settings is None:
            return False
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

        return not self._engine_initialization_failed

    def should_index(self, file_path: Path, *, check_target: bool = True) -> bool:
        """Return whether a path should enter the realtime indexing pipeline.

        A path is judged where it sits in the tree. When a link is on its way
        (the file itself, or a directory between it and the root), the indexer
        reads the link's target, so the target has to pass as well: a link
        ``src/a.py -> vendor/x.py`` with ``vendor/`` ignored is not indexed.
        The startup scan never indexes a symlinked file at all.

        ``check_target=False`` judges the path where it sits only. A delete
        reads nothing, so its target does not matter.
        """
        settings = self._settings
        if settings is None:
            return self._language_fallback(file_path)
        if not self._admits(file_path, settings):
            return False
        if not check_target:
            return True
        target = file_path.resolve()
        return target == self._engine_path(file_path) or self._admits(target, settings)

    def _admits(self, file_path: Path, settings: RealtimePathFilterSettings) -> bool:
        """The policy for ``file_path`` where it sits: ignore rules, then includes."""
        # A match rejects the file, and so does a policy that cannot be evaluated.
        if self._ignore_rule_matches(file_path, is_dir=False) is not False:
            return False

        if self._include_degraded:
            return False

        try:
            if self._include_patterns is None:
                from chunkhound.utils.file_patterns import (
                    normalize_include_patterns,
                    prepare_extension_filter,
                )

                if settings.include_patterns is None:
                    return self._language_fallback(file_path)

                includes = list(settings.include_patterns)
                self._include_patterns = normalize_include_patterns(includes)
                # Prepared once here because should_index() runs per filesystem
                # event — re-summarizing patterns per call would be wasteful.
                self._extension_filter = prepare_extension_filter(
                    self._include_patterns
                )

            if not self._include_patterns:
                return False

            from chunkhound.utils.file_patterns import should_include_file

            if not should_include_file(
                file_path,
                self._root,
                self._include_patterns,
                self._pattern_cache,
            ):
                return False

            from chunkhound.utils.file_patterns import passes_extension_filter

            return passes_extension_filter(
                file_path,
                self._include_patterns,
                settings.index_unknown_files,
                prepared=self._extension_filter,
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

        return Language.is_known_path(file_path)
