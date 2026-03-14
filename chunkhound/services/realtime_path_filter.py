"""Backend-neutral path filtering for realtime indexing."""

from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config


class RealtimePathFilter:
    """Apply ChunkHound's realtime path filtering independently of the backend."""

    def __init__(self, *, config: Config | None, root_path: Path | None = None) -> None:
        self._config = config
        self._engine: Any | None = None
        self._include_patterns: list[str] | None = None
        self._pattern_cache: dict[str, Any] = {}
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
        if not self._config:
            return self._language_fallback(file_path)

        try:
            if self._engine is None:
                from chunkhound.utils.ignore_engine import (
                    build_repo_aware_ignore_engine,
                )

                sources = self._config.indexing.resolve_ignore_sources()
                cfg_ex = self._config.indexing.get_effective_config_excludes()
                chf = self._config.indexing.chignore_file
                backend = str(
                    getattr(self._config.indexing, "gitignore_backend", "python")
                )
                overlay = bool(
                    getattr(self._config.indexing, "workspace_gitignore_nonrepo", False)
                )
                self._engine = build_repo_aware_ignore_engine(
                    self._root,
                    sources,
                    chf,
                    cfg_ex,
                    backend=backend,
                    workspace_root_only_gitignore=overlay,
                )
        except Exception:
            self._engine = None

        try:
            if self._engine is not None and self._engine.matches(
                file_path, is_dir=False
            ):
                return False
        except Exception:
            pass

        try:
            if self._include_patterns is None:
                from chunkhound.utils.file_patterns import normalize_include_patterns

                includes = list(self._config.indexing.include)
                self._include_patterns = normalize_include_patterns(includes)

            from chunkhound.utils.file_patterns import should_include_file

            return should_include_file(
                file_path,
                self._root,
                self._include_patterns,
                self._pattern_cache,
            )
        except Exception:
            return self._language_fallback(file_path)

    @staticmethod
    def _language_fallback(file_path: Path) -> bool:
        from chunkhound.core.types.common import Language

        if file_path.suffix.lower() in Language.get_all_extensions():
            return True
        if file_path.name.lower() in Language.get_all_filename_patterns():
            return True
        return False
