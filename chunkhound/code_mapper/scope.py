from __future__ import annotations

from pathlib import Path

from chunkhound.code_mapper.models import HydeConfig


def collect_scope_files(
    *,
    scope_path: Path,
    project_root: Path,
    hyde_cfg: HydeConfig,
    include_patterns: list[str] | None = None,
    indexing_excludes: list[str] | None = None,
    ignore_sources: list[str] | None = None,
    gitignore_backend: str = "python",
    workspace_root_only_gitignore: bool | None = None,
) -> list[str]:
    """Collect file paths within the scope, relative to project_root.

    This is filesystem-only and intentionally lightweight.
    """
    try:
        from chunkhound.core.config.indexing_config import IndexingConfig
        from chunkhound.utils.file_patterns import (
            normalize_include_patterns,
            walk_directory_tree,
        )
        from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

        if include_patterns is None:
            include_patterns = list(IndexingConfig().include)
        patterns = normalize_include_patterns(list(include_patterns))

        if indexing_excludes is None:
            indexing_excludes = IndexingConfig().get_effective_config_excludes()

        if ignore_sources is None:
            ignore_sources = ["gitignore"]

        ignore_engine = build_repo_aware_ignore_engine(
            root=project_root,
            sources=list(ignore_sources),
            chignore_file=".chignore",
            config_exclude=list(indexing_excludes),
            backend=gitignore_backend,
            workspace_root_only_gitignore=workspace_root_only_gitignore,
        )

        max_files = hyde_cfg.max_scope_files if hyde_cfg.max_scope_files > 0 else None
        files, _ = walk_directory_tree(
            scope_path,
            project_root,
            patterns,
            [],
            {project_root: []},
            ignore_engine=ignore_engine,
            max_files=max_files,
        )
        file_paths = []
        for path in files:
            try:
                rel = path.relative_to(project_root)
            except ValueError:
                continue
            file_paths.append(rel.as_posix())
    except (ImportError, OSError):
        file_paths = []

    return file_paths
