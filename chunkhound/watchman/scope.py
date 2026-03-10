"""Watchman scope planning for v1 live-indexing integration.

The v1 contract deliberately keeps the handled logical scope equal to
`config.target_dir` and returns a single-item plan so later steps can grow into
multi-scope subscriptions without redesigning the interface.

Future coarse optimizations live outside this step:
- repo roots from `detect_repo_roots()`
- anchored include prefixes from `_extract_include_prefixes()`
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class WatchmanSubscriptionScope:
    """A single Watchman subscription scope derived from `watch-project`."""

    requested_path: Path
    watch_root: Path
    relative_root: str | None


@dataclass(frozen=True)
class WatchmanScopePlan:
    """List-friendly Watchman scope plan for future scope splitting."""

    scopes: tuple[WatchmanSubscriptionScope, ...]

    @property
    def primary_scope(self) -> WatchmanSubscriptionScope:
        if not self.scopes:
            raise ValueError("Watchman scope plan must contain at least one scope")
        return self.scopes[0]


def build_watchman_scope_plan(
    target_dir: Path, watch_project_result: Mapping[str, object]
) -> WatchmanScopePlan:
    """Build the v1 Watchman scope plan for a logical live-indexing target."""

    requested_path = target_dir.expanduser().resolve()
    watch_root = _require_watch_root(watch_project_result)
    relative_root = _normalize_relative_root(watch_project_result.get("relative_path"))
    mapped_path = _resolve_mapped_path(watch_root, relative_root)

    if mapped_path != requested_path:
        raise ValueError(
            "watch-project result does not map back to the requested target_dir: "
            f"target_dir={requested_path} mapped_path={mapped_path}"
        )

    return WatchmanScopePlan(
        scopes=(
            WatchmanSubscriptionScope(
                requested_path=requested_path,
                watch_root=watch_root,
                relative_root=relative_root,
            ),
        )
    )


def _require_watch_root(watch_project_result: Mapping[str, object]) -> Path:
    watch_value = watch_project_result.get("watch")
    if not isinstance(watch_value, str) or not watch_value.strip():
        raise ValueError("watch-project result must include a non-empty 'watch' string")
    watch_root = Path(watch_value).expanduser()
    if not watch_root.is_absolute():
        raise ValueError("watch-project result 'watch' must be an absolute path")
    return watch_root.resolve()


def _normalize_relative_root(relative_path: object) -> str | None:
    if relative_path is None:
        return None
    if not isinstance(relative_path, str):
        raise ValueError(
            "watch-project result 'relative_path' must be a string when present"
        )

    normalized_input = relative_path.strip().replace("\\", "/")
    if normalized_input in {"", "."}:
        return None

    candidate = PurePosixPath(normalized_input)
    if candidate.is_absolute():
        raise ValueError("watch-project relative_path must not be absolute")
    if ".." in candidate.parts:
        raise ValueError("watch-project relative_path must not traverse parents")
    if not candidate.parts:
        return None
    return candidate.as_posix()


def _resolve_mapped_path(watch_root: Path, relative_root: str | None) -> Path:
    if relative_root is None:
        return watch_root.resolve()
    relative_path = Path(*PurePosixPath(relative_root).parts)
    return (watch_root / relative_path).resolve()
