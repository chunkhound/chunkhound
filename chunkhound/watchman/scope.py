"""Watchman scope planning for live-indexing integration.

The primary logical scope remains exactly `config.target_dir`, but Linux nested
mount roots may require their own explicit Watchman subscriptions so live file
events keep flowing across mount boundaries.

Future coarse optimizations still live outside this step:
- repo roots from `detect_repo_roots()`
- anchored include prefixes from `_extract_include_prefixes()`
"""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class WatchmanSubscriptionScope:
    """A single Watchman subscription scope for one logical path surface."""

    requested_path: Path
    watch_root: Path
    relative_root: str | None
    scope_kind: str = "primary"


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
    target_dir: Path,
    watch_project_result: Mapping[str, object],
    *,
    nested_mount_roots: Sequence[Path] | None = None,
    additional_scopes: Sequence[WatchmanSubscriptionScope] | None = None,
) -> WatchmanScopePlan:
    """Build the Watchman scope plan for a logical live-indexing target."""

    logical_requested_path = _lexical_absolute_path(target_dir)
    resolved_requested_path = target_dir.expanduser().resolve()
    primary_scope = _build_scope_from_watch_project(
        logical_requested_path,
        resolved_requested_path,
        watch_project_result,
    )
    mount_roots = (
        discover_nested_linux_mount_roots(logical_requested_path)
        if nested_mount_roots is None
        else _normalize_nested_mount_roots(logical_requested_path, nested_mount_roots)
    )
    normalized_additional_scopes = (
        ()
        if additional_scopes is None
        else _normalize_additional_scopes(logical_requested_path, additional_scopes)
    )

    scopes = [primary_scope]
    for mount_root in mount_roots:
        scopes.append(
            WatchmanSubscriptionScope(
                requested_path=mount_root,
                watch_root=mount_root.expanduser().resolve(),
                relative_root=None,
                scope_kind="nested_mount",
            )
        )
    scopes.extend(normalized_additional_scopes)
    return WatchmanScopePlan(scopes=tuple(scopes))


def discover_nested_linux_mount_roots(
    target_dir: Path,
    *,
    mountinfo_path: Path | None = None,
) -> tuple[Path, ...]:
    """Return nested Linux mount roots contained under the logical target."""

    if not sys.platform.startswith("linux"):
        return ()

    requested_path = _lexical_absolute_path(target_dir)
    resolved_mountinfo_path = mountinfo_path or Path("/proc/self/mountinfo")
    try:
        mountinfo_text = resolved_mountinfo_path.read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ()
    return _mount_roots_from_mountinfo(requested_path, mountinfo_text)


def discover_nested_windows_junction_scopes(
    target_dir: Path,
) -> tuple[WatchmanSubscriptionScope, ...]:
    """Return logical junction-backed scopes that resolve outside the target."""

    if not sys.platform.startswith("win"):
        return ()

    resolved_target_path = target_dir.expanduser().resolve()
    traversal_root = _lexical_absolute_path(target_dir)
    discovered_scopes: dict[
        tuple[str, str, str | None, str], WatchmanSubscriptionScope
    ] = {}
    pending_dirs = [traversal_root]

    while pending_dirs:
        current_dir = pending_dirs.pop()
        try:
            with os.scandir(current_dir) as entries:
                for entry in entries:
                    try:
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        logical_path = _lexical_absolute_path(Path(entry.path))
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue

                    is_reparse_point = bool(
                        getattr(entry_stat, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    )
                    if is_reparse_point and not logical_path.is_symlink():
                        try:
                            watch_root = logical_path.resolve()
                        except OSError:
                            continue
                        if watch_root == resolved_target_path:
                            continue
                        try:
                            watch_root.relative_to(resolved_target_path)
                        except ValueError:
                            scope = WatchmanSubscriptionScope(
                                requested_path=logical_path,
                                watch_root=watch_root,
                                relative_root=None,
                                scope_kind="nested_junction",
                            )
                            scope_key = (
                                _path_sort_key(scope.requested_path),
                                str(scope.watch_root),
                                scope.relative_root,
                                scope.scope_kind,
                            )
                            discovered_scopes[scope_key] = scope
                        continue

                    pending_dirs.append(logical_path)
        except OSError:
            continue

    return tuple(
        scope
        for _key, scope in sorted(
            discovered_scopes.items(),
            key=lambda item: (
                len(item[1].requested_path.parts),
                item[0][0],
                str(item[1].watch_root),
            ),
        )
    )


def _build_scope_from_watch_project(
    logical_requested_path: Path,
    resolved_requested_path: Path,
    watch_project_result: Mapping[str, object],
) -> WatchmanSubscriptionScope:
    watch_root = _require_watch_root(watch_project_result)
    relative_root = _normalize_relative_root(watch_project_result.get("relative_path"))
    mapped_path = _resolve_mapped_path(watch_root, relative_root)

    if mapped_path != resolved_requested_path:
        raise ValueError(
            "watch-project result does not map back to the requested target_dir: "
            f"target_dir={resolved_requested_path} mapped_path={mapped_path}"
        )

    return WatchmanSubscriptionScope(
        requested_path=logical_requested_path,
        watch_root=watch_root,
        relative_root=relative_root,
        scope_kind="primary",
    )


def _normalize_nested_mount_roots(
    requested_path: Path, nested_mount_roots: Sequence[Path]
) -> tuple[Path, ...]:
    normalized_mount_roots: set[Path] = set()
    for candidate in nested_mount_roots:
        mount_root = _lexical_absolute_path(Path(candidate))
        if mount_root == requested_path:
            continue
        try:
            mount_root.relative_to(requested_path)
        except ValueError:
            continue
        normalized_mount_roots.add(mount_root)
    return tuple(
        sorted(normalized_mount_roots, key=lambda path: (len(path.parts), str(path)))
    )


def _normalize_additional_scopes(
    requested_path: Path, additional_scopes: Sequence[WatchmanSubscriptionScope]
) -> tuple[WatchmanSubscriptionScope, ...]:
    normalized_scopes: dict[
        tuple[str, str, str | None, str], WatchmanSubscriptionScope
    ] = {}
    for candidate in additional_scopes:
        logical_requested_path = _lexical_absolute_path(candidate.requested_path)
        if logical_requested_path == requested_path:
            continue
        try:
            logical_requested_path.relative_to(requested_path)
        except ValueError:
            continue

        normalized_scope = WatchmanSubscriptionScope(
            requested_path=logical_requested_path,
            watch_root=Path(candidate.watch_root).expanduser().resolve(),
            relative_root=_normalize_relative_root(candidate.relative_root),
            scope_kind=candidate.scope_kind,
        )
        scope_key = (
            _path_sort_key(normalized_scope.requested_path),
            str(normalized_scope.watch_root),
            normalized_scope.relative_root,
            normalized_scope.scope_kind,
        )
        normalized_scopes[scope_key] = normalized_scope
    return tuple(
        scope
        for _key, scope in sorted(
            normalized_scopes.items(),
            key=lambda item: (
                len(item[1].requested_path.parts),
                item[0][0],
                str(item[1].watch_root),
            ),
        )
    )


def _mount_roots_from_mountinfo(
    requested_path: Path, mountinfo_text: str
) -> tuple[Path, ...]:
    discovered_mount_roots: set[Path] = set()
    for line in mountinfo_text.splitlines():
        mount_root = _parse_mountinfo_mount_root(line)
        if mount_root is None or mount_root == requested_path:
            continue
        try:
            mount_root.relative_to(requested_path)
        except ValueError:
            continue
        discovered_mount_roots.add(mount_root)
    return tuple(
        sorted(discovered_mount_roots, key=lambda path: (len(path.parts), str(path)))
    )


def _parse_mountinfo_mount_root(line: str) -> Path | None:
    head, separator, _tail = line.partition(" - ")
    if not separator:
        return None
    fields = head.split()
    if len(fields) < 5:
        return None
    mount_point = _unescape_mountinfo_field(fields[4])
    return _lexical_absolute_path(Path(mount_point))


def _unescape_mountinfo_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _require_watch_root(watch_project_result: Mapping[str, object]) -> Path:
    watch_value = watch_project_result.get("watch")
    if not isinstance(watch_value, str) or not watch_value.strip():
        raise ValueError("watch-project result must include a non-empty 'watch' string")
    watch_root = Path(watch_value).expanduser()
    if not watch_root.is_absolute():
        raise ValueError("watch-project result 'watch' must be an absolute path")
    return watch_root.resolve()


def _lexical_absolute_path(path: Path) -> Path:
    expanded_path = Path(path).expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return Path(os.path.abspath(str(expanded_path)))


def _path_sort_key(path: Path) -> str:
    rendered = str(path)
    return rendered.lower() if os.name == "nt" else rendered


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
