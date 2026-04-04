from __future__ import annotations

import os
from pathlib import Path

import pytest

import chunkhound.watchman.scope as watchman_scope_module
from chunkhound.watchman import (
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    discover_nested_linux_mount_roots,
)


def test_build_watchman_scope_plan_for_direct_root(tmp_path: Path) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()

    plan = build_watchman_scope_plan(target_dir, {"watch": str(target_dir)})

    assert isinstance(plan, WatchmanScopePlan)
    assert len(plan.scopes) == 1
    assert isinstance(plan.primary_scope, WatchmanSubscriptionScope)
    assert plan.primary_scope.requested_path == target_dir.resolve()
    assert plan.primary_scope.watch_root == target_dir.resolve()
    assert plan.primary_scope.relative_root is None


def test_build_watchman_scope_plan_for_subdirectory_target(tmp_path: Path) -> None:
    watch_root = tmp_path / "repo"
    target_dir = watch_root / "packages" / "api"
    target_dir.mkdir(parents=True)

    plan = build_watchman_scope_plan(
        target_dir,
        {"watch": str(watch_root), "relative_path": "packages/api"},
    )

    assert plan.primary_scope.requested_path == target_dir.resolve()
    assert plan.primary_scope.watch_root == watch_root.resolve()
    assert plan.primary_scope.relative_root == "packages/api"


def test_build_watchman_scope_plan_is_stable_across_repeated_calls(
    tmp_path: Path,
) -> None:
    watch_root = tmp_path / "repo"
    target_dir = watch_root / "services" / "watchman"
    target_dir.mkdir(parents=True)
    watch_project_result = {
        "watch": str(watch_root),
        "relative_path": "services/watchman",
    }

    first = build_watchman_scope_plan(target_dir, watch_project_result)
    second = build_watchman_scope_plan(target_dir, watch_project_result)

    assert first == second


@pytest.mark.parametrize(
    ("watch_project_result_factory", "message"),
    [
        (lambda watch_root: {}, "watch"),
        (lambda watch_root: {"watch": 123}, "watch"),
        (lambda watch_root: {"watch": "relative/root"}, "absolute"),
        (
            lambda watch_root: {"watch": str(watch_root), "relative_path": 123},
            "relative_path",
        ),
        (
            lambda watch_root: {
                "watch": str(watch_root),
                "relative_path": "../outside",
            },
            "traverse",
        ),
    ],
)
def test_build_watchman_scope_plan_rejects_malformed_results(
    tmp_path: Path,
    watch_project_result_factory,
    message: str,
) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()
    watch_project_result = watch_project_result_factory(target_dir)

    with pytest.raises(ValueError, match=message):
        build_watchman_scope_plan(target_dir, watch_project_result)


def test_build_watchman_scope_plan_rejects_inconsistent_mapping(
    tmp_path: Path,
) -> None:
    watch_root = tmp_path / "repo"
    target_dir = watch_root / "packages" / "api"
    target_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="target_dir"):
        build_watchman_scope_plan(
            target_dir,
            {"watch": str(watch_root), "relative_path": "packages/other"},
        )


def test_build_watchman_scope_plan_normalizes_dot_relative_root(tmp_path: Path) -> None:
    target_dir = tmp_path / "repo"
    target_dir.mkdir()

    plan = build_watchman_scope_plan(
        target_dir,
        {"watch": str(target_dir), "relative_path": "."},
    )

    assert plan.primary_scope.relative_root is None


def test_discover_nested_linux_mount_roots_reads_mountinfo_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = (tmp_path / "workspace_root").resolve()
    nested_mount = (target_dir / "chunkhound_workspace").resolve()
    unrelated_mount = (tmp_path / "elsewhere").resolve()
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)
    unrelated_mount.mkdir(parents=True)
    mountinfo_path = tmp_path / "mountinfo"
    mountinfo_path.write_text(
        "\n".join(
            (
                f"24 23 0:45 / {target_dir} rw,relatime - overlay overlay rw",
                f"25 24 0:46 / {nested_mount} rw,relatime - ext4 /dev/sdb rw",
                f"26 24 0:47 / {unrelated_mount} rw,relatime - ext4 /dev/sdc rw",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(watchman_scope_module.sys, "platform", "linux")

    assert discover_nested_linux_mount_roots(
        target_dir,
        mountinfo_path=mountinfo_path,
    ) == (nested_mount,)


def test_build_watchman_scope_plan_adds_nested_mount_scopes(tmp_path: Path) -> None:
    target_dir = (tmp_path / "workspace_root").resolve()
    nested_mount = (target_dir / "chunkhound_workspace").resolve()
    target_dir.mkdir(parents=True)
    nested_mount.mkdir(parents=True)

    plan = build_watchman_scope_plan(
        target_dir,
        {"watch": str(target_dir)},
        nested_mount_roots=(nested_mount,),
    )

    assert plan.primary_scope.requested_path == target_dir
    assert len(plan.scopes) == 2
    assert plan.scopes[1] == WatchmanSubscriptionScope(
        requested_path=nested_mount,
        watch_root=nested_mount,
        relative_root=None,
        scope_kind="nested_mount",
    )


def test_build_watchman_scope_plan_adds_nested_junction_scopes(tmp_path: Path) -> None:
    target_dir = (tmp_path / "workspace_root").resolve()
    logical_junction = target_dir / "linked_workspace"
    physical_root = (tmp_path / "external_workspace").resolve()
    target_dir.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)

    plan = build_watchman_scope_plan(
        target_dir,
        {"watch": str(target_dir)},
        additional_scopes=(
            WatchmanSubscriptionScope(
                requested_path=logical_junction,
                watch_root=physical_root,
                relative_root=None,
                scope_kind="nested_junction",
            ),
        ),
    )

    assert len(plan.scopes) == 2
    assert plan.scopes[1] == WatchmanSubscriptionScope(
        requested_path=logical_junction,
        watch_root=physical_root,
        relative_root=None,
        scope_kind="nested_junction",
    )


def test_build_watchman_scope_plan_preserves_logical_root_when_target_resolves_elsewhere(  # noqa: E501
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = tmp_path / "logical_workspace"
    logical_junction = target_dir / "linked_workspace"
    physical_root = tmp_path / "physical_workspace"
    physical_junction_root = tmp_path / "physical_linked_workspace"
    target_dir.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)
    physical_junction_root.mkdir(parents=True)

    original_resolve = watchman_scope_module.Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == target_dir:
            return physical_root
        if self == logical_junction:
            return physical_junction_root
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(watchman_scope_module.Path, "resolve", fake_resolve)

    plan = build_watchman_scope_plan(
        target_dir,
        {"watch": str(physical_root)},
        additional_scopes=(
            WatchmanSubscriptionScope(
                requested_path=logical_junction,
                watch_root=physical_junction_root,
                relative_root=None,
                scope_kind="nested_junction",
            ),
        ),
    )

    assert plan.primary_scope == WatchmanSubscriptionScope(
        requested_path=target_dir,
        watch_root=physical_root,
        relative_root=None,
        scope_kind="primary",
    )
    assert plan.scopes[1] == WatchmanSubscriptionScope(
        requested_path=logical_junction,
        watch_root=physical_junction_root,
        relative_root=None,
        scope_kind="nested_junction",
    )


def test_discover_nested_windows_junction_scopes_detects_external_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_dir = (tmp_path / "workspace_root").resolve()
    logical_junction = target_dir / "linked_workspace"
    physical_root = (tmp_path / "external_workspace").resolve()
    target_dir.mkdir(parents=True)
    logical_junction.mkdir(parents=True)
    physical_root.mkdir(parents=True)

    class _FakeDirEntry:
        path = str(logical_junction)

        def is_dir(self, *, follow_symlinks: bool = False) -> bool:
            return True

        def stat(self, *, follow_symlinks: bool = False):
            class _FakeStat:
                st_file_attributes = getattr(
                    watchman_scope_module.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
                )

            return _FakeStat()

    class _FakeScandir:
        def __init__(self, current_path: str | os.PathLike[str]):
            self._current_path = Path(current_path)

        def __enter__(self):
            if self._current_path == target_dir:
                return [_FakeDirEntry()]
            return []

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    original_resolve = watchman_scope_module.Path.resolve

    def fake_resolve(self: Path, strict: bool = False) -> Path:
        if self == logical_junction:
            return physical_root
        return original_resolve(self, strict=strict)

    monkeypatch.setattr(watchman_scope_module.sys, "platform", "win32")
    monkeypatch.setattr(watchman_scope_module.os, "scandir", _FakeScandir)
    monkeypatch.setattr(watchman_scope_module.Path, "resolve", fake_resolve)

    assert watchman_scope_module.discover_nested_windows_junction_scopes(
        target_dir
    ) == (
        WatchmanSubscriptionScope(
            requested_path=logical_junction,
            watch_root=physical_root,
            relative_root=None,
            scope_kind="nested_junction",
        ),
    )
