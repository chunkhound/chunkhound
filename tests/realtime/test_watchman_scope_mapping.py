from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.watchman import (
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
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
