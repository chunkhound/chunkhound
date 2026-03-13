from __future__ import annotations

import pytest

import hatch_build


def test_load_supported_watchman_platforms_matches_declared_slots() -> None:
    assert hatch_build._load_supported_watchman_platforms() == {
        "linux-x86_64",
        "windows-x86_64",
    }


@pytest.mark.parametrize(
    ("system_name", "machine_name", "expected_platform"),
    [
        ("Linux", "amd64", "linux-x86_64"),
        ("Windows", "AMD64", "windows-x86_64"),
    ],
)
def test_require_supported_build_host_accepts_declared_slots(
    system_name: str,
    machine_name: str,
    expected_platform: str,
) -> None:
    supported_platforms = hatch_build._load_supported_watchman_platforms()

    assert (
        hatch_build._require_supported_build_host(
            supported_platforms,
            system_name=system_name,
            machine_name=machine_name,
        )
        == expected_platform
    )


def test_require_supported_build_host_rejects_unsupported_host() -> None:
    supported_platforms = hatch_build._load_supported_watchman_platforms()

    with pytest.raises(RuntimeError, match="linux-arm64"):
        hatch_build._require_supported_build_host(
            supported_platforms,
            system_name="Linux",
            machine_name="aarch64",
        )


def test_custom_build_hook_hydrates_runtime_for_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "linux-x86_64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: (calls.append("hydrated") or {"src": "dst"}),
    )

    build_data: dict[str, object] = {}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("0.0.0", build_data)

    assert calls == ["hydrated"]
    assert build_data["force_include"] == {"src": "dst"}
    assert build_data["pure_python"] is False
    assert isinstance(build_data["tag"], str)


def test_custom_build_hook_skips_native_hydration_on_unsupported_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hatch_build,
        "_host_watchman_platform",
        lambda **_: "macos-arm64",
    )
    monkeypatch.setattr(
        hatch_build,
        "_hydrate_runtime_for_build",
        lambda: pytest.fail("unsupported hosts should not hydrate native runtime"),
    )

    build_data: dict[str, object] = {"force_include": {"existing": "entry"}}
    hook = object.__new__(hatch_build.CustomBuildHook)
    hook.initialize("0.0.0", build_data)

    assert build_data == {"force_include": {"existing": "entry"}}
