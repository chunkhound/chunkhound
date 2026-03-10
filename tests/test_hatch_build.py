from __future__ import annotations

import pytest

import hatch_build


def test_load_supported_watchman_platforms_matches_declared_slots() -> None:
    assert hatch_build._load_supported_watchman_platforms() == {
        "linux-x86_64",
        "macos-arm64",
        "macos-x86_64",
        "windows-x86_64",
    }


@pytest.mark.parametrize(
    ("system_name", "machine_name", "expected_platform"),
    [
        ("Linux", "amd64", "linux-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
        ("Darwin", "x86_64", "macos-x86_64"),
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
