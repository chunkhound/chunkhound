"""Automatic alpha allocation follows the latest chunkhound line and never guesses."""

import os
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.next_alpha_version import allocate, next_version, pypi_json

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/next_alpha_version.py"


def fake_pypi(projects: dict[str, list[str] | None]):
    def fetch(package: str):
        releases = projects[package]
        if releases is None:
            return None
        return {"releases": {version: [{"filename": "x"}] for version in releases}}

    return fetch


def run(projects):
    return allocate(fake_pypi(projects))


@pytest.mark.parametrize(
    ("main", "native", "expected"),
    [
        (["6.1.0"], [], "6.2.0a1"),
        (["6.1.0", "6.1.1"], ["6.1.0"], "6.2.0a1"),
        (["6.1.0"], ["6.2.0a1"], "6.2.0a2"),
        (["6.2.0a3"], ["6.2.0a1"], "6.2.0a4"),
        (["6.2.0a1"], ["6.2.0a7"], "6.2.0a8"),
        (["6.2.0a2"], ["6.2.0a5"], "6.2.0a6"),
        (["6.2.0a0"], [], "6.2.0a1"),
        (["6.2.0a1"], ["6.3.0a9", "6.1.0"], "6.2.0a2"),
        (["6.2.0a1", "6.3.0a4"], ["6.3.0a9"], "6.3.0a10"),
    ],
)
def test_alpha_line_follows_latest_chunkhound_and_counts_both_packages(
    main, native, expected
):
    assert run({"chunkhound": main, "chunkhound-native": native}) == expected


@pytest.mark.parametrize(
    ("main", "native", "expected"),
    [
        (["6.1.0"], ["6.2a2"], "6.2.0a3"),
        (["6.2.0a1"], ["6.2a2"], "6.2.0a3"),
        (["6.2.0a1"], ["6.2.0.0a7"], "6.2.0a8"),
        (["6.2a3"], ["6.2.0a5"], "6.2.0a6"),
        (["6.2.0.0a3"], ["6.2a5"], "6.2.0a6"),
    ],
)
def test_equivalent_pep440_cores_share_alpha_serials(main, native, expected):
    assert run({"chunkhound": main, "chunkhound-native": native}) == expected


@pytest.mark.parametrize("main", ["6.1.0", "6.2.0a1"])
@pytest.mark.parametrize("native", ["6.2b1", "6.2.0.0rc1", "6.2"])
def test_non_alpha_on_equivalent_pep440_core_blocks_allocation(main, native):
    with pytest.raises(ValueError, match="non-alpha release of 6.2.0"):
        run({"chunkhound": [main], "chunkhound-native": [native]})


def test_different_epoch_does_not_share_alpha_core():
    assert (
        run({"chunkhound": ["6.2.0a1"], "chunkhound-native": ["1!6.2a9", "1!6.2b1"]})
        == "6.2.0a2"
    )


def test_stable_latest_starts_next_minor_alpha_one():
    assert run({"chunkhound": ["6.1.0", "6.2.0"], "chunkhound-native": []}) == (
        "6.3.0a1"
    )


def test_output_is_canonical_alpha_one_not_zero_padded():
    assert run({"chunkhound": ["6.1.0"], "chunkhound-native": None}) == "6.2.0a1"
    assert "a01" not in run({"chunkhound": ["6.2.0a0"], "chunkhound-native": None})


@pytest.mark.parametrize(
    "latest",
    ["6.2.0b1", "6.2.0rc1", "6.2.0.post1", "6.2.0.dev3", "6.2.0+local"],
)
def test_non_alpha_non_stable_latest_fails_instead_of_guessing(latest):
    with pytest.raises(ValueError, match="refusing to guess"):
        run({"chunkhound": [latest], "chunkhound-native": []})


@pytest.mark.parametrize(
    ("main", "native", "expected"),
    [
        (["6.2.0", "6.2.1rc1"], [], "6.3.0a1"),
        (["6.2.0", "6.2.1b1"], ["6.2.1.dev2"], "6.3.0a1"),
        (["6.2.0a5", "6.2.1b1"], [], "6.2.0a6"),
        (["6.2.0a5", "6.2.1rc1"], ["6.2.0a7"], "6.2.0a8"),
    ],
)
def test_beta_or_rc_latest_does_not_choose_the_line(main, native, expected):
    assert run({"chunkhound": main, "chunkhound-native": native}) == expected


def test_beta_on_new_core_stops_alpha_with_clear_error():
    with pytest.raises(ValueError, match="non-alpha release of 6.3.0"):
        run({"chunkhound": ["6.2.0", "6.3.0b1"], "chunkhound-native": []})


def test_non_alpha_release_on_selected_core_makes_alpha_unsafe():
    with pytest.raises(ValueError, match="non-alpha release"):
        run({"chunkhound": ["6.2.0a1"], "chunkhound-native": ["6.2.0"]})


def test_chunkhound_without_published_releases_fails_clearly():
    with pytest.raises(ValueError, match="no published releases"):
        run({"chunkhound": None, "chunkhound-native": ["6.2.0a1"]})


def test_empty_release_entries_are_not_published():
    projects = {"chunkhound": None, "chunkhound-native": None}

    def fetch(package):
        if package == "chunkhound":
            return {"releases": {"6.1.0": [{"filename": "x"}], "6.2.0a9": []}}
        return projects[package]

    assert allocate(fetch) == "6.2.0a1"


def test_native_alpha_on_other_core_does_not_change_selected_line():
    assert run({"chunkhound": ["6.2.0a1"], "chunkhound-native": ["7.0.0a3"]}) == (
        "6.2.0a2"
    )


def test_malformed_json_shape_fails_clearly():
    def fetch(package):
        return {"releases": ["6.1.0"]}

    with pytest.raises(ValueError, match="no releases mapping"):
        allocate(fetch)


def test_malformed_version_string_fails_clearly():
    def fetch(package):
        return {"releases": {"not-a-version": [{"filename": "x"}]}}

    with pytest.raises(ValueError, match="malformed version"):
        allocate(fetch)


def test_non_alpha_native_release_on_next_stable_line_fails():
    with pytest.raises(ValueError, match="non-alpha release"):
        next_version([Version("6.1.0")], [Version("6.2.0")])


def test_http_errors_other_than_missing_project_fail(monkeypatch):
    def boom(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)

    monkeypatch.setattr("scripts.next_alpha_version.urllib.request.urlopen", boom)
    with pytest.raises(urllib.error.HTTPError):
        pypi_json("chunkhound")


def test_missing_project_is_reported_as_no_releases(monkeypatch):
    def not_found(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "nf", {}, None)

    monkeypatch.setattr("scripts.next_alpha_version.urllib.request.urlopen", not_found)
    assert pypi_json("chunkhound") is None


def test_cli_rejects_retired_core_flag(tmp_path):
    output = tmp_path / "outputs"
    env = {**os.environ, "GITHUB_OUTPUT": str(output)}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--core", "6.2.0"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
