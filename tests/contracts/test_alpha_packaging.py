"""Release artifacts retain native dependency semantics outside the checkout."""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

import hatch_build

ROOT = Path(__file__).resolve().parents[2]


def _build_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SETUPTOOLS_SCM_PRETEND_VERSION")
        and key
        not in {
            "CHUNKHOUND_BUILD_ALPHA_RELEASE",
            "CHUNKHOUND_BUILD_PACKAGED_WATCHMAN_RUNTIME",
        }
    }


def _native_requirement(metadata_text: str) -> Requirement:
    metadata = Parser().parsestr(metadata_text)
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    native = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "chunkhound-native"
    ]
    assert len(native) == 1
    return native[0]


@pytest.mark.parametrize(
    ("version", "alpha", "expected_specifier"),
    [
        ("6.2.0", False, None),
        ("6.2.0rc1", False, None),
        ("6.2.0a43", False, None),
        ("6.2.0a43", True, "==6.2.0a43"),
    ],
)
def test_release_artifacts_preserve_native_dependency_on_source_rebuild(
    tmp_path: Path, version: str, alpha: bool, expected_specifier: str | None
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, source / filename)
    shutil.copytree(ROOT / "chunkhound", source / "chunkhound")
    original_config = (source / "pyproject.toml").read_bytes()
    expected_dependencies = hatch_build.tomllib.loads(original_config.decode())[
        "tool"
    ]["hatch"]["metadata"]["hooks"]["custom"]["dependencies"]
    if expected_specifier is None:
        native = [
            Requirement(value)
            for value in expected_dependencies
            if canonicalize_name(Requirement(value).name) == "chunkhound-native"
        ]
        assert len(native) == 1
        expected_specifier = str(native[0].specifier)
    environment = _build_environment()
    environment["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
    if alpha:
        environment["CHUNKHOUND_BUILD_ALPHA_RELEASE"] = "1"
    artifacts = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--sdist", "--wheel", "--out-dir", str(artifacts)],
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (source / "pyproject.toml").read_bytes() == original_config

    wheel = next(artifacts.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        requirement = _native_requirement(archive.read(metadata_name).decode())
    assert str(requirement.specifier) == expected_specifier
    assert requirement.marker is None and not requirement.extras and not requirement.url

    sdist = artifacts / f"chunkhound-{version}.tar.gz"
    with tarfile.open(sdist) as archive:
        config = archive.extractfile(f"chunkhound-{version}/pyproject.toml")
        assert config is not None
        persisted = hatch_build.tomllib.loads(config.read().decode())
        dependencies = persisted["tool"]["hatch"]["metadata"]["hooks"]["custom"][
            "dependencies"
        ]
        if alpha:
            expected_dependencies = [
                f"chunkhound-native=={version}"
                if canonicalize_name(Requirement(value).name) == "chunkhound-native"
                else value
                for value in expected_dependencies
            ]
        assert dependencies == expected_dependencies
        package_info = archive.extractfile(f"chunkhound-{version}/PKG-INFO")
        assert package_info is not None
        assert str(_native_requirement(package_info.read().decode()).specifier) == (
            expected_specifier
        )

    # Use the real publication checker, which clears all release environment
    # variables before rebuilding. No Watchman payload is needed for this wheel.
    checker = runpy.run_path(str(ROOT / "scripts/validate_release_artifacts.py"))
    checker["check_sdist"](sdist, version, alpha)
    checker["rebuild_sdist"](sdist, version, alpha)


def test_alpha_release_build_rejects_a_stable_version(tmp_path: Path) -> None:
    environment = _build_environment()
    environment.update(
        SETUPTOOLS_SCM_PRETEND_VERSION="6.2.0", CHUNKHOUND_BUILD_ALPHA_RELEASE="1"
    )
    result = subprocess.run(
        ["uv", "build", "--sdist", "--out-dir", str(tmp_path)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "Alpha release builds require an alpha release version" in result.stderr
