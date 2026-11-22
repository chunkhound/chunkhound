"""Release artifacts retain native dependency semantics outside the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
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
            "CHUNKHOUND_BUILD_RELEASE_ARTIFACT",
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
    ("version", "release_artifact", "expected_specifier"),
    [
        ("6.2.0", False, ">=6.1.0"),
        ("6.2.0a10", False, ">=6.1.0"),
        ("6.2.0b1", False, ">=6.1.0"),
        ("6.2.0rc1", False, ">=6.1.0"),
        ("6.2.0", True, "==6.2.0"),
        ("6.2.0a10", True, "==6.2.0a10"),
        ("6.2.0b1", True, "==6.2.0b1"),
        ("6.2.0rc1", True, "==6.2.0rc1"),
    ],
)
def test_public_build_round_trip_preserves_native_dependency_policy(
    tmp_path: Path, version: str, release_artifact: bool, expected_specifier: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for filename in ("pyproject.toml", "hatch_build.py", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, source / filename)
    shutil.copytree(ROOT / "chunkhound", source / "chunkhound")
    original_config = (source / "pyproject.toml").read_bytes()
    environment = _build_environment()
    environment["SETUPTOOLS_SCM_PRETEND_VERSION"] = version
    if release_artifact:
        environment["CHUNKHOUND_BUILD_RELEASE_ARTIFACT"] = "1"
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
        native_config = [
            Requirement(value)
            for value in dependencies
            if canonicalize_name(Requirement(value).name) == "chunkhound-native"
        ]
        assert len(native_config) == 1
        assert str(native_config[0].specifier) == expected_specifier
        assert native_config[0].marker is None
        assert not native_config[0].extras and native_config[0].url is None
        info_member = archive.extractfile(f"chunkhound-{version}/PKG-INFO")
        assert info_member is not None
        assert str(_native_requirement(info_member.read().decode()).specifier) == (
            expected_specifier
        )

    with tempfile.TemporaryDirectory(prefix="chunkhound-test-sdist-") as directory:
        root = Path(directory)
        with tarfile.open(sdist) as archive:
            archive.extractall(root, filter="data")
        rebuilt = root / "rebuilt"
        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(rebuilt)],
            cwd=root / f"chunkhound-{version}",
            env=_build_environment(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        rebuilt_wheels = list(rebuilt.glob("*.whl"))
        assert len(rebuilt_wheels) == 1
        with zipfile.ZipFile(rebuilt_wheels[0]) as archive:
            metadata_path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            )
            rebuilt_requirement = _native_requirement(
                archive.read(metadata_path).decode()
            )
        assert str(rebuilt_requirement.specifier) == expected_specifier


@pytest.mark.parametrize("version", ["6.2.0.dev1", "6.2.0+local"])
def test_release_artifact_build_rejects_dev_and_local_versions(
    tmp_path: Path, version: str
) -> None:
    environment = _build_environment()
    environment.update(
        SETUPTOOLS_SCM_PRETEND_VERSION=version,
        CHUNKHOUND_BUILD_RELEASE_ARTIFACT="1",
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
    assert "Official release artifacts require a stable or a/b/rc version" in (
        result.stderr
    )
