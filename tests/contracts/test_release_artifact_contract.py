"""The publication gate checks artifacts, not whichever tag Git discovers."""

import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def make_artifacts(directory: Path, *, version: str, native_requirement: str) -> None:
    metadata = (
        f"Metadata-Version: 2.4\nName: chunkhound\nVersion: {version}\n"
        f"Requires-Dist: chunkhound-native{native_requirement}\n"
    )
    embedded = f"__version__ = version = {version!r}\n"
    with zipfile.ZipFile(
        directory / f"chunkhound-{version}-py3-none-any.whl", "w"
    ) as archive:
        archive.writestr(f"chunkhound-{version}.dist-info/METADATA", metadata)
        archive.writestr("chunkhound/_version.py", embedded)
    with zipfile.ZipFile(
        directory / f"chunkhound_native-{version}-cp310-abi3-manylinux_2_31_x86_64.whl",
        "w",
    ) as archive:
        archive.writestr(
            f"chunkhound_native-{version}.dist-info/METADATA",
            f"Name: chunkhound-native\nVersion: {version}\n",
        )
    with tarfile.open(directory / f"chunkhound-{version}.tar.gz", "w:gz") as archive:
        for filename, content in {
            "PKG-INFO": metadata,
            "chunkhound/_version.py": embedded,
        }.items():
            payload = content.encode()
            member = tarfile.TarInfo(f"chunkhound-{version}/{filename}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


@pytest.mark.parametrize(
    ("version", "tag", "alpha", "requirement", "valid"),
    [
        ("6.2.0a43", "v6.2.0a43", True, "==6.2.0a43", True),
        ("6.2.0", "v6.2.0", False, ">=5.2.0", True),
        ("6.2.0a43", "v6.2.0a44", True, "==6.2.0a43", False),
        ("6.2.0a43", "v6.2.0a43", True, ">=5.2.0", False),
        ("6.2.0", "v6.2.0", True, "==6.2.0", False),
    ],
)
def test_publication_gate_requires_requested_version_and_alpha_pairing(
    tmp_path, version, tag, alpha, requirement, valid
):
    make_artifacts(tmp_path, version=version, native_requirement=requirement)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_release_artifacts.py"),
            "--dist",
            str(tmp_path),
        ],
        env={**os.environ, "RELEASE_TAG": tag, "ALPHA_RELEASE": str(alpha).lower()},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == valid, result.stdout + result.stderr


def test_publication_gate_rejects_incorrect_embedded_version(tmp_path):
    make_artifacts(tmp_path, version="6.2.0", native_requirement=">=5.2.0")
    wheel = tmp_path / "chunkhound-6.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "chunkhound-6.2.0.dist-info/METADATA", "Name: chunkhound\nVersion: 6.2.0\n"
        )
        archive.writestr("chunkhound/_version.py", "__version__ = version = '5.2.0'\n")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_release_artifacts.py"),
            "--dist",
            str(tmp_path),
        ],
        env={**os.environ, "RELEASE_TAG": "v6.2.0", "ALPHA_RELEASE": "false"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Embedded main version" in result.stderr


def test_requested_tag_wins_when_multiple_tags_point_to_same_commit(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            "release source",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    for tag in ["v6.2.0a43", "v6.2.0", "v7.0.0"]:
        subprocess.run(["git", "tag", tag], cwd=tmp_path, check=True)
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/release-packages.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["prepare"]["steps"]
        if step.get("id") == "release"
    )
    script = step["run"].split("<<'PYCODE'\n", 1)[1].split("\nPYCODE", 1)[0]
    output = tmp_path / "outputs"
    environment = {
        **os.environ,
        "RELEASE_TAG": "v6.2.0a43",
        "SOURCE_SHA": source,
        "ALPHA_RELEASE": "true",
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "version=6.2.0a43\n" in output.read_text()
    environment["SOURCE_SHA"] = "0" * 40
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "does not resolve to requested source" in result.stderr


def test_release_workflows_keep_native_first_and_isolate_alpha_approvals():
    # PyYAML uses YAML 1.1, so the GitHub Actions `on` key is parsed as True.
    shared = yaml.safe_load(
        (ROOT / ".github/workflows/release-packages.yml").read_text()
    )
    inputs = shared[True]["workflow_call"]["inputs"]
    assert inputs["release_tag"] == {"required": True, "type": "string"}
    assert inputs["source_sha"] == {"required": True, "type": "string"}
    assert inputs["alpha"]["default"] is False
    jobs = shared["jobs"]
    assert "publish-native" in jobs["publish"]["needs"]
    assert {"validate-release", "validate-artifacts"} <= set(
        jobs["publish-native"]["needs"]
    )
    assert "pypi-alpha-native" in jobs["publish-native"]["environment"]
    assert "pypi-alpha" in jobs["publish"]["environment"]
    assert jobs["build-native-wheel"]["strategy"]["matrix"]["os"] == [
        "ubuntu-latest",
        "ubuntu-24.04-arm",
        "macos-latest",
        "windows-latest",
    ]
    assert jobs["build-runtime-wheels"]["strategy"]["matrix"]["os"] == [
        "ubuntu-latest",
        "windows-latest",
    ]
    stable = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    assert stable[True] == {"release": {"types": ["published"]}}
    assert stable["jobs"]["release"]["with"]["alpha"] is False
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                assert step["with"]["overwrite"] is True
            if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@"):
                assert step["with"]["skip-existing"] is True
