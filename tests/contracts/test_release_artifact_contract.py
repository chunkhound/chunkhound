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


@pytest.mark.parametrize(
    ("version", "alpha"),
    [
        ("6.2.0a43", "true"),
        ("6.2.0b1", "false"),
        ("6.2.0rc1", "false"),
        ("6.2.0", "false"),
    ],
)
def test_requested_tag_sets_version_and_channel_without_caller_flag(
    tmp_path, version, alpha
):
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
    for tag in ["v6.2.0a43", "v6.2.0b1", "v6.2.0rc1", "v6.2.0", "v7.0.0"]:
        subprocess.run(["git", "tag", tag], cwd=tmp_path, check=True)
    output = tmp_path / "outputs"
    environment = {
        **os.environ,
        "GITHUB_OUTPUT": str(output),
    }
    command = [
        sys.executable,
        str(ROOT / "scripts/release_tag.py"),
        "validate",
        "--tag",
        f"v{version}",
        "--source-sha",
        source,
    ]
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["version"] == version
    assert outputs["source_sha"] == source
    assert outputs["alpha"] == alpha
    command[-1] = "0" * 40
    result = subprocess.run(
        command,
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
    assert "alpha" not in inputs
    jobs = shared["jobs"]
    assert "publish-native" in jobs["publish"]["needs"]
    assert {"validate-release", "validate-artifacts"} <= set(
        jobs["publish-native"]["needs"]
    )
    assert jobs["prepare"]["outputs"]["alpha"] == "${{ steps.release.outputs.alpha }}"
    for job_name in [
        "build-sdist",
        "build-runtime-wheels",
        "validate-artifacts",
        "publish-native",
        "publish",
    ]:
        assert "prepare" in jobs[job_name]["needs"]
    for job_name in ["build-sdist", "build-runtime-wheels"]:
        build = next(
            step
            for step in jobs[job_name]["steps"]
            if "CHUNKHOUND_BUILD_ALPHA_RELEASE" in step.get("env", {})
        )
        assert build["env"]["CHUNKHOUND_BUILD_ALPHA_RELEASE"] == (
            "${{ needs.prepare.outputs.alpha == 'true' && '1' || '' }}"
        )
    validation = jobs["validate-artifacts"]["steps"][-1]
    assert validation["env"]["ALPHA_RELEASE"] == "${{ needs.prepare.outputs.alpha }}"
    for job_name, suffix in [("publish", ""), ("publish-native", "-native")]:
        assert jobs[job_name]["environment"] == (
            "${{ needs.prepare.outputs.alpha == 'true' && "
            f"'pypi-alpha{suffix}' || 'pypi{suffix}'"
            " }}"
        )
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
    assert (
        stable["jobs"]["release"]["uses"] == "./.github/workflows/release-packages.yml"
    )
    assert stable["jobs"]["release"]["with"]["release_tag"] == (
        "${{ github.event.release.tag_name }}"
    )
    assert "alpha" not in stable["jobs"]["release"]["with"]
    prerelease = yaml.safe_load((ROOT / ".github/workflows/release-rc.yml").read_text())
    assert "alpha" not in prerelease["jobs"]["release"]["with"]
    prepare = next(
        step for step in jobs["prepare"]["steps"] if step.get("id") == "release"
    )
    assert "scripts/release_tag.py validate" in prepare["run"]
    assert '--tag "$RELEASE_TAG" --source-sha "$SOURCE_SHA"' in prepare["run"]
    tag_gate = prerelease["jobs"]["validate-tag"]["steps"][-1]
    assert "scripts/release_tag.py validate" in tag_gate["run"]
    assert '--tag "$TAG_TO_VALIDATE" --prerelease-only' in tag_gate["run"]
    native_action = yaml.safe_load(
        (ROOT / ".github/actions/build-native-wheel/action.yml").read_text()
    )
    patch = next(
        step
        for step in native_action["runs"]["steps"]
        if step["name"] == "Patch Cargo.toml version from requested tag"
    )
    assert "scripts/release_tag.py" in patch["run"]
    assert 'patch-cargo --tag "$release_tag"' in patch["run"]
    wheel_gate = jobs["validate-release"]["steps"][-1]
    assert "scripts/validate_release_wheels.py" in wheel_gate["run"]
    assert "--main-dist main-dist --native-dist native-dist" in wheel_gate["run"]
    for job in jobs.values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/upload-artifact@"):
                assert step["with"]["overwrite"] is True
            if step.get("uses", "").startswith("pypa/gh-action-pypi-publish@"):
                assert step["with"]["skip-existing"] is True
                # PyPI rejects caller-workflow attestations when authentication
                # uses the reusable workflow's job_workflow_ref instead.
                assert step["with"]["attestations"] is False
