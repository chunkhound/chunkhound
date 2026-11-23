"""Release tools agree on tag syntax, package versions, and source identity."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.release_tag import parse_release_tag

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/release_tag.py"


@pytest.mark.parametrize(
    ("tag", "version", "cargo_version", "channel"),
    [
        ("v6.2.0", "6.2.0", "6.2.0", "stable"),
        ("v6.2.0a43", "6.2.0a43", "6.2.0-a43", "alpha"),
        ("v6.2.0b2", "6.2.0b2", "6.2.0-b2", "beta"),
        ("v6.2.0rc1", "6.2.0rc1", "6.2.0-rc1", "rc"),
    ],
)
def test_release_tag_agrees_with_python_and_cargo_versions(
    tmp_path, tag, version, cargo_version, channel
):
    release = parse_release_tag(tag)
    assert (release.version, release.cargo_version, release.channel) == (
        version,
        cargo_version,
        channel,
    )
    assert release.alpha == (channel == "alpha")
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        '[dependencies.other]\nversion = "1.0.0"\n\n'
        '[package]\nname = "chunkhound_native"\nversion = "0.1.0" # keep comment\n'
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "patch-cargo", "--tag", tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    parsed = tomllib.loads(cargo.read_text())
    assert parsed["package"]["version"] == cargo_version
    assert parsed["dependencies"]["other"]["version"] == "1.0.0"
    assert "# keep comment" in cargo.read_text()


@pytest.mark.parametrize(
    "tag",
    [
        "6.2.0",
        "v6.2",
        "v6.2.0-a1",
        "v6.2.0dev1",
        "v6.2.0.post1",
        "v6.2.0+local",
        "v6.2.0a",
        "v6.2.0rc1\n",
        "refs/tags/v6.2.0",
    ],
)
def test_release_tag_rejects_unsupported_versions_without_modifying_cargo(
    tmp_path, tag
):
    with pytest.raises(ValueError, match="Invalid release tag"):
        parse_release_tag(tag)
    cargo = tmp_path / "Cargo.toml"
    original = '[package]\nversion = "0.1.0"\n'
    cargo.write_text(original)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "patch-cargo", "--tag", tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Invalid release tag" in result.stderr
    assert cargo.read_text() == original


def test_prerelease_gate_checks_tag_and_checkout_before_writing_outputs(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, stderr=subprocess.PIPE
        ).strip()

    git("init")
    git(
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "release",
    )
    source = git("rev-parse", "HEAD")
    for tag in ("v6.2.0a1", "v6.2.0b1", "v6.2.0rc1", "v6.2.0"):
        git(
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.invalid",
            "tag",
            "-m",
            "release",
            tag,
        )
    output = tmp_path / "outputs"

    def validate(tag, *extra):
        output.unlink(missing_ok=True)
        return subprocess.run(
            [sys.executable, str(SCRIPT), "validate", "--tag", tag, *extra],
            cwd=tmp_path,
            env={**os.environ, "GITHUB_OUTPUT": str(output)},
            capture_output=True,
            text=True,
        )

    for tag in ("v6.2.0a1", "v6.2.0b1", "v6.2.0rc1"):
        result = validate(tag, "--prerelease-only")
        assert result.returncode == 0, result.stderr
        assert f"source_sha={source}\n" in output.read_text()
    result = validate("v6.2.0", "--prerelease-only")
    assert result.returncode != 0
    assert "pre-release tag" in result.stderr
    assert not output.exists()
    result = validate("v6.2.0", "--source-sha", source[:8])
    assert result.returncode != 0
    assert "full commit SHA" in result.stderr
    assert not output.exists()
    result = validate("v9.9.9")
    assert result.returncode != 0
    assert not output.exists()

    git(
        "-c",
        "user.name=CI",
        "-c",
        "user.email=ci@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "different checkout",
    )
    result = validate("v6.2.0", "--source-sha", source)
    assert result.returncode != 0
    assert "checkout does not match" in result.stderr
    assert not output.exists()


def test_explicit_alpha_version_is_validated_against_checkout_without_tag(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, stderr=subprocess.PIPE
        ).strip()

    git("init")
    identity = ["-c", "user.name=CI", "-c", "user.email=ci@example.invalid"]
    git(*identity, "commit", "--allow-empty", "-m", "source")
    source = git("rev-parse", "HEAD")
    output = tmp_path / "outputs"

    def validate(*args):
        output.unlink(missing_ok=True)
        return subprocess.run(
            [sys.executable, str(SCRIPT), "validate", *args],
            cwd=tmp_path,
            env={**os.environ, "GITHUB_OUTPUT": str(output)},
            capture_output=True,
            text=True,
        )

    result = validate("--version", "6.2.0a5", "--source-sha", source)
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["version"] == "6.2.0a5"
    assert outputs["cargo_version"] == "6.2.0-a5"
    assert outputs["alpha"] == "true"
    assert outputs["source_sha"] == source

    for version in ("6.2.0", "6.2.0b1", "6.2.0rc1", "v6.2.0a5", "6.2"):
        result = validate("--version", version, "--source-sha", source)
        assert result.returncode != 0
        assert "Invalid explicit version" in result.stderr
        assert not output.exists()

    result = validate(
        "--version", "6.2.0a5", "--tag", "v6.2.0a5", "--source-sha", source
    )
    assert result.returncode != 0
    assert "Exactly one of --tag or --version" in result.stderr
    result = validate("--source-sha", source)
    assert result.returncode != 0
    assert "Exactly one of --tag or --version" in result.stderr

    result = validate("--version", "6.2.0a5", "--source-sha", "0" * 40)
    assert result.returncode != 0
    assert "does not match requested source" in result.stderr
    assert not output.exists()


def test_explicit_alpha_version_patches_cargo_without_tag(tmp_path):
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text('[package]\nname = "chunkhound_native"\nversion = "0.1.0"\n')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "patch-cargo", "--version", "6.2.0a5"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert tomllib.loads(cargo.read_text())["package"]["version"] == "6.2.0-a5"
