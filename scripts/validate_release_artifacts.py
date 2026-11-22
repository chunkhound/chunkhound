"""Gate publication on the requested release tag, exact pins, and a clean sdist rebuild.

Run with RELEASE_TAG=vX.Y.Z[(a|b|rc)N]. The tag alone selects the version and
channel for every official release (alpha, beta, release candidate, or stable).
Each channel must pin chunkhound-native to the exact main version, both in the
wheel metadata and in the source distribution's persisted pyproject.toml.
The sdist rebuild deliberately receives neither a VCS version override nor any
build-release flag: the published source must retain both contracts on its own.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 validator environments
    import tomli as tomllib

# Support direct execution from any working directory, including sdist tests.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_tag import parse_release_tag

# Inherited build controls that must not influence a clean sdist rebuild.
RELEASE_BUILD_CONTROLS = {
    "CHUNKHOUND_BUILD_RELEASE_ARTIFACT",
    "CHUNKHOUND_BUILD_ALPHA_RELEASE",
    "CHUNKHOUND_BUILD_PACKAGED_WATCHMAN_RUNTIME",
}


def _normalize(value: object, label: str) -> Version:
    try:
        return Version(str(value))
    except InvalidVersion as exc:
        raise SystemExit(f"{label} is not a valid version: {value!r}") from exc


def _native_requirements(values: list[str]) -> list[Requirement]:
    return [
        requirement
        for requirement in map(Requirement, values)
        if canonicalize_name(requirement.name) == "chunkhound-native"
    ]


def _is_exact_pin(requirement: Requirement, version: Version) -> bool:
    specifiers = list(requirement.specifier)
    return (
        len(specifiers) == 1
        and specifiers[0].operator == "=="
        and not specifiers[0].version.endswith(".*")
        and Version(specifiers[0].version) == version
        and requirement.marker is None
        and not requirement.extras
        and requirement.url is None
    )


def check_metadata(text: str, name: str, version: str, release_artifact: bool) -> None:
    expected = _normalize(version, "Release version")
    metadata = Parser().parsestr(text)
    if canonicalize_name(metadata.get("Name", "")) != name:
        raise SystemExit(f"Expected package {name}, found {metadata.get('Name')!r}")
    actual = _normalize(metadata.get("Version", ""), f"{name} metadata version")
    if actual != expected:
        raise SystemExit(
            f"{name} metadata version {metadata.get('Version')!r}; expected {expected}"
        )
    if release_artifact and name == "chunkhound":
        native = _native_requirements(metadata.get_all("Requires-Dist", []))
        if len(native) != 1 or not _is_exact_pin(native[0], expected):
            raise SystemExit(
                "Official release must require "
                f"chunkhound-native=={expected} unconditionally"
            )


def check_embedded_version(source: str, version: str) -> None:
    """Read the generated version file without importing release code."""
    expected = _normalize(version, "Release version")
    values = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(
            isinstance(target, ast.Name) and target.id in {"version", "__version__"}
            for target in targets
        ) and isinstance(node.value, ast.Constant):
            values.append(node.value.value)
    if not values or any(
        not isinstance(value, str)
        or _normalize(value, "Embedded main version") != expected
        for value in values
    ):
        raise SystemExit(f"Embedded main version {values!r}; expected {expected}")


def _check_sdist_pyproject(text: str, version: str) -> None:
    expected = _normalize(version, "Release version")
    try:
        config = tomllib.loads(text)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        raise SystemExit("Source archive pyproject.toml is invalid TOML") from exc
    dependencies = (
        config.get("tool", {})
        .get("hatch", {})
        .get("metadata", {})
        .get("hooks", {})
        .get("custom", {})
        .get("dependencies", [])
    )
    native = _native_requirements(dependencies)
    if len(native) != 1 or not _is_exact_pin(native[0], expected):
        raise SystemExit(
            "Official source archive must persist "
            f"chunkhound-native=={expected} in pyproject.toml"
        )


def check_wheel(path: Path, version: str, release_artifact: bool) -> str:
    name, filename_version, _, _ = parse_wheel_filename(path.name)
    if name not in {"chunkhound", "chunkhound-native"}:
        raise SystemExit(f"Unexpected release package: {path.name}")
    expected = _normalize(version, "Release version")
    if filename_version != expected:
        raise SystemExit(f"{path.name} filename version does not match {expected}")
    with zipfile.ZipFile(path) as archive:
        metadata = [
            item for item in archive.namelist() if item.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise SystemExit(f"{path.name} must have exactly one METADATA file")
        check_metadata(
            archive.read(metadata[0]).decode(), name, str(expected), release_artifact
        )
        if name == "chunkhound":
            check_embedded_version(
                archive.read("chunkhound/_version.py").decode(), str(expected)
            )
    return name


def check_sdist(path: Path, version: str, release_artifact: bool) -> str:
    """Return the archive's root directory; sdist members live beneath it."""
    expected = _normalize(version, "Release version")
    prefix, suffix = "chunkhound-", ".tar.gz"
    if not (path.name.startswith(prefix) and path.name.endswith(suffix)):
        raise SystemExit(f"Unexpected source distribution: {path.name}")
    file_version = path.name[len(prefix) : -len(suffix)]
    if _normalize(file_version, "Source distribution version") != expected:
        raise SystemExit(f"{path.name} filename version does not match {expected}")
    root = path.name[: -len(suffix)]
    members = [
        (f"{root}/PKG-INFO", "PKG-INFO"),
        (f"{root}/chunkhound/_version.py", "_version.py"),
        (f"{root}/pyproject.toml", "pyproject.toml"),
    ]
    with tarfile.open(path) as archive:
        for member_path, label in members:
            member = archive.extractfile(member_path)
            if member is None:
                raise SystemExit(f"Missing {label} in source distribution")
            text = member.read().decode()
            if label == "PKG-INFO":
                check_metadata(text, "chunkhound", str(expected), release_artifact)
            elif label == "_version.py":
                check_embedded_version(text, str(expected))
            elif release_artifact:
                _check_sdist_pyproject(text, str(expected))
    return root


def rebuild_sdist(path: Path, version: str, release_artifact: bool) -> None:
    with tempfile.TemporaryDirectory(
        prefix="chunkhound-release-roundtrip-"
    ) as directory:
        work = Path(directory)
        with tarfile.open(path) as archive:
            archive.extractall(work, filter="data")
        source = work / check_sdist(path, version, release_artifact)
        environment = os.environ.copy()
        # Neither Git discovery nor inherited release controls may rescue an
        # incorrectly generated published sdist. The persisted pyproject.toml
        # alone supplies the native pin for the ordinary wheel rebuild.
        for key in list(environment):
            if key.startswith("SETUPTOOLS_SCM_PRETEND_VERSION") or (
                key in RELEASE_BUILD_CONTROLS
            ):
                environment.pop(key)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(work / "rebuilt")],
            cwd=source,
            env=environment,
            check=True,
        )
        wheels = list((work / "rebuilt").glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit("Expected exactly one wheel rebuilt from the final sdist")
        check_wheel(wheels[0], version, release_artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("artifacts"))
    parser.add_argument("--rebuild-sdist", action="store_true")
    args = parser.parse_args()
    try:
        release = parse_release_tag(os.environ["RELEASE_TAG"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    version = str(Version(release.version))
    release_artifact = True
    wheels = sorted(args.dist.glob("*.whl"))
    names = {check_wheel(wheel, version, release_artifact) for wheel in wheels}
    if names != {"chunkhound", "chunkhound-native"}:
        raise SystemExit("Both main and native wheel artifacts are required")
    sdists = list(args.dist.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise SystemExit("Expected exactly one source distribution")
    check_sdist(sdists[0], version, release_artifact)
    if args.rebuild_sdist:
        rebuild_sdist(sdists[0], version, release_artifact)


if __name__ == "__main__":
    main()
