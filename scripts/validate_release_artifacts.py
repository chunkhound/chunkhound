"""Gate publication on explicit-tag metadata and a clean sdist rebuild.

Run with RELEASE_TAG=vX.Y.Z[preN] and ALPHA_RELEASE=true for alpha pairing.
The sdist rebuild deliberately receives neither a VCS version override nor the
alpha build flag: the published source must retain both contracts on its own.
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

# Support direct execution from any working directory, including sdist tests.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_tag import parse_release_tag


def check_metadata(text: str, name: str, version: str, alpha: bool) -> None:
    metadata = Parser().parsestr(text)
    if canonicalize_name(metadata.get("Name", "")) != name:
        raise SystemExit(f"Expected package {name}, found {metadata.get('Name')!r}")
    if metadata.get("Version") != version:
        raise SystemExit(
            f"{name} metadata version {metadata.get('Version')!r}; expected {version!r}"
        )
    if alpha and name == "chunkhound":
        native = [
            Requirement(value)
            for value in metadata.get_all("Requires-Dist", [])
            if canonicalize_name(Requirement(value).name) == "chunkhound-native"
        ]
        if (
            len(native) != 1
            or str(native[0].specifier) != f"=={version}"
            or native[0].marker is not None
            or native[0].extras
            or native[0].url is not None
        ):
            raise SystemExit(
                f"Alpha must require chunkhound-native=={version} unconditionally"
            )


def check_embedded_version(source: str, version: str) -> None:
    """Read the generated version file without importing release code."""
    versions = []
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
        ):
            if isinstance(node.value, ast.Constant):
                versions.append(node.value.value)
    if not versions or any(value != version for value in versions):
        raise SystemExit(f"Embedded main version {versions!r}; expected {version!r}")


def check_wheel(path: Path, version: str, alpha: bool) -> str:
    name, filename_version, _, _ = parse_wheel_filename(path.name)
    if name not in {"chunkhound", "chunkhound-native"}:
        raise SystemExit(f"Unexpected release package: {path.name}")
    if str(filename_version) != version:
        raise SystemExit(f"{path.name} filename version does not match {version}")
    with zipfile.ZipFile(path) as archive:
        metadata = [
            item for item in archive.namelist() if item.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise SystemExit(f"{path.name} must have exactly one METADATA file")
        check_metadata(archive.read(metadata[0]).decode(), name, version, alpha)
        if name == "chunkhound":
            check_embedded_version(
                archive.read("chunkhound/_version.py").decode(), version
            )
    return name


def check_sdist(path: Path, version: str, alpha: bool) -> None:
    if path.name != f"chunkhound-{version}.tar.gz":
        raise SystemExit(f"Unexpected source distribution: {path.name}")
    root = f"chunkhound-{version}"
    with tarfile.open(path) as archive:
        for member_path, check in (
            (
                f"{root}/PKG-INFO",
                lambda text: check_metadata(text, "chunkhound", version, alpha),
            ),
            (
                f"{root}/chunkhound/_version.py",
                lambda text: check_embedded_version(text, version),
            ),
        ):
            member = archive.extractfile(member_path)
            if member is None:
                raise SystemExit(f"Missing {member_path} in source distribution")
            check(member.read().decode())


def rebuild_sdist(path: Path, version: str, alpha: bool) -> None:
    with tempfile.TemporaryDirectory(
        prefix="chunkhound-release-roundtrip-"
    ) as directory:
        root = Path(directory)
        with tarfile.open(path) as archive:
            archive.extractall(root, filter="data")
        source = root / f"chunkhound-{version}"
        environment = os.environ.copy()
        # Neither Git discovery nor a release-only environment may rescue an
        # incorrectly generated published sdist. Build the ordinary wheel here;
        # runtime hydration was already covered by the platform-wheel jobs.
        for key in list(environment):
            if key.startswith("SETUPTOOLS_SCM_PRETEND_VERSION") or key in {
                "CHUNKHOUND_BUILD_ALPHA_RELEASE",
                "CHUNKHOUND_BUILD_PACKAGED_WATCHMAN_RUNTIME",
            }:
                environment.pop(key)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(root / "rebuilt")],
            cwd=source,
            env=environment,
            check=True,
        )
        wheels = list((root / "rebuilt").glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit("Expected exactly one wheel rebuilt from the final sdist")
        check_wheel(wheels[0], version, alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("artifacts"))
    parser.add_argument("--rebuild-sdist", action="store_true")
    args = parser.parse_args()
    try:
        release = parse_release_tag(os.environ["RELEASE_TAG"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    version = release.version
    alpha = os.environ.get("ALPHA_RELEASE", "false") == "true"
    if alpha and not release.alpha:
        raise SystemExit("Alpha publication requires an alpha tag")
    wheels = sorted(args.dist.glob("*.whl"))
    names = {check_wheel(wheel, version, alpha) for wheel in wheels}
    if names != {"chunkhound", "chunkhound-native"}:
        raise SystemExit("Both main and native wheel artifacts are required")
    sdists = list(args.dist.glob("*.tar.gz"))
    if len(sdists) != 1:
        raise SystemExit("Expected exactly one source distribution")
    check_sdist(sdists[0], version, alpha)
    if args.rebuild_sdist:
        rebuild_sdist(sdists[0], version, alpha)


if __name__ == "__main__":
    main()
