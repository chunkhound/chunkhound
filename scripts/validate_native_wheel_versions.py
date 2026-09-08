"""Validate that downloaded chunkhound-native wheel artifacts match a release tag.

Checks both the wheel filename version and the package's actual dist-info
METADATA (name + version), guarding against a mismatched build artifact
reaching PyPI. Reads the release tag from the RELEASE_TAG env var and the
wheels from a "dist-native" directory relative to the current working
directory. Shared by release.yml and release-rc.yml's publish-native jobs,
which previously each carried their own byte-identical copy of this check.
"""

from __future__ import annotations

import os
import re
import zipfile
from email.parser import Parser
from pathlib import Path


def main() -> None:
    tag = os.environ["RELEASE_TAG"]
    match = re.fullmatch(r"v(\d+\.\d+\.\d+(?:[a-zA-Z].*)?)", tag)
    if match is None:
        raise SystemExit(f"Malformed release tag: {tag!r}")
    expected_version = match.group(1)

    wheels = sorted(Path("dist-native").glob("*.whl"))
    if not wheels:
        raise SystemExit("No native wheel artifacts were downloaded")

    for wheel in wheels:
        parts = wheel.name.removesuffix(".whl").split("-")
        if len(parts) != 5 or parts[0] != "chunkhound_native":
            raise SystemExit(f"Malformed native wheel filename: {wheel.name}")
        if parts[1] != expected_version:
            raise SystemExit(
                f"Native wheel {wheel.name} has filename version {parts[1]!r}; "
                f"expected {expected_version!r}"
            )

        try:
            with zipfile.ZipFile(wheel) as archive:
                metadata_files = [
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_files) != 1:
                    raise SystemExit(
                        f"Native wheel {wheel.name} has malformed metadata"
                    )
                metadata = Parser().parsestr(
                    archive.read(metadata_files[0]).decode("utf-8")
                )
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise SystemExit(f"Cannot read native wheel {wheel.name}: {exc}") from exc

        package_name = metadata.get("Name")
        normalized_name = re.sub(r"[-_.]+", "-", package_name or "").lower()
        if normalized_name != "chunkhound-native":
            raise SystemExit(
                f"Native wheel {wheel.name} has package name {package_name!r}"
            )
        if metadata.get("Version") != expected_version:
            raise SystemExit(
                f"Native wheel {wheel.name} has metadata version "
                f"{metadata.get('Version')!r}; expected {expected_version!r}"
            )


if __name__ == "__main__":
    main()
