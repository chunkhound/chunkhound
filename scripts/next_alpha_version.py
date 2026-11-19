"""Allocate the next automatic alpha version from PyPI state.

Usage: uv run scripts/next_alpha_version.py

Queries PyPI JSON for chunkhound and chunkhound-native. Only releases with at
least one published file count. The latest published chunkhound version picks
the line:

- latest stable X.Y.Z            -> X.(Y+1).0a1
- latest alpha X.Y.Z aN          -> X.Y.Z a(max N for that exact core + 1),
                                    counted across both packages
- anything else (beta, RC, dev,  -> fail; no guessing
  post, local, unsupported)

Writes version= to GITHUB_OUTPUT when set, otherwise prints it.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any

from packaging.version import InvalidVersion, Version

PACKAGES = ("chunkhound", "chunkhound-native")
MAIN_PACKAGE = "chunkhound"

Fetch = Callable[[str], dict[str, Any] | None]


def pypi_json(package: str) -> dict[str, Any] | None:
    """Return fresh PyPI JSON for a package, or None when the project is missing."""
    url = f"https://pypi.org/pypi/{package}/json?cache_bust={time.time_ns()}"
    request = urllib.request.Request(
        url,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def published_versions(package: str, fetch: Fetch = pypi_json) -> list[Version]:
    """Return parsed versions of a package that have at least one published file."""
    data = fetch(package)
    if data is None:
        return []
    releases = data.get("releases")
    if not isinstance(releases, dict):
        raise ValueError(f"PyPI JSON for {package} has no releases mapping")
    versions: list[Version] = []
    for text, files in releases.items():
        if not isinstance(files, list) or not files:
            continue  # empty release entry: not published
        try:
            versions.append(Version(text))
        except InvalidVersion as exc:
            raise ValueError(
                f"PyPI JSON for {package} has malformed version {text!r}"
            ) from exc
    return versions


def _is_plain_alpha(version: Version) -> bool:
    return (
        version.pre is not None
        and version.pre[0] == "a"
        and version.post is None
        and version.dev is None
        and version.local is None
    )


def _is_plain_stable(version: Version) -> bool:
    return (
        version.pre is None
        and version.post is None
        and version.dev is None
        and version.local is None
    )


def next_version(main: Iterable[Version], others: Iterable[Version]) -> str:
    """Return the next automatic alpha given chunkhound and native versions."""
    main_versions = list(main)
    if not main_versions:
        raise ValueError(
            f"{MAIN_PACKAGE} has no published releases; cannot determine release core"
        )
    latest = max(main_versions)
    all_versions = [*main_versions, *others]

    if _is_plain_stable(latest):
        core_text = f"{latest.major}.{latest.minor + 1}.0"
        # Normally this starts at a1. If native was partially published first,
        # advance past it rather than trying to reuse an immutable PyPI version.
        on_core = [v for v in all_versions if v.base_version == core_text]
        if any(not _is_plain_alpha(v) for v in on_core):
            raise ValueError(
                f"A non-alpha release of {core_text} already exists; "
                "an alpha of it is unsafe"
            )
        highest = max((v.pre[1] for v in on_core if _is_plain_alpha(v)), default=0)
        return str(Version(f"{core_text}a{highest + 1}"))

    if not _is_plain_alpha(latest):
        raise ValueError(
            f"Latest {MAIN_PACKAGE} version {latest} is not a stable or alpha "
            "release; refusing to guess the next alpha"
        )

    core_text = f"{latest.major}.{latest.minor}.{latest.micro}"
    on_core = [v for v in all_versions if v.base_version == core_text]
    if any(not _is_plain_alpha(v) for v in on_core):
        raise ValueError(
            f"A non-alpha release of {core_text} already exists; "
            "an alpha of it is unsafe"
        )
    highest = max((v.pre[1] for v in on_core if _is_plain_alpha(v)), default=0)
    return str(Version(f"{core_text}a{highest + 1}"))


def allocate(fetch: Fetch = pypi_json) -> str:
    main = published_versions(MAIN_PACKAGE, fetch)
    others: list[Version] = []
    for package in PACKAGES:
        if package != MAIN_PACKAGE:
            others.extend(published_versions(package, fetch))
    return next_version(main, others)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    version = allocate()
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a") as output:
            output.write(f"version={version}\n")
    else:
        print(f"version={version}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
