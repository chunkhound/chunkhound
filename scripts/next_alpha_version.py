"""Allocate the next automatic alpha version from PyPI state.

Usage: uv run scripts/next_alpha_version.py

Queries PyPI JSON for chunkhound and chunkhound-native. Only releases with at
least one published file count. The latest published chunkhound version picks
the line:

- latest stable X.Y.Z            -> X.(Y+1).0a1
- latest alpha X.Y.Z aN          -> X.Y.Z a(max N for that exact core + 1),
                                    counted across both packages

Only plain stable and plain alpha releases choose the line. Beta, RC, dev,
post, and local releases are ignored for that choice. If a beta or RC exists
on the selected core, the script fails instead of allocating an alpha.

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


def _next_alpha_on_core(core_text: str, all_versions: list[Version]) -> str:
    """Return the next alpha on core_text, refusing when a non-alpha release exists."""
    # Normally this starts at a1. If native was partially published first,
    # advance past it rather than trying to reuse an immutable PyPI version.
    # PEP 440 treats trailing release zeros as equivalent (6.2 == 6.2.0).
    # Compare Version objects so equivalent cores match without merging epochs.
    core = Version(core_text)
    on_core = [v for v in all_versions if Version(v.base_version) == core]
    if any(not _is_plain_alpha(v) for v in on_core):
        raise ValueError(
            f"A non-alpha release of {core_text} already exists (beta, RC, or "
            "stable); an alpha of it is unsafe, so no alpha is allocated"
        )
    highest = max((v.pre[1] for v in on_core if _is_plain_alpha(v)), default=0)
    return str(Version(f"{core_text}a{highest + 1}"))


def next_version(main: Iterable[Version], others: Iterable[Version]) -> str:
    """Return the next automatic alpha given chunkhound and native versions.

    Only plain stable and plain alpha chunkhound releases choose the line.
    Beta, RC, dev, post, and local releases are ignored for that choice.
    """
    main_versions = list(main)
    if not main_versions:
        raise ValueError(
            f"{MAIN_PACKAGE} has no published releases; cannot determine release core"
        )
    candidates = [v for v in main_versions if _is_plain_stable(v) or _is_plain_alpha(v)]
    if not candidates:
        raise ValueError(
            f"{MAIN_PACKAGE} has no stable or alpha release to pick the line from; "
            "refusing to guess the next alpha"
        )
    latest = max(candidates)
    all_versions = [*main_versions, *others]

    if _is_plain_stable(latest):
        core_text = f"{latest.major}.{latest.minor + 1}.0"
    else:
        core_text = f"{latest.major}.{latest.minor}.{latest.micro}"
    return _next_alpha_on_core(core_text, all_versions)


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
