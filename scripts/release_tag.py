"""Shared release-tag rules for Python packaging, Cargo, and publication.

Validate a checked-out release (writes GitHub outputs when GITHUB_OUTPUT is set):
    uv run --no-project python scripts/release_tag.py validate --tag v6.2.0a1

Patch the native package version before building:
    uv run --no-project python scripts/release_tag.py patch-cargo --tag v6.2.0a1
"""

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReleaseTag:
    version: str
    cargo_version: str
    channel: str

    @property
    def alpha(self) -> bool:
        return self.channel == "alpha"


def parse_release_tag(tag: str) -> ReleaseTag:
    """Require vX.Y.Z with an optional PEP 440 alpha, beta, or RC suffix."""
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)((a|b|rc)\d+)?", tag)
    if not match:
        raise ValueError(
            f"Invalid release tag: {tag!r}; expected vX.Y.Z(a|b|rc)N or vX.Y.Z"
        )
    base, suffix, channel = match.groups()
    return ReleaseTag(
        version=base + (suffix or ""),
        cargo_version=f"{base}-{suffix}" if suffix else base,
        channel={"a": "alpha", "b": "beta", "rc": "rc", None: "stable"}[channel],
    )


def validate_source(tag: str, source: str | None) -> str:
    """The requested tag and checkout must both identify the requested commit."""
    if source is not None and not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("source_sha must be a full commit SHA")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source = source or head
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"], text=True
    ).strip()
    if resolved != source or head != source:
        raise ValueError(
            f"Release tag {tag} does not resolve to requested source {source} "
            "or the checkout does not match"
        )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate", help="Validate tag and checked-out source"
    )
    validate.add_argument("--tag", required=True)
    validate.add_argument("--source-sha")
    validate.add_argument("--prerelease-only", action="store_true")
    patch = commands.add_parser("patch-cargo", help="Set the native package version")
    patch.add_argument("--tag", required=True)
    patch.add_argument("--cargo-toml", type=Path, default=Path("Cargo.toml"))
    args = parser.parse_args()

    release = parse_release_tag(args.tag)
    if args.command == "patch-cargo":
        # Cargo requires a hyphen before prerelease suffixes; maturin converts
        # that SemVer back to the matching PEP 440 wheel version.
        text = args.cargo_toml.read_text()
        text, count = re.subn(
            r'(?ms)(^\[package\][ \t]*\n(?:(?!^\[).)*?^version[ \t]*=[ \t]*)"[^"\n]*"',
            lambda match: f'{match.group(1)}"{release.cargo_version}"',
            text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"No package version found in {args.cargo_toml}")
        args.cargo_toml.write_text(text)
        return

    if args.prerelease_only and release.channel == "stable":
        raise ValueError("Expected a PEP 440 pre-release tag: vX.Y.Z(a|b|rc)N")
    source = validate_source(args.tag, args.source_sha)
    outputs = (
        f"version={release.version}\nsource_sha={source}\n"
        f"alpha={str(release.alpha).lower()}\n"
        f"cargo_version={release.cargo_version}\nchannel={release.channel}\n"
    )
    if output_path := os.environ.get("GITHUB_OUTPUT"):
        with open(output_path, "a") as output:
            output.write(outputs)
    else:
        # This is a standalone release tool, not an MCP server.
        print(outputs, end="")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
