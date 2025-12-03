#!/usr/bin/env uv run python3
"""
Create a new ChunkHound version tag with PEP 440 support.
"""

import argparse
import re
import subprocess
import sys


def get_current_version() -> str:
    """Get the current version from the latest git tag."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().lstrip("v")
    except subprocess.CalledProcessError:
        return "0.0.0"


def validate_version(version: str) -> None:
    """Validate version format (PEP 440 compliant).

    Supports:
    - Standard releases: 4.1.0
    - Pre-releases: 4.1.0a1, 4.1.0b2, 4.1.0rc3
    - Alternative spellings: 4.1.0alpha1, 4.1.0beta2
    """
    pattern = r"^\d+\.\d+\.\d+(?:(?:a|alpha|b|beta|rc|c|preview|pre)\d+)?$"
    if not re.match(pattern, version):
        raise ValueError(
            f"Invalid version format: {version}. "
            f"Expected: X.Y.Z or X.Y.Z{{a|b|rc}}N (e.g., 4.1.0, 4.1.0b1, 4.1.0rc3)"
        )


def parse_version(version_str: str) -> tuple[int, int, int, str | None]:
    """Parse version string into (major, minor, patch, prerelease) tuple.

    Examples:
        "4.1.0" -> (4, 1, 0, None)
        "4.1.0b2" -> (4, 1, 0, "b2")
        "4.1.0rc1" -> (4, 1, 0, "rc1")
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(.+)?$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    major, minor, patch, prerelease = match.groups()
    return (int(major), int(minor), int(patch), prerelease)


def bump_version(current: str, bump_type: str, prerelease: str | None = None) -> str:
    """Bump version based on type (major, minor, patch).

    Args:
        current: Current version string
        bump_type: One of 'major', 'minor', 'patch'
        prerelease: Optional pre-release suffix (e.g., 'a1', 'b2', 'rc1')

    Examples:
        bump_version("4.0.1", "minor") -> "4.1.0"
        bump_version("4.0.1", "minor", "b1") -> "4.1.0b1"
        bump_version("4.1.0b1", "patch") -> "4.1.1" (removes prerelease)
    """
    major, minor, patch, _ = parse_version(current)

    if bump_type == "major":
        new_version = f"{major + 1}.0.0"
    elif bump_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")

    if prerelease:
        new_version += prerelease

    return new_version


def create_tag(version: str) -> str:
    """Create annotated git tag.

    Args:
        version: Version string (e.g., "4.1.0", "4.1.0b1")

    Returns:
        Created tag name (e.g., "v4.1.0")

    Raises:
        RuntimeError: If tag already exists or working tree is dirty
    """
    tag = f"v{version}"

    # Check for uncommitted changes
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status_result.stdout.strip():
        raise RuntimeError(
            "Working directory has uncommitted changes. "
            "Commit or stash changes before creating a version tag."
        )

    # Check if tag already exists
    result = subprocess.run(
        ["git", "tag", "-l", tag],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"Tag {tag} already exists")

    # Create annotated tag
    subprocess.run(
        ["git", "tag", "-a", tag, "-m", f"Release {version}"],
        check=True,
    )
    print(f"Created tag {tag}")
    return tag


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create PEP 440 compliant version tags for ChunkHound",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 4.1.0           # Release version
  %(prog)s 4.1.0b1         # Beta pre-release
  %(prog)s 4.1.0rc2        # Release candidate
  %(prog)s --bump minor    # Bump to next minor
  %(prog)s --bump minor b1 # Bump to next minor beta
        """,
    )

    # Mutually exclusive: either explicit version OR bump
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "version",
        nargs="?",
        help="Explicit version to set (e.g., 4.1.0, 4.1.0b1)",
    )
    group.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version semantically",
    )

    parser.add_argument(
        "prerelease",
        nargs="?",
        help="Optional pre-release suffix (e.g., b1, rc2) for --bump",
    )

    args = parser.parse_args()

    try:
        if args.bump:
            # Bump mode
            current = get_current_version()
            version = bump_version(current, args.bump, args.prerelease)
            print(f"Bumping version from {current} to {version}")
        else:
            # Explicit version mode
            version = args.version
            validate_version(version)
            print(f"Creating version {version}")

        tag = create_tag(version)

        print(f"\nSuccessfully created version {version}")
        print("\nNext steps:")
        print(f"1. Push the tag: git push origin {tag}")
        print("2. Build will automatically use version from git tag")
        print("\nNote: Version is now managed via git tags (hatch-vcs)")
        print("      No need to manually update version files!")

    except (ValueError, RuntimeError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
