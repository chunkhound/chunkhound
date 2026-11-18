"""Allocate an immutable alpha tag for one CI run, using only Git remote refs.

An atomic push creates the annotated release tag AND a run-identity tag pointing
to a metadata blob. GitHub supports atomic pushes and immutable tag creation;
this also prevents concurrent executions of the same run allocating two versions.
Do not delete/move either tag namespace. Run IDs (not attempts or source SHAs)
identify builds, so reruns reuse a tag even after a new stable tag moves the base.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ALPHA_TAG = re.compile(r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)a[1-9]\d*")
STABLE_TAG = re.compile(r"v((?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*))")


def git(*args: str, input_text: str | None = None) -> str:
    # Git object input requires LF bytes, not Windows text-mode CRLF translation.
    result = subprocess.run(
        ["git", *args],
        input=input_text.encode("utf-8") if input_text is not None else None,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8").strip()


def remote_tags(remote: str) -> dict[str, str]:
    return {
        ref.removeprefix("refs/tags/"): oid
        for oid, ref in (
            line.split() for line in git("ls-remote", "--tags", remote).splitlines()
        )
        if not ref.endswith("^{}")
    }


def alpha_base(tags: dict[str, str]) -> str:
    """Return the next minor after the highest stable vX.Y.Z tag: (X, Y+1, 0)."""
    stable = [
        tuple(map(int, match[1].split(".")))
        for tag in tags
        if (match := STABLE_TAG.fullmatch(tag))
    ]
    if not stable:
        raise ValueError("No stable vX.Y.Z tag found on the remote; cannot pick a base")
    major, minor, _ = max(stable)
    return f"{major}.{minor + 1}.0"


def metadata(oid: str) -> dict:
    if git("cat-file", "-t", oid) != "tag":
        return {}
    message = git("cat-file", "tag", oid).partition("\n\n")[2]
    try:
        value = json.loads(message)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def find_existing(
    tags: dict[str, str], identity: str, repository: str, run_id: str, source_sha: str
) -> str | None:
    """Verify both immutable refs and reject ambiguous or inconsistent identities."""
    matches = []
    for tag, oid in tags.items():
        if not ALPHA_TAG.fullmatch(tag):
            continue
        data = metadata(oid)
        if data.get("repository") == repository and data.get("run_id") == run_id:
            matches.append(tag)
    if len(matches) > 1:
        raise ValueError(f"Multiple alpha tags claim this build: {matches}")
    if identity not in tags:
        if matches:
            raise ValueError("Build tag exists without its atomic run-identity tag")
        return None
    if len(matches) != 1:
        raise ValueError("Run identity has no corresponding annotated alpha tag")
    tag = matches[0]
    data = metadata(tags[tag])
    if (
        data
        != {
            "schema": 1,
            "repository": repository,
            "run_id": run_id,
            "source_sha": source_sha,
            "release_tag": tag,
        }
        or not isinstance(tag, str)
        or not ALPHA_TAG.fullmatch(tag)
        or matches != [tag]
        or git("cat-file", "-t", tags[identity]) != "blob"
        or git("cat-file", "blob", tags[identity]) != json.dumps(data, sort_keys=True)
        or git("rev-parse", f"{tags[tag]}^{{commit}}") != source_sha
    ):
        raise ValueError(
            "Run identity does not match the requested build and release tag"
        )
    return tag


def allocate(repository: str, run_id: str, source_sha: str, remote: str) -> str:
    identity = f"alpha-build/{repository}/{run_id}"
    for _ in range(100):
        tags = remote_tags(remote)
        # Use a private local namespace: never force-update release refs. Only
        # remote refs in this snapshot count, not possibly stale local tags.
        git(
            "fetch",
            "--quiet",
            "--no-tags",
            remote,
            "+refs/tags/*:refs/alpha-allocator/*",
        )
        existing = find_existing(tags, identity, repository, run_id, source_sha)
        if existing:
            return existing
        # Base and numbering both come from this same remote snapshot.
        base = alpha_base(tags)
        pattern = re.compile(rf"v{re.escape(base)}a([1-9]\d*)")
        numbers = [int(match[1]) for tag in tags if (match := pattern.fullmatch(tag))]
        tag = f"v{base}a{max(numbers, default=0) + 1}"
        message = json.dumps(
            {
                "schema": 1,
                "repository": repository,
                "run_id": run_id,
                "source_sha": source_sha,
                "release_tag": tag,
            },
            sort_keys=True,
        )
        oid = git(
            "mktag",
            input_text=(
                f"object {source_sha}\ntype commit\ntag {tag}\n"
                f"tagger ChunkHound CI <actions@chunkhound.com> "
                f"{int(time.time())} +0000\n\n"
                f"{message}\n"
            ),
        )
        identity_oid = git("hash-object", "-w", "--stdin", input_text=message)
        result = subprocess.run(
            [
                "git",
                "push",
                "--atomic",
                remote,
                f"{oid}:refs/tags/{tag}",
                # Blob identity is unique to this allocation, so a competing
                # tag for the same run cannot silently push an up-to-date ref.
                # Non-commit tags are ignored by Git describe; aliases of the
                # release object break its version discovery (misnamed tags).
                f"{identity_oid}:refs/tags/{identity}",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode == 0:
            return tag
        # Retry allocation races only, not permissions/network/atomic-support
        # failures. A lost acknowledgement is safe: the next pass reuses identity.
        current = remote_tags(remote)
        if tag not in current and identity not in current:
            raise ValueError(f"Atomic tag push failed: {result.stderr.strip()}")
    raise ValueError("Alpha allocation exceeded 100 concurrent collision retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
            raise ValueError("repository must be owner/name")
        if not re.fullmatch(r"[1-9]\d*", args.run_id):
            raise ValueError("run-id must be a positive GitHub run ID")
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
            raise ValueError("source-sha must be a full commit SHA")
        if git("rev-parse", "HEAD") != args.source_sha:
            raise ValueError("Checkout HEAD must be the exact CI-tested source SHA")
        tag = allocate(args.repository, args.run_id, args.source_sha, args.remote)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        sys.exit(f"Alpha allocation failed: {error}\n{details}")
    except (ValueError, OSError) as error:
        sys.exit(f"Alpha allocation failed: {error}")
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as stream:
            stream.write(f"release_tag={tag}\nsource_sha={args.source_sha}\n")
    sys.stdout.write(f"{tag}\n")


if __name__ == "__main__":
    main()
