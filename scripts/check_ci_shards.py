"""Verify actual pytest shard membership without running test bodies."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARD_PLUGIN = "scripts.pytest_equal_shard"


class ShardContractError(RuntimeError):
    pass


def require_shard_plugin() -> None:
    try:
        module = importlib.import_module(SHARD_PLUGIN)
    except ImportError as error:
        raise ShardContractError(f"cannot import {SHARD_PLUGIN}: {error}") from error
    if not callable(getattr(module, "pytest_collection_modifyitems", None)):
        raise ShardContractError(f"{SHARD_PLUGIN} lacks pytest_collection_modifyitems")


def capture(output: Path, arguments: list[str]) -> int:
    import pytest

    class Capture:
        def pytest_collection_finish(self, session: pytest.Session) -> None:
            output.write_text(
                json.dumps([item.nodeid for item in session.items], indent=2) + "\n",
                encoding="utf-8",
            )

    return int(
        pytest.main(
            ["-p", SHARD_PLUGIN, *arguments, "--collect-only", "-q"],
            plugins=[Capture()],
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("pytest_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.pytest_arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        parser.error("provide a test target after --")
    output = args.output.resolve()
    if args.capture:
        return capture(output, arguments)
    require_shard_plugin()
    if any(
        item == option or item.startswith(option + "=")
        for item in arguments
        for option in ("--group", "--splits", "--ci-group", "--ci-splits")
    ):
        parser.error("the checker owns shard groups; use the caller's test target only")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    memberships: dict[str, list[str]] = {}
    for group in range(4):
        name = "unsplit" if group == 0 else f"shard-{group}"
        shard_arguments = [] if group == 0 else ["--ci-splits=3", f"--ci-group={group}"]
        destination = output / f"{name}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--capture",
            "--output",
            str(destination),
            "--",
            *arguments,
            *shard_arguments,
        ]
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
                    ),
                },
            )
        (output / f"{name}.exit").write_text(
            str(completed.returncode) + "\n", encoding="utf-8"
        )
        if completed.returncode != 0 or not destination.is_file():
            sys.stderr.write(f"Collection failed: {name}; inspect {output}\n")
            return 1
        memberships[name] = json.loads(destination.read_text(encoding="utf-8"))
    unsplit = memberships["unsplit"]
    shards = [memberships[f"shard-{group}"] for group in range(1, 4)]
    combined = Counter(node for shard in shards for node in shard)
    sizes = [len(shard) for shard in shards]
    expected_sizes = [len(unsplit) // 3 + (i < len(unsplit) % 3) for i in range(3)]
    if (
        not unsplit
        or len(unsplit) != len(set(unsplit))
        or any(not shard or len(shard) != len(set(shard)) for shard in shards)
        or combined != Counter(unsplit)
        or any(count != 1 for count in combined.values())
        or sizes != expected_sizes
        or any(shard != unsplit[group::3] for group, shard in enumerate(shards))
    ):
        sys.stderr.write(f"Shards are not a complete, disjoint partition: {output}\n")
        return 1
    summary = {
        "unsplit": len(unsplit),
        "shards": sizes,
        "hub_tests": [
            sum(node.startswith("tests/hub/") for node in shard) for shard in shards
        ],
        "collection_seconds": time.monotonic() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ShardContractError as error:
        sys.stderr.write(f"Shard contract error: {error}\n")
        raise SystemExit(1) from error
