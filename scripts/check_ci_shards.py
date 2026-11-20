"""Verify actual pytest shard membership without running test bodies."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


def capture(output: Path, arguments: list[str]) -> int:
    import pytest

    class Capture:
        def pytest_collection_finish(self, session: pytest.Session) -> None:
            output.write_text(
                json.dumps([item.nodeid for item in session.items], indent=2) + "\n",
                encoding="utf-8",
            )

    return int(pytest.main([*arguments, "--collect-only", "-q"], plugins=[Capture()]))


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
    if any(
        item == option or item.startswith(option + "=")
        for item in arguments
        for option in ("--group", "--splits", "--store-durations")
    ):
        parser.error("the checker owns shard groups and never stores durations")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    memberships: dict[str, list[str]] = {}
    for group in range(4):
        name = "unsplit" if group == 0 else f"shard-{group}"
        shard_arguments = [] if group == 0 else ["--splits=3", f"--group={group}"]
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
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
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
    if (
        not unsplit
        or len(unsplit) != len(set(unsplit))
        or any(not shard or len(shard) != len(set(shard)) for shard in shards)
        or combined != Counter(unsplit)
        or any(count != 1 for count in combined.values())
    ):
        sys.stderr.write(f"Shards are not a complete, disjoint partition: {output}\n")
        return 1
    summary = {
        "unsplit": len(unsplit),
        "shards": [len(shard) for shard in shards],
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
    raise SystemExit(main())
