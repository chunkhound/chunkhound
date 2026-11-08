"""CI must execute every collected test in exactly one shard."""

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_ci_shards.py"


@pytest.mark.parametrize("broken_collection", [False, True])
def test_ci_shard_check_requires_a_complete_collection(
    tmp_path: Path, broken_collection: bool
) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    source = (
        "raise RuntimeError('collection-contract-failure')\n"
        if broken_collection
        else (
            "import pytest\n"
            "@pytest.mark.parametrize('value', range(11))\n"
            "def test_sample(value):\n"
            "    assert value >= 0\n"
        )
    )
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(evidence),
            "--",
            "test_sample.py",
            "--splitting-algorithm=least_duration",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if broken_collection:
        assert result.returncode != 0
        assert not (evidence / "summary.json").exists()
        assert "collection-contract-failure" in (evidence / "unsplit.log").read_text(
            encoding="utf-8"
        )
        return
    assert result.returncode == 0, result.stdout + result.stderr
    unsplit = json.loads((evidence / "unsplit.json").read_text(encoding="utf-8"))
    shards = [
        json.loads((evidence / f"shard-{group}.json").read_text(encoding="utf-8"))
        for group in range(1, 4)
    ]
    assert len(unsplit) == 11
    assert Counter(node for shard in shards for node in shard) == Counter(unsplit)
    assert all(len(shard) == len(set(shard)) for shard in shards)
    assert all(
        set(shards[left]).isdisjoint(shards[right])
        for left in range(3)
        for right in range(left + 1, 3)
    )
    assert all(shards)
