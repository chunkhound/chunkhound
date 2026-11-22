"""CI must execute every collected test in exactly one shard."""

import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_ci_shards.py"
EXPECTED_SHARDS = [
    [f"test_sample.py::test_sample[{value}]" for value in range(0, 4)],
    [f"test_sample.py::test_sample[{value}]" for value in range(4, 8)],
    [f"test_sample.py::test_sample[{value}]" for value in range(8, 11)],
]


def _write_sample(tmp_path: Path, source: str) -> None:
    (tmp_path / "pytest.ini").write_text("[pytest]\naddopts =\n", encoding="utf-8")
    (tmp_path / "test_sample.py").write_text(source, encoding="utf-8")


def _run_checker(tmp_path: Path, evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(evidence),
            "--",
            "test_sample.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _load_shards(evidence: Path) -> tuple[list[str], list[list[str]]]:
    unsplit = json.loads((evidence / "unsplit.json").read_text(encoding="utf-8"))
    shards = [
        json.loads((evidence / f"shard-{group}.json").read_text(encoding="utf-8"))
        for group in range(1, 4)
    ]
    return unsplit, shards


def test_ci_shards_are_deterministic_equal_contiguous_slices(tmp_path: Path) -> None:
    _write_sample(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(11))\n"
        "def test_sample(value):\n"
        "    assert value >= 0\n",
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _run_checker(tmp_path, first)
    second_result = _run_checker(tmp_path, second)
    assert first_result.returncode == 0, first_result.stdout + first_result.stderr
    assert second_result.returncode == 0, second_result.stdout + second_result.stderr

    unsplit, shards = _load_shards(first)
    _, repeated_shards = _load_shards(second)
    assert len(unsplit) == 11
    assert [len(shard) for shard in shards] == [4, 4, 3]
    assert shards == EXPECTED_SHARDS
    assert shards == repeated_shards
    assert shards[0] + shards[1] + shards[2] == unsplit
    assert Counter(node for shard in shards for node in shard) == Counter(unsplit)


def test_ci_shard_check_fails_when_shared_plugin_is_unavailable(
    tmp_path: Path,
) -> None:
    _write_sample(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(11))\n"
        "def test_sample(value):\n"
        "    assert value >= 0\n",
    )
    # A regular package at the temp root shadows the repo's namespace `scripts`.
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    checker = tmp_path / "checker.py"
    shutil.copy(SCRIPT, checker)
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            str(checker),
            "--output",
            str(evidence),
            "--",
            "test_sample.py",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "pytest_equal_shard" in result.stdout + result.stderr
    assert not (evidence / "summary.json").exists()


@pytest.mark.parametrize("broken_collection", [False, True])
def test_ci_shard_check_requires_a_complete_collection(
    tmp_path: Path, broken_collection: bool
) -> None:
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
    _write_sample(tmp_path, source)
    evidence = tmp_path / "evidence"
    result = _run_checker(tmp_path, evidence)
    if broken_collection:
        assert result.returncode != 0
        assert not (evidence / "summary.json").exists()
        assert "collection-contract-failure" in (evidence / "unsplit.log").read_text(
            encoding="utf-8"
        )
        return
    assert result.returncode == 0, result.stdout + result.stderr
    unsplit, shards = _load_shards(evidence)
    assert len(unsplit) == 11
    assert Counter(node for shard in shards for node in shard) == Counter(unsplit)
    assert all(len(shard) == len(set(shard)) for shard in shards)
    assert all(shards)
