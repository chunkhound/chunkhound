"""Public indexing and regex search must retain large source content."""

import os
import subprocess
from pathlib import Path

import pytest

from tests.utils.windows_subprocess import get_safe_subprocess_env


@pytest.fixture
def cli_env(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CHUNKHOUND_")
    }
    # Restore the pinned empty global config so CLI subprocesses stay isolated
    # from developer-wide settings (session-autouse fixture in conftest.py).
    if global_config := os.environ.get("CHUNKHOUND_GLOBAL_CONFIG_FILE"):
        env["CHUNKHOUND_GLOBAL_CONFIG_FILE"] = global_config
    env.update(
        CHUNKHOUND_NO_PROMPTS="1",
        CHUNKHOUND_NO_RICH="1",
        UV_CACHE_DIR=str(tmp_path / ".uv-cache"),
    )
    return get_safe_subprocess_env(env)


def run_cli(root: Path, env: dict[str, str], *args: str) -> str:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "chunkhound", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def source_block(kind: str, identifier: str, value: str) -> str:
    if kind == "python":
        return f'def {identifier}():\n    return "{value}"\n\n'
    if kind == "makefile":
        return f'{identifier}:\n\t@echo "{value}"\n\n'
    return f"{identifier}\n{value}\n\n"


def large_source(kind: str) -> tuple[str, list[str]]:
    markers = [
        f"retention_{position}_sentinel"
        for position in ("beginning", "middle", "end", "longline")
    ]
    blocks = []
    for index in range(151):
        identifier = {0: markers[0], 75: markers[1], 150: markers[2]}.get(
            index, f"unique_record_{index:04d}"
        )
        value = " ".join(f"record_{index:04d}_field_{field:03d}" for field in range(20))
        blocks.append(source_block(kind, identifier, value))
    long_value = " ".join(f"long_field_{index:04d}" for index in range(1000))
    blocks.insert(100, source_block(kind, "oversized_record", long_value + markers[3]))
    return "".join(blocks), markers


@pytest.mark.parametrize(
    "kind,filename",
    [
        ("python", "retained_source.py"),
        ("makefile", "Makefile"),
        ("text", "retained_source.txt"),
    ],
)
def test_indexed_content_remains_searchable(tmp_path, cli_env, kind, filename):
    source, markers = large_source(kind)
    # Stay below legacy Windows path limits; POSIX also exercises long paths.
    depth = 1 if os.name == "nt" else 4
    directory = tmp_path.joinpath(
        *[f"long_source_directory_{index}_" + "x" * 45 for index in range(depth)]
    )
    directory.mkdir(parents=True)
    (directory / filename).write_text(source, encoding="utf-8")
    run_cli(tmp_path, cli_env, "index", str(tmp_path), "--no-embeddings")
    for marker in markers:
        output = run_cli(tmp_path, cli_env, "search", marker, str(tmp_path), "--regex")
        # CLI previews can truncate content; the matching file proves availability.
        assert filename in output, f"Indexed content missing: {marker}\n{output}"
