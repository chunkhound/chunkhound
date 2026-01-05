from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 30,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(
        ["uv", "run", *cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=e,
    )


def test_full_run_profile_includes_git_counters(tmp_path: Path) -> None:
    # Create a tiny repo so backend can be forced to git_only
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    (repo / "a.py").write_text("print('ok')\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=ci@example.com",
            "-c",
            "user.name=CI",
            "commit",
            "-m",
            "init",
        ],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    env = {
        "CHUNKHOUND_NO_RICH": "1",
        "CHUNKHOUND_INDEXING__DISCOVERY_BACKEND": "git_only",
    }
    proc = _run(
        ["chunkhound", "index", str(repo), "--no-embeddings", "--profile-startup"],
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    # Parse last JSON block from stderr
    err_lines = proc.stderr.strip().splitlines()
    joined = "\n".join(err_lines[-30:])
    first = joined.find("{")
    assert first != -1, f"stderr tail did not contain JSON: {joined!r}"
    data = json.loads(joined[first:])
    assert "startup_profile" in data
    sp = data["startup_profile"]
    # Ensure counters are present
    assert isinstance(sp.get("git_rows_total"), int)
    assert isinstance(sp.get("git_pathspecs"), int)
    # Backend resolution should be visible too
    assert sp.get("resolved_backend") in ("git_only", "git")
