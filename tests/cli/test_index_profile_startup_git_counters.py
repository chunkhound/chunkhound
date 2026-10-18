from __future__ import annotations

import json
from pathlib import Path

from tests.helpers.cli import init_git_repository, run_cli, run_command


def test_full_run_profile_includes_git_counters(tmp_path: Path) -> None:
    # Create a tiny repo so backend can be forced to git_only
    repo = tmp_path / "repo"
    init_git_repository(repo)
    (repo / "a.py").write_text("print('ok')\n", encoding="utf-8")
    run_command(["git", "add", "-A"], cwd=repo).check_returncode()
    run_command(
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
        cwd=repo,
    ).check_returncode()

    env = {
        "CHUNKHOUND_NO_RICH": "1",
        "CHUNKHOUND_INDEXING__DISCOVERY_BACKEND": "git_only",
    }
    proc = run_cli(
        ["index", str(repo), "--no-embeddings", "--profile-startup"],
        cwd=repo,
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
