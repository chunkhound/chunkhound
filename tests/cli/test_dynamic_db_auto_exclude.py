from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _run_simulate(path: Path) -> list[str]:
    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    p = subprocess.run(
        ["uv", "run", "chunkhound", "index", "--simulate", str(path), "--sort", "path"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert p.returncode == 0, p.stderr
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def test_dynamic_db_path_is_excluded_when_inside_target(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    # Place a normal file that should be listed
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "src" / "a.py").write_text("print('ok')\n", encoding="utf-8")

    # Put database path inside workspace under a
    # custom directory (not default .chunkhound)
    db_dir = ws / "mydbdir"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / "test.duckdb"  # let DuckDB create this

    # Write local config pointing database.path to this FILE path
    cfg = {
        "database": {"provider": "duckdb", "path": str(db_file)},
        # Include everything so the db file would be listed if not dynamically excluded
        "indexing": {"include": ["**/*"], "exclude": []},
    }
    (ws / ".chunkhound.json").write_text(json.dumps(cfg), encoding="utf-8")

    lines = _run_simulate(ws)
    s = set(lines)
    # Ensure regular file is present
    assert "src/a.py" in s
    # Ensure db FILE is excluded dynamically
    assert "mydbdir/test.duckdb" not in s
