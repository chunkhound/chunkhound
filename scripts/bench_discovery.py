#!/usr/bin/env python3
"""Benchmark discovery backends (auto vs python, optionally git/git_only).

Generates heavy synthetic workspaces deterministically (in a temp dir) and/or
runs against provided real directories. Prints a JSON summary per target.

Examples:
  uv run python scripts/bench_discovery.py \
      --dirs /workspaces/chunkhound /tmp/gpw-bot \
      --scale 1 --trials 1

  uv run python scripts/bench_discovery.py --synthetic --scale 2 --trials 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _run_simulate(path: Path, backend: str) -> tuple[list[str], dict]:
    env = os.environ.copy()
    env["CHUNKHOUND_NO_RICH"] = "1"
    env["CHUNKHOUND_INDEXING__DISCOVERY_BACKEND"] = backend
    p = subprocess.run(
        [
            "uv",
            "run",
            "chunkhound",
            "index",
            "--simulate",
            str(path),
            "--profile-startup",
            "--sort",
            "path",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    files = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
    prof: dict[str, Any] = {}
    for ln in (p.stderr or "").splitlines()[::-1]:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict) and (
                "discovery_ms" in obj or "startup_profile" in obj
            ):
                prof = obj.get("startup_profile", obj)
                break
        except Exception:
            continue
    return files, prof


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
    )


def _git_init_commit(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"],
        cwd=str(repo),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repo, "config", "user.email", "ci@example.com")
    _git(repo, "config", "user.name", "CI")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")


def _w(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _generate_heavy_workspace(scale: int = 1) -> Path:
    """Create a mixed workspace with repos + non-repo datasets; deterministic."""
    temp = Path(tempfile.mkdtemp(prefix="bench-discovery-"))
    ws = temp / "ws"
    # repoA with tracked-under-ignored artifacts
    repoA = ws / "repoA"
    _w(
        repoA / ".gitignore",
        "\n".join(["runs/", "node_modules/", ".venv/", "dist/"]) + "\n",
    )
    # Source
    for i in range(100 * scale):
        _w(repoA / "src" / "pkg" / f"m{i:03d}.py", f"def f{i}():\n    return {i}\n")
    _git_init_commit(repoA)
    # Force-track artifacts in ignored runs/
    for t in range(20 * scale):
        p = repoA / "runs" / f"job{t:03d}" / f"kept{t:03d}.md"
        _w(p, f"# kept {t}\n")
    _git(
        repoA,
        "add",
        "-f",
        *[f"runs/job{t:03d}/kept{t:03d}.md" for t in range(20 * scale)],
    )
    _git(repoA, "commit", "-m", "keep tracked under ignored", check=False)

    # repoB plain
    repoB = ws / "repoB"
    for i in range(100 * scale):
        _w(repoB / "web" / "lib" / f"u{i:03d}.ts", f"export const v{i} = {i};\n")
    _git_init_commit(repoB)

    # Non-repo datasets
    for i in range(200 * scale):
        _w(ws / "datasets" / f"data{i:04d}.json", "{}\n")

    return ws


def bench_target(path: Path, trials: int = 1) -> dict[str, Any]:
    results: dict[str, Any] = {"path": str(path), "trials": trials, "runs": []}
    for _ in range(trials):
        auto_files, auto_prof = _run_simulate(path, "auto")
        py_files, py_prof = _run_simulate(path, "python")
        a, p = set(auto_files), set(py_files)
        diff_a = sorted(a - p)
        diff_p = sorted(p - a)
        results["runs"].append(
            {
                "auto": auto_prof,
                "python": py_prof,
                "counts": {"auto": len(a), "python": len(p)},
                "only_auto_count": len(diff_a),
                "only_python_count": len(diff_p),
                "only_auto_sample": diff_a[:10],
                "only_python_sample": diff_p[:10],
            }
        )
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark discovery backends")
    ap.add_argument(
        "--dirs", nargs="*", type=Path, help="Real directories to benchmark"
    )
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Also generate and benchmark a heavy synthetic workspace",
    )
    ap.add_argument(
        "--scale", type=int, default=1, help="Scale factor for synthetic generation"
    )
    ap.add_argument(
        "--trials", type=int, default=1, help="Trials per target (report each)"
    )
    args = ap.parse_args()

    reports: list[dict[str, Any]] = []

    if args.synthetic:
        ws = _generate_heavy_workspace(scale=max(1, args.scale))
        try:
            reports.append(bench_target(ws, trials=max(1, args.trials)))
        finally:
            # Clean up temp tree
            shutil.rmtree(ws.parent, ignore_errors=True)

    for d in args.dirs or []:
        reports.append(bench_target(d, trials=max(1, args.trials)))

    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
