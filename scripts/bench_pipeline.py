#!/usr/bin/env python3
"""Benchmark: Rust indexing pipeline vs Python indexing pipeline (full flow).

Implements the harness described in docs/rust-pipeline-architecture.html §13:
indexes the same directory with both pipelines, measuring wall-clock time,
peak RSS, and output DB size, across three scenarios:

  cold        empty DB, index the repo from scratch
  warm        re-index the repo into an already-fully-indexed DB (nothing changed)
  incremental re-index into an already-indexed DB with exactly one file modified

Each timed indexing run happens in a fresh child process (this same script,
re-invoked with --_child) so that peak RSS can be measured from the outside
via /proc/<pid>/status — measuring the harness's own interpreter would give a
misleading number. Run order alternates Python/Rust in an ABBA pattern
(P,R,R,P,P,R,R,P,...) across the requested number of runs to cancel
directional page-cache/allocator bias; only medians are reported. Before
printing results, DB-derived file/chunk/embedding counts are compared between
backends for every run — a mismatch aborts immediately.

Usage:
  uv run python scripts/bench_pipeline.py --repo /path/to/repo
  uv run python scripts/bench_pipeline.py --repo /path/to/repo --runs 5 \\
      --scenarios cold,warm
  uv run python scripts/bench_pipeline.py --repo /path/to/repo --embeddings
  uv run python scripts/bench_pipeline.py --repo /path/to/repo --python-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_MARKER = "##BENCH_RESULT##"
_SCENARIOS = ("cold", "warm", "incremental")
_BACKENDS = ("python", "rust")

# Extensions preferred as the mutation target for the incremental scenario —
# editing one of these is representative of a real code change.
_CODE_EXTENSIONS = {
    ".py",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".go",
    ".java",
    ".rb",
    ".php",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}


# ---------------------------------------------------------------------------
# Child process — runs exactly one indexing pass, prints one JSON result line
# ---------------------------------------------------------------------------


async def _child_run(
    project_root: Path, db_dir: Path, *, backend: str, skip_embeddings: bool
) -> dict[str, Any]:
    os.environ["CHUNKHOUND_USE_RUST"] = "1" if backend == "rust" else "0"
    os.environ["CHUNKHOUND_NO_RICH"] = "1"

    from chunkhound.core.config.config import Config
    from chunkhound.registry import configure_registry, create_indexing_coordinator
    from chunkhound.services.directory_indexing_service import DirectoryIndexingService

    db_dir.mkdir(parents=True, exist_ok=True)
    config = Config(
        target_dir=project_root,
        database={"provider": "duckdb", "path": str(db_dir)},
        embeddings_disabled=skip_embeddings,
    )
    configure_registry(config)
    coordinator = create_indexing_coordinator()
    service = DirectoryIndexingService(indexing_coordinator=coordinator, config=config)

    t0 = time.perf_counter()
    stats = await service.process_directory(project_root, no_embeddings=skip_embeddings)
    elapsed = time.perf_counter() - t0

    try:
        coordinator._db.disconnect()
    except Exception:
        pass

    counts = _query_db_counts(db_dir / "chunks.db")
    db_file = db_dir / "chunks.db"

    return {
        "backend": backend,
        "elapsed": elapsed,
        "files_processed": stats.files_processed,
        "chunks_written": stats.chunks_created,
        "embeddings_generated": stats.embeddings_generated,
        "errors": len(stats.errors_encountered or []),
        "db_bytes": db_file.stat().st_size if db_file.exists() else 0,
        **counts,
    }


def _connect_duckdb(db_path: Path, *, read_only: bool) -> Any:
    """Open a DuckDB connection with the vss extension loaded.

    The Rust pipeline always persists the HNSW/vss storage format (even with
    zero embeddings) — reopening the file without loading vss first raises a
    deserialization error, matching the setup every production connection
    (connection_manager.py, duckdb_provider.py) already performs.
    """
    import duckdb

    conn = duckdb.connect(str(db_path), read_only=read_only)
    try:
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
    except Exception:
        pass
    return conn


def _query_db_counts(db_path: Path) -> dict[str, int]:
    """Row counts straight from the DB — authoritative, independent of stats bugs."""
    if not db_path.exists():
        return {"db_files": 0, "db_chunks": 0, "db_embs": 0}
    conn = _connect_duckdb(db_path, read_only=True)
    try:
        n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        emb_tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name SIMILAR TO 'embeddings_[0-9]+'"
            ).fetchall()
        ]
        n_embs = sum(
            conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in emb_tables
        )
        return {"db_files": n_files, "db_chunks": n_chunks, "db_embs": n_embs}
    finally:
        conn.close()


def _child_main(args: argparse.Namespace) -> None:
    try:
        result = asyncio.run(
            _child_run(
                Path(args.project_root),
                Path(args.db_dir),
                backend=args.backend,
                skip_embeddings=args.skip_embeddings,
            )
        )
    except Exception as e:  # noqa: BLE001 — reported to parent, not swallowed
        print(f"{_MARKER}{json.dumps({'backend': args.backend, 'error': str(e)})}")
        sys.exit(1)
    print(f"{_MARKER}{json.dumps(result)}")


# ---------------------------------------------------------------------------
# Parent — spawns one child subprocess per timed run, tracks peak RSS
# ---------------------------------------------------------------------------


def _sample_peak_rss_mb(pid: int, stop: threading.Event) -> list[int]:
    """Poll /proc/<pid>/status VmHWM (Linux only) until *stop* is set."""
    peak_kb = [0]
    status_path = f"/proc/{pid}/status"
    while not stop.is_set():
        try:
            with open(status_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_kb[0] = max(peak_kb[0], int(line.split()[1]))
                        break
        except FileNotFoundError:
            break
        stop.wait(0.1)
    return peak_kb


def run_child_process(
    script_path: Path,
    project_root: Path,
    db_dir: Path,
    *,
    backend: str,
    skip_embeddings: bool,
) -> dict[str, Any]:
    """Spawn a fresh interpreter for one indexing pass; return its result + peak RSS."""
    argv = [
        sys.executable,
        str(script_path),
        "--_child",
        "--backend",
        backend,
        "--project-root",
        str(project_root),
        "--db-dir",
        str(db_dir),
    ]
    if skip_embeddings:
        argv.append("--skip-embeddings")

    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    peak_kb_holder: list[int] = [0]
    stop = threading.Event()
    sampler: threading.Thread | None = None
    if sys.platform.startswith("linux"):

        def _poll() -> None:
            peak_kb_holder[0] = _sample_peak_rss_mb(proc.pid, stop)[0]

        sampler = threading.Thread(target=_poll, daemon=True)
        sampler.start()

    stdout, stderr = proc.communicate()
    stop.set()
    if sampler:
        sampler.join(timeout=1)

    result_line = next(
        (line for line in stdout.splitlines() if line.startswith(_MARKER)), None
    )
    if result_line is None:
        raise RuntimeError(
            f"child process produced no result (exit={proc.returncode}):\n"
            f"{stderr[-4000:]}"
        )
    result = json.loads(result_line[len(_MARKER) :])
    if "error" in result:
        raise RuntimeError(f"child indexing run failed ({backend}): {result['error']}")

    result["peak_rss_mb"] = (peak_kb_holder[0] / 1024) if peak_kb_holder[0] else None
    return result


# ---------------------------------------------------------------------------
# Scenario setup
# ---------------------------------------------------------------------------


def _pick_mutation_target(db_path: Path, project_root: Path) -> Path:
    """Pick a real, already-indexed file to modify for the incremental scenario."""
    conn = _connect_duckdb(db_path, read_only=True)
    try:
        rows = conn.execute("SELECT path FROM files ORDER BY path").fetchall()
    finally:
        conn.close()

    for (path_str,) in rows:
        p = Path(path_str)
        if not p.is_absolute():
            p = project_root / p
        if p.suffix in _CODE_EXTENSIONS and p.is_file():
            return p
    if rows:
        p = Path(rows[0][0])
        return p if p.is_absolute() else project_root / p
    raise RuntimeError(f"no indexed files found under {project_root} to mutate")


def _touch_modify(path: Path) -> None:
    """Append one byte — changes content hash and mtime, safe for any text format."""
    with open(path, "ab") as f:
        f.write(b"\n")


def _setup_persistent_backend(
    script_path: Path,
    work_root: Path,
    backend: str,
    scenario: str,
    *,
    repo: Path,
    skip_embeddings: bool,
) -> tuple[Path, Path]:
    """Prepare (project_root, db_dir) for warm/incremental; build the baseline once."""
    if scenario == "incremental":
        project_root = work_root / f"repo_{backend}"
        shutil.copytree(repo, project_root)
    else:
        project_root = repo

    db_dir = work_root / f"db_{scenario}_{backend}"
    baseline = run_child_process(
        script_path,
        project_root,
        db_dir,
        backend=backend,
        skip_embeddings=skip_embeddings,
    )
    print(
        f"  [{backend}] baseline built for {scenario}: "
        f"{baseline['files_processed']} files, {baseline['chunks_written']} chunks"
    )
    return project_root, db_dir


def _backend_order(run_idx: int) -> tuple[str, str]:
    """ABBA alternation: run 0 -> (python, rust); run 1 -> (rust, python); ..."""
    return ("python", "rust") if run_idx % 2 == 0 else ("rust", "python")


def _validate_counts(
    run_idx: int, scenario: str, results: dict[str, dict[str, Any]]
) -> None:
    py, rust = results.get("python"), results.get("rust")
    if py is None or rust is None:
        return
    mismatches = [
        key for key in ("db_files", "db_chunks", "db_embs") if py[key] != rust[key]
    ]
    if mismatches:
        details = ", ".join(f"{k}: python={py[k]} rust={rust[k]}" for k in mismatches)
        raise SystemExit(
            f"ABORT: count mismatch on {scenario} run {run_idx + 1} — {details}"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_mb(mb: float | None) -> str:
    return f"{mb:,.1f} MB" if mb is not None else "N/A"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


def _print_scenario_table(
    scenario: str, samples: dict[str, list[dict[str, Any]]]
) -> None:
    print(f"\n=== Scenario: {scenario} ===")
    py_samples, rust_samples = samples.get("python", []), samples.get("rust", [])

    def _median(items: list[dict[str, Any]], key: str) -> float | None:
        vals = [i[key] for i in items if i.get(key) is not None]
        return statistics.median(vals) if vals else None

    py_wall = _median(py_samples, "elapsed")
    rust_wall = _median(rust_samples, "elapsed")
    py_rss = _median(py_samples, "peak_rss_mb")
    rust_rss = _median(rust_samples, "peak_rss_mb")
    py_db = _median(py_samples, "db_bytes")
    rust_db = _median(rust_samples, "db_bytes")

    speedup = (py_wall / rust_wall) if (py_wall and rust_wall) else None

    col_width = 16
    header = "Metric".ljust(22) + "Python".rjust(col_width) + "Rust".rjust(col_width)
    print(header)
    print("-" * len(header))

    def row(label: str, py_val: str, rust_val: str) -> None:
        print(f"{label:<22}{py_val:>{col_width}}{rust_val:>{col_width}}")

    row(
        "Wall time (median)",
        f"{py_wall:.3f} s" if py_wall else "N/A",
        f"{rust_wall:.3f} s" if rust_wall else "N/A",
    )
    row("Peak RSS (median)", _fmt_mb(py_rss), _fmt_mb(rust_rss))
    row(
        "DB size (median)",
        _fmt_bytes(int(py_db)) if py_db else "N/A",
        _fmt_bytes(int(rust_db)) if rust_db else "N/A",
    )
    row("Samples", str(len(py_samples)), str(len(rust_samples)))
    row("Speedup", "1.0x", f"{speedup:.2f}x" if speedup else "N/A")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark: Rust vs Python indexing pipeline"
    )
    p.add_argument("--repo", metavar="PATH", help="Directory to index (real codebase)")
    p.add_argument(
        "--runs",
        type=int,
        default=5,
        metavar="N",
        help="Timed runs per backend per scenario (default: 5)",
    )
    p.add_argument(
        "--scenarios",
        default="all",
        help="Comma-separated: cold,warm,incremental (default: all)",
    )
    p.add_argument(
        "--embeddings", action="store_true", help="Enable embeddings (default: skipped)"
    )
    p.add_argument("--python-only", action="store_true", help="Skip the Rust backend")
    p.add_argument("--rust-only", action="store_true", help="Skip the Python backend")
    p.add_argument(
        "--json", metavar="PATH", help="Write raw per-run results as JSON to PATH"
    )
    p.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Don't delete the temp work directory",
    )

    # Hidden child-mode flags (internal re-invocation, not for interactive use)
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--backend", choices=_BACKENDS, help=argparse.SUPPRESS)
    p.add_argument("--project-root", help=argparse.SUPPRESS)
    p.add_argument("--db-dir", help=argparse.SUPPRESS)
    p.add_argument("--skip-embeddings", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def _rust_available() -> bool:
    try:
        from chunkhound_native import IndexingPipeline
    except ImportError:
        return False
    return IndexingPipeline is not None


def main() -> None:
    args = parse_args()

    if args._child:
        _child_main(args)
        return

    if not args.repo:
        print("error: --repo is required", file=sys.stderr)
        sys.exit(2)

    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"error: --repo {repo} is not a directory", file=sys.stderr)
        sys.exit(2)

    scenarios = (
        _SCENARIOS
        if args.scenarios == "all"
        else tuple(s.strip() for s in args.scenarios.split(",") if s.strip())
    )
    for s in scenarios:
        if s not in _SCENARIOS:
            print(
                f"error: unknown scenario {s!r}, expected one of {_SCENARIOS}",
                file=sys.stderr,
            )
            sys.exit(2)

    backends = list(_BACKENDS)
    if args.python_only:
        backends = ["python"]
    elif args.rust_only:
        backends = ["rust"]
    if "rust" in backends and not _rust_available():
        print(
            "Rust pipeline unavailable (chunkhound_native.IndexingPipeline is "
            "None) — skipping Rust backend."
        )
        backends = [b for b in backends if b != "rust"]
    if not backends:
        print("Nothing to benchmark.", file=sys.stderr)
        sys.exit(1)

    script_path = Path(__file__).resolve()
    skip_embeddings = not args.embeddings

    work_root = Path(tempfile.mkdtemp(prefix="chunkhound_bench_pipeline_"))
    print(f"Repo: {repo}")
    print(f"Work dir: {work_root}")
    print(
        f"Scenarios: {', '.join(scenarios)}  |  "
        f"Backends: {', '.join(backends)}  |  Runs: {args.runs}"
    )

    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    try:
        for scenario in scenarios:
            print(f"\n--- Setting up scenario '{scenario}' ---")
            persistent: dict[str, tuple[Path, Path]] = {}
            if scenario in ("warm", "incremental"):
                for backend in backends:
                    persistent[backend] = _setup_persistent_backend(
                        script_path,
                        work_root,
                        backend,
                        scenario,
                        repo=repo,
                        skip_embeddings=skip_embeddings,
                    )

            samples: dict[str, list[dict[str, Any]]] = {b: [] for b in backends}

            for run_idx in range(args.runs):
                order = [b for b in _backend_order(run_idx) if b in backends]
                run_results: dict[str, dict[str, Any]] = {}
                for backend in order:
                    if scenario == "cold":
                        project_root = repo
                        db_dir = work_root / f"db_cold_{backend}_{run_idx}"
                    else:
                        project_root, db_dir = persistent[backend]
                        if scenario == "incremental":
                            target = _pick_mutation_target(
                                db_dir / "chunks.db", project_root
                            )
                            _touch_modify(target)

                    result = run_child_process(
                        script_path,
                        project_root,
                        db_dir,
                        backend=backend,
                        skip_embeddings=skip_embeddings,
                    )
                    run_results[backend] = result
                    samples[backend].append(result)
                    rss = _fmt_mb(result["peak_rss_mb"])
                    print(
                        f"  run {run_idx + 1}/{args.runs} [{backend:>6}] "
                        f"{result['elapsed']:.3f}s  files={result['files_processed']}  "
                        f"chunks={result['chunks_written']}  RSS={rss}"
                    )

                _validate_counts(run_idx, scenario, run_results)

            all_results[scenario] = samples
            _print_scenario_table(scenario, samples)
    finally:
        if not args.keep_work_dir:
            shutil.rmtree(work_root, ignore_errors=True)
        else:
            print(f"\nWork dir preserved at: {work_root}")

    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2))
        print(f"\nRaw results written to {args.json}")


if __name__ == "__main__":
    main()
