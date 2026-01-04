#!/usr/bin/env python3
"""
YAML parsing benchmark harness for ChunkHound.

This script compares multiple parsing backends (PyYAML, tree-sitter/cAST,
RapidYAML arena reuse, RapidYAML in-place reuse) across a repeatable set of
synthetic or user-supplied YAML documents. Results include latency, throughput,
and reproducible metadata (case sizes + hashes).

Usage:
    uv run python scripts/bench_yaml.py
    uv run python scripts/bench_yaml.py --cases-dir ./my-fixtures --iterations 10
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Benchmark case generation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BenchmarkCase:
    """Container for a benchmark input."""

    name: str
    description: str
    text: str
    origin: str
    size_bytes: int = field(init=False)
    sha256: str = field(init=False)
    _bytes: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        encoded = self.text.encode("utf-8")
        self._bytes = encoded
        self.size_bytes = len(encoded)
        self.sha256 = hashlib.sha256(encoded).hexdigest()

    @property
    def bytes(self) -> bytes:
        return self._bytes


def _render_service_block(
    services: int, env_vars: int, volumes: int, indent: int = 2
) -> str:
    pad = " " * indent
    lines: list[str] = ["services:"]
    for svc in range(services):
        lines.extend(
            [
                f"{pad}service_{svc}:",
                f"{pad * 2}image: ghcr.io/example/service:{svc % 7}",
                f"{pad * 2}restart: always",
                f"{pad * 2}deploy:",
                f"{pad * 3}replicas: {(svc % 5) + 1}",
                f"{pad * 3}resources:",
                f"{pad * 4}limits:",
                f"{pad * 5}cpus: '{0.5 + (svc % 4) / 10:.1f}'",
                f"{pad * 5}memory: '{256 + (svc % 8) * 32}Mi'",
                f"{pad * 3}placement:",
                f"{pad * 4}constraints:",
                f"{pad * 5}- 'node.labels.dc=={svc % 3}'",
                f"{pad * 2}environment:",
            ]
        )
        for env in range(env_vars):
            lines.append(f"{pad * 3}SERVICE_{svc}_VAR_{env}: value_{svc}_{env}")
        lines.append(f"{pad * 2}volumes:")
        for vol in range(volumes):
            lines.append(
                f"{pad * 3}- type: bind\n"
                f"{pad * 4}source: /data/service_{svc}/vol_{vol}\n"
                f"{pad * 4}target: /app/data/vol_{vol}"
            )
    return "\n".join(lines)


def _render_pipeline_block(stages: int, jobs_per_stage: int) -> str:
    lines = ["stages:"]
    lines.extend([f"  - stage_{idx}" for idx in range(stages)])
    lines.append("")
    for stage in range(stages):
        for job in range(jobs_per_stage):
            name = f"job_{stage}_{job}"
            lines.extend(
                [
                    f"{name}:",
                    f"  stage: stage_{stage}",
                    "  script:",
                    "    - make lint",
                    "    - make test",
                    "  cache:",
                    "    paths:",
                    "      - node_modules/",
                    "      - .mypy_cache/",
                    "  rules:",
                    "    - if: '$CI_COMMIT_BRANCH == \"main\"'",
                    "      when: always",
                    "    - when: manual",
                    "",
                ]
            )
    return "\n".join(lines)


def _render_multi_doc(documents: int, depth: int) -> str:
    lines: list[str] = []
    for doc in range(documents):
        lines.append("---")
        lines.append(f"document: {doc}")
        lines.append("matrix:")
        indent = "  "
        for level in range(depth):
            lines.append(f"{indent * (level + 1)}level_{level}:")
        for row in range(depth * 2):
            lines.append(f"{indent * depth}- item_{doc}_{row}")
    lines.append("...")
    return "\n".join(lines)


def generate_default_cases() -> list[BenchmarkCase]:
    """Produce deterministic synthetic YAML cases sized for benchmarking."""

    cases = [
        BenchmarkCase(
            name="ci_small",
            description="CI pipeline with 4 stages x 3 jobs (≈3KB)",
            text=_render_pipeline_block(stages=4, jobs_per_stage=3),
            origin="synthetic",
        ),
        BenchmarkCase(
            name="services_medium",
            description="Service mesh config with 35 services (≈60KB)",
            text=_render_service_block(services=35, env_vars=12, volumes=3),
            origin="synthetic",
        ),
        BenchmarkCase(
            name="services_large",
            description="Service mesh config with 200 services (≈320KB)",
            text=_render_service_block(services=200, env_vars=16, volumes=6),
            origin="synthetic",
        ),
        BenchmarkCase(
            name="multidoc_stack",
            description="50 multi-document blocks with nested sequences",
            text=_render_multi_doc(documents=50, depth=6),
            origin="synthetic",
        ),
    ]
    return cases


def load_cases_from_dir(case_dir: Path) -> list[BenchmarkCase]:
    """Load .yml/.yaml files from a directory recursively."""

    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    files = sorted(
        [
            path
            for pattern in ("*.yml", "*.yaml")
            for path in case_dir.rglob(pattern)
            if path.is_file()
        ]
    )
    cases: list[BenchmarkCase] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        cases.append(
            BenchmarkCase(
                name=path.stem,
                description=f"Fixture from {rel}",
                text=text,
                origin=str(rel),
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class BackendError(Exception):
    """Raised when a backend cannot be initialized."""


class BenchmarkBackend:
    """Interface for benchmark backends."""

    name: str

    def run(self, case: BenchmarkCase) -> None:
        raise NotImplementedError


class PyYAMLBackend(BenchmarkBackend):
    name = "pyyaml_safe_load"

    def __init__(self) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - controlled by env
            raise BackendError(
                "PyYAML is required; install with 'uv add pyyaml'"
            ) from exc
        self._yaml = yaml

    def run(self, case: BenchmarkCase) -> None:
        list(self._yaml.safe_load_all(case.text))


class TreeSitterBackend(BenchmarkBackend):
    name = "tree_sitter_universal"

    def __init__(self) -> None:
        from chunkhound.core.types.common import FileId, Language
        from chunkhound.parsers.parser_factory import create_parser_for_language

        parser = create_parser_for_language(Language.YAML)
        if parser is None:
            raise BackendError("Tree-sitter YAML parser not available.")
        self._parser = parser
        self._file_id = FileId(0)

    def run(self, case: BenchmarkCase) -> None:
        # parse_content returns list[Chunk]; we discard result
        self._parser.parse_content(case.text, None, self._file_id)


class RapidYamlArenaBackend(BenchmarkBackend):
    name = "rapidyaml_arena_reuse"

    def __init__(self) -> None:
        try:
            import ryml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendError(
                "RapidYAML bindings not found. Install via "
                "'uv pip install rapidyaml' (or `uv pip install \".[rapidyaml]\"`)."
            ) from exc

        self._ryml = ryml
        self._tree = ryml.Tree()

    def run(self, case: BenchmarkCase) -> None:
        buf = bytearray(case.bytes)
        self._ryml.parse_in_arena(buf, tree=self._tree)
        self._tree.clear()
        self._tree.clear_arena()


class RapidYamlInPlaceBackend(BenchmarkBackend):
    name = "rapidyaml_in_place_reuse"

    def __init__(self) -> None:
        try:
            import ryml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendError(
                "RapidYAML bindings not found. Install via "
                "'uv pip install rapidyaml' (or `uv pip install \".[rapidyaml]\"`)."
            ) from exc

        self._ryml = ryml
        self._tree = ryml.Tree()

    def run(self, case: BenchmarkCase) -> None:
        buf = bytearray(case.bytes)
        self._ryml.parse_in_place(buf, tree=self._tree)
        self._tree.clear()
        self._tree.clear_arena()


BACKEND_FACTORIES: dict[str, Callable[[], BenchmarkBackend]] = {
    PyYAMLBackend.name: PyYAMLBackend,
    TreeSitterBackend.name: TreeSitterBackend,
    RapidYamlArenaBackend.name: RapidYamlArenaBackend,
    RapidYamlInPlaceBackend.name: RapidYamlInPlaceBackend,
}


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def benchmark_case(
    backend: BenchmarkBackend,
    case: BenchmarkCase,
    iterations: int,
    warmup: int,
) -> dict:
    """Run benchmark for a backend/case pair."""

    for _ in range(max(0, warmup)):
        backend.run(case)

    timings: list[float] = []
    for _ in range(iterations):
        gc.collect()
        start = perf_counter()
        backend.run(case)
        timings.append(perf_counter() - start)

    avg = sum(timings) / len(timings)
    throughput = (case.size_bytes / (1024 * 1024)) / avg if avg > 0 else 0.0
    return {
        "backend": backend.name,
        "iterations": iterations,
        "mean_ms": avg * 1000,
        "median_ms": statistics.median(timings) * 1000,
        "min_ms": min(timings) * 1000,
        "max_ms": max(timings) * 1000,
        "stddev_ms": statistics.pstdev(timings) * 1000 if len(timings) > 1 else 0.0,
        "throughput_mb_s": throughput,
    }


def run_benchmarks(
    cases: Sequence[BenchmarkCase],
    backend_names: Sequence[str],
    iterations: int,
    warmup: int,
) -> tuple[list[dict], list[str]]:
    """Execute benchmarks for all cases/backends."""

    instantiated: list[BenchmarkBackend] = []
    warnings: list[str] = []
    for name in backend_names:
        factory = BACKEND_FACTORIES.get(name)
        if not factory:
            warnings.append(f"Unknown backend '{name}', skipping.")
            continue
        try:
            instantiated.append(factory())
        except BackendError as exc:
            warnings.append(f"{name}: {exc}")

    if not instantiated:
        raise SystemExit("No benchmark backends available. Install required deps.")

    results: list[dict] = []
    for case in cases:
        entry = {
            "name": case.name,
            "description": case.description,
            "origin": case.origin,
            "size_bytes": case.size_bytes,
            "sha256": case.sha256,
            "results": [],
        }
        for backend in instantiated:
            metrics = benchmark_case(backend, case, iterations, warmup)
            entry["results"].append(metrics)
        results.append(entry)

    return results, warnings


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark YAML parsing backends for ChunkHound."
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        help="Path containing YAML fixtures (defaults to synthetic cases).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of recorded iterations per backend/case.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up runs per backend/case (not counted).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=sorted(BACKEND_FACTORIES.keys()),
        default=list(BACKEND_FACTORIES.keys()),
        help="Subset of backends to benchmark.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results.",
    )
    return parser.parse_args(argv)


def collect_metadata() -> dict:
    """Record environment metadata for reproducibility."""

    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count() or 0,
        "git_commit": commit,
    }


def print_summary(results: list[dict], warnings: list[str]) -> None:
    if warnings:
        print("Warnings:")
        for msg in warnings:
            print(f"  - {msg}")
        print()

    for case in results:
        print(
            f"Case '{case['name']}' "
            f"({case['size_bytes'] / 1024:.1f} KiB, sha256={case['sha256'][:8]}…)"
        )
        print(f"  {case['description']} (origin: {case['origin']})")
        for entry in sorted(case["results"], key=lambda r: r["mean_ms"]):
            print(
                f"  - {entry['backend']:<26} "
                f"avg={entry['mean_ms']:.2f} ms  "
                f"median={entry['median_ms']:.2f} ms  "
                f"throughput={entry['throughput_mb_s']:.2f} MB/s"
            )
        print()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cases = (
        load_cases_from_dir(args.cases_dir)
        if args.cases_dir
        else generate_default_cases()
    )
    if not cases:
        raise SystemExit("No benchmark cases found.")

    results, warnings = run_benchmarks(
        cases=cases,
        backend_names=args.backends,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    metadata = collect_metadata()
    output_payload = {
        "metadata": metadata,
        "cases": results,
    }

    print_summary(results, warnings)

    if args.output:
        args.output.write_text(json.dumps(output_payload, indent=2))
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
