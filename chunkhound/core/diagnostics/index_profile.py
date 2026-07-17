"""Indexing-flow performance profiling (phases, DB ops, RSS).

Used by the profile harness and optional coordinator instrumentation.
All measurements are process-local and opt-in; no effect when unused.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


def _rss_mb() -> float | None:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:
        try:
            import resource

            # ru_maxrss is KB on Linux, bytes on macOS — report best-effort.
            val = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if val > 10_000_000:  # assume bytes (macOS-ish)
                return val / (1024 * 1024)
            return val / 1024.0  # assume KB (Linux)
        except Exception:
            return None


@dataclass
class DbOpStats:
    """Counters for serial DB write path."""

    merge_insert_calls: int = 0
    merge_insert_rows: int = 0
    merge_insert_s: float = 0.0
    optimize_calls: int = 0
    optimize_s: float = 0.0
    chunk_insert_batches: int = 0
    embedding_insert_batches: int = 0

    def record_merge_insert(self, rows: int, elapsed_s: float) -> None:
        self.merge_insert_calls += 1
        self.merge_insert_rows += max(0, int(rows))
        self.merge_insert_s += max(0.0, float(elapsed_s))

    def record_optimize(self, elapsed_s: float) -> None:
        self.optimize_calls += 1
        self.optimize_s += max(0.0, float(elapsed_s))

    def as_dict(self) -> dict[str, Any]:
        return {
            "merge_insert_calls": self.merge_insert_calls,
            "merge_insert_rows": self.merge_insert_rows,
            "merge_insert_s": round(self.merge_insert_s, 4),
            "optimize_calls": self.optimize_calls,
            "optimize_s": round(self.optimize_s, 4),
            "chunk_insert_batches": self.chunk_insert_batches,
            "embedding_insert_batches": self.embedding_insert_batches,
        }


@dataclass
class IndexProfile:
    """Phase timers and peak RSS for one indexing run."""

    phases_s: dict[str, float] = field(default_factory=dict)
    db: DbOpStats = field(default_factory=DbOpStats)
    peak_rss_mb: float | None = None
    start_rss_mb: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def __post_init__(self) -> None:
        self.start_rss_mb = _rss_mb()
        self.peak_rss_mb = self.start_rss_mb

    def sample_rss(self) -> None:
        rss = _rss_mb()
        if rss is None:
            return
        if self.peak_rss_mb is None or rss > self.peak_rss_mb:
            self.peak_rss_mb = rss

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.phases_s[name] = self.phases_s.get(name, 0.0) + elapsed
            self.sample_rss()

    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, Any]:
        self.sample_rss()
        return {
            "phases_s": {k: round(v, 4) for k, v in sorted(self.phases_s.items())},
            "total_s": round(self.total_s(), 4),
            "start_rss_mb": (
                round(self.start_rss_mb, 2) if self.start_rss_mb is not None else None
            ),
            "peak_rss_mb": (
                round(self.peak_rss_mb, 2) if self.peak_rss_mb is not None else None
            ),
            "db": self.db.as_dict(),
            "meta": dict(self.meta),
        }

    def format_report(self) -> str:
        d = self.as_dict()
        lines = ["=== index profile ==="]
        for k, v in d["phases_s"].items():
            lines.append(f"  {k:20s} {v:8.3f}s")
        lines.append(f"  {'TOTAL':20s} {d['total_s']:8.3f}s")
        lines.append(
            f"  rss start={d['start_rss_mb']} peak={d['peak_rss_mb']} MB"
        )
        db = d["db"]
        lines.append(
            "  db merge_insert calls={merge_insert_calls} rows={merge_insert_rows} "
            "s={merge_insert_s} | optimize calls={optimize_calls} s={optimize_s}".format(
                **db
            )
        )
        lines.append(
            f"  db chunk_batches={db['chunk_insert_batches']} "
            f"embed_batches={db['embedding_insert_batches']}"
        )
        if d["meta"]:
            for k, v in d["meta"].items():
                lines.append(f"  {k}={v}")
        return "\n".join(lines)
