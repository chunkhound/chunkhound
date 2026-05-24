#!/usr/bin/env python3
"""Demo: DuckDB compaction lifecycle.

Creates a synthetic fragmented database, runs compaction, and shows
before/after stats. No external dependencies beyond chunkhound itself.

Usage:
    uv run python scripts/demo_compaction.py
    uv run python scripts/demo_compaction.py --size large
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence loguru before any chunkhound import
import loguru
loguru.logger.remove()

from chunkhound.core.config.config import Config
from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.models import Chunk, File
from chunkhound.core.types.common import ChunkType, Language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.compaction_service import CompactionService


SIZES = {
    "small":  dict(n_files=30,  n_chunks=40),
    "medium": dict(n_files=120, n_chunks=80),
    "large":  dict(n_files=400, n_chunks=150),
}

LINE = "─" * 56


def _hr():
    print(LINE)


def _build_fragmented_db(db_path: Path, n_files: int, n_chunks: int) -> DuckDBProvider:
    print(f"  Creating {n_files} files × {n_chunks} chunks/file...", flush=True)
    provider = DuckDBProvider(str(db_path), base_directory=db_path.parent)
    provider.connect()

    for i in range(n_files):
        f = File(
            path=f"demo_{i:04d}.py",
            mtime=1_700_000_000.0 + i,
            language=Language.PYTHON,
            size_bytes=n_chunks * 512,
        )
        file_id = provider.insert_file(f)
        chunks = [
            Chunk(
                file_id=file_id,
                code=f"def demo_{i}_{j}(): return {j}  # demo payload " + "z" * 350,
                start_line=j * 7 + 1,
                end_line=j * 7 + 5,
                chunk_type=ChunkType.FUNCTION,
                symbol=f"demo_{i}_{j}",
                language=Language.PYTHON,
            )
            for j in range(n_chunks)
        ]
        provider.insert_chunks_batch(chunks)

    provider.optimize_tables()

    n_delete = int(n_files * 0.75)
    print(f"  Deleting {n_delete}/{n_files} files (75% delete → fragmentation)...", flush=True)
    for i in range(n_delete):
        provider.delete_file_completely(f"demo_{i:04d}.py")
    provider.optimize_tables()

    return provider


def _fmt_mb(b: int) -> str:
    return f"{b / (1024 * 1024):.2f} MB"


def _fmt_pct(f: float) -> str:
    return f"{f:.0%}"


async def _run_demo(size: str) -> None:
    params = SIZES[size]
    n_files = params["n_files"]
    n_chunks = params["n_chunks"]

    with tempfile.TemporaryDirectory(prefix="chunkhound_demo_") as tmpdir:
        db_path = Path(tmpdir) / "demo.duckdb"

        _hr()
        print(f"  ChunkHound Compaction Demo  [{size}]")
        _hr()
        print()

        t_setup = time.perf_counter()
        provider = _build_fragmented_db(db_path, n_files, n_chunks)
        t_setup = time.perf_counter() - t_setup

        stats_before = provider.get_storage_stats()
        size_before = db_path.stat().st_size

        print()
        _hr()
        print("  BEFORE compaction:")
        print(f"    DB size         : {_fmt_mb(size_before)}")
        print(f"    Row waste       : {_fmt_pct(stats_before['row_waste_ratio'])}")
        print(f"    Free blocks     : {_fmt_pct(stats_before['free_ratio'])}")
        print(f"    Effective waste : {_fmt_pct(stats_before['effective_waste'])}")
        would = stats_before["effective_waste"] >= 0.50
        print(f"    Auto-compact?   : {'YES ✓' if would else 'no (below 50% threshold)'}")
        _hr()

        config = Config(
            database=DatabaseConfig(
                path=str(db_path),
                provider="duckdb",
                compaction_enabled=True,
                compaction_threshold=0.01,
                compaction_min_size_mb=0,
            )
        )

        print()
        print("  Running compaction...", flush=True)
        svc = CompactionService(db_path, config)
        t0 = time.perf_counter()
        performed = await svc.compact_blocking(provider)
        elapsed = time.perf_counter() - t0

        if not performed:
            print("  (compaction skipped — nothing to reclaim)")
            provider.disconnect()
            return

        stats_after = provider.get_storage_stats()
        size_after = db_path.stat().st_size
        reduction = (1 - size_after / size_before) * 100

        print()
        _hr()
        print(f"  AFTER compaction  ({elapsed:.1f}s):")
        print(f"    DB size         : {_fmt_mb(size_after)}  ({reduction:.0f}% reduction)")
        print(f"    Row waste       : {_fmt_pct(stats_after['row_waste_ratio'])}")
        print(f"    Free blocks     : {_fmt_pct(stats_after['free_ratio'])}")
        print(f"    Effective waste : {_fmt_pct(stats_after['effective_waste'])}")
        _hr()
        print()

        # Search verification
        kept_idx = int(n_files * 0.75)
        pattern = f"demo_{kept_idx}_{0}"
        results = provider.search_chunks_regex(pattern)
        ok = "✓" if results else "✗ FAILED"
        print(f"  Search verification: {len(results)} result(s) for {pattern!r}  {ok}")
        print()

        provider.disconnect()

    print(f"  Setup: {t_setup:.1f}s | Compaction: {elapsed:.1f}s")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="ChunkHound compaction demo")
    parser.add_argument(
        "--size",
        choices=list(SIZES),
        default="medium",
        help="Database size (default: medium)",
    )
    args = parser.parse_args()
    asyncio.run(_run_demo(args.size))


if __name__ == "__main__":
    main()
