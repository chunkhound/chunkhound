#!/usr/bin/env python3
"""
Database benchmark for ChunkHound - compares LanceDB vs DuckDB.

Usage:
    uv run scripts/benchmark_database.py --repo tests/fixtures --provider lancedb
    uv run scripts/benchmark_database.py --repo . --provider duckdb --output results.md
"""

import argparse
import asyncio
import os
import random
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import psutil
import pyarrow as pa

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services


def format_size(size_bytes: float) -> str:
    """Auto-scale bytes to B/KB/MB/GB."""
    for unit, div in [("B", 1), ("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)]:
        if size_bytes < div * 1024:
            return f"{size_bytes / div:.2f} {unit}"
    return f"{size_bytes / 1024**3:.2f} GB"


def format_time(seconds: float) -> str:
    """Format time in s or ms."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.3f} s"


async def benchmark_provider(repo_path: Path, provider: str, output: Path | None):
    """Run benchmark for provider."""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        db_path = tmp_dir / ".chunkhound" / f"bench.{provider}.db"
        db_path.parent.mkdir(exist_ok=True)

        args = SimpleNamespace(path=repo_path)
        config = Config(args=args, database={"path": str(db_path), "provider": provider})

        services = create_services(db_path, config)
        services.provider.connect()

        coord = services.indexing_coordinator

        process = psutil.Process()
        baseline_rss = process.memory_info().rss

        # Index
        index_start = perf_counter()
        result = await coord.process_directory(
            repo_path,
            patterns=["**/*"],
            exclude_patterns=[".git/**", "**/.git/**", "**/.chunkhound/**", "**/__pycache__/**"],
        )
        index_time = perf_counter() - index_start

        # Update peak
        current_rss = process.memory_info().rss
        peak_rss = current_rss

        stats = services.provider.get_stats()
        chunks = stats.get("chunks", 0)
        files = stats.get("files", 0)

        if chunks == 0:
            raise ValueError("No chunks indexed")

        # DB size
        db_size_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())

        # Mock embed
        embed_time = 0.0
        inserted = 0
        try:
            # Get chunk_ids
            chunks_df = services.provider._chunks_table.to_pandas() if hasattr(services.provider, '_chunks_table') else None
            if chunks_df is not None:
                chunks_df = chunks_df.drop_duplicates(subset=['id'])
                chunk_ids = chunks_df["id"].tolist()[:500]
                dim = 768
                mock_embeds = [
                    {
                        "chunk_id": cid,
                        "provider": "mock",
                        "model": "mock-768",
                        "dims": dim,
                        "embedding": [random.random() * 2 - 1 for _ in range(dim)],
                    }
                    for cid in chunk_ids
                ]
                embed_start = perf_counter()
                inserted = services.provider.insert_embeddings_batch(mock_embeds)
                embed_time = perf_counter() - embed_start
        except Exception as e:
            print(f"Mock embed skipped: {e}")

        # Update peak
        current_rss = process.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

        # Search latency
        queries = ["def ", "class ", "function", "import ", "return "]
        latencies = []
        for q in queries:
            s_start = perf_counter()
            res, _ = services.provider.search_regex(q, page_size=50)
            lat = perf_counter() - s_start
            latencies.append(lat)

        avg_lat = sum(latencies) / len(latencies) * 1000  # ms

        # Final peak
        current_rss = process.memory_info().rss
        peak_rss = max(peak_rss, current_rss)

        mem_delta_mb = (peak_rss - baseline_rss) / 1024 / 1024

        # Schema migration for LanceDB
        migration = "N/A"
        if provider == "lancedb":
            schema = services.provider._chunks_table.schema
            embedding_field = next((f for f in schema if f.name == "embedding"), None)
            if embedding_field:
                fixed = pa.types.is_fixed_size_list(embedding_field.type)
                migration = "fixed" if fixed else "variable"

        # Results
        results = {
            "repo": str(repo_path),
            "provider": provider,
            "chunks": chunks,
            "files": files,
            "index_time": index_time,
            "db_size_bytes": db_size_bytes,
            "embed_time": embed_time,
            "inserted": inserted,
            "avg_search_lat_ms": avg_lat,
            "mem_delta_mb": mem_delta_mb,
            "migration": migration,
        }

        # Improved aligned MD table
        headers = ["Provider", "Index Time", "DB/chunk", "Embed Time", "Schema", "Search Lat", "Mem Delta"]
        widths = [10, 12, 12, 12, 10, 12, 12]
        values = [
            provider,
            format_time(index_time),
            format_size(db_size_bytes / chunks),
            format_time(embed_time),
            migration,
            f"{avg_lat:.1f} ms",
            format_size(mem_delta_mb * 1024**2)
        ]
        header_row = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
        separator_row = "| " + " | ".join("-" * w for w in widths) + " |"
        data_row = "| " + " | ".join(f"{v:<{w}}" for v, w in zip(values, widths)) + " |"

        if output and output.exists():
            with open(output, "a") as f:
                f.write(data_row + "\n")
        else:
            if output is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output = Path(f"db_bench_result_{provider}_{timestamp}.md")
            with open(output, "w") as f:
                f.write(f"# Database Benchmark: {repo_path} ({chunks} chunks, {files} files)\n\n")
                f.write(header_row + "\n")
                f.write(separator_row + "\n")
                f.write(data_row + "\n")

        # Disconnect to allow cleanup
        try:
            services.provider.disconnect()
        except Exception:
            pass

        print(f"Benchmark complete: {provider} - {chunks} chunks, {index_time:.2f}s index, {db_size_bytes/1024**2:.2f} MB DB")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Benchmark ChunkHound databases.")
    parser.add_argument("--repo", type=Path, default="tests/fixtures", help="Repo to index")
    parser.add_argument("--provider", choices=["lancedb", "duckdb"], default="lancedb")
    parser.add_argument("--output", type=Path, help="Output MD file (default: db_bench_result_{provider}_{timestamp}.md)")
    args = parser.parse_args()

    asyncio.run(benchmark_provider(args.repo, args.provider, args.output))


if __name__ == "__main__":
    main()