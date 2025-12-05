"""
Probe LanceDB multi-connection concurrency behavior.

This script runs concurrent reader and (optional) writer processes, each with
its own LanceDB connection, against a shared database directory. It measures
throughput and exceptions to assess feasibility of concurrent access.

Notes:
- This is a standalone experiment; it does NOT use ChunkHound providers.
- It is safe to run locally. Do not run inside MCP stdio server.
- Requires `lancedb`, `pyarrow`, and `numpy`.

Usage examples:
  uv run python operations/experiments/lancedb_concurrency_probe.py \
      --db /tmp/ldbt --rows 20000 --readers 8 --writers 1 --duration 20

  uv run python operations/experiments/lancedb_concurrency_probe.py \
      --db /tmp/ldbt --rows 20000 --readers 8 --writers 0 --duration 20
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue, get_context
from pathlib import Path
from typing import Any


def _import_lance() -> tuple[Any, Any, Any]:
    try:
        import lancedb  # type: ignore
        import pyarrow as pa  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing dependencies for LanceDB experiment: {e}.\n"
            "Install dev deps or run with 'uv run'."
        )
    return lancedb, pa, np


def connect(db_dir: Path):
    lancedb, _, _ = _import_lance()
    db_dir.mkdir(parents=True, exist_ok=True)
    # LanceDB connects to a directory path
    return lancedb.connect(str(db_dir))


def ensure_schema(conn: Any, dims: int) -> tuple[Any, Any]:
    """Open or create files/chunks tables with expected schema."""
    _, pa, _ = _import_lance()

    # Files schema
    files_schema = pa.schema(
        [
            ("id", pa.int64()),
            ("path", pa.string()),
            ("size", pa.int64()),
            ("modified_time", pa.float64()),
            ("content_hash", pa.string()),
            ("indexed_time", pa.float64()),
            ("language", pa.string()),
            ("encoding", pa.string()),
            ("line_count", pa.int64()),
        ]
    )

    # Chunks schema with fixed-size list for embedding
    embedding_field = pa.list_(pa.float32(), dims)
    chunks_schema = pa.schema(
        [
            ("id", pa.int64()),
            ("file_id", pa.int64()),
            ("content", pa.string()),
            ("start_line", pa.int64()),
            ("end_line", pa.int64()),
            ("chunk_type", pa.string()),
            ("language", pa.string()),
            ("name", pa.string()),
            ("embedding", embedding_field),
            ("provider", pa.string()),
            ("model", pa.string()),
            ("created_time", pa.float64()),
        ]
    )

    try:
        files = conn.open_table("files")
    except Exception:
        files = conn.create_table("files", schema=files_schema)

    try:
        chunks = conn.open_table("chunks")
    except Exception:
        chunks = conn.create_table("chunks", schema=chunks_schema)

    return files, chunks


def seed_data(conn: Any, rows: int, dims: int, provider: str, model: str) -> list[int]:
    """Insert initial data with embeddings in batches."""
    _, pa, np = _import_lance()
    files, chunks = ensure_schema(conn, dims)

    # Seed files
    file_rows = []
    for i in range(rows):
        file_rows.append(
            {
                "id": i + 1,
                "path": f"file_{i}.txt",
                "size": 0,
                "modified_time": time.time(),
                "content_hash": "",
                "indexed_time": time.time(),
                "language": "text",
                "encoding": "utf-8",
                "line_count": 0,
            }
        )
    if file_rows:
        files.add(pa.Table.from_pylist(file_rows))

    # Seed chunks with embeddings
    ids: list[int] = []
    batch = 1000
    cur = 0
    rng = np.random.default_rng(42)
    while cur < rows:
        take = min(batch, rows - cur)
        data = []
        for j in range(take):
            cid = cur + j + 1
            ids.append(cid)
            vec = rng.normal(size=dims).astype("float32").tolist()
            data.append(
                {
                    "id": cid,
                    "file_id": cid,
                    "content": "",
                    "start_line": 0,
                    "end_line": 0,
                    "chunk_type": "code",
                    "language": "text",
                    "name": f"sym_{cid}",
                    "embedding": vec,
                    "provider": provider,
                    "model": model,
                    "created_time": time.time(),
                }
            )
        chunks.add(pa.Table.from_pylist(data))
        cur += take
    return ids


def ensure_vector_index(conn: Any, dims: int, index_type: str | None = None) -> None:
    """Create IVF/HNSW vector index if not present."""
    # Probe: try a search; if it fails, index likely missing → create it
    try:
        chunks = conn.open_table("chunks")
        chunks.search([0.0] * dims, vector_column_name="embedding").limit(1).to_list()
        return
    except Exception:
        pass

    try:
        chunks = conn.open_table("chunks")
        if index_type == "IVF_HNSW_SQ":
            chunks.create_index(vector_column_name="embedding", index_type="IVF_HNSW_SQ", metric="cosine")
        else:
            chunks.create_index(vector_column_name="embedding", metric="cosine")
    except Exception:
        # If index creation fails (e.g., dataset too small), continue without
        pass


@dataclass
class Metrics:
    reads_ok: int = 0
    reads_err: int = 0
    writes_ok: int = 0
    writes_err: int = 0
    # Number of search iterations that returned at least one newly inserted row (id > seed_rows)
    fresh_read_iterations: int = 0
    # Total count of result rows that came from newly inserted ids
    fresh_read_results: int = 0
    # Number of newly inserted rows created by writers (beyond initial seed_data rows)
    new_rows_inserted: int = 0


def reader_worker(
    db_dir: Path,
    dims: int,
    provider: str,
    model: str,
    seed_rows: int,
    duration: float,
    start_evt: Event,
    out_q: Queue,
) -> None:
    _, _, np = _import_lance()
    conn = connect(db_dir)
    ensure_schema(conn, dims)
    ensure_vector_index(conn, dims)

    metrics = Metrics()
    rng = np.random.default_rng()
    start_evt.wait()
    end_time = time.time() + duration
    while time.time() < end_time:
        try:
            q = rng.normal(size=dims).astype("float32").tolist()
            # Filter by provider/model to stress index + predicate
            rows = (
                conn.open_table("chunks")
                .search(q, vector_column_name="embedding")
                .where(f"provider = '{provider}' AND model = '{model}' AND embedding IS NOT NULL")
                .limit(5)
                .to_list()
            )
            fresh = 0
            for row in rows:
                row_id = row.get("id")
                if isinstance(row_id, int) and row_id > seed_rows:
                    fresh += 1

            if fresh > 0:
                metrics.fresh_read_iterations += 1
                metrics.fresh_read_results += fresh

            metrics.reads_ok += 1
        except Exception:
            metrics.reads_err += 1
    out_q.put(metrics)


def writer_worker(
    db_dir: Path,
    dims: int,
    provider: str,
    model: str,
    seed_rows: int,
    ids: list[int],
    duration: float,
    start_evt: Event,
    out_q: Queue,
) -> None:
    _, pa, np = _import_lance()
    conn = connect(db_dir)
    ensure_schema(conn, dims)
    ensure_vector_index(conn, dims)

    metrics = Metrics()
    rng = np.random.default_rng(123)
    start_evt.wait()
    end_time = time.time() + duration

    chunks = conn.open_table("chunks")
    # Moderate batch size to keep writer busy
    batch = 256
    next_id = seed_rows + 1
    while time.time() < end_time:
        try:
            # 1) Update a subset of existing rows (simulates re-embedding)
            if ids:
                sample = random.sample(ids, k=min(batch // 2, len(ids)))
                updates = []
                for cid in sample:
                    vec = rng.normal(size=dims).astype("float32").tolist()
                    updates.append(
                        {
                            "id": cid,
                            "embedding": vec,
                            "provider": provider,
                            "model": model,
                        }
                    )
                chunks.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(updates)

            # 2) Insert brand new rows with embeddings (simulates new chunks being indexed)
            new_rows = []
            for _ in range(batch // 2):
                cid = next_id
                next_id += 1
                ids.append(cid)
                vec = rng.normal(size=dims).astype("float32").tolist()
                new_rows.append(
                    {
                        "id": cid,
                        "file_id": cid,
                        "content": "new_chunk",
                        "start_line": 0,
                        "end_line": 0,
                        "chunk_type": "code",
                        "language": "text",
                        "name": f"rw_sym_{cid}",
                        "embedding": vec,
                        "provider": provider,
                        "model": model,
                        "created_time": time.time(),
                    }
                )

            if new_rows:
                chunks.add(pa.Table.from_pylist(new_rows))
                metrics.new_rows_inserted += len(new_rows)

            metrics.writes_ok += 1
        except Exception:
            metrics.writes_err += 1
    out_q.put(metrics)


def run_probe(db_dir: Path, dims: int, rows: int, readers: int, writers: int, duration: float, index_type: str | None) -> dict[str, int]:
    conn = connect(db_dir)
    ids = seed_data(conn, rows, dims, provider="prov", model="mdl")
    ensure_vector_index(conn, dims, index_type=index_type)

    # Use spawn start method to avoid inheriting LanceDB state across forks.
    ctx = get_context("spawn")
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs: list[Process] = []

    for _ in range(readers):
        p = ctx.Process(
            target=reader_worker,
            args=(db_dir, dims, "prov", "mdl", rows, duration, start_evt, out_q),
        )
        p.daemon = True
        procs.append(p)

    for _ in range(writers):
        p = ctx.Process(
            target=writer_worker,
            args=(db_dir, dims, "prov", "mdl", rows, ids, duration, start_evt, out_q),
        )
        p.daemon = True
        procs.append(p)

    for p in procs:
        p.start()

    # Begin run
    start_evt.set()
    for p in procs:
        p.join()

    # Aggregate metrics
    agg = Metrics()
    while not out_q.empty():
        m = out_q.get()
        agg.reads_ok += m.reads_ok
        agg.reads_err += m.reads_err
        agg.writes_ok += m.writes_ok
        agg.writes_err += m.writes_err
        agg.fresh_read_iterations += m.fresh_read_iterations
        agg.fresh_read_results += m.fresh_read_results
        agg.new_rows_inserted += m.new_rows_inserted

    return {
        "reads_ok": agg.reads_ok,
        "reads_err": agg.reads_err,
        "writes_ok": agg.writes_ok,
        "writes_err": agg.writes_err,
        "fresh_read_iterations": agg.fresh_read_iterations,
        "fresh_read_results": agg.fresh_read_results,
        "new_rows_inserted": agg.new_rows_inserted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LanceDB concurrency probe")
    parser.add_argument("--db", type=Path, default=Path("/tmp/lancedb_probe"), help="Database directory")
    parser.add_argument("--dims", type=int, default=1536, help="Embedding dimensions")
    parser.add_argument("--rows", type=int, default=10000, help="Initial rows to seed")
    parser.add_argument("--readers", type=int, default=4, help="Number of reader processes")
    parser.add_argument("--writers", type=int, default=0, help="Number of writer processes")
    parser.add_argument("--duration", type=float, default=15.0, help="Run duration in seconds")
    parser.add_argument(
        "--index-type",
        type=str,
        default=None,
        choices=[None, "IVF_HNSW_SQ", "AUTO"],
        help="Vector index type (None/AUTO/IVF_HNSW_SQ)",
    )
    args = parser.parse_args()

    # Normalize index type
    index_type = None if args.index_type in (None, "AUTO") else args.index_type

    # Make sure parent exists
    args.db.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats = run_probe(args.db, args.dims, args.rows, args.readers, args.writers, args.duration, index_type)
    elapsed = time.time() - t0

    # Print structured summary (JSON-ish for easy parsing)
    print(
        {
            "db": str(args.db),
            "dims": args.dims,
            "rows": args.rows,
            "readers": args.readers,
            "writers": args.writers,
            "duration_sec": args.duration,
            "elapsed_sec": round(elapsed, 3),
            **stats,
        }
    )


if __name__ == "__main__":
    main()
