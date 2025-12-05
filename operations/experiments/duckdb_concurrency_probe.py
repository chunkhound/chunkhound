"""
Probe DuckDB multi-connection concurrency behavior on ChunkHound-like data.

This script mirrors the LanceDB concurrency probe but uses DuckDB as the
backend. It runs one writer process and multiple reader processes, each with
its own DuckDB connection to a shared database file. The writer continuously
updates existing chunk embeddings and inserts new chunks; readers run queries
that filter on provider/model and track when they observe newly inserted rows.

Notes:
- This is a standalone experiment; it does NOT use ChunkHound providers.
- It is NOT an endorsement of concurrent DuckDB access in ChunkHound. The
  project architecture still assumes a single owning process per database.
- Requires `duckdb` and `numpy`.

Usage example:
  uv run python operations/experiments/duckdb_concurrency_probe.py \\
      --db /tmp/duckdb_probe.duckdb --rows 5000 --readers 4 --writers 1 --duration 10
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from multiprocessing import Event, Process, Queue, get_context
from pathlib import Path
from typing import Any


def _import_duckdb() -> tuple[Any, Any]:
    try:
        import duckdb  # type: ignore
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing dependencies for DuckDB experiment: {e}.\n"
            "Install dev deps or run with 'uv run'."
        )
    return duckdb, np


def connect(db_path: Path, read_only: bool = False):
    duckdb, _ = _import_duckdb()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # DuckDB uses file locks to protect the database file. read_only=True
    # allows shared access between multiple readers when no writer holds
    # an exclusive lock.
    return duckdb.connect(str(db_path), read_only=read_only)


def ensure_schema(conn: Any, dims: int) -> None:
    """Create files/chunks tables if they do not exist.

    Embeddings are stored as FLOAT[] arrays to roughly approximate the layout
    used in ChunkHound's DuckDB provider (without depending on its code).
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            id BIGINT,
            path TEXT,
            size BIGINT,
            modified_time DOUBLE,
            content_hash TEXT,
            indexed_time DOUBLE,
            language TEXT,
            encoding TEXT,
            line_count BIGINT
        )
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGINT,
            file_id BIGINT,
            content TEXT,
            start_line BIGINT,
            end_line BIGINT,
            chunk_type TEXT,
            language TEXT,
            name TEXT,
            embedding FLOAT[{dims}],
            provider TEXT,
            model TEXT,
            created_time DOUBLE
        )
        """
    )


def seed_data(conn: Any, rows: int, dims: int, provider: str, model: str) -> list[int]:
    """Insert initial data with embeddings in batches."""
    _, np = _import_duckdb()
    ensure_schema(conn, dims)

    ids: list[int] = []
    batch = 1000
    cur = 0
    while cur < rows:
        take = min(batch, rows - cur)
        data = []
        for j in range(take):
            cid = cur + j + 1
            ids.append(cid)
            vec = np.random.default_rng(42 + cid).normal(size=dims).astype("float32").tolist()
            data.append(
                (
                    cid,  # id
                    cid,  # file_id
                    "",  # content
                    0,  # start_line
                    0,  # end_line
                    "code",  # chunk_type
                    "text",  # language
                    f"sym_{cid}",  # name
                    vec,  # embedding
                    provider,
                    model,
                    time.time(),
                )
            )
        conn.executemany(
            """
            INSERT INTO chunks (
                id, file_id, content, start_line, end_line,
                chunk_type, language, name, embedding, provider,
                model, created_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            data,
        )
        cur += take
    return ids


@dataclass
class Metrics:
    reads_ok: int = 0
    reads_err: int = 0
    writes_ok: int = 0
    writes_err: int = 0
    fresh_read_iterations: int = 0
    fresh_read_results: int = 0
    new_rows_inserted: int = 0


def reader_worker(
    db_path: Path,
    dims: int,
    provider: str,
    model: str,
    seed_rows: int,
    duration: float,
    start_evt: Event,
    out_q: Queue,
) -> None:
    _, np = _import_duckdb()
    conn = connect(db_path, read_only=True)
    ensure_schema(conn, dims)

    metrics = Metrics()
    rng = np.random.default_rng()
    start_evt.wait()
    end_time = time.time() + duration

    while time.time() < end_time:
        try:
            # Use a simple random probe; we care about visibility of new rows,
            # not perfect similarity search.
            res = conn.execute(
                """
                SELECT id, provider, model
                FROM chunks
                WHERE provider = ? AND model = ?
                ORDER BY random()
                LIMIT 5
                """,
                [provider, model],
            ).fetchall()

            fresh = 0
            for row_id, row_provider, row_model in res:
                if row_provider == provider and row_model == model and isinstance(row_id, int) and row_id > seed_rows:
                    fresh += 1

            if fresh > 0:
                metrics.fresh_read_iterations += 1
                metrics.fresh_read_results += fresh

            metrics.reads_ok += 1
        except Exception:
            metrics.reads_err += 1
    out_q.put(metrics)


def writer_worker(
    db_path: Path,
    dims: int,
    provider: str,
    model: str,
    seed_rows: int,
    ids: list[int],
    duration: float,
    start_evt: Event,
    out_q: Queue,
) -> None:
    _, np = _import_duckdb()
    conn = connect(db_path)
    ensure_schema(conn, dims)

    metrics = Metrics()
    rng = np.random.default_rng(123)
    start_evt.wait()
    end_time = time.time() + duration
    next_id = seed_rows + 1

    while time.time() < end_time:
        try:
            # 1) Update embeddings for a subset of existing rows
            if ids:
                sample = random.sample(ids, k=min(64, len(ids)))
                update_data = []
                for cid in sample:
                    vec = rng.normal(size=dims).astype("float32").tolist()
                    update_data.append((vec, cid))
                conn.executemany(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    update_data,
                )

            # 2) Insert new rows with embeddings
            new_rows = []
            for _ in range(64):
                cid = next_id
                next_id += 1
                ids.append(cid)
                vec = rng.normal(size=dims).astype("float32").tolist()
                new_rows.append(
                    (
                        cid,
                        cid,
                        "new_chunk",
                        0,
                        0,
                        "code",
                        "text",
                        f"rw_sym_{cid}",
                        vec,
                        provider,
                        model,
                        time.time(),
                    )
                )

            if new_rows:
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        id, file_id, content, start_line, end_line,
                        chunk_type, language, name, embedding, provider,
                        model, created_time
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    new_rows,
                )
                metrics.new_rows_inserted += len(new_rows)

            metrics.writes_ok += 1
        except Exception:
            metrics.writes_err += 1
    out_q.put(metrics)


def run_probe(
    db_path: Path,
    dims: int,
    rows: int,
    readers: int,
    writers: int,
    duration: float,
) -> dict[str, int]:
    # Seed initial data in a single owning connection, then close it so that
    # reader/writer processes can open their own connections.
    conn = connect(db_path)
    ids = seed_data(conn, rows, dims, provider="prov", model="mdl")
    conn.close()

    # Use spawn start method to avoid inheriting DuckDB state across forks.
    ctx = get_context("spawn")
    start_evt = ctx.Event()
    out_q = ctx.Queue()
    procs: list[Process] = []

    for _ in range(readers):
        p = ctx.Process(
            target=reader_worker,
            args=(db_path, dims, "prov", "mdl", rows, duration, start_evt, out_q),
        )
        p.daemon = True
        procs.append(p)

    for _ in range(writers):
        p = ctx.Process(
            target=writer_worker,
            args=(db_path, dims, "prov", "mdl", rows, ids, duration, start_evt, out_q),
        )
        p.daemon = True
        procs.append(p)

    for p in procs:
        p.start()

    start_evt.set()
    for p in procs:
        p.join()

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
    parser = argparse.ArgumentParser(description="DuckDB concurrency probe")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("/tmp/duckdb_probe.duckdb"),
        help="DuckDB database file",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=1536,
        help="Embedding dimensions",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help="Initial rows to seed",
    )
    parser.add_argument(
        "--readers",
        type=int,
        default=4,
        help="Number of reader processes",
    )
    parser.add_argument(
        "--writers",
        type=int,
        default=1,
        help="Number of writer processes",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Run duration in seconds",
    )
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stats = run_probe(args.db, args.dims, args.rows, args.readers, args.writers, args.duration)
    elapsed = time.time() - t0

    print(
        {
            "backend": "duckdb",
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
