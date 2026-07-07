#!/usr/bin/env python3
"""Benchmark: Rust RustDbWriter vs Python direct DuckDB writes.

Two modes:
  Real DB:    --db /path/to/chunks.db  [--sample-files N]
  Synthetic:  --files N --chunks-per-file N --embedding-dims N

Both paths write to fresh temporary databases and report throughput.

Usage examples:
  uv run python scripts/bench_db_writer.py --files 500 --chunks-per-file 50 --embedding-dims 1536
  uv run python scripts/bench_db_writer.py --db ~/.chunkhound/db/chunks.db --sample-files 200
  uv run python scripts/bench_db_writer.py --files 200 --chunks-per-file 30 --python-only
"""

from __future__ import annotations

import argparse
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import duckdb

# ---------------------------------------------------------------------------
# Production batch-size logic (mirrors indexing_coordinator._determine_db_batch_size)
# ---------------------------------------------------------------------------

def _mem_available_bytes() -> int:
    try:
        import psutil
        return int(getattr(psutil.virtual_memory(), "available", 0)) or 0
    except Exception:
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    return 0


def determine_batch_size(files: list[dict]) -> int:
    """Compute a file-level batch size matching production memory behaviour.

    Priority:
      1. CHUNKHOUND_DB_BATCH_SIZE env var (interpreted as files/batch)
      2. Dynamic: budget 10% of available RAM against estimated bytes per file
         (code + embedding floats × 4 B + overhead), clamped to [100, 5000] files.

    The production indexer uses a chunk-level batch size [1000, 20000]; we
    convert to file-level here because the benchmark iterates over files.
    """
    try:
        env_bs = int(os.environ.get("CHUNKHOUND_DB_BATCH_SIZE", "0") or "0")
        if env_bs > 0:
            return max(1, min(env_bs, 20000))
    except Exception:
        pass

    if not files:
        return 1000

    avail = _mem_available_bytes()
    if avail <= 0:
        avail = 512 * 1024 * 1024
    budget = int(avail * 0.10)
    budget = max(64 * 1024 * 1024, min(budget, 512 * 1024 * 1024))

    # Estimate bytes per file from a sample (code + embedding storage + overhead)
    sample_files = files[: min(50, len(files))]
    total_bytes = 0
    total_counted = 0
    for f in sample_files:
        for c in f.get("chunks", []):
            code_bytes = len((c.get("code") or "").encode("utf-8", errors="ignore"))
            emb = c.get("embedding")
            emb_bytes = len(emb) * 4 if emb else 0  # float32
            total_bytes += code_bytes + emb_bytes + 512  # 512 B overhead per chunk
            total_counted += 1

    if total_counted == 0:
        return 1000

    avg_bytes_per_chunk = max(512, total_bytes // total_counted)
    avg_chunks_per_file = total_counted / len(sample_files)
    avg_bytes_per_file = max(1024, int(avg_bytes_per_chunk * avg_chunks_per_file))

    est = max(1, budget // avg_bytes_per_file)
    return max(100, min(int(est), 5000))


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_CREATE_SCHEMA = """
CREATE SEQUENCE IF NOT EXISTS files_id_seq START 1;
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    extension TEXT,
    size INTEGER,
    modified_time TIMESTAMP,
    content_hash TEXT,
    language TEXT,
    skip_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1;
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
    file_id INTEGER REFERENCES files(id),
    chunk_type TEXT NOT NULL,
    symbol TEXT,
    code TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    start_byte INTEGER,
    end_byte INTEGER,
    language TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq START 1;
CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
"""


def _create_embedding_table(conn: Any, dims: int) -> None:
    table = f"embeddings_{dims}"
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
            chunk_id INTEGER NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding FLOAT[{dims}],
            dims INTEGER NOT NULL DEFAULT {dims},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_{dims}_chunk_provider_model_unique
        ON {table} (chunk_id, provider, model)
    """)


# ---------------------------------------------------------------------------
# Data loading / generation
# ---------------------------------------------------------------------------

def load_from_db(db_path: str, sample_files: int | None) -> list[dict]:
    """Load real files+chunks+embeddings from an existing DuckDB (read-only).

    Samples only files that have at least one chunk, using bulk queries.
    """
    from datetime import datetime as _dt

    conn = duckdb.connect(db_path, read_only=True)
    try:
        limit = sample_files or 200
        # Sample files that have chunks; use USING SAMPLE for randomness at scale
        files_rows = conn.execute(f"""
            SELECT f.id, f.path, f.modified_time, f.size, f.content_hash, f.language
            FROM files f
            WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = f.id)
            LIMIT {limit}
        """).fetchall()

        if not files_rows:
            return []

        file_ids = [r[0] for r in files_rows]
        placeholders = ",".join(str(i) for i in file_ids)

        # Bulk load all chunks for sampled files in one query
        chunk_rows = conn.execute(f"""
            SELECT id, file_id, symbol, start_line, end_line, code, chunk_type, language
            FROM chunks
            WHERE file_id IN ({placeholders})
        """).fetchall()

        chunk_ids = [r[0] for r in chunk_rows]

        # Discover embedding tables
        emb_tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name SIMILAR TO 'embeddings_[0-9]+'"
            ).fetchall()
        ]

        # Bulk load embeddings for all chunks
        vectors: dict[int, tuple[list[float], str, str]] = {}  # chunk_id → (emb, provider, model)
        if emb_tables and chunk_ids:
            chunk_id_csv = ",".join(str(i) for i in chunk_ids)
            for tbl in emb_tables:
                rows = conn.execute(
                    f"SELECT chunk_id, embedding, provider, model FROM {tbl} "
                    f"WHERE chunk_id IN ({chunk_id_csv}) AND embedding IS NOT NULL"
                ).fetchall()
                for cid, emb, provider, model in rows:
                    vectors[cid] = (list(emb), provider, model)

        # Group chunks by file_id
        from collections import defaultdict
        chunks_by_file: dict[int, list] = defaultdict(list)
        for row in chunk_rows:
            chunks_by_file[row[1]].append(row)

        result = []
        for file_id, path, mtime, size, content_hash, language in files_rows:
            mtime_float = None
            if mtime is not None:
                try:
                    if isinstance(mtime, _dt):
                        mtime_float = mtime.timestamp()
                    else:
                        mtime_float = float(mtime)
                except Exception:
                    pass

            file_chunks = chunks_by_file.get(file_id, [])
            chunk_dicts = []
            for c in file_chunks:
                cid = c[0]
                emb_info = vectors.get(cid)
                chunk_dicts.append({
                    "chunk_type": c[6] or "text",
                    "symbol": c[2],
                    "start_line": c[3],
                    "end_line": c[4],
                    "code": c[5] or "",
                    "language": c[7],
                    "metadata": None,
                    "embedding": emb_info[0] if emb_info else None,
                    "provider": emb_info[1] if emb_info else None,
                    "model": emb_info[2] if emb_info else None,
                })

            result.append({
                "existing_file_id": None,
                "path": path,
                "mtime": mtime_float,
                "size_bytes": size,
                "content_hash": content_hash,
                "language": language,
                "chunks": chunk_dicts,
            })
        return result
    finally:
        conn.close()


def generate_synthetic(
    n_files: int,
    chunks_per_file: int,
    embedding_dims: int,
) -> list[dict]:
    """Generate fake files+chunks+embeddings deterministically."""
    rng = random.Random(42)
    files = []
    for i in range(n_files):
        path = f"/fake/repo/module_{i // 10}/file_{i}.py"
        chunks = []
        for j in range(chunks_per_file):
            # Hash-based deterministic embedding
            seed = hash((i, j)) & 0xFFFFFFFF
            r = random.Random(seed)
            emb = [r.gauss(0.0, 1.0) for _ in range(embedding_dims)]
            # Normalise to unit sphere (mimics real embeddings)
            norm = sum(x * x for x in emb) ** 0.5 or 1.0
            emb = [x / norm for x in emb]
            chunks.append({
                "chunk_type": "function",
                "symbol": f"func_{j}",
                "start_line": j * 10 + 1,
                "end_line": j * 10 + 9,
                "code": f"def func_{j}(x):\n    return x + {j}\n",
                "language": "python",
                "metadata": None,
                "embedding": emb,
                "provider": "openai",
                "model": "text-embedding-3-small",
            })
        files.append({
            "existing_file_id": None,
            "path": path,
            "mtime": 1_700_000_000.0 + i,
            "size_bytes": chunks_per_file * 50,
            "content_hash": f"hash_{i:08x}",
            "language": "python",
            "chunks": chunks,
        })
    return files


# ---------------------------------------------------------------------------
# Python direct path
# ---------------------------------------------------------------------------

def _python_write_files(conn: Any, files: list[dict]) -> tuple[int, int]:
    """Write all files+chunks+embeddings using the Python duckdb API directly."""
    dims_seen: set[int] = set()
    total_chunks = 0
    total_embs = 0

    for file in files:
        # Upsert file
        rows = conn.execute(
            """INSERT INTO files (path, name, extension, size, modified_time, content_hash, language)
               VALUES (?, ?, ?, ?, CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, ?, ?)
               ON CONFLICT (path) DO UPDATE SET
                   size = EXCLUDED.size,
                   modified_time = EXCLUDED.modified_time,
                   content_hash = EXCLUDED.content_hash,
                   language = EXCLUDED.language,
                   updated_at = now()
               RETURNING id""",
            [
                file["path"],
                Path(file["path"]).name,
                Path(file["path"]).suffix,
                file.get("size_bytes"),
                file.get("mtime"),
                file.get("mtime"),
                file.get("content_hash"),
                file.get("language"),
            ],
        ).fetchone()
        file_id = rows[0]

        chunks = file.get("chunks", [])
        if not chunks:
            continue

        # Bulk insert chunks via temp table
        conn.execute("""
            CREATE TEMPORARY TABLE IF NOT EXISTS py_temp_chunks (
                file_id INTEGER, chunk_type TEXT, symbol TEXT, code TEXT,
                start_line INTEGER, end_line INTEGER, start_byte INTEGER, end_byte INTEGER,
                language TEXT, metadata TEXT
            )
        """)
        conn.execute("DELETE FROM py_temp_chunks")
        conn.executemany(
            "INSERT INTO py_temp_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    file_id,
                    c.get("chunk_type", "text"),
                    c.get("symbol"),
                    c.get("code", ""),
                    c.get("start_line"),
                    c.get("end_line"),
                    c.get("start_byte"),
                    c.get("end_byte"),
                    c.get("language"),
                    c.get("metadata"),
                )
                for c in chunks
            ],
        )
        chunk_ids = conn.execute(
            """INSERT INTO chunks (file_id, chunk_type, symbol, code, start_line, end_line,
               start_byte, end_byte, language, metadata)
               SELECT * FROM py_temp_chunks RETURNING id"""
        ).fetchall()
        total_chunks += len(chunk_ids)
        chunk_id_list = [r[0] for r in chunk_ids]

        # Insert embeddings grouped by dims
        emb_chunks = [
            (cid, c)
            for cid, c in zip(chunk_id_list, chunks)
            if c.get("embedding")
        ]
        by_dims: dict[int, list[tuple[int, dict]]] = {}
        for cid, c in emb_chunks:
            d = len(c["embedding"])
            by_dims.setdefault(d, []).append((cid, c))

        for d, items in by_dims.items():
            if d not in dims_seen:
                _create_embedding_table(conn, d)
                dims_seen.add(d)
            table = f"embeddings_{d}"
            # Temp table for embeddings
            conn.execute(f"""
                CREATE TEMPORARY TABLE IF NOT EXISTS py_temp_emb_{d} (
                    chunk_id INTEGER, provider TEXT, model TEXT,
                    embedding TEXT, dims INTEGER
                )
            """)
            conn.execute(f"DELETE FROM py_temp_emb_{d}")
            conn.executemany(
                f"INSERT INTO py_temp_emb_{d} VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        cid,
                        c.get("provider", "unknown"),
                        c.get("model", "unknown"),
                        __import__("json").dumps(c["embedding"]),
                        d,
                    )
                    for cid, c in items
                ],
            )
            n = conn.execute(
                f"""INSERT INTO {table} (chunk_id, provider, model, embedding, dims)
                   SELECT chunk_id, provider, model,
                          embedding::FLOAT[{d}],
                          dims
                   FROM py_temp_emb_{d}
                   ON CONFLICT (chunk_id, provider, model) DO UPDATE
                   SET embedding = EXCLUDED.embedding, dims = EXCLUDED.dims"""
            ).rowcount
            if n is not None:
                total_embs += n
            else:
                total_embs += len(items)

    return total_chunks, total_embs


def run_python_path(files: list[dict], db_path: str, batch_size: int) -> dict:
    """Run the Python direct write path in production-sized batches."""
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()

    conn = duckdb.connect(db_path)
    try:
        conn.execute(_CREATE_SCHEMA)
        total_chunks = total_embs = 0
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            c, e = _python_write_files(conn, batch)
            total_chunks += c
            total_embs += e
        conn.execute("CHECKPOINT")
    finally:
        conn.close()

    wall = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    db_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0

    return {
        "wall": wall,
        "chunks": total_chunks,
        "files": len(files),
        "embs": total_embs,
        "rss_delta_kb": rss_after - rss_before,
        "db_bytes": db_size,
    }


# ---------------------------------------------------------------------------
# Rust path
# ---------------------------------------------------------------------------

def run_rust_path(files: list[dict], db_path: str, batch_size: int) -> dict:
    """Run the Rust RustDbWriter path in production-sized batches."""
    try:
        from chunkhound_native import RustDbWriter
    except ImportError:
        return {"error": "chunkhound_native not built — run: uv run maturin develop"}

    db_config = {"db_path": db_path, "compaction_batch_threshold": 9999}

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()

    writer = RustDbWriter(db_config)
    writer.open()
    try:
        total_chunks = total_embs = 0
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            result = writer.write_batch({"files": batch, "delete_paths": []})
            total_chunks += result.get("chunks_written", 0)
            total_embs += result.get("embeddings_written", 0)
    finally:
        writer.close()

    wall = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    db_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0

    return {
        "wall": wall,
        "chunks": total_chunks,
        "files": len(files),
        "embs": total_embs,
        "rss_delta_kb": rss_after - rss_before,
        "db_bytes": db_size,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def _fmt_rss(kb: int) -> str:
    # ru_maxrss is in KB on Linux, bytes on macOS
    if sys.platform == "darwin":
        kb = kb // 1024
    mb = kb / 1024
    return f"+{mb:.1f} MB" if mb >= 0 else f"{mb:.1f} MB"


def print_table(py_result: dict | None, rust_result: dict | None) -> None:
    py_err = py_result.get("error") if py_result else None
    rust_err = rust_result.get("error") if rust_result else None

    def _cell(r: dict | None, err: str | None, key: str, fmt=str) -> str:
        if err:
            return err if key == "wall" else "N/A"
        if r is None:
            return "skipped"
        v = r.get(key)
        if v is None:
            return "N/A"
        return fmt(v)

    if py_result and not py_err:
        n_files = py_result["files"]
        n_chunks = py_result["chunks"]
        py_wall = py_result["wall"]
        py_tput = n_chunks / py_wall if py_wall > 0 else 0
        py_fps = n_files / py_wall if py_wall > 0 else 0
    else:
        py_wall = py_tput = py_fps = None

    if rust_result and not rust_err:
        n_files_r = rust_result["files"]
        n_chunks_r = rust_result["chunks"]
        rust_wall = rust_result["wall"]
        rust_tput = n_chunks_r / rust_wall if rust_wall > 0 else 0
        rust_fps = n_files_r / rust_wall if rust_wall > 0 else 0
    else:
        rust_wall = rust_tput = rust_fps = None

    speedup = (
        (py_wall / rust_wall) if (py_wall and rust_wall and rust_wall > 0) else None
    )

    SEP = "─" * 20
    W = 18

    def col(v: str) -> str:
        return str(v).rjust(W)

    header_py = "Python direct".rjust(W)
    header_rust = "Rust".rjust(W)

    print()
    print("ChunkHound DB Writer Benchmark")
    if py_result:
        n = py_result.get("files", 0)
        c = py_result.get("chunks", 0)
        print(f"Config: {n} files × {c // n if n else 0} chunks")
    print()
    print(f"{'Metric':<22}{header_py}{header_rust}")
    print(f"{SEP:<22}{SEP}{SEP}")

    def row(label: str, py_val: str, rust_val: str) -> None:
        print(f"{label:<22}{py_val.rjust(W)}{rust_val.rjust(W)}")

    row(
        "Wall time",
        f"{py_wall:.3f} s" if py_wall is not None else (py_err or "skipped"),
        f"{rust_wall:.3f} s" if rust_wall is not None else (rust_err or "skipped"),
    )
    row(
        "Throughput (c/s)",
        f"{py_tput:,.0f}" if py_tput is not None else "N/A",
        f"{rust_tput:,.0f}" if rust_tput is not None else "N/A",
    )
    row(
        "Files/sec",
        f"{py_fps:,.1f}" if py_fps is not None else "N/A",
        f"{rust_fps:,.1f}" if rust_fps is not None else "N/A",
    )
    row(
        "Peak RSS delta",
        _fmt_rss(py_result["rss_delta_kb"]) if py_result and not py_err else "N/A",
        _fmt_rss(rust_result["rss_delta_kb"]) if rust_result and not rust_err else "N/A",
    )
    row(
        "DB size after",
        _fmt_bytes(py_result["db_bytes"]) if py_result and not py_err else "N/A",
        _fmt_bytes(rust_result["db_bytes"]) if rust_result and not rust_err else "N/A",
    )
    print(f"{SEP:<22}{SEP}{SEP}")
    row(
        "Speedup",
        "1.0×",
        f"{speedup:.1f}×" if speedup is not None else "N/A",
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark RustDbWriter vs Python direct DuckDB writes"
    )

    # Source
    source = p.add_mutually_exclusive_group()
    source.add_argument("--db", metavar="PATH", help="Real DuckDB source (read-only)")
    source.add_argument(
        "--files",
        type=int,
        default=200,
        metavar="N",
        help="Synthetic: number of files (default: 200)",
    )

    p.add_argument(
        "--sample-files",
        type=int,
        default=None,
        metavar="N",
        help="Cap files loaded from --db (default: all)",
    )
    p.add_argument(
        "--chunks-per-file",
        type=int,
        default=50,
        metavar="N",
        help="Synthetic: chunks per file (default: 50)",
    )
    p.add_argument(
        "--embedding-dims",
        type=int,
        default=1536,
        metavar="N",
        help="Synthetic: embedding dimensions (default: 1536)",
    )
    p.add_argument(
        "--python-only",
        action="store_true",
        help="Skip Rust path (useful for Python baseline before building)",
    )
    p.add_argument(
        "--rust-only",
        action="store_true",
        help="Skip Python path",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of timed runs (default: 1)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load or generate data
    if args.db:
        print(f"Loading data from {args.db} ...")
        files = load_from_db(args.db, args.sample_files)
        print(f"  Loaded {len(files)} files")
    else:
        n_files = args.files
        cpf = args.chunks_per_file
        dims = args.embedding_dims
        total_chunks = n_files * cpf
        print(f"Generating synthetic data: {n_files} files × {cpf} chunks × {dims} dims "
              f"= {total_chunks:,} chunks total ...")
        files = generate_synthetic(n_files, cpf, dims)

    if not files:
        print("No data to benchmark.")
        sys.exit(1)

    batch_size = determine_batch_size(files)
    print(f"  Batch size: {batch_size:,} files/batch (production heuristic)")

    best_py: dict | None = None
    best_rust: dict | None = None

    for run_idx in range(args.runs):
        if args.runs > 1:
            print(f"\n── Run {run_idx + 1}/{args.runs} ──")

        if not args.rust_only:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                py_db = f.name
            os.unlink(py_db)  # DuckDB creates fresh DB if file absent
            try:
                print("Running Python direct path ...", end=" ", flush=True)
                py_r = run_python_path(files, py_db, batch_size)
                if "error" not in py_r:
                    print(f"{py_r['wall']:.3f}s")
                else:
                    print(f"ERROR: {py_r['error']}")
                if best_py is None or py_r.get("wall", 1e9) < best_py.get("wall", 1e9):
                    best_py = py_r
            finally:
                try:
                    os.unlink(py_db)
                except OSError:
                    pass
        else:
            best_py = None

        if not args.python_only:
            with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
                rust_db = f.name
            os.unlink(rust_db)  # DuckDB creates fresh DB if file absent
            try:
                print("Running Rust path ...", end=" ", flush=True)
                rust_r = run_rust_path(files, rust_db, batch_size)
                if "error" not in rust_r:
                    print(f"{rust_r['wall']:.3f}s")
                else:
                    print(f"ERROR: {rust_r['error']}")
                if best_rust is None or rust_r.get("wall", 1e9) < best_rust.get("wall", 1e9):
                    best_rust = rust_r
            finally:
                try:
                    os.unlink(rust_db)
                except OSError:
                    pass
        else:
            best_rust = None

    print_table(best_py, best_rust)


if __name__ == "__main__":
    main()
