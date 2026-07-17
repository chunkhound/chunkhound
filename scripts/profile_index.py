#!/usr/bin/env python3
"""Profile indexing flow with FakeEmbeddingProvider (no API keys).

Measures phase wall times, DB merge_insert/optimize counters, and peak RSS.

Examples:

  # Synthetic seed + stream-embed (DB-bound)
  uv run python scripts/profile_index.py --mode soak --chunks 5000

  # Full index of a directory (parse + store + residual embed)
  uv run python scripts/profile_index.py --mode index --root . --defer-write

  # Compare classic two-write vs deferred single-write (synthetic)
  uv run python scripts/profile_index.py --mode soak --chunks 10000 --defer-write
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _quiet_logs() -> None:
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


async def _run_soak(
    *,
    chunks: int,
    page_size: int,
    work_dir: Path,
    defer_write: bool,
) -> dict:
    """Synthetic insert + stream embed (or single-write path simulation)."""
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.core.diagnostics.index_profile import IndexProfile
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.embedding_service import EmbeddingService
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    profile = IndexProfile()
    profile.meta.update(
        {
            "mode": "soak",
            "chunks": chunks,
            "page_size": page_size,
            "defer_write": defer_write,
            "provider": "lancedb",
            "embed": "fake",
        }
    )

    cfg = DatabaseConfig(
        path=work_dir,
        provider="lancedb",
        lancedb_optimize_fragment_threshold=50,
    )
    db = LanceDBProvider(
        str(cfg.get_db_path()), base_directory=work_dir, config=cfg
    )
    db.set_index_profile(profile)
    db.connect()
    try:
        fake = FakeEmbeddingProvider(dims=32, batch_size=page_size)
        files_n = max(1, chunks // 100)
        per = chunks // files_n
        rem = chunks % files_n

        # Per-file deferred flush (L1). Cross-file batching (L2) raised wall.
        profile.meta["flush_policy"] = "per_file" if defer_write else "classic"

        with profile.phase("seed_insert"):
            all_chunks: list[Chunk] = []
            all_ids: list[int] = []
            for f in range(files_n):
                n = per + (1 if f < rem else 0)
                file_id = int(
                    db.insert_file(
                        File(
                            path=f"soak/f_{f:05d}.py",
                            mtime=1_700_000_000.0 + f,
                            language=Language.PYTHON,
                            size_bytes=64 * n,
                        )
                    )
                )
                batch = [
                    Chunk(
                        file_id=file_id,
                        code=f"def soak_{f}_{i}():\n    return {i}\n",
                        start_line=i * 3 + 1,
                        end_line=i * 3 + 2,
                        chunk_type=ChunkType.FUNCTION,
                        language=Language.PYTHON,
                        symbol=f"soak_{f}_{i}",
                    )
                    for i in range(n)
                ]
                if defer_write:
                    texts = [c.code or "" for c in batch]
                    vecs = await fake.embed_batch(texts)
                    vec_lists = [list(v) for v in vecs]
                    ids = db.insert_chunks_with_embeddings_batch(
                        batch,
                        vec_lists,
                        fake.name,
                        fake.model,
                    )
                    all_ids.extend(int(x) for x in ids)
                else:
                    all_ids.extend(int(x) for x in db.insert_chunks_batch(batch))
                    all_chunks.extend(batch)

        if not defer_write:
            with profile.phase("stream_embed"):
                service = EmbeddingService(
                    database_provider=db,
                    embedding_provider=fake,  # type: ignore[arg-type]
                    embedding_batch_size=page_size,
                    db_batch_size=page_size,
                    max_concurrent_batches=4,
                )
                result = await service.generate_missing_embeddings()
                profile.meta["embed_result"] = {
                    "status": result.get("status"),
                    "generated": result.get("generated"),
                }
        else:
            profile.meta["embed_result"] = {
                "status": "deferred_in_seed",
                "generated": len(all_ids),
            }

        remaining = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=10
        )
        profile.meta["remaining_missing"] = len(remaining)
        profile.meta["ids"] = len(all_ids)
        profile.sample_rss()
        return profile.as_dict()
    finally:
        db.disconnect()


async def _run_index(
    *,
    root: Path,
    work_dir: Path,
    defer_write: bool,
    page_size: int,
) -> dict:
    """Full IndexingCoordinator process_directory + generate_missing."""
    from chunkhound.core.config.config import Config
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.indexing_config import IndexingConfig
    from chunkhound.core.diagnostics.index_profile import IndexProfile
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    profile = IndexProfile()
    profile.meta.update(
        {
            "mode": "index",
            "root": str(root.resolve()),
            "defer_write": defer_write,
            "page_size": page_size,
            "provider": "lancedb",
            "embed": "fake",
        }
    )

    cfg = Config(
        database=DatabaseConfig(
            path=work_dir,
            provider="lancedb",
            lancedb_optimize_fragment_threshold=50,
        ),
        indexing=IndexingConfig(
            defer_chunk_write=defer_write,
            db_batch_size=max(100, page_size),
            cleanup=False,
        ),
        embedding=EmbeddingConfig(
            provider="openai",  # unused — we inject FakeEmbeddingProvider
            model="fake-embeddings",
            batch_size=page_size,
        ),
    )

    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=root.resolve(),
        config=cfg.database,
    )
    db.set_index_profile(profile)
    db.connect()
    fake = FakeEmbeddingProvider(dims=32, batch_size=page_size)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=root.resolve(),
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        coord.attach_index_profile(profile)

        with profile.phase("process_directory"):
            # Explicit patterns so discovery does not require full CLI config layer.
            dir_result = await coord.process_directory(
                root.resolve(),
                patterns=["**/*"],
                exclude_patterns=[
                    "**/.git/**",
                    "**/node_modules/**",
                    "**/.venv/**",
                    "**/venv/**",
                    "**/.chunkhound/**",
                    "**/__pycache__/**",
                    "**/target/**",
                    "**/dist/**",
                    "**/.pytest_cache/**",
                ],
            )
        profile.meta["dir_result"] = {
            k: dir_result.get(k)
            for k in (
                "status",
                "files_processed",
                "total_chunks",
                "skipped",
                "skipped_unchanged",
            )
            if k in dir_result
        }

        if not defer_write:
            with profile.phase("generate_missing"):
                emb_result = await coord.generate_missing_embeddings()
            profile.meta["embed_result"] = {
                "status": emb_result.get("status"),
                "generated": emb_result.get("generated"),
            }
        else:
            # Residual pass should be near-empty for new files under defer.
            with profile.phase("generate_missing"):
                emb_result = await coord.generate_missing_embeddings()
            profile.meta["embed_result"] = {
                "status": emb_result.get("status"),
                "generated": emb_result.get("generated"),
            }

        remaining = db.get_chunks_without_embeddings_paginated(
            fake.name, fake.model, limit=20
        )
        profile.meta["remaining_missing"] = len(remaining)
        profile.sample_rss()
        return profile.as_dict()
    finally:
        db.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["soak", "index"],
        default="soak",
        help="soak=synthetic DB path; index=full coordinator over --root",
    )
    parser.add_argument("--chunks", type=int, default=5000)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory to index (mode=index). Defaults to repo root.",
    )
    parser.add_argument(
        "--defer-write",
        action="store_true",
        help="Use embed-then-single-write for new chunks",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help=(
            "Run soak ladder 2k/10k/25k/50k (fake embed) for both classic and "
            "defer paths; prints a comparison table. Ignores --chunks."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON report",
    )
    parser.add_argument(
        "--keep-dir",
        type=Path,
        default=None,
        help="Keep work DB directory (default: temp, deleted on exit)",
    )
    args = parser.parse_args()
    _quiet_logs()

    work = args.keep_dir
    cleanup = False
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="chunkhound-profile-"))
        cleanup = True
    else:
        work.mkdir(parents=True, exist_ok=True)

    def _print_report(report: dict) -> None:
        print("=== index profile ===")
        for k, v in report.get("phases_s", {}).items():
            print(f"  {k:20s} {v:8.3f}s")
        print(f"  {'TOTAL':20s} {report.get('total_s', 0):8.3f}s")
        print(
            f"  rss start={report.get('start_rss_mb')} "
            f"peak={report.get('peak_rss_mb')} MB"
        )
        db = report.get("db", {})
        print(
            "  db merge_insert calls={merge_insert_calls} "
            "rows={merge_insert_rows} s={merge_insert_s} | "
            "optimize calls={optimize_calls} s={optimize_s}".format(
                merge_insert_calls=db.get("merge_insert_calls", 0),
                merge_insert_rows=db.get("merge_insert_rows", 0),
                merge_insert_s=db.get("merge_insert_s", 0),
                optimize_calls=db.get("optimize_calls", 0),
                optimize_s=db.get("optimize_s", 0),
            )
        )
        print(
            f"  db chunk_batches={db.get('chunk_insert_batches')} "
            f"embed_batches={db.get('embedding_insert_batches')}"
        )
        for k, v in report.get("meta", {}).items():
            print(f"  {k}={v}")
        print(f"  wall_s={report.get('wall_s')}")

    try:
        if args.scale:
            # DB scale ladder: only FakeEmbeddingProvider, measure classic vs defer.
            sizes = [2_000, 10_000, 25_000, 50_000]
            rows: list[dict] = []
            for n in sizes:
                for defer in (False, True):
                    sub = work / f"n{n}_{'defer' if defer else 'classic'}"
                    if sub.exists():
                        shutil.rmtree(sub, ignore_errors=True)
                    sub.mkdir(parents=True, exist_ok=True)
                    t0 = time.perf_counter()
                    report = asyncio.run(
                        _run_soak(
                            chunks=n,
                            page_size=args.page_size,
                            work_dir=sub,
                            defer_write=defer,
                        )
                    )
                    report["wall_s"] = round(time.perf_counter() - t0, 4)
                    rows.append(report)
                    if not args.json:
                        print(
                            f"\n--- n={n} defer={defer} wall={report['wall_s']}s "
                            f"peak_rss={report.get('peak_rss_mb')} ---"
                        )
                        _print_report(report)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                print("\n=== scale summary (fake embed, LanceDB) ===")
                print(
                    f"{'n':>8} {'path':>8} {'total_s':>8} {'mi_s':>8} "
                    f"{'mi_calls':>8} {'mi_rows':>8} {'peak_mb':>8}"
                )
                for r in rows:
                    db = r.get("db", {})
                    print(
                        f"{r['meta']['chunks']:8d} "
                        f"{'defer' if r['meta']['defer_write'] else 'classic':>8} "
                        f"{r.get('total_s', 0):8.3f} "
                        f"{db.get('merge_insert_s', 0):8.3f} "
                        f"{db.get('merge_insert_calls', 0):8d} "
                        f"{db.get('merge_insert_rows', 0):8d} "
                        f"{r.get('peak_rss_mb') or 0:8.1f}"
                    )
            ok = all(r.get("meta", {}).get("remaining_missing", 1) == 0 for r in rows)
            return 0 if ok else 1

        t0 = time.perf_counter()
        if args.mode == "soak":
            report = asyncio.run(
                _run_soak(
                    chunks=args.chunks,
                    page_size=args.page_size,
                    work_dir=work,
                    defer_write=args.defer_write,
                )
            )
        else:
            root = args.root or _REPO_ROOT
            report = asyncio.run(
                _run_index(
                    root=root,
                    work_dir=work,
                    defer_write=args.defer_write,
                    page_size=args.page_size,
                )
            )
        report["wall_s"] = round(time.perf_counter() - t0, 4)

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)

        remaining = report.get("meta", {}).get("remaining_missing", 0)
        return 0 if remaining == 0 else 1
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
