#!/usr/bin/env python3
"""Profile indexing flow with FakeEmbeddingProvider (no API keys).

Two harnesses share one entrypoint:

  soak   — synthetic DB seed (no discovery/parse); isolates write+embed path.
  full   — product IndexingCoordinator path (discover → change-detect →
           parse → store → residual embed). Fake vectors only — embeddings are
           meaningless for search quality but realistic for wall / RSS / Lance
           write cost when --dims matches production (default 1024).

Modes:
  soak   DB-only synthetic seed (legacy scale / optimize ladders)
  full   Full product index path (preferred for end-to-end tuning)
  index  Alias of full (kept for older docs / scripts)

Examples:

  # DB soak (write path only)
  uv run python scripts/profile_index.py --mode soak --chunks 5000
  uv run python scripts/profile_index.py --mode soak --chunks 10000 --defer-write
  uv run python scripts/profile_index.py --scale --page-size 256
  uv run python scripts/profile_index.py --optimize-ladder --chunks 50000 --page-size 512

  # Full-flow cold start on synthetic corpus (recommended gate)
  uv run python scripts/profile_index.py --mode full --corpus synthetic \\
      --files 200 --funcs-per-file 20 --defer-write

  # Full-flow on a real tree (still fake embeds)
  uv run python scripts/profile_index.py --mode full --root . --defer-write

  # Cold then resume (unchanged files should skip) into --keep-dir
  uv run python scripts/profile_index.py --mode full --corpus synthetic \\
      --files 100 --scenario cold --keep-dir .bench-full
  uv run python scripts/profile_index.py --mode full --corpus synthetic \\
      --files 100 --scenario resume --keep-dir .bench-full

  # Full-flow scale ladder (synthetic cold, fake embed)
  uv run python scripts/profile_index.py --full-scale --defer-write

  # F5: full-flow optimize-threshold ladder (gate size example)
  uv run python scripts/profile_index.py --full-optimize-ladder \\
      --files 1000 --funcs-per-file 20 --defer-flush-chunks 1000

Isolation:
  This harness never loads global/user config (~/.config/chunkhound, etc.),
  project .chunkhound.json, or CHUNKHOUND_* env config layers. Config is built
  only from CLI flags + in-process defaults (controlled mode).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Voyage-code-3-ish width: vector payload size matters for Lance add / merge.
_DEFAULT_FULL_DIMS = 1024
_DEFAULT_SOAK_DIMS = 32
# Soft safety for --corpus root: refuse accidental huge trees (e.g. full AOSP).
_DEFAULT_MAX_TREE_FILES = 150_000

_DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/venv/**",
    "**/.chunkhound/**",
    "**/__pycache__/**",
    "**/target/**",
    "**/dist/**",
    "**/.pytest_cache/**",
    "**/.bench-full/**",
    "**/chunkhound-profile-*/**",
    "**/chunkhound-full-*/**",
]


def _isolate_from_global_config() -> None:
    """Prevent Config hierarchical loader from seeing global/env config.

    Even when callers build nested DatabaseConfig/IndexingConfig, ``Config()``
    still deep-merges env + ~/.config/chunkhound + local .chunkhound.json.
    Clear the env knobs that enable that before any Config construction.
    """
    for key in list(os.environ):
        if key == "CHUNKHOUND_GLOBAL_CONFIG_FILE" or key == "CHUNKHOUND_CONFIG_FILE":
            os.environ.pop(key, None)
        elif key.startswith("CHUNKHOUND_"):
            # Leave nothing from the user's shell global/env profile.
            os.environ.pop(key, None)


def _build_isolated_config(
    *,
    db_dir: Path,
    optimize_threshold: int,
    defer_write: bool,
    defer_flush_chunks: int,
    page_size: int,
    cleanup: bool,
    force_reindex: bool,
    per_file_timeout_seconds: float,
    target_dir: Path,
) -> object:
    """Build Config without hierarchical global/local/env merge.

    Uses ``model_construct`` so ``Config.__init__`` (which loads globals) is
    never invoked.
    """
    from chunkhound.core.config.config import Config
    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.indexing_config import IndexingConfig

    database = DatabaseConfig(
        path=db_dir,
        provider="lancedb",
        lancedb_optimize_fragment_threshold=optimize_threshold,
    )
    indexing = IndexingConfig(
        defer_chunk_write=defer_write,
        defer_flush_chunks=max(1, int(defer_flush_chunks)),
        db_batch_size=max(100, page_size),
        cleanup=cleanup,
        force_reindex=force_reindex,
        per_file_timeout_seconds=per_file_timeout_seconds,
        per_file_timeout_min_size_kb=128,
    )
    embedding = EmbeddingConfig(
        provider="openai",  # unused — FakeEmbeddingProvider is injected
        model="fake-embeddings",
        batch_size=page_size,
    )
    return Config.model_construct(
        database=database,
        indexing=indexing,
        embedding=embedding,
        target_dir=target_dir.resolve(),
        config_file=None,
        local_config_file=None,
        global_config_file=None,
    )


def _count_tree_files(root: Path, *, max_files: int) -> int:
    """Count files under root (skip .git). Stops early if over max_files."""
    n = 0
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune .git and other heavy non-source dirs for counting.
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {".git", "node_modules", ".venv", "venv", "__pycache__"}
        ]
        n += len(filenames)
        if n > max_files:
            return n
    return n


def _quiet_logs() -> None:
    try:
        from loguru import logger

        logger.remove()
    except Exception:
        pass


def _write_synthetic_corpus(
    root: Path,
    *,
    files: int,
    funcs_per_file: int,
    seed: int = 0,
) -> dict:
    """Write a deterministic Python tree that exercises real parse+chunk.

    Each file has ``funcs_per_file`` top-level functions plus a small class so
    tree-sitter yields multiple chunks per file. Content is unique per path so
    content-hash chunk ids stay distinct.
    """
    if files < 1:
        raise ValueError("files must be >= 1")
    if funcs_per_file < 1:
        raise ValueError("funcs_per_file must be >= 1")

    root = root.resolve()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    # Spread across packages so discovery sees multiple top-level dirs
    # (enables parallel discovery when threshold is met).
    n_pkgs = max(4, min(16, files // 25 or 4))
    total_lines = 0
    for i in range(files):
        pkg = f"pkg_{i % n_pkgs:02d}"
        pkg_dir = root / pkg
        pkg_dir.mkdir(parents=True, exist_ok=True)
        # Package markers keep import-like structure without affecting parse much.
        init = pkg_dir / "__init__.py"
        if not init.exists():
            init.write_text(f'"""Synthetic package {pkg} (seed={seed})."""\n', encoding="utf-8")
            total_lines += 1

        body_lines: list[str] = [
            f'"""Synthetic module m_{i:05d} (seed={seed})."""',
            "from __future__ import annotations",
            "",
            f"MODULE_ID = {i}",
            f"SEED = {seed}",
            "",
        ]
        for j in range(funcs_per_file):
            body_lines.extend(
                [
                    f"def fn_{i:05d}_{j:03d}(x: int = {j}, y: int = {i}) -> int:",
                    f'    """Doc for fn_{i:05d}_{j:03d}."""',
                    f"    # unique payload {seed}-{i}-{j}",
                    f"    acc = x + y + {j} + MODULE_ID",
                    "    for k in range(3):",
                    "        acc += k * SEED",
                    "    return acc",
                    "",
                ]
            )
        body_lines.extend(
            [
                f"class Widget_{i:05d}:",
                f'    """Class body for file {i}."""',
                f"    tag = {i}",
                "",
                "    def run(self, n: int = 1) -> int:",
                f"        return fn_{i:05d}_000(n) + self.tag",
                "",
                f"def main_{i:05d}() -> None:",
                f"    w = Widget_{i:05d}()",
                "    print(w.run())",
                "",
                f'if __name__ == "__main__":',
                f"    main_{i:05d}()",
                "",
            ]
        )
        text = "\n".join(body_lines)
        path = pkg_dir / f"m_{i:05d}.py"
        path.write_text(text, encoding="utf-8")
        total_lines += text.count("\n") + 1

    return {
        "files": files,
        "funcs_per_file": funcs_per_file,
        "packages": n_pkgs,
        "approx_lines": total_lines,
        "seed": seed,
        "root": str(root),
    }


async def _run_soak(
    *,
    chunks: int,
    page_size: int,
    work_dir: Path,
    defer_write: bool,
    optimize_threshold: int = 100,
    dims: int = _DEFAULT_SOAK_DIMS,
) -> dict:
    """Synthetic insert + stream embed (or single-write path simulation)."""
    _isolate_from_global_config()
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
            "dims": dims,
            "optimize_threshold": optimize_threshold,
        }
    )

    cfg = DatabaseConfig(
        path=work_dir,
        provider="lancedb",
        # Default matches DatabaseConfig product default (was hard-coded 50).
        lancedb_optimize_fragment_threshold=optimize_threshold,
    )
    db = LanceDBProvider(
        str(cfg.get_db_path()), base_directory=work_dir, config=cfg
    )
    db.set_index_profile(profile)
    db.connect()
    try:
        fake = FakeEmbeddingProvider(dims=dims, batch_size=page_size)
        files_n = max(1, chunks // 100)
        per = chunks // files_n
        rem = chunks % files_n

        # Per-file deferred chunk flush (L1). File rows use batch insert (P3).
        profile.meta["flush_policy"] = "per_file" if defer_write else "classic"
        profile.meta["file_insert"] = "batch"

        # Sub-phases under seed_insert (Instr): attribute wall, not just merge_s.
        # Nested phase() accumulates; seed_insert ≈ sum of sub-phases.
        with profile.phase("seed_insert"):
            all_chunks: list[Chunk] = []
            all_ids: list[int] = []
            file_models: list[File] = []
            per_file_counts: list[int] = []
            for f in range(files_n):
                n = per + (1 if f < rem else 0)
                per_file_counts.append(n)
                file_models.append(
                    File(
                        path=f"soak/f_{f:05d}.py",
                        mtime=1_700_000_000.0 + f,
                        language=Language.PYTHON,
                        size_bytes=64 * n,
                    )
                )

            with profile.phase("seed_file_insert"):
                file_ids = [int(x) for x in db.insert_files_batch(file_models)]

            for f, file_id in enumerate(file_ids):
                n = per_file_counts[f]
                with profile.phase("seed_chunk_build"):
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
                    with profile.phase("seed_embed"):
                        texts = [c.code or "" for c in batch]
                        vecs = await fake.embed_batch(texts)
                        vec_lists = [list(v) for v in vecs]
                    with profile.phase("seed_chunk_write"):
                        ids = db.insert_chunks_with_embeddings_batch(
                            batch,
                            vec_lists,
                            fake.name,
                            fake.model,
                        )
                        all_ids.extend(int(x) for x in ids)
                else:
                    with profile.phase("seed_chunk_write"):
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


async def _run_full(
    *,
    root: Path,
    db_dir: Path,
    defer_write: bool,
    page_size: int,
    optimize_threshold: int = 100,
    dims: int = _DEFAULT_FULL_DIMS,
    scenario: str = "cold",
    force_reindex: bool = False,
    cleanup: bool = False,
    per_file_timeout_seconds: float = 60.0,
    defer_flush_chunks: int = 1000,
    max_tree_files: int = _DEFAULT_MAX_TREE_FILES,
) -> dict:
    """Full product path: discover → change-detect → parse → store → residual.

    Embeddings use FakeEmbeddingProvider (no network). Vectors are not useful for
    semantic quality, but vector width (``dims``) still stresses Lance writes.

    Config is fully isolated: no global/user/project config layers.
    """
    from chunkhound.core.diagnostics.index_profile import IndexProfile
    from chunkhound.providers.database.lancedb_provider import LanceDBProvider
    from chunkhound.services.indexing_coordinator import IndexingCoordinator
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    _isolate_from_global_config()

    root = root.resolve()
    db_dir = db_dir.resolve()
    db_dir.mkdir(parents=True, exist_ok=True)

    tree_files = _count_tree_files(root, max_files=max_tree_files)
    if tree_files > max_tree_files:
        raise RuntimeError(
            f"Tree too large for controlled profile: {tree_files} files under "
            f"{root} (max_tree_files={max_tree_files}). Refuse to run."
        )

    if scenario == "cold":
        # Fresh DB so change-detect does not skip anything.
        for child in list(db_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    profile = IndexProfile()
    profile.meta.update(
        {
            "mode": "full",
            "scenario": scenario,
            "root": str(root),
            "db_dir": str(db_dir),
            "defer_write": defer_write,
            "page_size": page_size,
            "provider": "lancedb",
            "embed": "fake",
            "dims": dims,
            "optimize_threshold": optimize_threshold,
            "force_reindex": force_reindex,
            "cleanup": cleanup,
            "defer_flush_chunks": defer_flush_chunks,
            "config_isolation": "model_construct+env_cleared",
            "tree_files_precheck": tree_files,
            "max_tree_files": max_tree_files,
        }
    )

    cfg = _build_isolated_config(
        db_dir=db_dir,
        optimize_threshold=optimize_threshold,
        defer_write=defer_write,
        defer_flush_chunks=defer_flush_chunks,
        page_size=page_size,
        cleanup=cleanup,
        force_reindex=force_reindex,
        per_file_timeout_seconds=per_file_timeout_seconds,
        target_dir=root,
    )

    db = LanceDBProvider(
        str(cfg.database.get_db_path()),
        base_directory=root,
        config=cfg.database,
    )
    db.set_index_profile(profile)
    db.connect()
    fake = FakeEmbeddingProvider(dims=dims, batch_size=page_size)
    try:
        coord = IndexingCoordinator(
            database_provider=db,
            base_directory=root,
            embedding_provider=fake,  # type: ignore[arg-type]
            config=cfg,
        )
        coord.attach_index_profile(profile)

        # Coordinator already records discover / change_detect / parse_store /
        # store / store_optimize. Outer phase is total product wall for the
        # directory pass (gate metric with generate_missing).
        with profile.phase("process_directory"):
            dir_result = await coord.process_directory(
                root,
                patterns=["**/*"],
                exclude_patterns=list(_DEFAULT_EXCLUDES),
            )
        profile.meta["dir_result"] = {
            k: dir_result.get(k)
            for k in (
                "status",
                "files_processed",
                "total_chunks",
                "skipped",
                "skipped_unchanged",
                "errors",
            )
            if k in dir_result
        }

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

        files_n = int(dir_result.get("files_processed") or 0)
        chunks_n = int(dir_result.get("total_chunks") or 0)
        profile.meta["files_processed"] = files_n
        profile.meta["total_chunks"] = chunks_n
        if fake._embeddings_generated:
            profile.meta["fake_embeddings_generated"] = fake._embeddings_generated
            profile.meta["fake_embed_requests"] = fake._requests_made

        # Bottleneck helpers for large-tree analysis
        phases = profile.phases_s
        store_s = float(phases.get("store", 0.0))
        parse_store_s = float(phases.get("parse_store", 0.0))
        wall_hint = max(profile.total_s(), 1e-9)
        profile.meta["bottleneck"] = {
            "parse_approx_s": round(max(0.0, parse_store_s - store_s), 4),
            "store_minus_lance_s": round(
                max(0.0, store_s - float(profile.db.merge_insert_s)), 4
            ),
            "lance_write_s": round(float(profile.db.merge_insert_s), 4),
            "optimize_s": round(float(profile.db.optimize_s), 4),
            "chunks_per_embed_request": (
                round(chunks_n / fake._requests_made, 2)
                if fake._requests_made
                else None
            ),
            "phase_share_of_total": {
                k: round(v / wall_hint, 4) for k, v in sorted(phases.items())
            },
        }

        profile.sample_rss()
        return profile.as_dict()
    finally:
        db.disconnect()


def _throughput(report: dict) -> dict:
    """Derive files/s and chunks/s from wall and meta counts."""
    wall = float(report.get("wall_s") or report.get("total_s") or 0.0)
    meta = report.get("meta") or {}
    files_n = int(meta.get("files_processed") or 0)
    chunks_n = int(meta.get("total_chunks") or meta.get("chunks") or meta.get("ids") or 0)
    out: dict = {}
    if wall > 0 and files_n > 0:
        out["files_per_s"] = round(files_n / wall, 2)
    if wall > 0 and chunks_n > 0:
        out["chunks_per_s"] = round(chunks_n / wall, 1)
    return out


def _print_report(report: dict) -> None:
    print("=== index profile ===")
    # Nested phases are listed flat; do not sum all phases_s keys — use
    # total_s/wall_s, or process_directory / seed_insert as parents.
    for k, v in report.get("phases_s", {}).items():
        print(f"  {k:22s} {v:8.3f}s")
    print(f"  {'TOTAL':22s} {report.get('total_s', 0):8.3f}s")
    thr = _throughput(report)
    if thr:
        print(
            f"  throughput files/s={thr.get('files_per_s', 'n/a')} "
            f"chunks/s={thr.get('chunks_per_s', 'n/a')}"
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["soak", "index", "full"],
        default="soak",
        help=(
            "soak=synthetic DB path; full/index=product IndexingCoordinator "
            "(discover+parse+store+residual) with fake embeds"
        ),
    )
    parser.add_argument("--chunks", type=int, default=5000, help="Soak chunk count")
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help=(
            f"Fake embedding dims (soak default {_DEFAULT_SOAK_DIMS}; "
            f"full default {_DEFAULT_FULL_DIMS} ≈ voyage-code-3 width)"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory to index (mode=full/index, corpus=root). Default: repo root.",
    )
    parser.add_argument(
        "--corpus",
        choices=["root", "synthetic"],
        default="root",
        help=(
            "full mode source tree: root=--root (or repo); synthetic=generated "
            "Python packages under the work dir"
        ),
    )
    parser.add_argument(
        "--files",
        type=int,
        default=100,
        help="Synthetic corpus file count (excluding __init__.py packages)",
    )
    parser.add_argument(
        "--funcs-per-file",
        type=int,
        default=15,
        help="Functions per synthetic file (drives chunks/file roughly)",
    )
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=0,
        help="Seed baked into synthetic source (content uniqueness)",
    )
    parser.add_argument(
        "--scenario",
        choices=["cold", "resume"],
        default="cold",
        help=(
            "cold=wipe DB then index; resume=reuse DB under --keep-dir "
            "(unchanged files should skip). Resume requires --keep-dir."
        ),
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="full mode: reprocess all files even if mtime/size match",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="full mode: enable orphan cleanup (default off for clean benches)",
    )
    parser.add_argument(
        "--defer-write",
        action="store_true",
        help="Use embed-then-single-write for new chunks (product default path)",
    )
    parser.add_argument(
        "--defer-flush-chunks",
        type=int,
        default=1000,
        help=(
            "F1: flush deferred (chunk,vector) buffer after this many chunks "
            "across new files (1=per-file L1). Default 1000."
        ),
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help=(
            "Soak ladder 2k/10k/25k/50k (fake embed) classic vs defer. "
            "Ignores --chunks."
        ),
    )
    parser.add_argument(
        "--full-scale",
        action="store_true",
        help=(
            "Full-flow synthetic cold ladder (files=50/200/500/1000) with "
            "fake embeds; wall is the gate. Implies --mode full --corpus synthetic."
        ),
    )
    parser.add_argument(
        "--optimize-threshold",
        type=int,
        default=100,
        help=(
            "LanceDB fragment count before mid-write optimize "
            "(default 100 = product default; thr=50 over-optimizes large trees)"
        ),
    )
    parser.add_argument(
        "--optimize-ladder",
        action="store_true",
        help=(
            "L4 A/B: soak at --chunks for thresholds 50/100/200/500/10000 "
            "(defer only). Wall is the gate metric."
        ),
    )
    parser.add_argument(
        "--full-optimize-ladder",
        action="store_true",
        help=(
            "F5 A/B: full synthetic cold path for thresholds 50/100/200/500/10000 "
            "with F1 defer flush (default --defer-flush-chunks). Wall is the gate. "
            "Uses --files / --funcs-per-file / --dims. "
            "Gate size: --files 1000 --funcs-per-file 20."
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
        help=(
            "Work directory for DB (+ synthetic corpus). Default: temp, deleted "
            "on exit. Required for --scenario resume."
        ),
    )
    parser.add_argument(
        "--max-tree-files",
        type=int,
        default=_DEFAULT_MAX_TREE_FILES,
        help=(
            "Refuse --corpus root if tree exceeds this many files "
            f"(default {_DEFAULT_MAX_TREE_FILES}; 200k is considered too large)."
        ),
    )
    args = parser.parse_args()
    if args.optimize_threshold < 0:
        parser.error("--optimize-threshold must be >= 0")
    if args.scenario == "resume" and args.keep_dir is None:
        parser.error("--scenario resume requires --keep-dir (reuse prior cold DB)")
    if args.files < 1:
        parser.error("--files must be >= 1")
    if args.funcs_per_file < 1:
        parser.error("--funcs-per-file must be >= 1")
    if args.dims is not None and args.dims < 1:
        parser.error("--dims must be >= 1")
    if args.max_tree_files < 1:
        parser.error("--max-tree-files must be >= 1")
    # Controlled mode: drop global/env config before any Config/import side paths.
    _isolate_from_global_config()
    _quiet_logs()

    work = args.keep_dir
    cleanup = False
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="chunkhound-full-"))
        cleanup = True
    else:
        work.mkdir(parents=True, exist_ok=True)

    def _dims_for(mode: str) -> int:
        if args.dims is not None:
            return args.dims
        return _DEFAULT_FULL_DIMS if mode in ("full", "index") else _DEFAULT_SOAK_DIMS

    try:
        if args.full_optimize_ladder:
            # F5: product full-flow thr A/B under F1 store cadence (wall gate).
            thresholds = [50, 100, 200, 500, 10_000]
            rows: list[dict] = []
            for thr in thresholds:
                sub = work / f"full_opt_thr{thr}"
                if sub.exists():
                    shutil.rmtree(sub, ignore_errors=True)
                sub.mkdir(parents=True, exist_ok=True)
                corpus = sub / "corpus"
                db_dir = sub / "db"
                corpus_meta = _write_synthetic_corpus(
                    corpus,
                    files=args.files,
                    funcs_per_file=args.funcs_per_file,
                    seed=args.corpus_seed,
                )
                t0 = time.perf_counter()
                report = asyncio.run(
                    _run_full(
                        root=corpus,
                        db_dir=db_dir,
                        defer_write=True,
                        page_size=args.page_size,
                        optimize_threshold=thr,
                        dims=_dims_for("full"),
                        scenario="cold",
                        force_reindex=False,
                        cleanup=False,
                        defer_flush_chunks=args.defer_flush_chunks,
                        max_tree_files=args.max_tree_files,
                    )
                )
                report["wall_s"] = round(time.perf_counter() - t0, 4)
                report["meta"]["corpus"] = corpus_meta
                report.update(_throughput(report))
                rows.append(report)
                if not args.json:
                    db = report.get("db", {})
                    phases = report.get("phases_s", {})
                    print(
                        f"\n--- full thr={thr} wall={report['wall_s']}s "
                        f"store={phases.get('store', 0):.2f}s "
                        f"opt={db.get('optimize_calls')}/{db.get('optimize_s')}s "
                        f"batches={db.get('chunk_insert_batches')} "
                        f"peak={report.get('peak_rss_mb')} ---"
                    )
                    _print_report(report)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                print(
                    "\n=== F5 full-flow optimize-threshold ladder "
                    f"(synthetic cold, F1 flush={args.defer_flush_chunks}, "
                    "isolated config) ==="
                )
                print(
                    f"{'thr':>6} {'wall':>8} {'store':>8} {'opt_n':>6} "
                    f"{'opt_s':>8} {'mi_s':>8} {'batches':>8} {'peak_mb':>8} "
                    f"{'ch/s':>8}"
                )
                for r in rows:
                    db = r.get("db", {})
                    phases = r.get("phases_s", {})
                    thr_m = _throughput(r)
                    print(
                        f"{r['meta'].get('optimize_threshold', 0):6d} "
                        f"{r.get('wall_s', 0):8.2f} "
                        f"{phases.get('store', 0):8.2f} "
                        f"{db.get('optimize_calls', 0):6d} "
                        f"{db.get('optimize_s', 0):8.2f} "
                        f"{db.get('merge_insert_s', 0):8.2f} "
                        f"{db.get('chunk_insert_batches', 0):8d} "
                        f"{r.get('peak_rss_mb') or 0:8.1f} "
                        f"{thr_m.get('chunks_per_s') or 0:8.1f}"
                    )
            ok = all(
                r.get("meta", {}).get("remaining_missing", 1) == 0
                and (r.get("meta", {}).get("dir_result") or {}).get("status")
                in ("success", "no_files")
                and (r.get("meta", {}).get("embed_result") or {}).get("status")
                in ("success", "complete", "deferred_in_seed")
                for r in rows
            )
            return 0 if ok else 1

        if args.optimize_ladder:
            # L4: fragment-threshold A/B under defer (wall gate).
            thresholds = [50, 100, 200, 500, 10_000]
            rows: list[dict] = []
            for thr in thresholds:
                sub = work / f"opt_thr{thr}"
                if sub.exists():
                    shutil.rmtree(sub, ignore_errors=True)
                sub.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                report = asyncio.run(
                    _run_soak(
                        chunks=args.chunks,
                        page_size=args.page_size,
                        work_dir=sub,
                        defer_write=True,
                        optimize_threshold=thr,
                        dims=_dims_for("soak"),
                    )
                )
                report["wall_s"] = round(time.perf_counter() - t0, 4)
                report.update(_throughput(report))
                rows.append(report)
                if not args.json:
                    db = report.get("db", {})
                    print(
                        f"\n--- thr={thr} wall={report['wall_s']}s "
                        f"opt={db.get('optimize_calls')}/{db.get('optimize_s')}s "
                        f"mi_s={db.get('merge_insert_s')} "
                        f"peak={report.get('peak_rss_mb')} ---"
                    )
                    _print_report(report)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                print("\n=== L4 optimize-threshold ladder (defer, fake embed) ===")
                print(
                    f"{'thr':>6} {'wall':>8} {'opt_n':>6} {'opt_s':>8} "
                    f"{'mi_s':>8} {'write_s':>8} {'peak_mb':>8}"
                )
                for r in rows:
                    db = r.get("db", {})
                    phases = r.get("phases_s", {})
                    print(
                        f"{r['meta'].get('optimize_threshold', 0):6d} "
                        f"{r.get('wall_s', 0):8.2f} "
                        f"{db.get('optimize_calls', 0):6d} "
                        f"{db.get('optimize_s', 0):8.2f} "
                        f"{db.get('merge_insert_s', 0):8.2f} "
                        f"{phases.get('seed_chunk_write', 0):8.2f} "
                        f"{r.get('peak_rss_mb') or 0:8.1f}"
                    )
            ok = all(r.get("meta", {}).get("remaining_missing", 1) == 0 for r in rows)
            return 0 if ok else 1

        if args.scale:
            # DB scale ladder: only FakeEmbeddingProvider, measure classic vs defer.
            sizes = [2_000, 10_000, 25_000, 50_000]
            rows = []
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
                            optimize_threshold=args.optimize_threshold,
                            dims=_dims_for("soak"),
                        )
                    )
                    report["wall_s"] = round(time.perf_counter() - t0, 4)
                    report.update(_throughput(report))
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

        if args.full_scale:
            # Full product path scale ladder (synthetic cold only).
            file_counts = [50, 200, 500, 1000]
            rows = []
            for n_files in file_counts:
                sub = work / f"full_n{n_files}"
                if sub.exists():
                    shutil.rmtree(sub, ignore_errors=True)
                sub.mkdir(parents=True, exist_ok=True)
                corpus = sub / "corpus"
                db_dir = sub / "db"
                corpus_meta = _write_synthetic_corpus(
                    corpus,
                    files=n_files,
                    funcs_per_file=args.funcs_per_file,
                    seed=args.corpus_seed,
                )
                t0 = time.perf_counter()
                report = asyncio.run(
                    _run_full(
                        root=corpus,
                        db_dir=db_dir,
                        defer_write=args.defer_write,
                        page_size=args.page_size,
                        optimize_threshold=args.optimize_threshold,
                        dims=_dims_for("full"),
                        scenario="cold",
                        force_reindex=False,
                        cleanup=False,
                        defer_flush_chunks=args.defer_flush_chunks,
                        max_tree_files=args.max_tree_files,
                    )
                )
                report["wall_s"] = round(time.perf_counter() - t0, 4)
                report["meta"]["corpus"] = corpus_meta
                report.update(_throughput(report))
                rows.append(report)
                if not args.json:
                    print(
                        f"\n--- full files={n_files} defer={args.defer_write} "
                        f"wall={report['wall_s']}s peak={report.get('peak_rss_mb')} ---"
                    )
                    _print_report(report)
            if args.json:
                print(json.dumps(rows, indent=2))
            else:
                print("\n=== full-flow scale (synthetic cold, fake embed) ===")
                print(
                    f"{'files':>8} {'chunks':>8} {'wall':>8} {'parse_store':>12} "
                    f"{'store':>8} {'residual':>8} {'mi_s':>8} {'peak_mb':>8} "
                    f"{'ch/s':>8}"
                )
                for r in rows:
                    db = r.get("db", {})
                    phases = r.get("phases_s", {})
                    meta = r.get("meta", {})
                    thr = _throughput(r)
                    print(
                        f"{meta.get('files_processed', 0):8d} "
                        f"{meta.get('total_chunks', 0):8d} "
                        f"{r.get('wall_s', 0):8.2f} "
                        f"{phases.get('parse_store', 0):12.2f} "
                        f"{phases.get('store', 0):8.2f} "
                        f"{phases.get('generate_missing', 0):8.2f} "
                        f"{db.get('merge_insert_s', 0):8.2f} "
                        f"{r.get('peak_rss_mb') or 0:8.1f} "
                        f"{thr.get('chunks_per_s') or 0:8.1f}"
                    )
            ok = all(
                r.get("meta", {}).get("remaining_missing", 1) == 0
                and (r.get("meta", {}).get("dir_result") or {}).get("status")
                in ("success", "no_files")
                and (r.get("meta", {}).get("embed_result") or {}).get("status")
                in ("success", "complete", "deferred_in_seed")
                for r in rows
            )
            return 0 if ok else 1

        t0 = time.perf_counter()
        if args.mode == "soak":
            report = asyncio.run(
                _run_soak(
                    chunks=args.chunks,
                    page_size=args.page_size,
                    work_dir=work,
                    defer_write=args.defer_write,
                    optimize_threshold=args.optimize_threshold,
                    dims=_dims_for("soak"),
                )
            )
        else:
            # full / index — product path
            if args.corpus == "synthetic":
                corpus = work / "corpus"
                # Resume keeps corpus; cold regenerates for determinism.
                if args.scenario == "cold" or not corpus.exists():
                    corpus_meta = _write_synthetic_corpus(
                        corpus,
                        files=args.files,
                        funcs_per_file=args.funcs_per_file,
                        seed=args.corpus_seed,
                    )
                else:
                    corpus_meta = {
                        "files": args.files,
                        "funcs_per_file": args.funcs_per_file,
                        "seed": args.corpus_seed,
                        "root": str(corpus),
                        "reused": True,
                    }
                root = corpus
                db_dir = work / "db"
            else:
                root = (args.root or _REPO_ROOT).resolve()
                db_dir = work / "db" if args.keep_dir else work
                corpus_meta = {"kind": "root", "root": str(root)}

            report = asyncio.run(
                _run_full(
                    root=root,
                    db_dir=db_dir,
                    defer_write=args.defer_write,
                    page_size=args.page_size,
                    optimize_threshold=args.optimize_threshold,
                    dims=_dims_for("full"),
                    scenario=args.scenario,
                    force_reindex=args.force_reindex,
                    cleanup=args.cleanup,
                    defer_flush_chunks=args.defer_flush_chunks,
                    max_tree_files=args.max_tree_files,
                )
            )
            report["meta"]["corpus"] = corpus_meta
        report["wall_s"] = round(time.perf_counter() - t0, 4)
        report.update(_throughput(report))

        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_report(report)

        remaining = report.get("meta", {}).get("remaining_missing", 0)
        if remaining != 0:
            return 1
        if args.mode in ("full", "index"):
            dir_status = (report.get("meta", {}).get("dir_result") or {}).get(
                "status"
            )
            emb_status = (report.get("meta", {}).get("embed_result") or {}).get(
                "status"
            )
            if dir_status not in (None, "success", "no_files"):
                return 1
            if emb_status not in (None, "success", "complete", "deferred_in_seed"):
                return 1
        return 0
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
