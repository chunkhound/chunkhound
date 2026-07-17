#!/usr/bin/env python3
"""Optional local large-index soak for LanceDB / DuckDB (not CI).

Usage (from repo root):

  uv run python scripts/soak_large_index.py --provider lancedb --chunks 5000
  uv run python scripts/soak_large_index.py --provider duckdb --chunks 10000 --page-size 100

Measures wall time for insert + stream-embed and verifies residual empty.
Does not publish or require API keys (FakeEmbeddingProvider).
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Ensure repo root is importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


async def _run(provider_name: str, chunk_count: int, page_size: int, work_dir: Path) -> int:
    # Keep stdout metrics clean (loguru writes to stderr by default).
    try:
        from loguru import logger as _logger

        _logger.remove()
    except Exception:
        pass

    from chunkhound.core.config.database_config import DatabaseConfig
    from chunkhound.core.models import Chunk, File
    from chunkhound.core.types.common import ChunkType, Language
    from chunkhound.services.embedding_service import EmbeddingService
    from tests.fixtures.fake_providers import FakeEmbeddingProvider

    if provider_name == "lancedb":
        from chunkhound.providers.database.lancedb_provider import LanceDBProvider

        cfg = DatabaseConfig(
            path=work_dir,
            provider="lancedb",
            # Match product DatabaseConfig default (L4: thr=50 no better, thr≥200 hurts wall)
            lancedb_optimize_fragment_threshold=100,
        )
        provider = LanceDBProvider(str(cfg.get_db_path()), base_directory=work_dir, config=cfg)
    else:
        from chunkhound.providers.database.duckdb_provider import DuckDBProvider

        provider = DuckDBProvider(work_dir / "soak.db", base_directory=work_dir)

    provider.connect()
    try:
        t0 = time.perf_counter()
        # Seed
        files = max(1, chunk_count // 100)
        per = chunk_count // files
        rem = chunk_count % files
        all_ids: list[int] = []
        for f in range(files):
            n = per + (1 if f < rem else 0)
            file_id = int(
                provider.insert_file(
                    File(
                        path=f"soak/f_{f:05d}.py",
                        mtime=1_700_000_000.0 + f,
                        language=Language.PYTHON,
                        size_bytes=64 * n,
                    )
                )
            )
            chunks = [
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
            all_ids.extend(int(x) for x in provider.insert_chunks_batch(chunks))
        t_insert = time.perf_counter() - t0
        if len(all_ids) != chunk_count:
            print(
                f"seed mismatch: expected {chunk_count} ids, got {len(all_ids)}",
                file=sys.stderr,
            )
            return 1

        # Guard the same hot-path invariant as CI soak tests.
        from unittest.mock import MagicMock

        original_all = provider.get_all_chunks_with_metadata
        all_calls = MagicMock(side_effect=original_all)
        provider.get_all_chunks_with_metadata = all_calls  # type: ignore[method-assign]

        page_calls = MagicMock(side_effect=provider.get_chunks_without_embeddings_paginated)
        provider.get_chunks_without_embeddings_paginated = page_calls  # type: ignore[method-assign]

        service = EmbeddingService(
            database_provider=provider,
            embedding_provider=FakeEmbeddingProvider(  # type: ignore[arg-type]
                dims=32, batch_size=page_size
            ),
            embedding_batch_size=page_size,
            db_batch_size=page_size,
            max_concurrent_batches=4,
        )
        t1 = time.perf_counter()
        result = await service.generate_missing_embeddings()
        t_embed = time.perf_counter() - t1

        remaining = provider.get_chunks_without_embeddings_paginated(
            "fake", "fake-embeddings", limit=10
        )
        full_table_loads = all_calls.call_count
        pages = page_calls.call_count
        ok = (
            result.get("status") == "success"
            and int(result.get("generated", 0)) == chunk_count
            and remaining == []
            and full_table_loads == 0
            and pages >= 2
        )

        print(f"provider={provider_name}")
        print(f"chunks={chunk_count} page_size={page_size}")
        print(f"insert_s={t_insert:.3f}")
        print(f"embed_s={t_embed:.3f}")
        print(f"embed_rate={chunk_count / t_embed:.1f} chunks/s" if t_embed > 0 else "embed_rate=n/a")
        print(f"result={result}")
        print(f"remaining={len(remaining)}")
        print(f"full_table_metadata_calls={full_table_loads}")
        print(f"paginated_missing_calls={pages}")
        print(f"ok={ok}")
        return 0 if ok else 1
    finally:
        provider.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["lancedb", "duckdb"],
        default="lancedb",
    )
    parser.add_argument("--chunks", type=int, default=5000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--keep-dir",
        type=Path,
        default=None,
        help=(
            "Keep work directory (default: temp, deleted on exit). "
            "Use a fresh directory each run — reusing a prior DB can "
            "inflate generated counts beyond --chunks."
        ),
    )
    args = parser.parse_args()
    if args.chunks < 1 or args.page_size < 1:
        print("chunks and page-size must be >= 1", file=sys.stderr)
        return 2

    work = args.keep_dir
    cleanup = False
    if work is None:
        work = Path(tempfile.mkdtemp(prefix="chunkhound-soak-"))
        cleanup = True
    else:
        work.mkdir(parents=True, exist_ok=True)

    try:
        return asyncio.run(
            _run(args.provider, args.chunks, args.page_size, work)
        )
    finally:
        if cleanup:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
