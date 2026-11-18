"""Contract test: an indexing run reclaims dead-row waste, whichever writer ran.

Every assertion reads the DuckDB file and nothing else — no provider API, no
config internals, no Python module paths. The same scenario therefore holds for
the Python provider and the Rust pipeline, and survives moving the write path
between them.

Decision coverage (when *should* compaction fire) lives in the pure threshold
rule tests; this file only pins the outcome users can see: bloat disappears,
live chunks survive, index state is untouched.
"""

import shutil
import tempfile
from collections import Counter
from pathlib import Path

import pytest

from tests.contracts.compaction_seed import dead_row_count, seed_compaction_waste
from tests.contracts.pipeline_harness import (
    IndexResult,
    collect_chunk_tuples_from_duckdb,
    hnsw_index_names,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"
_PROBE_FILE = "zz_reclaim_probe.py"


async def _run(
    backend: str,
    source_dir: Path,
    db_dir: Path,
    *,
    compaction_threshold: float,
) -> IndexResult:
    """Index with the backend under test; both paths stay behind one call."""
    if backend == "rust":
        return index_with_rust(
            source_dir,
            db_dir,
            skip_embeddings=True,
            # Zero gates: the scenario, not the Rust-only defaults, decides.
            compaction_threshold=compaction_threshold,
            compaction_min_size_mb=0,
            duckdb_hnsw_enabled=False,
        )
    return await index_with_python(source_dir, db_dir, skip_embeddings=True)


@pytest.mark.parametrize("backend", ["python", "rust"])
async def test_index_run_reclaims_dead_row_waste(backend: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = Path(tmp) / "src"
        db_dir = Path(tmp) / "db"
        shutil.copytree(FIXTURE_DIR, source_dir)

        # Arrange: a real store, built so compaction has no reason to fire.
        arranged = await _run(backend, source_dir, db_dir, compaction_threshold=1.0)
        assert not arranged.errors, arranged.errors
        seed_compaction_waste(db_dir)

        # Precondition: without real waste the scenario proves nothing, and a
        # stray opportunistic compaction would make the assertion vacuous.
        assert dead_row_count(db_dir) > 0, "no dead-row waste to reclaim"

        chunks_before = collect_chunk_tuples_from_duckdb(db_dir)
        hnsw_before = hnsw_index_names(db_dir)

        # Unchanged files alone give the Python path nothing to do (it only
        # compacts after real work), so add one file to be indexed.
        (source_dir / _PROBE_FILE).write_text(
            "def reclaim_probe():\n    return 1\n", encoding="utf-8"
        )

        # Act: the run must notice the waste and rewrite the store.
        result = await _run(backend, source_dir, db_dir, compaction_threshold=0.0)

        assert not result.errors, result.errors
        assert dead_row_count(db_dir) == 0, "dead-row waste survived the run"
        assert hnsw_index_names(db_dir) == hnsw_before

        chunks_after = collect_chunk_tuples_from_duckdb(db_dir)
        lost = Counter(chunks_before) - Counter(chunks_after)
        assert not lost, f"chunks lost during compaction: {sorted(lost)[:3]}"
        assert any(path.endswith(_PROBE_FILE) for path, *_ in chunks_after), (
            "measured run indexed nothing, so the reclaim assertion is vacuous"
        )
