import asyncio
from pathlib import Path

import pytest

from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.core.types.common import Language
from chunkhound.services.batch_processor import ParsedFileResult


def _pfr(path: Path, chunks: list[dict], ok: bool = True) -> ParsedFileResult:
    return ParsedFileResult(
        file_path=path,
        chunks=chunks if ok else [],
        language=Language.YAML,
        file_size=10,
        file_mtime=0.0,
        status="ok" if ok else "error",
        error=None if ok else "simulated",
        content_hash=None,
    )


def test_per_file_transaction_isolated(tmp_path: Path):
    db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
    db.connect()
    try:
        _run_test(db, tmp_path)
    finally:
        db.disconnect()


def _run_test(db: DuckDBProvider, tmp_path: Path) -> None:
    coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)

    good_file = tmp_path / "good.yaml"
    bad_file = tmp_path / "bad.yaml"
    good_chunks = [
        {
            "symbol": "a",
            "code": "x: 1",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "key_value",
            "language": Language.YAML.value,
        }
    ]
    bad_chunks = [
        {
            "symbol": "b",
            "code": "y: 2",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "key_value",
            "language": Language.YAML.value,
        }
    ]

    # Monkeypatch _store_file_record to fail for bad_file
    original_store = coord._store_file_record

    async def _failing_store(path, *args, **kwargs):
        if Path(path) == bad_file:
            raise RuntimeError("boom")
        return await original_store(path, *args, **kwargs)

    coord._store_file_record = _failing_store  # type: ignore[assignment]

    results = [
        _pfr(good_file, good_chunks, ok=True),
        _pfr(bad_file, bad_chunks, ok=True),
    ]

    import asyncio
    res = asyncio.run(coord._store_parsed_results(results))  # type: ignore[arg-type]
    stats = res[0] if isinstance(res, tuple) else res

    # Good file should be stored; bad file should be in errors
    assert stats["total_files"] == 1
    assert stats["errors"] and any("boom" in e.get("error", "") for e in stats["errors"])  # noqa: SIM115


@pytest.mark.asyncio
async def test_concurrent_store_no_interleaved_transactions(tmp_path: Path):
    """Concurrent _store_parsed_results calls must not interleave transactions."""
    db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
    db.connect()
    try:
        coord = IndexingCoordinator(database_provider=db, base_directory=tmp_path)
        n_files = 5

        file_results = []
        for i in range(n_files):
            path = tmp_path / f"file_{i}.yaml"
            chunks = [
                {
                    "symbol": f"sym_{i}",
                    "code": f"key_{i}: {i}",
                    "start_line": 1,
                    "end_line": 1,
                    "chunk_type": "key_value",
                    "language": Language.YAML.value,
                }
            ]
            file_results.append(_pfr(path, chunks))

        results = await asyncio.gather(
            *[coord._store_parsed_results([pfr]) for pfr in file_results]
        )

        # Every call stored exactly one file with no errors
        for res in results:
            assert res["total_files"] == 1, f"Expected 1 file stored, got {res}"
            assert not res["errors"], f"Unexpected errors: {res['errors']}"

        # Total chunks across all results match
        total_chunks = sum(r["total_chunks"] for r in results)
        assert total_chunks == n_files

        # All files are queryable in the DB
        for i in range(n_files):
            rel_path = f"file_{i}.yaml"
            record = db.get_file_by_path(rel_path)
            assert record is not None, f"{rel_path} not found in DB"
    finally:
        db.disconnect()
