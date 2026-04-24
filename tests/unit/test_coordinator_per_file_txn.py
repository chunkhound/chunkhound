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
    assert stats["errors"] and any(
        "boom" in e.get("error", "") for e in stats["errors"]
    )  # noqa: SIM115


@pytest.mark.asyncio
async def test_concurrent_store_serializes_transaction_span_entry(tmp_path: Path):
    """Coordinator B cannot begin its transaction until coordinator A releases the span."""
    db = DuckDBProvider(db_path=tmp_path / "db", base_directory=tmp_path)
    db.connect()
    try:
        coord_a = IndexingCoordinator(database_provider=db, base_directory=tmp_path)
        coord_b = IndexingCoordinator(database_provider=db, base_directory=tmp_path)
        begin_calls: list[int] = []
        first_begin_entered = asyncio.Event()
        allow_first_begin_to_continue = asyncio.Event()
        second_begin_entered = asyncio.Event()
        original_begin = db.begin_transaction_async

        async def _tracked_begin() -> None:
            begin_calls.append(len(begin_calls) + 1)
            if len(begin_calls) == 1:
                first_begin_entered.set()
                await allow_first_begin_to_continue.wait()
            else:
                second_begin_entered.set()
            await original_begin()

        db.begin_transaction_async = _tracked_begin  # type: ignore[method-assign]

        result_a = _pfr(
            tmp_path / "file_a.yaml",
            [
                {
                    "symbol": "sym_a",
                    "code": "key_a: 1",
                    "start_line": 1,
                    "end_line": 1,
                    "chunk_type": "key_value",
                    "language": Language.YAML.value,
                }
            ],
        )
        result_b = _pfr(
            tmp_path / "file_b.yaml",
            [
                {
                    "symbol": "sym_b",
                    "code": "key_b: 2",
                    "start_line": 1,
                    "end_line": 1,
                    "chunk_type": "key_value",
                    "language": Language.YAML.value,
                }
            ],
        )

        task_a = asyncio.create_task(coord_a._store_parsed_results([result_a]))
        await first_begin_entered.wait()

        task_b = asyncio.create_task(coord_b._store_parsed_results([result_b]))
        await asyncio.sleep(0)

        assert not second_begin_entered.is_set(), (
            "Second coordinator reached begin_transaction_async before the first "
            "transaction span was released"
        )

        allow_first_begin_to_continue.set()
        stats_a, stats_b = await asyncio.gather(task_a, task_b)

        assert second_begin_entered.is_set()
        assert begin_calls == [1, 2]
        assert stats_a["total_files"] == 1 and not stats_a["errors"]
        assert stats_b["total_files"] == 1 and not stats_b["errors"]
        assert db.get_file_by_path("file_a.yaml") is not None
        assert db.get_file_by_path("file_b.yaml") is not None
    finally:
        db.disconnect()
