"""Unit tests for DuckDB indexed-root identity sidecar guard (Step 101).

Covers provider-level sidecar write/read/validation behavior in isolation:
- sidecar write/read round-trip
- :memory: no-op
- missing sidecar, allow_claim_if_missing=False -> return without writing
- missing sidecar, allow_claim_if_missing=True  -> claim and write
- matching sidecar -> return
- mismatching sidecar -> DuckDBIndexedRootMismatchError
- malformed sidecar cases -> plain RuntimeError
- connect-time mismatch propagates DuckDBIndexedRootMismatchError
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import chunkhound.providers.database.duckdb_provider as duckdb_provider_module
from chunkhound.providers.database.duckdb_provider import (
    DuckDBIndexedRootMismatchError,
    DuckDBProvider,
    _indexed_root_sidecar_path,
    _normalize_indexed_root,
)


def _fresh_provider(db_path: Path | str, base: Path) -> DuckDBProvider:
    provider = DuckDBProvider(db_path, base_directory=base)
    provider.connect()
    return provider


def test_memory_db_is_noop(tmp_path: Path) -> None:
    provider = _fresh_provider(":memory:", tmp_path)
    try:
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=False
        )
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=True
        )
        # No sidecar should exist for :memory:
        assert _indexed_root_sidecar_path(":memory:") is None
    finally:
        provider.disconnect()


def test_missing_sidecar_validate_only_is_noop(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    provider = _fresh_provider(db_path, tmp_path)
    try:
        sidecar = _indexed_root_sidecar_path(db_path)
        assert sidecar is not None
        if sidecar.exists():
            sidecar.unlink()
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=False
        )
        assert not sidecar.exists()
    finally:
        provider.disconnect()


def test_missing_sidecar_claim_writes_current_root(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    provider = _fresh_provider(db_path, tmp_path)
    try:
        sidecar = _indexed_root_sidecar_path(db_path)
        assert sidecar is not None
        if sidecar.exists():
            sidecar.unlink()
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=True
        )
        assert sidecar.exists()
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        assert data == {
            "version": 1,
            "indexed_root_path": _normalize_indexed_root(tmp_path),
        }
    finally:
        provider.disconnect()


def test_matching_sidecar_returns(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    provider = _fresh_provider(db_path, tmp_path)
    try:
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=True
        )
        provider.ensure_indexed_root_identity(
            requested_root=tmp_path, allow_claim_if_missing=False
        )
    finally:
        provider.disconnect()


def test_mismatching_sidecar_raises_typed_error(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()

    provider = _fresh_provider(db_path, root_a)
    try:
        provider.ensure_indexed_root_identity(
            requested_root=root_a, allow_claim_if_missing=True
        )
        with pytest.raises(DuckDBIndexedRootMismatchError) as excinfo:
            provider.ensure_indexed_root_identity(
                requested_root=root_b, allow_claim_if_missing=True
            )
        msg = str(excinfo.value)
        assert str(db_path) in msg
        assert _normalize_indexed_root(root_a) in msg
        assert _normalize_indexed_root(root_b) in msg
    finally:
        provider.disconnect()


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "{not-json",
        json.dumps({"version": 1}),  # missing indexed_root_path
        json.dumps({"version": 2, "indexed_root_path": "/tmp"}),
        json.dumps({"indexed_root_path": "/tmp"}),  # missing version
        json.dumps({"version": 1, "indexed_root_path": ""}),
    ],
)
def test_malformed_sidecar_raises_plain_runtime_error(
    tmp_path: Path, payload: str
) -> None:
    db_path = tmp_path / "chunks.db"
    provider = _fresh_provider(db_path, tmp_path)
    try:
        sidecar = _indexed_root_sidecar_path(db_path)
        assert sidecar is not None
        sidecar.write_text(payload, encoding="utf-8")
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_indexed_root_identity(
                requested_root=tmp_path, allow_claim_if_missing=False
            )
        assert not isinstance(excinfo.value, DuckDBIndexedRootMismatchError)
    finally:
        provider.disconnect()


def test_connect_time_mismatch_propagates(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()

    provider = DuckDBProvider(db_path, base_directory=root_a)
    provider.connect()
    try:
        provider.ensure_indexed_root_identity(
            requested_root=root_a, allow_claim_if_missing=True
        )
    finally:
        provider.disconnect()

    provider_b = DuckDBProvider(db_path, base_directory=root_b)
    with pytest.raises(DuckDBIndexedRootMismatchError):
        provider_b.connect()


def test_connect_time_mismatch_does_not_touch_db_file(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()

    sidecar = _indexed_root_sidecar_path(db_path)
    assert sidecar is not None
    sidecar.write_text(
        json.dumps(
            {
                "version": 1,
                "indexed_root_path": _normalize_indexed_root(root_a),
            }
        ),
        encoding="utf-8",
    )

    provider = DuckDBProvider(db_path, base_directory=root_b)
    with pytest.raises(DuckDBIndexedRootMismatchError):
        provider.connect()

    assert not db_path.exists()


def test_missing_sidecar_claim_is_exclusive_across_roots(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.db"
    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    root_a.mkdir()
    root_b.mkdir()

    provider_a = DuckDBProvider(db_path, base_directory=root_a)
    provider_b = DuckDBProvider(db_path, base_directory=root_b)
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def _claim(name: str, provider: DuckDBProvider, root: Path) -> None:
        try:
            barrier.wait(timeout=5.0)
            provider.ensure_indexed_root_identity(
                requested_root=root,
                allow_claim_if_missing=True,
            )
            results[name] = "claimed"
        except DuckDBIndexedRootMismatchError:
            results[name] = "mismatch"
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=_claim, args=("a", provider_a, root_a))
    second = threading.Thread(target=_claim, args=("b", provider_b, root_b))
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert sorted(results.values()) == ["claimed", "mismatch"]

    sidecar = _indexed_root_sidecar_path(db_path)
    assert sidecar is not None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["indexed_root_path"] in {
        _normalize_indexed_root(root_a),
        _normalize_indexed_root(root_b),
    }


def test_missing_sidecar_claim_does_not_publish_partial_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "chunks.db"
    root = tmp_path / "root"
    root.mkdir()
    provider = DuckDBProvider(db_path, base_directory=root)
    sidecar = _indexed_root_sidecar_path(db_path)
    assert sidecar is not None

    original_fsync = duckdb_provider_module.os.fsync
    started = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def blocking_fsync(fd: int) -> None:
        started.set()
        release.wait(timeout=5.0)
        original_fsync(fd)

    def claim() -> None:
        try:
            provider.ensure_indexed_root_identity(
                requested_root=root,
                allow_claim_if_missing=True,
            )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(duckdb_provider_module.os, "fsync", blocking_fsync)

    thread = threading.Thread(target=claim)
    thread.start()
    assert started.wait(timeout=5.0)
    assert not sidecar.exists()
    release.set()
    thread.join()

    assert errors == []
    assert sidecar.exists()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "version": 1,
        "indexed_root_path": _normalize_indexed_root(root),
    }
