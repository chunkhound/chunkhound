"""Rust and Python must agree on index-format and installer-owned file contracts."""

import re
from pathlib import Path

import duckdb

from chunkhound.core import index_format
from chunkhound.hub.client import _INSTALL_SUFFIXES
from chunkhound.providers.database import index_format as provider_format

ROOT = Path(__file__).resolve().parents[2]


def _rust(path: str) -> str:
    return (ROOT / "src" / path).read_text(encoding="utf-8")


def test_index_format_constants_match_native():
    source = _rust("db/duckdb_backend/schema.rs")
    for name in ("INDEX_FORMAT_VERSION", "LEGACY_INDEX_FORMAT_VERSION"):
        match = re.search(rf"const {name}: i64 = (\d+);", source)
        assert match is not None, name
        assert int(match.group(1)) == getattr(index_format, name), name


def test_installer_suffixes_match_native():
    source = _rust("database_coordination/install.rs")
    for backend, const in (
        ("duckdb", "DUCKDB_SUFFIXES"),
        ("lancedb", "LANCEDB_SUFFIXES"),
    ):
        body = re.search(rf"const {const}: &\[&str\] = &\[(.*?)\];", source, re.S)
        assert body is not None, const
        assert (
            tuple(re.findall(r'"([^"]*)"', body.group(1))) == _INSTALL_SUFFIXES[backend]
        )


def test_python_stamp_writes_the_current_format_version(monkeypatch):
    monkeypatch.setattr(provider_format, "INDEX_FORMAT_VERSION", 7)
    conn = duckdb.connect(":memory:")
    provider_format.stamp_duckdb(conn, catalog="memory")
    assert conn.execute("SELECT version FROM schema_version").fetchall() == [(7,)]
