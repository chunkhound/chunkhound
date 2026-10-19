"""Missing VSS preserves native writes and compaction without false build phases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    collect_table_counts,
    hnsw_index_names,
    index_with_rust,
)

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _ROOT / "tests" / "fixtures" / "pipeline"


class _DenyProxy(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _DenyRequest)
        self.requests: list[str] = []


class _DenyRequest(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        cast(_DenyProxy, self.server).requests.append(self.path)
        self.send_error(403, "Native extension downloads are blocked by this test")

    def do_CONNECT(self) -> None:
        self.do_GET()

    def log_message(self, format: str, *args: object) -> None:
        pass


def _offline_environment(home: Path, port: int) -> dict[str, str]:
    proxy = f"http://127.0.0.1:{port}"
    environment = os.environ | {"HOME": str(home), "USERPROFILE": str(home)}
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        environment[name] = proxy
    for name in ("NO_PROXY", "no_proxy"):
        environment[name] = ""
    return environment


def _native_case(home: Path, compact: bool) -> tuple[IndexResult, list[str]]:
    phases: list[str] = []
    db_dir = home / ("compact" if compact else "normal")
    result = index_with_rust(
        _FIXTURES,
        db_dir,
        skip_embeddings=False,
        duckdb_hnsw_enabled=True,
        compaction_threshold=0.0 if compact else 1.0,
        compaction_min_size_mb=0 if compact else 10**9,
        progress_callback=lambda phase, *counts: phases.append(phase),
    )
    _assert_persisted(result, db_dir)
    return result, phases


def _assert_persisted(result: IndexResult, db_dir: Path) -> None:
    assert not result.errors, result.errors
    assert result.files_processed > 0 and result.chunks_written > 0
    assert result.embeddings_generated == len(result.embedding_tuples) > 0
    counts = collect_table_counts(db_dir)
    assert counts["files"] == result.files_processed
    assert counts["chunks"] == len(result.chunk_tuples) == result.chunks_written
    assert counts["embeddings"] == result.embeddings_generated
    assert hnsw_index_names(db_dir) == set()


def _native_cases(home: Path) -> dict[str, list[str]]:
    normal, normal_phases = _native_case(home, compact=False)
    compact, compact_phases = _native_case(home, compact=True)
    assert_identical(normal, compact)
    # HOME isolates DuckDB's extension tree; no VSS binary may leak into it.
    assert not list((home / ".duckdb").rglob("vss.duckdb_extension*"))
    return {"normal": normal_phases, "compact": compact_phases}


def _subprocess_cases(home: Path, port: int) -> dict[str, list[str]]:
    code = (
        "import json, os; from pathlib import Path; "
        "from tests.contracts.test_hnsw_unavailable_progress import _native_cases; "
        "print(json.dumps(_native_cases(Path(os.environ['HOME']))))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_ROOT,
        env=_offline_environment(home, port),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "VSS extension unavailable" in process.stderr
    return json.loads(process.stdout)


@pytest.fixture(scope="module")
def unavailable_native_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, list[str]]:
    """Run real native writes with private HOME and a non-forwarding HTTP proxy."""
    home = tmp_path_factory.mktemp("offline-home")
    with _DenyProxy() as proxy:
        thread = Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            cases = _subprocess_cases(home, proxy.server_port)
        finally:
            proxy.shutdown()
            thread.join()
    # Assert that native INSTALL really hit the deny proxy, not ambient network.
    assert len(proxy.requests) >= 2
    assert all("vss.duckdb_extension" in url for url in proxy.requests)
    return cases


@pytest.mark.parametrize("mode", ["normal", "compact"])
def test_unavailable_vss_preserves_native_data_and_progress(
    unavailable_native_runs: dict[str, list[str]],
    mode: str,
) -> None:
    phases = unavailable_native_runs[mode]
    expected = f"write-{'compact' if mode == 'compact' else 'index'}-unavailable"
    other = f"write-{'index' if mode == 'compact' else 'compact'}-unavailable"
    assert expected in phases and other not in phases
    assert phases.index("write-data") < phases.index(expected)
    assert phases.index(expected) < phases.index("write-done")
    assert "write-index" not in phases and "write-compact" not in phases
    assert not any(phase.endswith("-disabled") for phase in phases)
