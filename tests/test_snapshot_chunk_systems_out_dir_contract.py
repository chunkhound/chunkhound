#!/usr/bin/env python3
"""Contract tests for `chunkhound snapshot --chunk-systems` artifacts (v1)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from chunkhound.core.types.common import ChunkType
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from tests.utils.windows_subprocess import get_safe_subprocess_env


def _write_minimal_project_config_no_llm(root: Path) -> Path:
    cfg_path = root / ".chunkhound.json"
    db_dir = root / ".chunkhound" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "database": {"path": str(db_dir), "provider": "duckdb"},
        "indexing": {"include": ["*.py"]},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg_path


def _index_no_embeddings(root: Path) -> None:
    res = subprocess.run(
        ["uv", "run", "chunkhound", "index", ".", "--no-embeddings"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=60,
        env=get_safe_subprocess_env(),
    )
    assert res.returncode == 0, res.stderr


def _insert_code_embeddings(root: Path, *, provider: str, model: str, dims: int) -> int:
    db_file = root / ".chunkhound" / "db" / "chunks.db"
    assert db_file.exists()

    db = DuckDBProvider(db_file, base_directory=root)
    db.connect()
    try:
        chunks = db.get_all_chunks_with_metadata()
        code_chunks = [
            c
            for c in chunks
            if ChunkType.from_string(str(c.get("chunk_type", ""))).is_code
        ]
        embeddings_data: list[dict] = []
        for c in code_chunks:
            chunk_id = int(c["chunk_id"])
            vec = [float(((chunk_id * 31) + i) % 97) / 97.0 for i in range(dims)]
            embeddings_data.append(
                {
                    "chunk_id": chunk_id,
                    "provider": provider,
                    "model": model,
                    "dims": dims,
                    "embedding": vec,
                }
            )
        return db.insert_embeddings_batch(embeddings_data)
    finally:
        db.disconnect()


def test_snapshot_chunk_systems_out_dir_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        # Ensure snapshot does not read/parse files: delete sources after indexing.
        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        run_payload = json.loads((out_dir / "snapshot.run.json").read_text("utf-8"))
        assert isinstance(run_payload.get("metrics"), dict)

        chunk_payload = json.loads(
            (out_dir / "snapshot.chunk_systems.json").read_text("utf-8")
        )
        assert chunk_payload["schema_version"] == "snapshot.chunk_systems.v1"
        assert chunk_payload["schema_revision"] == "2026-02-17"

        chunk_md = (out_dir / "snapshot.chunk_systems.md").read_text("utf-8")
        assert chunk_md.startswith("# Snapshot Chunk Systems\n")
        # Default view prunes singleton clusters from the operator report.
        assert "## View\n" in chunk_md
        assert "min_cluster_size" in chunk_md
        # Default min_cluster_size=2 emits view artifacts.
        assert (out_dir / "snapshot.chunk_systems.pruned.json").exists()
        assert (out_dir / "snapshot.chunk_systems.dropped.json").exists()


def test_snapshot_chunk_systems_optional_graph_and_adjacency_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out_graph"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--chunk-systems-write-graph",
                "--chunk-systems-write-adjacency",
                "--chunk-systems-adjacency-evidence-k",
                "3",
                "--chunk-systems-adjacency-max-neighbors",
                "10",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        assert (out_dir / "snapshot.chunk_systems.graph.nodes.jsonl").exists()
        assert (out_dir / "snapshot.chunk_systems.graph.edges.jsonl").exists()
        assert (out_dir / "snapshot.chunk_systems.system_adjacency.json").exists()

        # Sanity: adjacency JSON must be parseable.
        adjacency = json.loads(
            (out_dir / "snapshot.chunk_systems.system_adjacency.json").read_text("utf-8")
        )
        assert adjacency.get("schema_version") == "snapshot.chunk_systems.system_adjacency.v1"


def test_snapshot_chunk_systems_optional_viz_html_implies_adjacency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out_viz"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--chunk-systems-viz",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        # Viz implies adjacency JSON, and the HTML should be present.
        assert (out_dir / "snapshot.chunk_systems.system_adjacency.json").exists()
        assert (out_dir / "snapshot.chunk_systems.viz.html").exists()

        # Sanity: both artifacts must be readable/parseable.
        adjacency = json.loads(
            (out_dir / "snapshot.chunk_systems.system_adjacency.json").read_text("utf-8")
        )
        assert adjacency.get("schema_version") == "snapshot.chunk_systems.system_adjacency.v1"
        html = (out_dir / "snapshot.chunk_systems.viz.html").read_text("utf-8")
        assert "<!doctype html>" in html.lower()
        assert "chunkhound-data" in html


def test_snapshot_chunk_systems_system_group_labels_dry_run_writes_artifact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(root, provider="testprov", model="testmodel", dims=8)
        assert inserted > 0

        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out_group_labels"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--chunk-systems-viz",
                "--chunk-systems-system-groups-labels",
                "--llm-dry-run",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        group_labels_path = out_dir / "snapshot.chunk_systems.system_group_labels.json"
        assert group_labels_path.exists()
        group_labels = json.loads(group_labels_path.read_text("utf-8"))
        assert (
            group_labels.get("schema_version")
            == "snapshot.chunk_systems.system_group_labels.v1"
        )
        assert isinstance(group_labels.get("partitions"), list)
        assert group_labels.get("params", {}).get("prompt_mode") == "systems_only"

        run_payload = json.loads((out_dir / "snapshot.run.json").read_text("utf-8"))
        assert (
            run_payload.get("config_snapshot", {})
            .get("system_group_labels", {})
            .get("prompt_mode")
            == "systems_only"
        )

        html = (out_dir / "snapshot.chunk_systems.viz.html").read_text("utf-8")
        assert "system_group_labels" in html


def test_snapshot_chunk_systems_min_cluster_size_writes_pruned_and_dropped() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        # Ensure snapshot does not read/parse files: delete sources after indexing.
        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--chunk-systems-tau",
                "1.0",
                "--chunk-systems-min-cluster-size",
                "2",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        pruned = json.loads(
            (out_dir / "snapshot.chunk_systems.pruned.json").read_text("utf-8")
        )
        dropped = json.loads(
            (out_dir / "snapshot.chunk_systems.dropped.json").read_text("utf-8")
        )

        assert pruned["schema_revision"] == "2026-02-17"
        assert dropped["schema_revision"] == "2026-02-17"

        assert pruned["view"]["kind"] == "pruned"
        assert pruned["view"]["min_cluster_size"] == 2
        assert dropped["view"]["kind"] == "dropped"
        assert dropped["view"]["min_cluster_size"] == 2

        assert all(int(c.get("size") or 0) >= 2 for c in pruned.get("clusters") or [])
        assert all(int(c.get("size") or 0) < 2 for c in dropped.get("clusters") or [])

        chunk_md = (out_dir / "snapshot.chunk_systems.md").read_text("utf-8")
        assert "## View\n" in chunk_md
        assert "min_cluster_size" in chunk_md


def test_snapshot_chunk_systems_accepts_explicit_labeler_heuristic() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        (pkg / "a.py").unlink()

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--labeler",
                "heuristic",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr
        assert (out_dir / "snapshot.chunk_systems.json").exists()
        assert (out_dir / "snapshot.chunk_systems.md").exists()


def test_snapshot_chunk_systems_llm_dry_run_emits_labels_and_overlays_markdown() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_minimal_project_config_no_llm(root)

        pkg = root / "pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "a.py").write_text("class A:\n    def foo(self):\n        return 1\n")
        (pkg / "b.py").write_text("def bar(x: int) -> int:\n    return x + 1\n")

        _index_no_embeddings(root)
        inserted = _insert_code_embeddings(
            root, provider="testprov", model="testmodel", dims=8
        )
        assert inserted > 0

        (pkg / "a.py").unlink()
        (pkg / "b.py").unlink()

        out_dir = root / "out"
        res = subprocess.run(
            [
                "uv",
                "run",
                "chunkhound",
                "snapshot",
                ".",
                "--out-dir",
                str(out_dir),
                "--no-themes",
                "--no-systems",
                "--chunk-systems",
                "--chunk-systems-tau",
                "0.0",
                "--labeler",
                "llm",
                "--llm-dry-run",
                "--md-labels",
                "llm",
                "--embedding-provider",
                "testprov",
                "--embedding-model",
                "testmodel",
                "--embedding-dims",
                "8",
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=120,
            env=get_safe_subprocess_env(),
        )
        assert res.returncode == 0, res.stderr

        labels_payload = json.loads(
            (out_dir / "snapshot.labels.json").read_text("utf-8")
        )
        assert labels_payload["schema_version"] == "snapshot.labels.v1"
        assert labels_payload["schema_revision"] == "2026-02-17"
        assert labels_payload["dry_run"] is True
        chunk_systems = labels_payload.get("chunk_systems")
        assert isinstance(chunk_systems, dict)
        assert chunk_systems

        first_cluster_id = next(iter(chunk_systems.keys()))
        lab = (chunk_systems.get(first_cluster_id) or {}).get("label")
        assert lab == f"CHUNK_SYSTEM_{int(first_cluster_id)}"

        chunk_md = (out_dir / "snapshot.chunk_systems.md").read_text("utf-8")
        assert lab in chunk_md
