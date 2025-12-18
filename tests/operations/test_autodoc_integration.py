from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.autodoc as autodoc_mod
from chunkhound.core.config.config import Config


class DummyProvider:
    """Minimal provider stub exposing chunks for coverage stats."""

    def __init__(self) -> None:
        self._chunks = [
            {"file_path": "scope/a.py", "start_line": 1, "end_line": 10},
            {"file_path": "scope/b.py", "start_line": 5, "end_line": 15},
        ]

    def get_scope_stats(self, scope_prefix: str | None) -> tuple[int, int]:
        # Include one extra indexed file that is never referenced in the fake
        # deep-research results so autodoc can emit an "unreferenced files" artifact.
        if scope_prefix == "scope/":
            return 3, 2
        return 0, 0

    def get_scope_file_paths(self, scope_prefix: str | None) -> list[str]:
        if scope_prefix == "scope/":
            return ["scope/a.py", "scope/b.py", "scope/c.py"]
        return []

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        return list(self._chunks)


class DummyServices:
    """Services stub that only exposes a provider for coverage stats."""

    def __init__(self) -> None:
        self.provider = DummyProvider()


class DummyLLMManager:
    """Placeholder LLM manager for autodoc tests."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._configured = True

    def is_configured(self) -> bool:
        return self._configured


@pytest.mark.asyncio
async def test_autodoc_end_to_end_writes_index_and_topics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Autodoc should emit a combined doc and per-topic files when out-dir is set.

    This is a lightweight integration test that exercises the autodoc command
    with stubbed services and deep-research behavior. It verifies that:
    - the main document is printed to stdout with a metadata header,
    - coverage stats are computed without crashing, and
    - an index file plus one markdown file per point-of-interest are written.
    """

    project_root = tmp_path / "repo"
    scope_path = project_root / "scope"
    scope_path.mkdir(parents=True, exist_ok=True)
    (scope_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (scope_path / "b.py").write_text("print('b')\n", encoding="utf-8")
    (scope_path / "c.py").write_text("print('c')\n", encoding="utf-8")

    # Use a minimal config; database/embedding/llm implementations are stubbed.
    config = Config(
        target_dir=project_root,
        database={"path": project_root / ".chunkhound" / "db", "provider": "duckdb"},
        embedding={
            "provider": "openai",
            "api_key": "test",
            "model": "text-embedding-3-small",
        },
        llm={"provider": "openai", "api_key": "test"},
    )

    # Stub out database, services, embeddings, LLM, and deep research.
    seen_max_points: list[int] = []

    async def fake_overview(
        llm_manager: Any,
        target_dir: Path,
        scope_path: Path,
        scope_label: str,
        max_points: int = 10,
        comprehensiveness: str = "medium",
        out_dir: Path | None = None,
        assembly_provider: Any | None = None,
        indexing_cfg: Any | None = None,
    ) -> tuple[str, list[str]]:
        # Record the requested max_points so we can assert comprehensiveness mapping.
        seen_max_points.append(max_points)
        overview = (
            "1. **Core Flow**: High-level data flow.\n"
            "2. **Error Handling**: How failures are surfaced.\n"
        )
        return overview, [
            "Core Flow: High-level data flow.",
            "Error Handling: How failures are surfaced.",
        ][:max_points]

    async def fake_deep_research_impl(
        services: Any,
        embedding_manager: Any,
        llm_manager: Any,
        query: str,
        progress: Any,
        path: str | None = None,
    ) -> dict[str, Any]:
        # Return a minimal answer and sources metadata for coverage.
        # For one of the bullets, simulate an empty answer so it is skipped.
        # This specifically skips the FIRST POI to regression-test that topic
        # headings and content remain aligned (no zip-based mispairing).
        if "Core Flow" in query:
            return {
                "answer": "",
                "metadata": {
                    "sources": {
                        "files": [],
                        "chunks": [],
                    },
                    "aggregation_stats": {"files_total": 2, "chunks_total": 2},
                },
            }

        return {
            "answer": f"Section for query: {query[:40]}",
            "metadata": {
                "sources": {
                    "files": ["scope/a.py", "scope/b.py"],
                    "chunks": [
                        {"file_path": "scope/a.py", "start_line": 1, "end_line": 10},
                        {"file_path": "scope/b.py", "start_line": 5, "end_line": 15},
                    ],
                },
                "aggregation_stats": {"files_total": 2, "chunks_total": 2},
            },
        }

    monkeypatch.setattr(
        autodoc_mod, "verify_database_exists", lambda cfg: cfg.database.get_db_path()
    )
    monkeypatch.setattr(
        autodoc_mod,
        "create_services",
        lambda db_path, config, embedding_manager: DummyServices(),
    )
    monkeypatch.setattr(
        autodoc_mod, "LLMManager", DummyLLMManager
    )
    monkeypatch.setattr(
        autodoc_mod, "_run_autodoc_overview_hyde", fake_overview, raising=True
    )
    monkeypatch.setattr(
        autodoc_mod, "deep_research_impl", fake_deep_research_impl, raising=True
    )

    class Args:
        def __init__(self) -> None:
            self.path = scope_path
            self.verbose = False
            self.overview_only = False
            self.out_dir = tmp_path / "out"
            self.comprehensiveness = "low"

    args = Args()

    await autodoc_mod.autodoc_command(args, config)
    captured = capsys.readouterr().out

    # Main document should include the metadata header and top-level heading.
    assert "agent_doc_metadata:" in captured
    assert "autodoc_comprehensiveness: low" in captured
    assert "# AutoDoc for" in captured
    assert "## Coverage Summary" in captured
    # Skipping the first POI should renumber topics contiguously and keep
    # headings aligned with the surviving result.
    assert "## 1. Error Handling" in captured

    # Out-dir should contain an index and topic files.
    out_dir = args.out_dir
    index_files = list(out_dir.glob("*_autodoc_index.md"))
    assert index_files, "Expected an autodoc index file to be written"
    index_content = index_files[0].read_text(encoding="utf-8")
    assert "unreferenced_in_scope: 1" in index_content
    assert "total_indexed: 3" in index_content
    assert "coverage: 66.67%" in index_content
    assert "scope_scope_unreferenced_files.txt" in index_content

    topic_files = list(out_dir.glob("*_topic_*.md"))
    # One of the deep research calls returns an empty answer and should be skipped.
    assert len(topic_files) == 1, "Expected only non-empty topics to be written"
    topic_content = topic_files[0].read_text(encoding="utf-8")
    assert (
        topic_content.startswith("# Error Handling")
        or "\n# Error Handling\n" in topic_content
    )
    unref_files = list(out_dir.glob("*_scope_unreferenced_files.txt"))
    assert unref_files, "Expected unreferenced files artifact to be written"
    unref_content = unref_files[0].read_text(encoding="utf-8")
    assert "scope/c.py" in unref_content

    # Comprehensiveness=low should map to fewer max_points for the overview.
    assert seen_max_points, "Expected fake_overview to be called"
    assert seen_max_points[0] == 5
