from pathlib import Path
from typing import Any

import pytest

import chunkhound.api.cli.commands.code_mapper as code_mapper_mod
from chunkhound.code_mapper import service as code_mapper_service
from chunkhound.code_mapper.models import CodeMapperPOI
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
        # deep-research results so code_mapper can emit an "unreferenced files"
        # artifact.
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
    """Placeholder LLM manager for code_mapper tests."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._configured = True

    def is_configured(self) -> bool:
        return self._configured


async def _run_code_mapper_with_stubs(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    set_combined: bool,
    cli_combined: bool | None = None,
    comprehensiveness: str = "low",
) -> tuple[Path, str, list[int]]:
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
        meta: Any | None = None,
        context: str | None = None,
        max_points: int = 10,
        comprehensiveness: str = "medium",
        out_dir: Path | None = None,
        map_hyde_provider: Any | None = None,
        indexing_cfg: Any | None = None,
    ) -> tuple[str, list[CodeMapperPOI]]:
        _ = (llm_manager, target_dir, scope_path, scope_label)
        _ = (meta, context, comprehensiveness, out_dir, map_hyde_provider, indexing_cfg)
        # Record the requested max_points so we can assert comprehensiveness mapping.
        seen_max_points.append(max_points)
        overview = (
            "1. **Core Flow**: High-level data flow.\n"
            "2. **Error Handling**: How failures are surfaced.\n"
        )
        points = [
            CodeMapperPOI(
                mode="architectural",
                text="Core Flow: High-level data flow.",
            ),
            CodeMapperPOI(
                mode="architectural",
                text="Error Handling: How failures are surfaced.",
            ),
        ]
        return overview, points[:max_points]

    async def fake_run_deep_research(
        *,
        query: str,
        **__: Any,
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
        code_mapper_mod,
        "verify_database_exists",
        lambda cfg: cfg.database.get_db_path(),
    )
    monkeypatch.setattr(
        code_mapper_mod,
        "create_services",
        lambda db_path, config, embedding_manager: DummyServices(),
    )
    monkeypatch.setattr(code_mapper_mod, "LLMManager", DummyLLMManager)
    monkeypatch.setattr(
        code_mapper_service,
        "run_code_mapper_overview_hyde",
        fake_overview,
        raising=True,
    )
    monkeypatch.setattr(
        code_mapper_service, "run_deep_research", fake_run_deep_research, raising=True
    )

    if set_combined:
        monkeypatch.setenv("CH_CODE_MAPPER_WRITE_COMBINED", "1")
    else:
        monkeypatch.delenv("CH_CODE_MAPPER_WRITE_COMBINED", raising=False)

    class Args:
        def __init__(self) -> None:
            self.path = scope_path
            self.verbose = False
            self.overview_only = False
            self.out = tmp_path / "out"
            self.comprehensiveness = comprehensiveness
            self.combined = cli_combined

    args = Args()

    await code_mapper_mod.code_mapper_command(args, config)
    captured = capsys.readouterr().out
    return args.out, captured, seen_max_points


@pytest.mark.asyncio
async def test_code_mapper_end_to_end_default_omits_combined_doc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Code Mapper should skip the combined doc by default and still write topics.

    This is a lightweight integration test that exercises the code_mapper command
    with stubbed services and deep-research behavior. It verifies that:
    - the main document is not printed to stdout,
    - coverage stats are computed without crashing,
    - an index file plus one markdown file per point-of-interest are written, and
    - the combined doc is omitted unless explicitly enabled.
    """

    (
        out_dir,
        captured,
        seen_max_points,
    ) = await _run_code_mapper_with_stubs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        set_combined=False,
    )
    # Default behavior: avoid printing the full document to stdout.
    assert "# Code Mapper for" not in captured
    assert "## Coverage Summary" not in captured

    combined_docs = list(out_dir.glob("*_code_mapper.md"))
    assert not combined_docs, "Combined Code Mapper doc should be disabled by default"

    index_files = list(out_dir.glob("*_code_mapper_index.md"))
    assert index_files, "Expected a Code Mapper index file to be written"
    index_content = index_files[0].read_text(encoding="utf-8")
    assert "unreferenced_in_scope: 1" in index_content
    assert "total_research_calls: 2" in index_content
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


@pytest.mark.asyncio
async def test_code_mapper_end_to_end_combined_doc_requires_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Combined Code Mapper doc should be written only when the env flag is set."""

    (
        out_dir,
        captured,
        _seen_max_points,
    ) = await _run_code_mapper_with_stubs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        set_combined=True,
    )

    # Default behavior: avoid printing the full document to stdout.
    assert "# Code Mapper for" not in captured
    assert "## Coverage Summary" not in captured

    combined_docs = list(out_dir.glob("*_code_mapper.md"))
    assert combined_docs, "Expected a combined Code Mapper document to be written"
    combined_content = combined_docs[0].read_text(encoding="utf-8")
    assert "agent_doc_metadata:" in combined_content
    assert "code_mapper_comprehensiveness: low" in combined_content
    assert "# Code Mapper for" in combined_content
    assert "## Coverage Summary" in combined_content
    # Skipping the first POI should renumber topics contiguously and keep
    # headings aligned with the surviving result.
    assert "## 1. Error Handling" in combined_content


@pytest.mark.asyncio
async def test_code_mapper_end_to_end_combined_cli_overrides_env_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--no-combined should disable combined output even if env is set."""
    (
        out_dir,
        _captured,
        _seen_max_points,
    ) = await _run_code_mapper_with_stubs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        set_combined=True,
        cli_combined=False,
    )

    combined_docs = list(out_dir.glob("*_code_mapper.md"))
    assert not combined_docs


@pytest.mark.asyncio
async def test_code_mapper_end_to_end_combined_cli_overrides_env_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--combined should enable combined output even if env is unset."""
    (
        out_dir,
        _captured,
        _seen_max_points,
    ) = await _run_code_mapper_with_stubs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        set_combined=False,
        cli_combined=True,
    )

    combined_docs = list(out_dir.glob("*_code_mapper.md"))
    assert combined_docs


@pytest.mark.asyncio
async def test_code_mapper_ultra_requests_max_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Comprehensiveness=ultra should request more points-of-interest."""
    (
        out_dir,
        _captured,
        seen_max_points,
    ) = await _run_code_mapper_with_stubs(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        capsys=capsys,
        set_combined=False,
        comprehensiveness="ultra",
    )

    assert out_dir.exists()
    assert seen_max_points, "Expected overview to be called"
    assert seen_max_points[0] == 20
