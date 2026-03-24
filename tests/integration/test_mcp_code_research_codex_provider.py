import asyncio
from pathlib import Path

import pytest

from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import execute_tool


class _DummyEmbeddingProvider:
    name = "dummy"
    model = "dummy"
    dims = 1
    distance = "cosine"
    batch_size = 8

    def supports_reranking(self) -> bool:  # deep_research requirement
        return True

    async def embed(self, texts):  # pragma: no cover - trivial
        return [[0.0] for _ in texts]


@pytest.mark.asyncio
async def test_code_research_uses_codex_cli_for_synthesis(monkeypatch, tmp_path: Path):
    """Ensure MCP code_research tool calls Codex CLI provider during synthesis.

    We avoid invoking the real `codex` binary by monkeypatching `_run_exec` and
    providing a stub embedding provider that satisfies reranking checks.
    """

    # 1) Minimal DB and services
    db_path = tmp_path / "test.db"

    # Minimal EmbeddingManager with a reranking-capable stub
    em = EmbeddingManager()
    em.register_provider(_DummyEmbeddingProvider(), set_default=True)

    # Create services (connect later via provider.connect inside create_services)
    services = create_services(db_path=db_path, config={}, embedding_manager=em)
    services.provider.connect()

    # 2) LLM Manager configured to use codex-cli for synthesis
    util_conf = {"provider": "codex-cli", "model": "codex"}
    synth_conf = {"provider": "codex-cli", "model": "codex"}
    llm = LLMManager(util_conf, synth_conf)

    # 3) Monkeypatch CodexCLIProvider._run_exec to avoid real subprocess
    called = {"count": 0, "last_prompt": None}

    async def _stub_run_exec(self, text, cwd=None, max_tokens=1024, timeout=None, model=None):
        called["count"] += 1
        called["last_prompt"] = text
        return "SYNTH_OK: codex-cli invoked"

    import chunkhound.providers.llm.codex_cli_provider as codex_mod

    monkeypatch.setattr(codex_mod.CodexCLIProvider, "_run_exec", _stub_run_exec, raising=True)

    # 4) Monkeypatch tool implementation to force synthesis call regardless of DB/search
    from chunkhound.mcp_server import tools as tools_mod

    async def _stub_deep_research_impl(*, services, embedding_manager, llm_manager, query, progress=None):
        prov = llm_manager.get_synthesis_provider()
        resp = await prov.complete(prompt=f"probe: {query}")
        return {"answer": resp.content}

    # Replace both the function and the registered tool implementation
    monkeypatch.setattr(tools_mod, "deep_research_impl", _stub_deep_research_impl, raising=True)
    tools_mod.TOOL_REGISTRY["code_research"].implementation = _stub_deep_research_impl

    # 5) Execute the MCP tool directly
    result = await execute_tool(
        tool_name="code_research",
        services=services,
        embedding_manager=em,
        arguments={"query": "dummy question"},
        scan_progress=None,
        llm_manager=llm,
    )

    # 6) Assertions: provider path used and output contains our stub marker
    assert (
        isinstance(result, str) and "SYNTH_OK" in result
    ), "code_research did not use codex-cli synthesis path"
    assert called["count"] >= 1, "Codex CLI provider was not invoked"


@pytest.mark.asyncio
async def test_code_research_forwards_path_argument_to_impl(monkeypatch, tmp_path: Path):
    """Ensure execute_tool forwards path argument to deep_research_impl."""

    # Minimal DB and services
    db_path = tmp_path / "test.db"

    em = EmbeddingManager()
    em.register_provider(_DummyEmbeddingProvider(), set_default=True)

    services = create_services(db_path=db_path, config={}, embedding_manager=em)
    services.provider.connect()

    # LLM manager can be any configured provider; stub won't use it
    util_conf = {"provider": "codex-cli", "model": "codex"}
    synth_conf = {"provider": "codex-cli", "model": "codex"}
    llm = LLMManager(util_conf, synth_conf)

    # Capture arguments received by deep_research_impl
    captured: dict[str, Any] = {}

    from chunkhound.mcp_server import tools as tools_mod

    async def _stub_deep_research_impl(
        *,
        services,
        embedding_manager,
        llm_manager,
        query,
        progress=None,
        path=None,
    ):
        captured["services"] = services
        captured["embedding_manager"] = embedding_manager
        captured["llm_manager"] = llm_manager
        captured["query"] = query
        captured["path"] = path
        return {"answer": "PATH_OK"}

    # Replace implementation and registry entry
    monkeypatch.setattr(tools_mod, "deep_research_impl", _stub_deep_research_impl, raising=True)
    tools_mod.TOOL_REGISTRY["code_research"].implementation = _stub_deep_research_impl

    result = await execute_tool(
        tool_name="code_research",
        services=services,
        embedding_manager=em,
        arguments={"query": "dummy question", "path": "repo-a"},
        scan_progress=None,
        llm_manager=llm,
    )

    # execute_tool flattens code_research dict responses to raw markdown string
    assert result == "PATH_OK"
    assert captured.get("query") == "dummy question"
    assert captured.get("path") == "repo-a"
