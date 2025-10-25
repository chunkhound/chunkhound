"""Test-only sitecustomize hook to patch Codex CLI in subprocesses.

This module is auto-imported by Python when present on PYTHONPATH.
We use it in E2E tests to avoid invoking the real `codex` binary from
child processes (e.g., the MCP stdio server).

It is activated only when CH_TEST_PATCH_CODEX=1 in the environment.
"""

from __future__ import annotations

import os


def _patch_codex_cli_provider() -> None:
    try:
        from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore
    except Exception:
        return

    async def _stub_run_exec(self, text, cwd=None, max_tokens=1024, timeout=None, model=None):
        mark = os.getenv("CH_TEST_CODEX_MARK_FILE")
        if mark:
            try:
                with open(mark, "a", encoding="utf-8") as f:
                    f.write("CALLED\n")
            except Exception:
                pass
        return "SYNTH_OK: codex-cli invoked"

    # Avoid availability checks causing warnings
    def _stub_available(self) -> bool:  # pragma: no cover - trivial
        return True

    try:
        CodexCLIProvider._run_exec = _stub_run_exec  # type: ignore[attr-defined]
        CodexCLIProvider._codex_available = _stub_available  # type: ignore[attr-defined]
    except Exception:
        # Best-effort; tests will still fail clearly if not patched
        pass


def _force_code_research_synthesis() -> None:
    """Replace code_research implementation to call synthesis directly.

    This avoids dependencies on embeddings/search results in E2E tests while
    still exercising the MCP tool path and LLM provider wiring.
    """
    try:
        from chunkhound.mcp_server import tools as tools_mod  # type: ignore
    except Exception:
        return

    async def _stub_deep_research_impl(*, services, embedding_manager, llm_manager, query, depth, progress=None):
        # Ensure we have an LLM manager even if server didn't configure one
        if llm_manager is None:
            try:
                from chunkhound.llm_manager import LLMManager  # type: ignore
                llm_manager = LLMManager(
                    {"provider": "codex-cli", "model": "codex"},
                    {"provider": "codex-cli", "model": "codex"},
                )
            except Exception:
                return {"answer": "LLM manager unavailable"}
        prov = llm_manager.get_synthesis_provider()
        resp = await prov.complete(prompt=f"E2E: {query}")
        return {"answer": resp.content}

    try:
        tools_mod.deep_research_impl = _stub_deep_research_impl  # type: ignore[assignment]
        tools_mod.TOOL_REGISTRY["code_research"].implementation = _stub_deep_research_impl  # type: ignore[index]
    except Exception:
        pass


if os.getenv("CH_TEST_PATCH_CODEX") == "1":  # activate only for tests that request it
    _patch_codex_cli_provider()
    if os.getenv("CH_TEST_FORCE_SYNTHESIS") == "1":
        _force_code_research_synthesis()
