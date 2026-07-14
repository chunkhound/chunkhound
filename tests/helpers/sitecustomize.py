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

    async def _stub_deep_research_impl(*, services, embedding_manager, llm_manager, query, progress=None):
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


def _patch_capability_checks() -> None:
    """Stub capability checks to return True in test mode.

    In synthesis test mode, we bypass embedding/reranker validation
    because the stubbed implementation doesn't need real providers.
    """
    if os.getenv("CH_TEST_FORCE_SYNTHESIS") != "1":
        return

    try:
        # Patch the tool registry entry to not require embeddings/reranker
        from chunkhound.mcp_server.tools import TOOL_REGISTRY
        if "code_research" in TOOL_REGISTRY:
            tool = TOOL_REGISTRY["code_research"]
            tool.requires_embeddings = False
            tool.requires_reranker = False
    except ImportError:
        pass


def _patch_websearch_for_tests() -> None:
    """Stub the websearch pipeline so stdio integration tests run offline.

    Activated by CH_TEST_WEBSEARCH_STUB=1. Flips capability gating off for the
    websearch tool and replaces the three lazy-imported helpers with trivial
    stubs: search returns fixed results, fetch_and_save is a no-op, and the
    subprocess launch runs a one-line `print('ANSWER')` command.
    """
    try:
        from chunkhound.mcp_server.tools import TOOL_REGISTRY

        if "websearch" in TOOL_REGISTRY:
            tool = TOOL_REGISTRY["websearch"]
            tool.requires_embeddings = False
            tool.requires_llm = False
            tool.requires_reranker = False
    except ImportError:
        pass

    try:
        import sys as _sys

        from chunkhound.utils import websearch_core as ws_core

        # Touch each symbol before rebinding so a rename surfaces as
        # AttributeError at import time instead of silently leaving the
        # stub inactive while tests hit the real network.
        ws_core.search  # noqa: B018
        ws_core.fetch_and_save  # noqa: B018
        ws_core.build_quickresearch_argv_core  # noqa: B018

        def _stub_search(query, limit=30, progress_callback=None):
            return [
                ("Stub Result One", "https://example.invalid/one", "first stub"),
                ("Stub Result Two", "https://example.invalid/two", "second stub"),
                ("Stub Result Three", "https://example.invalid/three", "third stub"),
            ][:limit]

        async def _stub_fetch_and_save(
            urls, tmpdir, progress_callback=None, warning_callback=None,
            mapping=None,
        ):
            # Write minimal .md files so _quickresearch (stubbed separately)
            # has input should it ever run.
            for i, url in enumerate(urls):
                name = f"stub_{i}.md"
                (tmpdir / name).write_text("stub content", encoding="utf-8")
                if mapping is not None:
                    mapping[name] = url

        def _stub_build_argv(query, tmpdir, config, parent_pid):
            # -S skips site init so this sitecustomize isn't re-loaded in the
            # grandchild — avoids a recursive chunkhound cold-import that
            # blew past the 30s tools/call budget on Windows.
            return [_sys.executable, "-S", "-c", "print('ANSWER')"]

        ws_core.search = _stub_search  # type: ignore[assignment]
        ws_core.fetch_and_save = _stub_fetch_and_save  # type: ignore[assignment]
        ws_core.build_quickresearch_argv_core = _stub_build_argv  # type: ignore[assignment]
    except ImportError:
        pass


if os.getenv("CH_TEST_PATCH_CODEX") == "1":  # activate only for tests that request it
    _patch_codex_cli_provider()
    if os.getenv("CH_TEST_FORCE_SYNTHESIS") == "1":
        _force_code_research_synthesis()
        _patch_capability_checks()

if os.getenv("CH_TEST_WEBSEARCH_STUB") == "1":
    _patch_websearch_for_tests()
