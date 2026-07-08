"""Registry-isolation integration test for the websearch MCP tool.

Invariant: after `websearch` runs, a follow-up `search` call on the same
stdio session must still serve the originally indexed project (no pollution
from the transient web-content index). The happy-path roundtrip is already
covered by the lighter `test_mcp_websearch_stdio_mocked` smoke test; this
file focuses solely on isolation.

With `CH_TEST_WEBSEARCH_STUB=1` the research subprocess is replaced by a
one-liner (`python -c "print('ANSWER')"`), so the check verifies the
parent-process half of the invariant: that `websearch_impl` itself does not
mutate the registry before delegating to the child.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from tests.utils import (
    SubprocessJsonRpcClient,
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)
from tests.utils.windows_compat import get_fs_event_timeout, windows_safe_tempdir


@pytest.mark.asyncio
async def test_websearch_registry_isolation():
    async def run_index(temp_dir: Path, cfg_path: Path, db_path: Path) -> None:
        cmd = [
            "uv",
            "run",
            "chunkhound",
            "index",
            str(temp_dir),
            "--no-embeddings",
            "--config",
            str(cfg_path),
            "--db",
            str(db_path),
        ]
        proc = await create_subprocess_exec_safe(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=get_safe_subprocess_env(os.environ.copy()),
        )
        stdout, stderr = await proc.communicate()
        assert proc.returncode == 0, (
            f"indexing failed\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )

    with windows_safe_tempdir() as temp_dir:
        src_dir = temp_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        # The search probe below looks for "alpha" inside app.py.
        (src_dir / "app.py").write_text(
            "def alpha():\n    return 1\n", encoding="utf-8"
        )

        cfg_path = temp_dir / ".chunkhound.json"
        db_path = temp_dir / ".chunkhound" / "db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        helpers_dir = Path("tests/helpers").resolve()
        prior_pp = os.environ.get("PYTHONPATH", "")
        env = get_safe_subprocess_env(
            {
                **os.environ,
                "PYTHONPATH": f"{helpers_dir}{os.pathsep}{prior_pp}",
                "CH_TEST_WEBSEARCH_STUB": "1",
                "CHUNKHOUND_MCP_MODE": "1",
            }
        )

        await run_index(temp_dir, cfg_path, db_path)

        mcp_cmd = [
            "uv",
            "run",
            "chunkhound",
            "mcp",
            str(temp_dir),
            "--stdio",
            "--no-daemon",
            "--no-embeddings",
            "--config",
            str(cfg_path),
        ]
        proc = await create_subprocess_exec_safe(
            *mcp_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(temp_dir),
        )

        client = SubprocessJsonRpcClient(proc)
        await client.start()
        try:
            init_timeout = max(10.0, get_fs_event_timeout())
            init = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "websearch-int", "version": "1.0"},
                },
                timeout=init_timeout,
            )
            assert "serverInfo" in init

            await client.send_notification("notifications/initialized")

            tools_result = await client.send_request("tools/list", timeout=10.0)
            tool_names = [t["name"] for t in tools_result.get("tools", [])]
            assert "websearch" in tool_names, (
                f"websearch missing from tools/list: {tool_names}"
            )

            # Drive websearch through the real MCP dispatch path and confirm
            # the stubbed ANSWER sentinel reaches the client. Without this
            # assertion, a stub-load failure (e.g. PYTHONPATH not propagated)
            # would silently fall through to the real network while the
            # follow-up search probe still passes.
            ws_call = await client.send_request(
                "tools/call",
                {
                    "name": "websearch",
                    "arguments": {"query": "isolation probe", "limit": 3},
                },
                timeout=30.0,
            )
            ws_contents = ws_call.get("content") or []
            ws_text = "\n".join(
                c.get("text", "") for c in ws_contents if isinstance(c, dict)
            )
            assert "ANSWER" in ws_text, (
                f"stub sentinel missing from websearch response — did the "
                f"stub load? response={ws_call!r}"
            )

            # Spec §9.4: follow-up search on the SAME session must still
            # hit the originally indexed project, not the transient web
            # index the subprocess just built.
            search_result = await client.send_request(
                "tools/call",
                {
                    "name": "search",
                    "arguments": {"query": "alpha", "type": "regex"},
                },
                timeout=15.0,
            )
            contents = search_result.get("content") or []
            assert contents, f"search response missing content: {search_result!r}"
            search_text = contents[0]["text"]
            assert "No results found" not in search_text, (
                f"follow-up search returned no hits — registry may have been "
                f"polluted by websearch: {search_text!r}"
            )
            paths = re.findall(r"^## `([^`]+)`", search_text, re.MULTILINE)
            assert any(p.endswith("app.py") for p in paths), (
                f"follow-up search did not return the indexed project file; "
                f"paths={paths!r}"
            )
        finally:
            await client.close()
