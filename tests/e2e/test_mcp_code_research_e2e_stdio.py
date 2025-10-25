import asyncio
import json
import os
from pathlib import Path

import pytest

from tests.utils import SubprocessJsonRpcClient, create_subprocess_exec_safe, get_safe_subprocess_env
from tests.utils.windows_compat import windows_safe_tempdir


@pytest.mark.asyncio
async def test_mcp_code_research_uses_codex_cli_via_stdio():
    """E2E: spawn MCP stdio server and call code_research.

    - Creates a tiny project and DB in a temp directory
    - Indexes without embeddings (fast)
    - Starts MCP stdio server pointed at that directory
    - Uses a sitecustomize-based patch to stub codex exec and force synthesis
    - Calls the code_research tool and asserts the stubbed codex path was used
    """
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
        assert proc.returncode == 0, f"indexing failed\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"

    with windows_safe_tempdir() as temp:
        temp_dir = temp
        src_dir = temp_dir / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "app.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")

        # Minimal config with database path (duckdb)
        cfg_path = temp_dir / ".chunkhound.json"
        db_path = temp_dir / ".chunkhound" / "db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "database": {"path": str(db_path), "provider": "duckdb"},
            "indexing": {"include": ["*.py"]},
        }
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

        # Build an environment that enables our codex stub in the child process
        mark_file = temp_dir / "codex_called.txt"
        env = get_safe_subprocess_env({
            **os.environ,
            "PYTHONPATH": f"{Path('tests/helpers').resolve()}:{os.environ.get('PYTHONPATH','')}",
            "CH_TEST_PATCH_CODEX": "1",
            "CH_TEST_FORCE_SYNTHESIS": "1",
            "CH_TEST_CODEX_MARK_FILE": str(mark_file),
            "CHUNKHOUND_MCP_MODE": "1",
            "CHUNKHOUND_DEBUG": "1",
            "CHUNKHOUND_DEBUG_FILE": str(temp_dir / "mcp_debug.log"),
        })

        # 1) Index the tiny repo (no embeddings)
        await run_index(temp_dir, cfg_path, db_path)

        # 2) Start MCP stdio server pointing at temp_dir, with codex-cli synthesis
        mcp_cmd = [
            "uv",
            "run",
            "chunkhound",
            "mcp",
            str(temp_dir),
            "--stdio",
            "--llm-synthesis-provider",
            "codex-cli",
            "--llm-synthesis-model",
            "codex",
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
            # Handshake
            init = await client.send_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "e2e", "version": "1.0"},
                },
                timeout=10.0,
            )
            assert "serverInfo" in init

            await client.send_notification("notifications/initialized")

            # Call code_research
            call = await client.send_request(
                "tools/call",
                {"name": "code_research", "arguments": {"query": "alpha", "depth": "shallow"}},
                timeout=30.0,
            )
            # Response content is a list of text items; check for stub marker
            result = call.get("result") or {}
            # Some servers nest under 'content' or return 'content' directly
            contents = result.get("content") or result.get("result") or []
            if isinstance(contents, dict):
                contents = contents.get("content", [])
            # Fallback: sometimes server returns a single text field
            if not contents and isinstance(result, dict) and "text" in result:
                contents = [{"text": result["text"]}]
            full_text = "\n".join([c.get("text", "") for c in contents if isinstance(c, dict)])
            if not full_text:
                # Last resort: stringify the whole response for debugging
                full_text = json.dumps(call)
            assert "SYNTH_OK" in full_text, f"unexpected tool output: {full_text!r}"

            # Also verify our stub marked a file (proves cross-process patch executed)
            assert mark_file.exists(), "codex stub did not run in child process"
            assert "CALLED" in mark_file.read_text(encoding="utf-8")
        finally:
            await client.close()
