"""Retired MP settings remain visible without corrupting MCP stdout."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WARNING = (
    "CHUNKHOUND_MP_START_METHOD is no longer supported and is ignored; "
    "ChunkHound parser workers always use spawn. Remove this setting."
)


@pytest.mark.parametrize("value", [None, "", "fork"])
def test_startup_warns_once_when_retired_setting_is_present(
    tmp_path: Path, value: str | None
) -> None:
    env = dict(os.environ)
    env.pop("CHUNKHOUND_MP_START_METHOD", None)
    if value is not None:
        env["CHUNKHOUND_MP_START_METHOD"] = value
    code = (
        "import sys\n"
        "from chunkhound.api.cli.main import main\n"
        "sys.argv = ['chunkhound', '--version']\n"
        "for attempt in range(2):\n"
        "    try:\n"
        "        main()\n"
        "    except SystemExit as error:\n"
        "        assert error.code == 0\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count(WARNING) == (0 if value is None else 1)
    assert "CHUNKHOUND_MP_START_METHOD" not in result.stdout


@pytest.mark.asyncio
async def test_empty_retired_setting_preserves_real_mcp_framing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    global_config = tmp_path / "global.json"
    global_config.write_text("{}", encoding="utf-8")
    (project / ".chunkhound.json").write_text(
        json.dumps(
            {
                "database": {
                    "provider": "duckdb",
                    "path": str(project / ".chunkhound" / "db"),
                },
                "analytics": {"enabled": False},
                "indexing": {"include": ["*.py"]},
            }
        ),
        encoding="utf-8",
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CHUNKHOUND_", "OPENAI_", "VOYAGE_"))
    }
    env.update(
        CHUNKHOUND_GLOBAL_CONFIG_FILE=str(global_config),
        CHUNKHOUND_MP_START_METHOD="",
        CHUNKHOUND_NO_RICH="1",
        CHUNKHOUND_USE_RUST="0",
        CHUNKHOUND_NO_PROMPTS="1",
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "chunkhound.api.cli.main",
        "mcp",
        str(project),
        "--no-daemon",
        cwd=ROOT,
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())
    try:
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "retired-setting-test", "version": "1"},
            },
        }
        process.stdin.write((json.dumps(request) + "\n").encode())
        await process.stdin.drain()
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
            assert line, "MCP exited before replying to initialize"
            envelope = json.loads(line)
            assert envelope["jsonrpc"] == "2.0"
            if envelope.get("id") == 1:
                assert "serverInfo" in envelope["result"]
                break
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        stderr = (await stderr_task).decode(errors="replace")
    assert stderr.count(WARNING) == 1
