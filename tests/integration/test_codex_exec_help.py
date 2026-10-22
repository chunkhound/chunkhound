"""Local CLI help and explicitly configured real Codex requests."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
from tests.helpers.isolation import run_control
from tests.helpers.live import require_cli


def _run(binary, arguments, home=None):
    env = os.environ.copy()
    if home:
        env["CODEX_HOME"] = home
    return subprocess.run(
        [binary, "exec", *arguments],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path.cwd(),
        timeout=120,
        check=False,
    )


@pytest.mark.integration
def test_codex_exec_help_available():
    binary = run_control("CHUNKHOUND_CODEX_BIN") or shutil.which("codex")
    if not binary:
        pytest.skip("Local Codex CLI unavailable")
    result = _run(binary, ["--help"])
    output = (result.stdout + result.stderr).lower()
    assert result.returncode == 0, output
    assert "usage" in output and "codex exec" in output


@pytest.fixture
def codex_inputs(hermetic_environment):
    data = require_cli("codex")
    if not isinstance(data.get("auth"), dict) or not data.get("model"):
        pytest.fail("Live prerequisite missing: codex.auth object and codex.model")
    (Path(os.environ["CODEX_HOME"]) / "auth.json").write_text(json.dumps(data["auth"]))
    return data


@pytest.fixture
def codex_overlay(codex_inputs):
    provider = CodexCLIProvider(model=codex_inputs["model"], reasoning_effort="medium")
    overlay = provider._build_overlay_home()
    try:
        yield codex_inputs, overlay
    finally:
        shutil.rmtree(overlay)


@pytest.mark.live
@pytest.mark.integration
def test_codex_exec_simple_prompt(codex_overlay):
    data, overlay = codex_overlay
    result = _run(
        data.get("binary", "codex"),
        [
            "--skip-git-repo-check",
            "Output exactly the uppercase string OK and nothing else.",
        ],
        overlay,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout.lower(), result.stdout
    content = (Path(overlay) / "config.toml").read_text()
    assert "mcp_servers" not in content.lower()
    assert f'model = "{data["model"]}"' in content


@pytest.mark.live
@pytest.mark.integration
def test_codex_exec_status_reports_overlay_model(codex_overlay):
    data, overlay = codex_overlay
    result = _run(
        data.get("binary", "codex"), ["--skip-git-repo-check", "/status"], overlay
    )
    assert result.returncode == 0, result.stderr
    output = (result.stdout + result.stderr).lower()
    assert f"model: {data['model']}" in output
    assert "reasoning effort: medium" in output
