import os
import shutil
import subprocess
from pathlib import Path

import pytest

from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider


@pytest.mark.integration
def test_codex_exec_help_available():
    """Smoke-check that `codex exec --help` runs successfully.

    Skips if Codex CLI is not available on PATH and `CHUNKHOUND_CODEX_BIN` is unset.
    Uses a temporary `CODEX_HOME` to avoid touching user configuration/history.
    """
    codex_bin = os.getenv("CHUNKHOUND_CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        pytest.skip("Codex CLI not found; set CHUNKHOUND_CODEX_BIN or install `codex`.")

    env = os.environ.copy()
    provider = CodexCLIProvider(model="codex")
    base_home = provider._get_base_codex_home()
    if not base_home:
        pytest.xfail("No base CODEX_HOME found to inherit auth from.")

    overlay = provider._build_overlay_home()
    try:
        env["CODEX_HOME"] = overlay
        proc = subprocess.run(
            [codex_bin, "exec", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=30,
            check=False,
        )
    finally:
        shutil.rmtree(overlay, ignore_errors=True)

    # `--help` should succeed and print usage text
    combined = (proc.stdout + proc.stderr).decode("utf-8", errors="ignore").lower()
    assert proc.returncode == 0, f"codex exec --help failed: rc={proc.returncode}, out={combined!r}"
    assert "usage" in combined and "codex exec" in combined, (
        "Help output did not contain expected usage text. Output was: " + combined
    )


@pytest.mark.integration
def test_codex_exec_simple_prompt():
    """Run a tiny non-interactive prompt through `codex exec`.

    Dynamically discovers the cheapest available model via `codex debug models`
    to stay resilient across model catalog changes. Skips if Codex is unavailable
    and xfails if the CLI is not authenticated.
    """
    codex_bin = os.getenv("CHUNKHOUND_CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        pytest.skip("Codex CLI not found; set CHUNKHOUND_CODEX_BIN or install `codex`.")

    env = os.environ.copy()

    prompt = 'Output exactly the uppercase string OK and nothing else.'

    def run_cmd(args):
        return subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            check=False,
        )

    provider = CodexCLIProvider(model="codex")
    base_home = provider._get_base_codex_home()
    if not base_home:
        pytest.xfail("No base CODEX_HOME found to inherit auth from.")

    # Dynamically discover the cheapest available model
    cheapest_model = CodexCLIProvider.get_highest_priority_available_model()
    if not cheapest_model:
        pytest.xfail("Could not discover available Codex models.")

    overlay = provider._build_overlay_home(model_override=cheapest_model)
    try:
        env["CODEX_HOME"] = overlay

        # Verify overlay config enforces our requirements
        cfg = Path(overlay) / "config.toml"
        content = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        assert "history" in content and "persistence" in content and "none" in content.lower(), (
            "Overlay config.toml does not disable history persistence."
        )
        assert "mcp_servers" not in content.lower(), "Overlay config.toml must not define MCP servers."
        assert f'model = "{cheapest_model}"' in content, (
            f"Overlay config.toml must set model to {cheapest_model}."
        )
        assert "model_reasoning_effort" in content and "low" in content.lower(), (
            "Overlay config.toml must set model_reasoning_effort to low."
        )

        # Try explicit model/effort flags first; gracefully remove if unsupported
        base = [codex_bin, "exec", prompt]
        flags = ["--model", cheapest_model, "--model-reasoning-effort", "low", "--skip-git-repo-check"]

        def try_exec(args):
            p = run_cmd(args)
            return (
                p,
                p.stdout.decode("utf-8", errors="ignore").strip(),
                p.stderr.decode("utf-8", errors="ignore").strip().lower(),
            )

        proc, out, err = try_exec(base + flags)
        if proc.returncode != 0 and "unexpected argument '--model-reasoning-effort'" in err:
            flags = ["--model", cheapest_model, "--skip-git-repo-check"]
            proc, out, err = try_exec(base + flags)
        if proc.returncode != 0 and "unexpected argument '--model'" in err:
            # Try only skip-git flag
            proc, out, err = try_exec(base + ["--skip-git-repo-check"])
        if proc.returncode != 0 and "unexpected argument '--skip-git-repo-check'" in err:
            # Last resort: no flags at all
            proc, out, err = try_exec(base)

        # If not authenticated, xfail instead of failing the suite
        auth_hints = ("login", "authenticate", "not logged in", "sign in", "unauthorized")
        if proc.returncode != 0 and any(h in err for h in auth_hints):
            pytest.xfail("Codex CLI not authenticated in this environment.")

        assert proc.returncode == 0, f"codex exec failed: rc={proc.returncode}, stderr={err!r}"
        assert out, "codex exec produced no output"
        assert out.strip() == "OK" or "ok" in out.lower(), (
            f"Unexpected output from codex exec. Expected 'OK', got: {out!r}"
        )

        # Ensure MCP servers are still absent post-run
        content_post = (Path(overlay) / "config.toml").read_text(encoding="utf-8")
        assert "mcp_servers" not in content_post.lower()
    finally:
        shutil.rmtree(overlay, ignore_errors=True)


@pytest.mark.integration
def test_codex_exec_status_reports_overlay_model(monkeypatch):
    """Ensure `codex exec` sees the overlay model/effort configuration.

    Runs `codex exec "/status"` against a provider-built overlay and asserts
    that the reported model and reasoning effort match the overlay configuration.
    Uses the cheapest available model discovered dynamically.
    """
    codex_bin = os.getenv("CHUNKHOUND_CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        pytest.skip("Codex CLI not found; set CHUNKHOUND_CODEX_BIN or install `codex`.")

    # Ensure no env overrides interfere with the test-specific configuration
    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)

    env = os.environ.copy()

    # Dynamically discover the cheapest available model
    cheapest_model = CodexCLIProvider.get_highest_priority_available_model()
    if not cheapest_model:
        pytest.xfail("Could not discover available Codex models.")

    # Use the discovered model with medium reasoning effort to verify overlay wiring
    provider = CodexCLIProvider(model=cheapest_model, reasoning_effort="medium")
    base_home = provider._get_base_codex_home()
    if not base_home:
        pytest.xfail("No base CODEX_HOME found to inherit auth from.")

    overlay = provider._build_overlay_home()
    try:
        env["CODEX_HOME"] = overlay

        cfg = Path(overlay) / "config.toml"
        cfg_text = cfg.read_text(encoding="utf-8")
        # Sanity: overlay config should encode the expected model/effort
        assert f'model = "{cheapest_model}"' in cfg_text
        assert 'model_reasoning_effort = "medium"' in cfg_text

        proc = subprocess.run(
            [codex_bin, "exec", "--skip-git-repo-check", "/status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            check=False,
        )

        out = proc.stdout.decode("utf-8", errors="ignore")
        err = proc.stderr.decode("utf-8", errors="ignore").lower()

        auth_hints = ("login", "authenticate", "not logged in", "sign in", "unauthorized")
        if proc.returncode != 0 and any(h in err for h in auth_hints):
            pytest.xfail("Codex CLI not authenticated in this environment.")

        combined = f"{out}\n{err}".lower()
        # `/status` should report the effective model and reasoning effort
        assert f"model: {cheapest_model}" in combined, combined
        assert "reasoning effort: medium" in combined, combined
    finally:
        shutil.rmtree(overlay, ignore_errors=True)
