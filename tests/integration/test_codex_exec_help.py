import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider


def _should_xfail_codex_env_limitation(stderr: str, *, explicit_model_requested: bool) -> bool:
    err = stderr.lower()
    auth_hints = ("login", "authenticate", "not logged in", "sign in", "unauthorized")
    if any(h in err for h in auth_hints):
        return True
    if not explicit_model_requested and (
        "model is not supported when using codex with a chatgpt account" in err
        or "not supported when using codex with a chatgpt account" in err
    ):
        return True
    return False


def _xfail_reason(stderr: str, *, explicit_model_requested: bool) -> str:
    err = stderr.lower()
    if any(h in err for h in ("login", "authenticate", "not logged in", "sign in", "unauthorized")):
        return "Codex CLI not authenticated in this environment."
    if not explicit_model_requested and (
        "model is not supported when using codex with a chatgpt account" in err
        or "not supported when using codex with a chatgpt account" in err
    ):
        return "Codex CLI provider-default model is not available for this authenticated account."
    return "Codex CLI environment limitation."


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

    Uses the provider-default model path unless CHUNKHOUND_CODEX_DEFAULT_MODEL
    is explicitly set. Skips if Codex is unavailable and xfails on expected
    environment limitations such as missing auth or account/model incompatibility.
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
    explicit_model_requested = bool(os.getenv("CHUNKHOUND_CODEX_DEFAULT_MODEL"))

    overlay = provider._build_overlay_home()
    try:
        env["CODEX_HOME"] = overlay

        # Verify overlay config enforces our requirements
        cfg = Path(overlay) / "config.toml"
        content = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
        assert "history" in content and "persistence" in content and "none" in content.lower(), (
            "Overlay config.toml does not disable history persistence."
        )
        assert "mcp_servers" not in content.lower(), "Overlay config.toml must not define MCP servers."
        if explicit_model_requested:
            assert 'model = "' in content, "Overlay config.toml must set the explicit override model."
        else:
            assert 'model = "' not in content, "Overlay config.toml must omit model to use the provider default."
        assert "model_reasoning_effort" in content and "low" in content.lower(), (
            "Overlay config.toml must set model_reasoning_effort to low."
        )

        base = [codex_bin, "exec", prompt]
        flags = ["--model-reasoning-effort", "low", "--skip-git-repo-check"]
        if explicit_model_requested:
            flags = ["--model", os.environ["CHUNKHOUND_CODEX_DEFAULT_MODEL"], *flags]

        def try_exec(args):
            p = run_cmd(args)
            return (
                p,
                p.stdout.decode("utf-8", errors="ignore").strip(),
                p.stderr.decode("utf-8", errors="ignore").strip().lower(),
            )

        proc, out, err = try_exec(base + flags)
        if proc.returncode != 0 and "unexpected argument '--model-reasoning-effort'" in err:
            flags = ["--skip-git-repo-check"]
            if explicit_model_requested:
                flags = ["--model", os.environ["CHUNKHOUND_CODEX_DEFAULT_MODEL"], *flags]
            proc, out, err = try_exec(base + flags)
        if proc.returncode != 0 and "unexpected argument '--model'" in err:
            # Try only skip-git flag
            proc, out, err = try_exec(base + ["--skip-git-repo-check"])
        if proc.returncode != 0 and "unexpected argument '--skip-git-repo-check'" in err:
            # Last resort: no flags at all
            proc, out, err = try_exec(base)

        if proc.returncode != 0 and _should_xfail_codex_env_limitation(
            err,
            explicit_model_requested=explicit_model_requested,
        ):
            pytest.xfail(
                _xfail_reason(err, explicit_model_requested=explicit_model_requested)
            )

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
    """Ensure `codex exec` sees the overlay effort configuration.

    Runs `codex exec "/status"` against a provider-built overlay and asserts
    that the reported reasoning effort matches the overlay configuration.
    """
    codex_bin = os.getenv("CHUNKHOUND_CODEX_BIN") or shutil.which("codex")
    if not codex_bin:
        pytest.skip("Codex CLI not found; set CHUNKHOUND_CODEX_BIN or install `codex`.")

    # Ensure no env overrides interfere with the test-specific configuration
    monkeypatch.delenv("CHUNKHOUND_CODEX_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("CHUNKHOUND_CODEX_REASONING_EFFORT", raising=False)

    env = os.environ.copy()
    provider = CodexCLIProvider(model="codex", reasoning_effort="medium")
    base_home = provider._get_base_codex_home()
    if not base_home:
        pytest.xfail("No base CODEX_HOME found to inherit auth from.")
    explicit_model_requested = bool(os.getenv("CHUNKHOUND_CODEX_DEFAULT_MODEL"))

    overlay = provider._build_overlay_home()
    try:
        env["CODEX_HOME"] = overlay

        cfg = Path(overlay) / "config.toml"
        cfg_text = cfg.read_text(encoding="utf-8")
        if explicit_model_requested:
            assert 'model = "' in cfg_text
        else:
            assert 'model = "' not in cfg_text
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

        if proc.returncode != 0 and _should_xfail_codex_env_limitation(
            err,
            explicit_model_requested=explicit_model_requested,
        ):
            pytest.xfail(
                _xfail_reason(err, explicit_model_requested=explicit_model_requested)
            )

        combined = f"{out}\n{err}".lower()
        assert "model:" in combined, combined
        assert "reasoning effort: medium" in combined, combined
    finally:
        shutil.rmtree(overlay, ignore_errors=True)
