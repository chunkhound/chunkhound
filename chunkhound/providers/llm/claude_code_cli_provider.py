"""Claude Code CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the Claude Code CLI (claude --print) to enable deep research
using the user's existing Claude subscription instead of API credits.

Note: This provider is configured for vanilla LLM behavior:
- All tools disabled via bare ``--disallowedTools`` (CLI: flag alone = deny all;
  avoids empty ``--tools ""`` which Windows ``.cmd`` shims garble)
- MCP servers disabled via empty --mcp-config (temp JSON file on disk so
  Windows ``claude.cmd`` shims do not garble inline JSON quotes)
- Workspace isolation (runs from temp directory to prevent context gathering)
- Clean API access without workspace overhead
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_SENTINEL,
    CLAUDE_OPUS_SENTINEL,
    CLAUDE_SONNET_SENTINEL,
    resolve_claude_cli_model,
)
from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.base_cli_provider import (
    BaseCLIProvider,
    build_cli_argv,
    resolve_cli_binary,
    terminate_cli_process,
)
from chunkhound.utils.text_sanitization import sanitize_error_text

# Empty MCP config: no servers. Passed as a file path (not inline JSON) so
# Windows cmd.exe / .cmd batch reparse cannot mangle embedded quotes.
_EMPTY_MCP_CONFIG_JSON = b'{"mcpServers":{}}\n'


def _write_empty_mcp_config_file() -> Path:
    """Write a unique empty MCP config JSON file; caller must unlink it."""
    fd, name = tempfile.mkstemp(
        prefix="chunkhound_claude_mcp_",
        suffix=".json",
        text=False,
    )
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_EMPTY_MCP_CONFIG_JSON)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path.resolve()


class ClaudeCodeCLIProvider(BaseCLIProvider):
    """Claude Code CLI provider using subprocess calls to claude --print."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = CLAUDE_HAIKU_SENTINEL,
        base_url: str | None = None,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize Claude Code CLI provider.

        The CLI natively resolves aliases (``haiku``, ``sonnet``, ``opus``)
        to the latest available model. ChunkHound still honors its own
        ``CHUNKHOUND_CLAUDE_DEFAULT_{HAIKU,SONNET,OPUS}_MODEL`` overrides
        before passing anything to the CLI. Without a ChunkHound override,
        sentinels are mapped to bare aliases so the CLI can stay fresh.
        Explicit full names (e.g. ``claude-sonnet-4-5-20250929``) pass
        through unchanged.

        Args:
            api_key: Not used (subscription-based authentication)
            model: Model sentinel or full name.
                Sentinels: ``claude-haiku``, ``claude-sonnet``, ``claude-opus``
                Full names: ``claude-sonnet-4-5-20250929`` (pinned to version)
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        super().__init__(api_key, model, base_url, timeout, max_retries)
        self._model = resolve_claude_cli_model(model)

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "claude-code-cli"

    def _map_model_to_cli_arg(self, model: str) -> str:
        """Map ChunkHound sentinel to CLI bare alias.

        The Claude Code CLI accepts bare aliases (``haiku``, ``sonnet``,
        ``opus``) and full model names (``claude-sonnet-4-5-20250929``).
        ChunkHound's sentinels are ``claude-``-prefixed; strip that prefix
        for the CLI.  Partial names like ``sonnet-4-5`` are **not** accepted
        by the CLI (only bare aliases or full dated names).

        Args:
            model: Model sentinel or full name.

        Returns:
            CLI-compatible ``--model`` argument.
        """
        sentinel_to_cli = {
            CLAUDE_HAIKU_SENTINEL: "haiku",
            CLAUDE_SONNET_SENTINEL: "sonnet",
            CLAUDE_OPUS_SENTINEL: "opus",
        }
        return sentinel_to_cli.get(model.strip().lower(), model)

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run claude CLI command and return output.

        Args:
            prompt: User prompt
            system: Optional system prompt (appended to default)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        # Resolve via PATH/PATHEXT (Windows: claude.cmd npm shims). Bare "claude"
        # with create_subprocess_exec fails with WinError 2 when only .cmd exists.
        try:
            claude_bin = resolve_cli_binary("claude")
        except FileNotFoundError as e:
            raise RuntimeError(str(e)) from e

        model_arg = self._map_model_to_cli_arg(self._model)

        # Inline JSON on --mcp-config is mangled by Windows cmd.exe / .cmd
        # batch reparse (nested quotes). CLI accepts a file path instead.
        mcp_config_path = _write_empty_mcp_config_file()
        try:
            # Prompt is passed via stdin, not CLI args (avoids ARG_MAX limit).
            cli_args = [
                "--print",
                "--model",
                model_arg,
                "--output-format",
                "text",
                # Empty MCP servers via file path (not inline JSON).
                "--mcp-config",
                str(mcp_config_path),
                "--strict-mcp-config",
                # Prevent session persistence (avoid context bleed between calls).
                "--no-session-persistence",
            ]
            if system:
                cli_args.extend(["--append-system-prompt", system])
            # Deny all tools (flag alone per claude --help). Non-empty token so
            # Windows .cmd reparse cannot garble an empty --tools "" value.
            # Always last so a value-taking parser cannot swallow a following
            # --option (e.g. --append-system-prompt) as a tool name.
            cli_args.append("--disallowedTools")
            cmd = build_cli_argv(claude_bin, *cli_args)

            # Set environment for subscription-based auth
            env = os.environ.copy()
            env["CLAUDE_USE_SUBSCRIPTION"] = "true"

            # Suppress the CLI's auto-updater: updates should be driven by the
            # user's interactive `claude` sessions, not ChunkHound subprocess calls.
            env["DISABLE_AUTOUPDATER"] = "1"

            # Remove ANTHROPIC_API_KEY if present to force subscription auth
            env.pop("ANTHROPIC_API_KEY", None)

            # Use provided timeout or default
            request_timeout = timeout if timeout is not None else self._timeout

            # Run command with retry logic
            last_error = None
            for attempt in range(self._max_retries):
                process = None
                try:
                    # Create subprocess with neutral CWD to prevent workspace scanning
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=subprocess.PIPE,  # Pass prompt via stdin (avoids ARG_MAX)
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                        cwd=tempfile.gettempdir(),  # Cross-platform temp directory
                    )

                    # Wrap communicate() with timeout (this is the long-running part)
                    # Pass prompt via stdin to avoid OS ARG_MAX limits (~256KB on macOS)
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=prompt.encode("utf-8")),
                        timeout=request_timeout,
                    )

                    if process.returncode != 0:
                        raw_err = (stderr or stdout or b"").decode(
                            "utf-8", errors="ignore"
                        )
                        error_msg = (
                            sanitize_error_text(raw_err.strip())
                            or f"Exit code {process.returncode}"
                        )
                        last_error = RuntimeError(
                            f"CLI command failed (exit {process.returncode}): "
                            f"{error_msg}"
                        )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"CLI attempt {attempt + 1} failed, retrying: "
                                f"{error_msg}"
                            )
                            continue
                        raise last_error

                    return stdout.decode("utf-8").strip()

                except asyncio.TimeoutError as e:
                    # Kill cmd.exe + Node/CLI tree when wrapping a .cmd shim
                    if process is not None:
                        try:
                            await terminate_cli_process(process)
                        except ProcessLookupError:
                            pass

                    last_error = RuntimeError(
                        f"CLI command timed out after {request_timeout}s"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"CLI attempt {attempt + 1} timed out, retrying"
                        )
                        continue
                    raise last_error from e

                except Exception as e:
                    if isinstance(e, RuntimeError):
                        raise
                    if process is not None:
                        try:
                            await terminate_cli_process(process)
                        except ProcessLookupError:
                            pass

                    last_error = RuntimeError(f"CLI command failed: {e}")
                    if attempt < self._max_retries - 1:
                        logger.warning(f"CLI attempt {attempt + 1} failed: {e}")
                        continue
                    raise last_error from e

            # Should not reach here, but just in case
            raise last_error or RuntimeError("CLI command failed after retries")
        finally:
            try:
                mcp_config_path.unlink(missing_ok=True)
            except OSError as e:
                logger.debug(
                    "Failed to remove temp MCP config {}: {}",
                    mcp_config_path,
                    e,
                )
