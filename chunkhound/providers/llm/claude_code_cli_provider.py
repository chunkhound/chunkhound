"""Claude Code CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the Claude Code CLI (claude --print) to enable deep research
using the user's existing Claude subscription instead of API credits.

Note: This provider is configured for vanilla LLM behavior:
- All tools disabled via --tools ""
- MCP servers disabled via empty --mcp-config
- Workspace isolation (runs from temp directory to prevent context gathering)
- Clean API access without workspace overhead
"""

import asyncio
import os
import subprocess
import tempfile

from loguru import logger

from chunkhound.core.config.claude_model_resolution import (
    CLAUDE_HAIKU_SENTINEL,
    CLAUDE_OPUS_SENTINEL,
    CLAUDE_SONNET_SENTINEL,
    resolve_claude_cli_model,
)
from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.text_sanitization import sanitize_error_text


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
        # Build CLI command
        model_arg = self._map_model_to_cli_arg(self._model)
        cmd = ["claude", "--print", "--model", model_arg, "--output-format", "text"]

        # Disable all tools for vanilla LLM behavior.
        cmd.extend(["--tools", ""])

        # Prevent MCP server loading for clean LLM access
        cmd.extend(["--mcp-config", '{"mcpServers":{}}'])
        cmd.append("--strict-mcp-config")  # Ignore user/project MCP configs

        # Prevent session persistence (avoid context bleed between calls)
        cmd.append("--no-session-persistence")

        # Add system prompt if provided (appends to default)
        if system:
            cmd.extend(["--append-system-prompt", system])

        # Note: prompt is passed via stdin, not CLI args (avoids ARG_MAX limit)

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
                    raw_err = (stderr or stdout or b"").decode("utf-8", errors="ignore")
                    error_msg = (
                        sanitize_error_text(raw_err.strip())
                        or f"Exit code {process.returncode}"
                    )
                    last_error = RuntimeError(
                        f"CLI command failed (exit {process.returncode}): {error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"CLI attempt {attempt + 1} failed, retrying: {error_msg}"
                        )
                        continue
                    raise last_error

                return stdout.decode("utf-8").strip()

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(
                    f"CLI command timed out after {request_timeout}s"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} timed out, retrying")
                    continue
                raise last_error from e

            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(f"CLI command failed: {e}")
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("CLI command failed after retries")
