import asyncio
import os
import subprocess
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.windows_constants import get_utf8_env


class AntigravityCLIProvider(BaseCLIProvider):
    """CLI LLM provider wrapping the agy / antigravity CLI command."""

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "antigravity-cli"

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run CLI command and return output.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        request_timeout = timeout if timeout is not None else self._timeout

        import shutil

        binary = "agy"
        if not shutil.which("agy") and shutil.which("antigravity"):
            binary = "antigravity"

        # Build CLI command
        cmd = [binary, "--print", "--sandbox"]
        if self._model:
            cmd.extend(["--model", self._model])

        # Merge system prompt if provided
        merged_prompt = self._merge_prompts(prompt, system)

        # Clone and sanitize environment variables to prevent credentials/plugin hijacking
        safe_keys = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "USERPROFILE",
            "TMPDIR",
            "TMP",
            "TEMP",
            "TERM",
            "SHELL",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "LC_MESSAGES",
            "SystemRoot",
            "SystemDrive",
            "ComSpec",
            "PATHEXT",
            "WINDIR",
            "APPDATA",
            "LOCALAPPDATA",
        }
        base_env = {k: v for k, v in os.environ.items() if k in safe_keys}
        env = get_utf8_env(base_env)

        logger.debug(f"Executing CLI command: {' '.join(cmd)} in sandboxed mode")

        process = None
        try:
            # Create subprocess with neutral CWD to prevent local config scans
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=tempfile.gettempdir(),
            )

            # Wait for completion and stream prompt via stdin
            # (bypasses OS arg length limits)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=merged_prompt.encode("utf-8")),
                timeout=request_timeout,
            )

            if process.returncode != 0:
                raw_err = (
                    (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
                )
                logger.error(
                    "Antigravity CLI command failed "
                    f"(exit code {process.returncode}): {raw_err}"
                )
                raise RuntimeError(raw_err or f"Exit code {process.returncode}")

            return stdout.decode("utf-8", errors="ignore")

        except FileNotFoundError as fnf:
            logger.error(f"Antigravity CLI binary not found: {fnf}")
            raise RuntimeError(
                "Antigravity CLI binary ('agy' or 'antigravity') not found. "
                "Ensure it is installed and configured in your system PATH."
            ) from fnf
        except asyncio.TimeoutError as te:
            logger.error(f"Antigravity CLI command timed out after {request_timeout}s")
            if process:
                try:
                    process.kill()
                    await process.wait()
                except Exception as kill_err:
                    logger.warning(
                        f"Failed to kill timed out Antigravity CLI process: {kill_err}"
                    )
            raise RuntimeError(
                f"Antigravity CLI command timed out after {request_timeout}s"
            ) from te
        except Exception as e:
            if not isinstance(e, RuntimeError):
                logger.error(f"Antigravity CLI execution failed: {e}")
                raise RuntimeError(f"Antigravity CLI execution failed: {e}") from e
            raise

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations."""
        return 1
