import asyncio
import os
import signal
import subprocess
import sys
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.text_sanitization import sanitize_error_text
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
            max_completion_tokens: Maximum tokens to generate.
                Note: Unsupported by the underlying CLI and ignored.
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        if max_completion_tokens is not None and max_completion_tokens != 4096:
            logger.warning(
                "Antigravity CLI does not support limiting output tokens "
                "via max_completion_tokens. "
                f"Requested limit of {max_completion_tokens} is ignored."
            )

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
        captured_process_pid: int | None = None
        try:
            # Create subprocess with neutral CWD to prevent local config scans
            if sys.platform == "win32":
                create_new_process_group = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=tempfile.gettempdir(),
                    creationflags=create_new_process_group,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=tempfile.gettempdir(),
                    start_new_session=True,
                )

            if sys.platform != "win32":
                captured_process_pid = process.pid

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
                sanitized_err = sanitize_error_text(raw_err)
                logger.error(
                    "Antigravity CLI command failed "
                    f"(exit code {process.returncode}): {sanitized_err}"
                )
                raise RuntimeError(sanitized_err or f"Exit code {process.returncode}")

            return stdout.decode("utf-8", errors="ignore")

        except FileNotFoundError as fnf:
            logger.error(f"Antigravity CLI binary not found: {fnf}")
            raise RuntimeError(
                "Antigravity CLI binary ('agy' or 'antigravity') not found. "
                "Ensure it is installed and configured in your system PATH."
            ) from fnf
        except asyncio.TimeoutError as te:
            logger.error(f"Antigravity CLI command timed out after {request_timeout}s")
            raise RuntimeError(
                f"Antigravity CLI command timed out after {request_timeout}s"
            ) from te
        except Exception as e:
            if not isinstance(e, RuntimeError):
                logger.error(f"Antigravity CLI execution failed: {e}")
                raise RuntimeError(f"Antigravity CLI execution failed: {e}") from e
            raise
        finally:
            if process and process.returncode is None:
                await self._kill_process_tree(process, pgid=captured_process_pid)

    async def _kill_process_tree(
        self, process: asyncio.subprocess.Process, *, pgid: int | None = None
    ) -> None:
        """Terminate an antigravity subprocess and its descendants."""
        if process.returncode is not None:
            return

        if sys.platform == "win32":
            try:
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/T", "/PID", str(process.pid), "/F"],
                    check=False,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.SubprocessError, OSError):
                logger.debug("Windows taskkill failed during Antigravity CLI cleanup")
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            return

        process_group_id = pgid if pgid is not None else process.pid
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

        if process.returncode is None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations."""
        return 1
