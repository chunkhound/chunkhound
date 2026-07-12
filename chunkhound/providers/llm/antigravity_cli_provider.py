import asyncio
from typing import Any
import os
import shutil
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

        binary_path = shutil.which("agy")
        if not binary_path:
            binary_path = shutil.which("antigravity")

        if not binary_path:
            # Raise the actionable guidance directly. BaseCLIProvider.complete()
            # re-raises RuntimeError unchanged, so the user sees this message;
            # the shutil.which lookup happens before the try block below, so the
            # in-loop `except FileNotFoundError` handler cannot cover this case.
            raise RuntimeError(
                "Antigravity CLI binary ('agy' or 'antigravity') not found. "
                "Ensure it is installed and configured in your system PATH."
            )

        # Merge system prompt if provided
        merged_prompt = self._merge_prompts(prompt, system)

        # Build CLI command. The prompt is delivered via stdin and NOT via the
        # --print flag. agy v1.1.1 changed print-mode input handling: when a
        # prompt is supplied through a flag it "no longer reads stdin" (changelog
        # 1.1.1), and --print/-p consume the following argv token as their value.
        # So including --print here both (a) leaks the prompt into process
        # listings and (b) makes agy swallow the next flag (e.g. --model) as the
        # prompt. Omitting --print lets agy auto-detect the piped, non-TTY stdin
        # and run headless, reading the prompt from stdin — keeping source
        # snippets, paths, and secrets out of argv/`ps` and avoiding ARG_MAX
        # limits, consistent with the other CLI providers.
        cmd = [binary_path, "--sandbox"]
        if self._model:
            cmd.extend(["--model", self._model])

        # Clone and sanitize environment variables to prevent credentials/plugin hijacking
        safe_keys = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "USERPROFILE",
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

        process = None
        captured_process_pid: int | None = None
        temp_dir = tempfile.mkdtemp(prefix="chunkhound-antigravity-")

        # Note: We intentionally preserve HOME, USERPROFILE, APPDATA, and LOCALAPPDATA
        # rather than redirecting them to the temp directory. While this exposes user-level
        # configuration to the CLI (a documented sandboxing tradeoff), it is strictly
        # required for the CLI to locate its authentication credentials.

        # The prompt is delivered via stdin, so it is not part of cmd and cannot
        # leak through this log. Only the binary and flags are logged; the prompt
        # is never logged (it can carry source snippets, paths, or secrets from
        # the user's workspace) and is summarized by length.
        logger.debug(
            f"Executing CLI command: {' '.join(cmd)} "
            f"(prompt via stdin: {len(merged_prompt)} chars) in sandboxed mode"
        )
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
                    cwd=temp_dir,
                    creationflags=create_new_process_group,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=temp_dir,
                    start_new_session=True,
                )

            if sys.platform != "win32":
                captured_process_pid = process.pid

            # Wait for completion. The prompt is written to stdin (agy reads the
            # piped, non-TTY stdin as the prompt when no prompt flag is given);
            # communicate() writes it, closes stdin (EOF), and drains stdout/stderr.
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
            try:
                if process and process.returncode is None:
                    await asyncio.shield(
                        self._kill_process_tree(
                            process, pgid=captured_process_pid, env=env
                        )
                    )
            finally:
                # Runs even if the shielded await above re-raises CancelledError
                # (cancellation during process-tree teardown), so the per-call temp
                # working directory is never leaked. rmtree with ignore_errors=True
                # is synchronous and cannot itself raise.
                shutil.rmtree(temp_dir, ignore_errors=True)

    async def _kill_process_tree(
        self,
        process: asyncio.subprocess.Process,
        *,
        pgid: int | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        """Terminate an antigravity subprocess and its descendants."""
        if sys.platform == "win32":
            taskkill_success = False
            try:
                taskkill_path = shutil.which("taskkill") or "taskkill"
                res = await asyncio.to_thread(
                    subprocess.run,
                    [taskkill_path, "/T", "/PID", str(process.pid), "/F"],
                    capture_output=True,
                    check=False,
                    timeout=10,
                    env=env,
                )
                if res.returncode != 0:
                    logger.warning(
                        "Windows taskkill exited with code "
                        f"{res.returncode}: "
                        f"{res.stderr.decode('utf-8', errors='ignore')}"
                    )
                else:
                    taskkill_success = True
            except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
                logger.debug(
                    f"Windows taskkill failed during Antigravity CLI cleanup: {e}"
                )

            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

            if not taskkill_success:
                try:
                    import psutil

                    try:
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        parent.kill()
                    except psutil.NoSuchProcess:
                        pass
                except ImportError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            return

        process_group_id = pgid if pgid is not None else process.pid
        if isinstance(process_group_id, (int, float)) and process_group_id <= 1:
            logger.warning(
                f"Invalid process group ID for termination: {process_group_id}. "
                "Skipping group kill."
            )
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return

        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass

    async def health_check(self) -> dict[str, Any]:
        """Perform health check by attempting a simple completion.

        This overrides the base class implementation to avoid triggering the
        max_completion_tokens warning.
        """
        try:
            response = await self.complete(
                "Say 'OK'",
                max_completion_tokens=4096,
                timeout=self.HEALTH_CHECK_TIMEOUT,
            )
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
            }

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations."""
        return 1
