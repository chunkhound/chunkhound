from __future__ import annotations

import contextlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile

from tests.site.process_runner import run_text_process

ROOT = pathlib.Path(__file__).resolve().parents[2]
NPM: str = shutil.which("npm") or "npm"
_SUBPROCESS_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SystemRoot",
    "ComSpec",
    "PATHEXT",
    "APPDATA",
    "LOCALAPPDATA",
)


def _base_subprocess_env(**overrides: str) -> dict[str, str]:
    """Allowlisted host env without the npm cache entry."""
    env = {
        key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ
    }
    env.update(overrides)
    return env


def sanitized_subprocess_env(**overrides: str) -> dict[str, str]:
    """Build a hermetic runtime env for site subprocess tests."""
    env = _base_subprocess_env(**overrides)
    # npm exec reads the host npm cache; a broken/hostile cache must not fail
    # unrelated tests, so every subprocess gets its own private cache dir.
    # Prefer isolated_subprocess_env() for new call sites: it owns the cache
    # dir lifetime via TemporaryDirectory instead of leaving mkdtemp cleanup
    # to the caller.
    env.setdefault("npm_config_cache", tempfile.mkdtemp(prefix="npm-cache-"))
    return env


@contextlib.contextmanager
def isolated_subprocess_env(**overrides: str):
    """Yield a hermetic env whose private npm cache dir is deleted on exit."""
    with tempfile.TemporaryDirectory(prefix="npm-cache-") as cache_dir:
        env = _base_subprocess_env(**overrides)
        env.setdefault("npm_config_cache", cache_dir)
        yield env


def _absolute_site_imports(script: str) -> str:
    """Rewrite './site/...' repo-relative specifiers to absolute file:// URIs.

    The temp script lives outside the repo, so repo-relative specifiers no
    longer resolve against it; absolute URIs target the same files.
    """
    root_uri = ROOT.as_uri()
    return script.replace("'./site/", f"'{root_uri}/site/").replace(
        '"./site/', f'"{root_uri}/site/'
    )


def _run_npm_tsx(
    temp_path: pathlib.Path, env: dict[str, str], timeout: float, check: bool
) -> subprocess.CompletedProcess:
    """Run the temp script via npm exec tsx with a hung-process guard."""
    try:
        return run_text_process(
            [NPM, "exec", "--prefix", "site", "--", "tsx", str(temp_path)],
            cwd=ROOT,
            env=env,
            timeout=timeout,
            check=check,
        )
    # A stalled tsx subprocess must fail the test, not hang CI.
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"tsx run timed out after {timeout}s: {temp_path.name}"
        ) from e


def run_tsx_raw(script: str, **kwargs) -> subprocess.CompletedProcess:
    """Write script to a temp .mts file in the system temp dir and run via npm exec tsx.

    Accepts timeout=, env=, and check=; the rest of the subprocess options are
    fixed here. Output is decoded as UTF-8 (see process_runner). Defaults to an
    isolated env with a per-call npm cache dir (deleted after the run); pass
    env=... to override.
    """
    # Inline -e breaks on Windows: npm.CMD (batch file) treats newlines as
    # command separators, truncating the script to an empty string. The system
    # temp dir is used so read-only checkouts stay runnable.
    # Validate kwargs before writing the temp file so an error path can't leak it.
    timeout = kwargs.pop("timeout", 120)
    check = kwargs.pop("check", False)
    custom_env = kwargs.pop("env", None)
    if kwargs:
        raise TypeError(f"unexpected run_tsx_raw kwargs: {sorted(kwargs)}")
    script = _absolute_site_imports(script)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mts", delete=False, encoding="utf-8"
    ) as f:
        temp_path = pathlib.Path(f.name)
        f.write(script)
    try:
        if custom_env is not None:
            return _run_npm_tsx(temp_path, custom_env, timeout, check)
        # Default path owns its npm cache dir; TemporaryDirectory deletes it.
        with isolated_subprocess_env() as env:
            return _run_npm_tsx(temp_path, env, timeout, check)
    finally:
        temp_path.unlink(missing_ok=True)


def run_tsx_json(script: str) -> dict:
    """Execute a repo-local tsx snippet from the site workspace and parse JSON."""
    return json.loads(run_tsx_raw(script, check=True).stdout)
