"""Observable CI evidence contracts, using real subprocess/JSON-RPC boundaries."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.windows_compat import windows_safe_tempdir


def test_collection_and_mcp_contracts_work_without_zendriver():
    # A fresh interpreter prevents already-imported zendriver from hiding an
    # eager dependency import. No packages or environment files are changed.
    script = """
import importlib.abc
import sys

class MissingZendriver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'zendriver' or fullname.startswith('zendriver.'):
            raise ModuleNotFoundError("No module named 'zendriver'", name=fullname)

sys.meta_path.insert(0, MissingZendriver())
import chunkhound.utils.websearch_core as websearch
websearch._CHROME_PATHS = []
import pytest
sys.exit(pytest.main([
    '-q', '-p', 'no:cacheprovider',
    'tests/test_startup_diagnostics.py',
    'tests/integration/test_websearch_fetch_page.py',
    '-k', 'not without_zendriver',
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout  # MCP contracts must run, not skip.
    assert "skipped" in result.stdout


@pytest.mark.asyncio
async def test_mcp_failure_keeps_bounded_evidence_before_temp_cleanup(capsys):
    from tests.utils.startup_diagnostics import diagnosed_mcp_client

    failure = RuntimeError("original startup failure")
    with windows_safe_tempdir() as temp:
        debug = temp / "debug.log"
        debug.write_text("discard-debug-prefix" + "d" * 40000 + "DEBUG-END")
        script = (
            "import json,sys; "
            "sys.stderr.write('discard-stderr-prefix' + 'x' * 200000 + 'STDERR-END'); "
            "sys.stderr.flush(); "
            "req=json.loads(sys.stdin.readline()); "
            "print(json.dumps({'jsonrpc':'2.0','id':req['id'],"
            "'result':{'serverInfo':{'name':'real-child'}}}),flush=True); "
            "sys.stdin.read()"
        )
        with pytest.raises(RuntimeError) as caught:
            async with diagnosed_mcp_client(
                [sys.executable, "-u", "-c", script], debug_file=debug
            ) as client:
                result = await client.send_request("initialize", {}, timeout=10.0)
                assert result == {"serverInfo": {"name": "real-child"}}
                raise failure
        assert caught.value is failure
    assert not debug.exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    evidence = captured.err
    assert "stderr tail:\n" in evidence
    assert "existing library-read tail" not in evidence
    assert "STDERR-END" in evidence and "DEBUG-END" in evidence
    assert "discard-stderr-prefix" not in evidence
    assert "discard-debug-prefix" not in evidence
    assert "original startup failure" in evidence
    assert "pid=" in evidence and "returncode=None" in evidence
    assert "launch.begin" in evidence and "initialize.end" in evidence
    assert "elapsed=" in evidence
    assert len(evidence.encode()) < 60000


@pytest.mark.asyncio
async def test_mcp_launch_failure_reports_original_exception(capsys, tmp_path):
    from tests.utils.startup_diagnostics import diagnosed_mcp_client

    with pytest.raises(FileNotFoundError):
        async with diagnosed_mcp_client([str(tmp_path / "missing-program")]):
            pytest.fail("missing executable must not start")
    evidence = capsys.readouterr().err
    assert "FileNotFoundError" in evidence
    assert "launch.begin" in evidence
    assert "pid=None" in evidence


@pytest.mark.asyncio
async def test_mcp_success_leaves_no_diagnostic_reader_or_output(capsys):
    from tests.utils.startup_diagnostics import diagnosed_mcp_client

    existing = set(asyncio.all_tasks())
    async with diagnosed_mcp_client(
        [sys.executable, "-u", "-c", "import sys; sys.stdin.read()"]
    ):
        pass
    assert set(asyncio.all_tasks()) <= existing
    assert capsys.readouterr() == ("", "")


@pytest.mark.asyncio
async def test_chrome_failure_reports_existing_probes_stderr_and_exit(capsys):
    pytest.importorskip("zendriver")
    from tests.utils.chrome_diagnostics import diagnosed_chrome

    # Python is a real executable but rejects Chrome arguments, so zendriver's
    # real launch and existing HTTP probes fail without requiring Chrome.
    with pytest.raises(Exception, match="Failed to connect to browser"):
        async with diagnosed_chrome(sys.executable):
            pytest.fail("non-Chrome executable must not become ready")
    evidence = capsys.readouterr().err
    assert "existing library-read tail, potentially incomplete:\n" in evidence
    assert sys.executable in evidence
    assert "probe.begin" in evidence and "probe.error" in evidence
    assert "URLError" in evidence
    assert "unknown option" in evidence or "Unknown option" in evidence
    assert "returncode=2" in evidence
    assert "before-stop" in evidence
    assert "elapsed=" in evidence
    assert len(evidence.encode()) < 60000


@pytest.mark.asyncio
async def test_chrome_failure_retains_existing_version_probe(capsys, monkeypatch):
    pytest.importorskip("zendriver")
    import chunkhound.utils.websearch_core as websearch
    from tests.utils.chrome_diagnostics import diagnosed_chrome, resolve_chrome_for_test

    original_subprocess = websearch.subprocess
    monkeypatch.setattr(websearch, "_CHROME_PATHS", [sys.executable])
    # True external version-command boundary; no extra probe is needed to report it.
    monkeypatch.setattr(
        websearch.subprocess, "check_output", lambda *a, **kw: "Chrome 140.0.123\n"
    )
    path, version = resolve_chrome_for_test()
    assert websearch.subprocess is original_subprocess
    with pytest.raises(Exception, match="Failed to connect to browser"):
        async with diagnosed_chrome(path, version=version):
            pytest.fail("non-Chrome executable must not become ready")
    assert "Chrome 140.0.123" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_chrome_launch_exception_survives_cleanup_and_restores_hooks(
    capsys, monkeypatch
):
    pytest.importorskip("zendriver")
    from zendriver.core import browser, util

    from tests.utils.chrome_diagnostics import diagnosed_chrome

    failure = OSError("original launch failure")
    original_util = browser.util

    def fail_launch(*args):
        raise failure

    monkeypatch.setattr(util, "_start_process", fail_launch)
    with pytest.raises(OSError) as caught:
        async with diagnosed_chrome(sys.executable):
            pytest.fail("launch should fail")
    assert caught.value is failure
    assert browser.util is original_util
    evidence = capsys.readouterr().err
    assert "original launch failure" in evidence
    assert "launch.begin" in evidence and "pid=None" in evidence


@pytest.mark.asyncio
@pytest.mark.parametrize("first_exit", ["failure", "cancelled"])
async def test_chrome_overlapping_start_rejected_and_next_start_recovers(
    monkeypatch, first_exit
):
    pytest.importorskip("zendriver")
    from zendriver.core import browser, util

    from tests.utils.chrome_diagnostics import diagnosed_chrome

    originals = browser.Browser, browser.HTTPApi, browser.util
    original_launch = util._start_process
    launched = asyncio.Event()

    def launch(*args):
        process = original_launch(*args)
        launched.set()
        return process

    monkeypatch.setattr(util, "_start_process", launch)

    async def start():
        async with diagnosed_chrome(sys.executable):
            pytest.fail("non-Chrome executable must not become ready")

    first = asyncio.create_task(start())
    try:
        await asyncio.wait_for(launched.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="Overlapping diagnosed Chrome starts"):
            await start()
        if first_exit == "cancelled":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            with pytest.raises(Exception, match="Failed to connect to browser"):
                await first
        assert (browser.Browser, browser.HTTPApi, browser.util) == originals
        # Rejection and the first startup's failure/cancellation must not leave
        # hooks or a busy guard behind: the next real launch reaches zendriver.
        with pytest.raises(Exception, match="Failed to connect to browser"):
            await start()
        assert (browser.Browser, browser.HTTPApi, browser.util) == originals
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        # Keep a failed regression run from contaminating unrelated contracts.
        browser.Browser, browser.HTTPApi, browser.util = originals


@pytest.mark.asyncio
async def test_chrome_verbose_failure_is_bounded_even_with_cleanup_error(
    capsys, monkeypatch
):
    pytest.importorskip("zendriver")
    from zendriver.core import browser, util

    from tests.utils.chrome_diagnostics import diagnosed_chrome

    original_stop = browser.Browser.stop
    original_browser = browser.Browser
    original_http = browser.HTTPApi
    process = None

    def launch(*args):
        nonlocal process
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; "
                "sys.stderr.write('discard-prefix' + 'x'*60000 + 'CHROME-END')",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process

    async def broken_cleanup(self):
        await original_stop(self)
        raise OSError("cleanup broke")

    monkeypatch.setattr(util, "_start_process", launch)
    monkeypatch.setattr(browser.Browser, "stop", broken_cleanup)
    # The library itself raises this cleanup exception before constructing its
    # generic connection error. It must not be replaced by diagnostic cleanup.
    with pytest.raises(OSError, match="cleanup broke"):
        async with diagnosed_chrome(sys.executable):
            pytest.fail("non-Chrome executable must not become ready")
    assert process.poll() is not None
    assert browser.Browser is original_browser and browser.HTTPApi is original_http
    evidence = capsys.readouterr().err
    assert "CHROME-END" in evidence and "discard-prefix" not in evidence
    assert "cleanup broke" in evidence
    assert len(evidence.encode()) < 60000
