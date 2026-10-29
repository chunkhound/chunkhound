import asyncio
import json
import shutil

import pytest

from chunkhound.hub.inspect import inspect_database
from tests.hub.activation_support import cli, cli_env, publish, write_config
from tests.hub.artifacts import build_database
from tests.utils import SubprocessJsonRpcClient


def source_text(marker):
    # Distinct, sizeable functions avoid a tiny-file/chunk coalescing fixture.
    return "\n\n".join(
        f'def function_{i}():\n'
        f'    """{marker}: document a meaningful example with independent inputs, '
        f'validation, returned values and stable error handling for operation {i}."""\n'
        f'    return "{marker}_{i}"\n'
        for i in range(100)
    )


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
def test_moved_checkout_overlays_unchanged_new_changed_and_deleted_files(
    tmp_path, hub_http, embedding_http, backend, rust
):
    publisher = tmp_path / "publisher-checkout"
    write_config(publisher, backend, None, embedding_http.url)
    for filename, marker in [("unchanged.py", "UNCHANGED_BASE"),
                             ("changed.py", "OLD_CHANGED_BASE"),
                             ("deleted.py", "DELETED_BASE")]:
        (publisher / filename).write_text(source_text(marker), encoding="utf-8")
    built = cli(publisher, "index", publisher, rust=rust)
    assert built.returncode == 0, built.stdout + built.stderr
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    publisher_target = publisher / ".chunkhound" / "db" / name
    publish(hub_http, publisher_target)
    checkout = tmp_path / "different-parent" / "moved-checkout"
    write_config(checkout, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    for filename in ("unchanged.py", "changed.py", "deleted.py"):
        shutil.copy2(publisher / filename, checkout / filename)
    pulled = cli(checkout, "pull", checkout, rust=rust)
    assert pulled.returncode == 0, pulled.stdout + pulled.stderr
    before = cli(checkout, "search", "UNCHANGED_BASE", checkout, "--regex", rust=rust)
    assert before.returncode == 0 and "unchanged.py" in before.stdout + before.stderr
    (checkout / "changed.py").write_text(source_text("NEW_CHANGED_LOCAL"), encoding="utf-8")
    (checkout / "new.py").write_text(source_text("NEW_LOCAL_FILE"), encoding="utf-8")
    (checkout / "deleted.py").unlink()
    embedding_http.requests.clear()
    hub_http.requests.clear()
    overlaid = cli(checkout, "index", checkout, "--verbose", rust=rust)
    assert overlaid.returncode == 0, overlaid.stdout + overlaid.stderr
    assert hub_http.requests == []  # Aligned base, no second pull.
    assert embedding_http.requests
    embedded_text = json.dumps(embedding_http.requests)
    assert "UNCHANGED_BASE" not in embedded_text  # Real hash-based unchanged skip.
    assert "NEW_CHANGED_LOCAL" in embedded_text
    assert "NEW_LOCAL_FILE" in embedded_text
    for marker, filename in [("UNCHANGED_BASE", "unchanged.py"),
                             ("NEW_CHANGED_LOCAL", "changed.py"),
                             ("NEW_LOCAL_FILE", "new.py")]:
        result = cli(checkout, "search", marker, checkout, "--regex", rust=rust)
        assert result.returncode == 0, result.stdout + result.stderr
        assert filename in result.stdout + result.stderr
        assert "No results found" not in result.stdout + result.stderr
    for marker in ("OLD_CHANGED_BASE", "DELETED_BASE"):
        result = cli(checkout, "search", marker, checkout, "--regex", rust=rust)
        assert result.returncode == 0
        assert "No results found" in result.stdout + result.stderr
    semantic = cli(checkout, "search", "UNCHANGED_BASE", checkout,
                   "--single-hop", rust=rust)
    assert semantic.returncode == 0, semantic.stdout + semantic.stderr
    assert "No results found" not in semantic.stdout + semantic.stderr
    assert hub_http.requests == []
    assert {g.model for g in inspect_database(
        checkout / ".chunkhound" / "db" / name, backend
    ).groups} == {"hub-test"}


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_mcp_and_search_do_not_pull_even_when_configured_model_differs(
    tmp_path, hub_http, embedding_http, backend
):
    project = tmp_path / "checkout"
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published", backend))
    pulled = await asyncio.to_thread(cli, project, "pull", project)
    assert pulled.returncode == 0, pulled.stdout + pulled.stderr
    # Keep the source represented by the synthetic base available to MCP indexing.
    (project / "source.py").write_text(source_text("MCP_LOCAL"), encoding="utf-8")
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url, model="different-configured-model")
    hub_http.requests.clear()
    searched = await asyncio.to_thread(cli, project, "search", "function_0", project,
                                       "--regex")
    assert searched.returncode == 0, searched.stdout + searched.stderr
    assert hub_http.requests == []
    process = await asyncio.create_subprocess_exec(
        "uv", "run", "chunkhound", "mcp", str(project), "--no-daemon",
        cwd=project, env=cli_env(project), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(process.stderr.read())
    client = SubprocessJsonRpcClient(process)
    await client.start()
    try:
        initialized = await client.send_request(
            "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "hub-acceptance", "version": "1"}},
            timeout=60,
        )
        assert "serverInfo" in initialized
        await client.send_notification("notifications/initialized", {})
        result = await client.send_request(
            "tools/call", {"name": "search", "arguments":
                           {"type": "regex", "query": "function_0"}}, timeout=60,
        )
        assert not result.get("isError", False), result
        assert hub_http.requests == []
    finally:
        await client.close()
        await stderr_task


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
def test_interrupted_auto_refuses_and_explicit_repull_restores_search(
    tmp_path, hub_http, embedding_http, backend, rust
):
    project = tmp_path / "checkout"
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    (project / "local.py").write_text("def local_reindex_work(): return 1\n")
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    target = project / ".chunkhound" / "db" / name
    target.parent.mkdir(parents=True)
    marker = target.with_name(target.name + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    publish(hub_http, build_database(tmp_path / "published", backend))
    refused = cli(project, "index", project, "--verbose", rust=rust)
    assert refused.returncode == 1
    assert "explicit chunkhound pull" in refused.stderr
    assert "Traceback" not in refused.stdout + refused.stderr
    assert hub_http.requests == []
    assert marker.read_bytes() == b"{" and not target.exists()
    restored = cli(project, "pull", project, rust=rust)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert not marker.exists()
    result = cli(project, "search", "function_0", project, "--regex", rust=rust)
    assert result.returncode == 0
    assert "source.py" in result.stdout + result.stderr
    assert "No results found" not in result.stdout + result.stderr
