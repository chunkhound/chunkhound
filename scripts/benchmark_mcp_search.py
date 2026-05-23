#!/usr/bin/env python3
"""Benchmark ChunkHound MCP search latency via stdio.

Spawns `uv run chunkhound mcp stdio` in the current working directory,
performs the MCP handshake, then runs timed search calls to produce
machine-local advisory latency figures.

Usage:
    uv run python scripts/benchmark_mcp_search.py

The codebase must already be indexed.  Run `uv run chunkhound index .` first
if needed.
"""

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Allow importing the shared JSON-RPC client without installing the test package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.subprocess_jsonrpc import SubprocessJsonRpcClient
from tests.utils.windows_subprocess import (
    create_subprocess_exec_safe,
    get_safe_subprocess_env,
)

# ── config ──────────────────────────────────────────────────────────────────
WARMUP_RUNS = 3
MEASURED_RUNS = 10
INIT_TIMEOUT = 30.0   # first call may wait for services to start
SEARCH_TIMEOUT = 15.0
SETTLE_DELAY = 2.0

REGEX_QUERIES = [
    "def complete",
    "RuntimeError",
    "async def",
    "import asyncio",
]

SEMANTIC_QUERIES = [
    "error handling in LLM providers",
    "database connection management",
    "MCP tool registration",
    "embedding vector search",
]
# ────────────────────────────────────────────────────────────────────────────


class BenchmarkStartupError(RuntimeError):
    """Raised when the benchmark cannot establish a valid MCP search session."""


def _launch_args() -> tuple[str, ...]:
    return ("uv", "run", "chunkhound", "mcp", "stdio")


async def _launch_mcp_server(cwd: Path) -> asyncio.subprocess.Process:
    return await create_subprocess_exec_safe(
        *_launch_args(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=get_safe_subprocess_env(os.environ.copy()),
    )


def _require_search_tool(tools_result: dict[str, object]) -> list[str]:
    raw_tools = tools_result.get("tools")
    if not isinstance(raw_tools, list):
        raise BenchmarkStartupError(
            "tools/list response did not contain a valid tools array."
        )

    tools: dict[str, dict[str, Any]] = {
        tool["name"]: tool
        for tool in raw_tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }
    if "search" not in tools:
        raise BenchmarkStartupError(
            "'search' tool not found in tools/list response; "
            "benchmark requires MCP search."
        )

    search_schema = tools["search"].get("inputSchema", {})
    if not isinstance(search_schema, dict):
        return ["regex"]

    search_properties = search_schema.get("properties", {})
    if not isinstance(search_properties, dict):
        return ["regex"]

    search_type = search_properties.get("type", {})
    if not isinstance(search_type, dict):
        return ["regex"]

    raw_enum = search_type.get("enum", ["regex"])
    if not isinstance(raw_enum, list) or not all(
        isinstance(item, str) for item in raw_enum
    ):
        return ["regex"]

    return raw_enum


async def _settle_before_measurement(delay_seconds: float = SETTLE_DELAY) -> None:
    """Let stdio startup and caches settle before steady-state timing begins."""
    await asyncio.sleep(delay_seconds)


async def _handshake(client: SubprocessJsonRpcClient) -> list[str]:
    """MCP initialize + tools/list.  Returns available search type enum values."""
    await client.send_request(
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "benchmark", "version": "1.0"},
        },
        timeout=INIT_TIMEOUT,
    )
    await client.send_notification("notifications/initialized")

    result = await client.send_request("tools/list", timeout=INIT_TIMEOUT)
    return _require_search_tool(result)


async def _timed_call(
    client: SubprocessJsonRpcClient, search_type: str, query: str
) -> float:
    """Return wall-clock milliseconds for one search call."""
    t0 = time.perf_counter()
    await client.send_request(
        "tools/call",
        {
            "name": "search",
            "arguments": {"type": search_type, "query": query, "page_size": 10},
        },
        timeout=SEARCH_TIMEOUT,
    )
    return (time.perf_counter() - t0) * 1000


async def _run_queries(
    client: SubprocessJsonRpcClient,
    queries: list[str],
    search_type: str,
) -> dict[str, list[float]]:
    """Warm up then measure each query. Returns {query: [times_ms]}."""
    results: dict[str, list[float]] = {}
    for query in queries:
        for _ in range(WARMUP_RUNS):
            await _timed_call(client, search_type, query)
        results[query] = [
            await _timed_call(client, search_type, query)
            for _ in range(MEASURED_RUNS)
        ]
    return results


def _pct(data: list[float], p: float) -> float:
    s = sorted(data)
    idx = p / 100 * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _print_block(label: str, times: list[float]) -> None:
    print(
        f"  {label:<18}  "
        f"mean={statistics.mean(times):6.1f}ms  "
        f"p50={_pct(times,50):6.1f}ms  "
        f"p95={_pct(times,95):6.1f}ms  "
        f"p99={_pct(times,99):6.1f}ms  "
        f"n={len(times)}"
    )


async def main() -> None:
    cwd = Path.cwd()
    print(f"Target directory : {cwd}")
    print(f"Warmup / measured: {WARMUP_RUNS} / {MEASURED_RUNS} runs per query\n")
    print("Note: benchmark numbers are advisory measurements for this machine only.\n")

    proc = await _launch_mcp_server(cwd)

    client = SubprocessJsonRpcClient(proc)
    await client.start()

    try:
        print("Connecting to MCP server…")
        available = await _handshake(client)
        print(f"Search types available: {available}\n")
        print(
            f"Settling for {SETTLE_DELAY:.1f}s before steady-state measurements...\n"
        )
        await _settle_before_measurement()

        regex_flat: list[float] = []
        semantic_flat: list[float] = []
        runs_label = f"{WARMUP_RUNS}+{MEASURED_RUNS}"

        if "regex" in available:
            print(f"Regex  ({len(REGEX_QUERIES)} queries × {runs_label})…")
            r = await _run_queries(client, REGEX_QUERIES, "regex")
            for q, times in r.items():
                print(f"  '{q}': p50={_pct(times, 50):.1f}ms")
                regex_flat.extend(times)

        if "semantic" in available:
            print(f"\nSemantic ({len(SEMANTIC_QUERIES)} queries × {runs_label})…")
            s = await _run_queries(client, SEMANTIC_QUERIES, "semantic")
            for q, times in s.items():
                print(f"  '{q}': p50={_pct(times, 50):.1f}ms")
                semantic_flat.extend(times)

        # ── summary ─────────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("RESULTS")
        print("=" * 72)
        if regex_flat:
            _print_block("regex", regex_flat)
        if semantic_flat:
            _print_block("semantic", semantic_flat)

        # ── suggested ProvenAtScale values ──────────────────────────────────
        print("\n" + "─" * 72)
        print("Suggested ProvenAtScale.astro data-target values (manual input only)")
        print("(bar labels represent codebase scale, not validated claim evidence)")
        print("─" * 72)

        if regex_flat:
            r_p50 = round(_pct(regex_flat, 50))
            regex_p50 = _pct(regex_flat, 50)
            print(
                f'  1K files  row  → data-target="{r_p50}"   '
                f"(regex p50 = {regex_p50:.1f}ms)"
            )
        if semantic_flat:
            s_p50 = round(_pct(semantic_flat, 50))
            s_p95 = round(_pct(semantic_flat, 95))
            semantic_p50 = _pct(semantic_flat, 50)
            semantic_p95 = _pct(semantic_flat, 95)
            print(
                f'  50K files row  → data-target="{s_p50}"  '
                f"(semantic p50 = {semantic_p50:.1f}ms)"
            )
            print(
                f'  50M+ LOC  row  → data-target="{s_p95}"  '
                f"(semantic p95 = {semantic_p95:.1f}ms)"
            )

        print()
        print("These are measurements on THIS machine against THIS codebase.")
        print("Use them as local guidance only, then validate any public scale claims")
        print("with independent evidence before updating site copy.")

    finally:
        await client.close()


def _run_cli() -> None:
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    _run_cli()
