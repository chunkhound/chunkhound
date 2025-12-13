from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def count_mcp_tool_calls(trace_path: Path) -> tuple[int, Counter]:
    """Scan a Codex JSONL trace and count MCP tool calls.

    We stream the file line-by-line to avoid loading it fully into memory.
    Each unique `item.id` with `item.type == "mcp_tool_call"` is counted once,
    and we also aggregate counts by `item.tool`.
    """
    seen_ids: set[str] = set()
    per_tool: Counter = Counter()

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") not in ("item.started", "item.completed"):
                continue

            item = obj.get("item") or {}
            if item.get("type") != "mcp_tool_call":
                continue

            item_id = item.get("id")
            tool_name = item.get("tool")
            if not isinstance(item_id, str):
                continue

            if item_id not in seen_ids:
                seen_ids.add(item_id)
                if isinstance(tool_name, str):
                    per_tool[tool_name] += 1

    return len(seen_ids), per_tool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a Codex JSONL trace and report MCP tool call counts."
    )
    parser.add_argument(
        "trace",
        type=Path,
        help="Path to the *.trace.jsonl file.",
    )
    args = parser.parse_args()

    trace_path: Path = args.trace
    if not trace_path.exists():
        raise SystemExit(f"Trace file not found: {trace_path}")

    total_calls, per_tool = count_mcp_tool_calls(trace_path)

    print(f"Trace file: {trace_path}")
    print(f"Total MCP tool calls (unique items): {total_calls}")
    if per_tool:
        print("Breakdown by tool:")
        for tool, count in sorted(per_tool.items()):
            print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()

