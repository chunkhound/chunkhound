"""Websearch command for ChunkHound CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.error

from chunkhound.core.config.config import Config
from chunkhound.utils.websearch_core import websearch_timeout
from chunkhound.utils.websearch_pipeline import (
    EmptyWebSearchError,
    NoPagesFetchedError,
    run_websearch,
)

from ..utils.provider_setup import setup_embedding_manager, setup_llm_manager
from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


async def websearch_command(args: argparse.Namespace, config: Config) -> None:
    """Fetch DuckDuckGo results for the given query."""
    # Empty-string → None coercion at the surface boundary, so every
    # downstream consumer sees the two-valued "real string or None" contract.
    previous_query = args.previous_query or None
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))
    embedding_manager = setup_embedding_manager(formatter, config)
    llm_manager = setup_llm_manager(formatter, config)

    def _warning_to_stderr(message: str) -> None:
        print(f"[WARN] {message}", file=sys.stderr)

    def _on_query_failure(q: str, e: urllib.error.URLError) -> None:
        _warning_to_stderr(
            f"DDG query failed ({q!r}): {e.reason}; continuing with remaining queries"
        )

    def _progress_to_stderr(message: str) -> None:
        print(f"[PROGRESS] {message}", file=sys.stderr)

    timeout_s = websearch_timeout()
    try:
        # Progress is append-only noise; keep stdout for the answer so
        # piped `chunkhound websearch` stays machine-readable.
        with TreeProgressDisplay(output=sys.stderr) as research_progress:
            result, warnings = await run_websearch(
                args.query,
                limit=args.limit,
                config=config,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                previous_query=previous_query,
                progress_callback=_progress_to_stderr,
                progress=research_progress,
                failure_callback=_on_query_failure,
                timeout_s=timeout_s,
            )
    except EmptyWebSearchError as e:
        formatter.error(str(e))
        # 10 = empty results (distinct from 1=fetch/research error, 124=timeout).
        sys.exit(10)
    except urllib.error.URLError as e:
        formatter.error(f"Web search failed: {e.reason}")
        sys.exit(1)
    except (TimeoutError, asyncio.TimeoutError) as e:
        formatter.error(str(e))
        sys.exit(124)
    except NoPagesFetchedError as e:
        formatter.error(str(e))
        sys.exit(1)
    except Exception as exc:
        formatter.error(f"Research failed: {exc}")
        sys.exit(1)
    for warning in warnings:
        _warning_to_stderr(warning)
    formatter.text_block(str(result.get("answer", "")))
