"""Websearch command for ChunkHound CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.error
from collections.abc import AsyncIterator

from chunkhound.core.config.config import Config
from chunkhound.services.web_research_service import Page, research_web_pages
from chunkhound.utils.websearch_core import (
    fetch_pages,
    search_multi,
    websearch_timeout,
)
from chunkhound.utils.websearch_expansion import expand_web_queries

from ..utils.provider_setup import setup_embedding_manager, setup_llm_manager
from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


async def websearch_command(args: argparse.Namespace, config: Config) -> None:
    """Fetch DuckDuckGo results for the given query."""
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))
    embedding_manager = setup_embedding_manager(formatter, config)
    llm_manager = setup_llm_manager(formatter, config)

    def _on_query_failure(q: str, e: urllib.error.URLError) -> None:
        formatter.warning(
            f"DDG query failed ({q!r}): {e.reason}; continuing with remaining queries"
        )

    timeout_s = websearch_timeout()

    async def _run() -> tuple[dict, list[str]] | None:
        queries = await expand_web_queries(args.query, llm_manager)
        results = await search_multi(
            queries,
            args.limit,
            formatter.progress_indicator,
            failure_callback=_on_query_failure,
        )
        if not results:
            formatter.error(
                f"No results found for {args.query!r} — "
                "DDG HTML structure may have changed"
            )
            return None
        formatter.progress_indicator(
            f"Found {len(results)} results, fetching content..."
        )
        warnings: list[str] = []
        got_page = False

        async def pages() -> AsyncIterator[Page]:
            nonlocal got_page
            async for page in fetch_pages(
                [url for _, url, _ in results],
                formatter.progress_indicator,
                warnings.append,
            ):
                got_page = True
                yield page

        with TreeProgressDisplay(output=sys.stdout) as research_progress:
            result = await research_web_pages(
                args.query,
                pages(),
                config,
                embedding_manager,
                llm_manager,
                progress=research_progress,
                warning_callback=warnings.append,
            )
        if not got_page:
            formatter.error(f"No pages could be fetched for {args.query!r}")
            raise RuntimeError("no pages fetched")
        return result, warnings

    try:
        outcome = await asyncio.wait_for(_run(), timeout=timeout_s)
    except urllib.error.URLError as e:
        formatter.error(f"Web search failed: {e.reason}")
        sys.exit(1)
    except asyncio.TimeoutError:
        formatter.error(f"websearch timed out after {timeout_s:.0f}s")
        sys.exit(124)
    except Exception as exc:
        formatter.error(f"Research failed: {exc}")
        sys.exit(1)
    if outcome is None:
        # 10 = empty results (distinct from 1=fetch/research error, 124=timeout).
        sys.exit(10)
    result, warnings = outcome
    for warning in warnings:
        formatter.warning(warning)
    formatter.text_block(str(result.get("answer", "")))
