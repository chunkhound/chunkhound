"""Websearch command for ChunkHound CLI."""

from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.error

from chunkhound.core.config.config import Config
from chunkhound.services.web_research_service import research_web_pages
from chunkhound.utils.websearch_core import (
    fetch_pages,
    search_multi,
    websearch_timeout,
)
from chunkhound.utils.websearch_expansion import expand_web_queries

from ..utils.provider_setup import setup_embedding_manager, setup_llm_manager
from ..utils.rich_output import RichOutputFormatter


async def websearch_command(args: argparse.Namespace, config: Config) -> None:
    """Fetch DuckDuckGo results for the given query."""
    formatter = RichOutputFormatter(verbose=getattr(args, "verbose", False))
    embedding_manager = setup_embedding_manager(formatter, config)
    llm_manager = setup_llm_manager(formatter, config)

    def _on_query_failure(q: str, e: urllib.error.URLError) -> None:
        formatter.warning(
            f"DDG query failed ({q!r}): {e.reason}; "
            "continuing with remaining queries"
        )

    try:
        queries = await expand_web_queries(args.query, llm_manager)
        results = await search_multi(
            queries,
            args.limit,
            formatter.progress_indicator,
            failure_callback=_on_query_failure,
        )
    except urllib.error.URLError as e:
        formatter.error(f"Web search failed: {e.reason}")
        sys.exit(1)
    if not results:
        formatter.error(
            f"No results found for {args.query!r} — DDG HTML structure may have changed"
        )
        # 10 = empty results (distinct from 1=fetch/research error, 124=timeout).
        sys.exit(10)
    formatter.progress_indicator(
        f"Found {len(results)} results, fetching content..."
    )
    warnings: list[str] = []
    pages = [
        page
        async for page in fetch_pages(
            [url for _, url, _ in results],
            formatter.progress_indicator,
            warnings.append,
        )
    ]
    for warning in warnings:
        formatter.warning(warning)
    if not pages:
        formatter.error(f"No pages could be fetched for {args.query!r}")
        sys.exit(1)

    timeout_s = websearch_timeout()
    try:
        result = await asyncio.wait_for(
            research_web_pages(
                args.query,
                pages,
                config,
                embedding_manager,
                llm_manager,
                progress=formatter.progress_indicator,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        formatter.error(f"websearch timed out after {timeout_s:.0f}s")
        sys.exit(124)
    except Exception as exc:
        formatter.error(f"Research failed: {exc}")
        sys.exit(1)
    formatter.text_block(str(result.get("answer", "")))
