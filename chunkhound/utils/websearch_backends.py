"""Multi-engine websearch orchestration: transport, adapters, health, selection.

One engine's refusal stops being an outage: engines are tried in a
configurable order (or a shuffle among healthy ones), and the first engine
that serves wins. Per-response verdicts (see ``websearch_verdict``) are
evidence; the registry turns accumulated evidence into cooldowns. A refusal
is distinct from an empty SERP — ``AllEnginesBlockedError`` means every
attempted engine refused, ``AllEnginesUnavailableError`` means no engine
served without a genuine-empty signal, and ``[]`` means the query
genuinely had no hits.
"""

from __future__ import annotations

import asyncio
import base64
import html
import html.parser
import itertools
import random
import re
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from typing import Any

from loguru import logger

from chunkhound.utils.websearch_core import _http_request, _normalize_url
from chunkhound.utils.websearch_verdict import (
    EngineResponse,
    EngineRun,
    Verdict,
    status_evidence,
    verdict_from_status,
)

BING_SEARCH_URL = "https://www.bing.com/search"
# Bing's <li class="b_algo"> rows carry redirect hrefs (/ck/a) whose real
# destination is base64-encoded in the ``u`` query param as "a1a<base64url>".
_BING_REDIRECT_PATH = "/ck/a"
# Results per Bing page. Modest so pagination stays cheap and the pager
# (``first`` param) advances in predictable steps.
_BING_PAGE_SIZE = 10
# Pagination cap: Bing never owes us more than 5 pages; a runaway pager
# must not eat the whole per-engine budget.
_BING_MAX_PAGES = 5

_UA = "Mozilla/5.0"

# Class-attribute match for Bing's soft-empty marker; plain substring
# matching would also hit JS config keys like "b_nobody".
_BING_NO_CLASS = re.compile(r'class="[^"]*\bb_no\b')
_BING_RESULTS_CONTAINER = 'id="b_results"'
_BING_NO_RESULTS_TEXT = "there are no results"


class AllEnginesError(Exception):
    """Base for terminal all-engines failures; subclasses set _message_prefix."""

    _message_prefix = ""

    def __init__(self, reasons: list[tuple[str, str]]) -> None:
        self.reasons = list(reasons)
        self.engines = [name for name, _ in self.reasons]
        details = "; ".join(f"{name}: {evidence}" for name, evidence in self.reasons)
        prefix = self._message_prefix.format(engines=", ".join(self.engines))
        super().__init__(f"{prefix} ({details})")


class AllEnginesBlockedError(AllEnginesError):
    """Every attempted engine refused to serve the search."""

    _message_prefix = "Blocked by {engines}; every engine refused"


class AllEnginesUnavailableError(AllEnginesError):
    """No engine served; at least one was unavailable (not a genuine empty)."""

    _message_prefix = "No engine served; engines unavailable"


class EngineRegistry:
    """Per-process engine health: consecutive refusals cool an engine down.

    One refusal is not enough — soft signals (empty, drift, transient) are
    ambiguous and neither strike nor reset; only accumulated refusal
    evidence blocks an engine, and a served run clears it immediately.
    """

    def __init__(self, cooldown_seconds: int = 900) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._blocked_until: dict[str, float] = {}
        self._strikes: dict[str, int] = {}

    def is_healthy(self, engine: str) -> bool:
        return time.monotonic() >= self._blocked_until.get(engine, 0.0)

    def recover_at(self, engine: str) -> float:
        """Monotonic time when the engine becomes usable again."""
        return self._blocked_until.get(engine, 0.0)

    def strikes(self, engine: str) -> int:
        return self._strikes.get(engine, 0)

    def mark_blocked(self, engine: str, *, cooldown_seconds: int | None = None) -> None:
        seconds = (
            cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        )
        self._blocked_until[engine] = time.monotonic() + seconds

    def record(
        self,
        engine: str,
        run: EngineRun,
        *,
        cooldown_seconds: int | None = None,
    ) -> None:
        """Accumulate one run's evidence into the health state."""
        if run.verdict is Verdict.SERVED:
            self._strikes[engine] = 0
            # A served run proves the engine works: lift any cooldown,
            # even when a later query in the same scan was refused.
            self._blocked_until.pop(engine, None)
        elif run.strike:
            count = self._strikes.get(engine, 0) + 1
            self._strikes[engine] = count
            if count >= 2:
                self.mark_blocked(
                    engine,
                    cooldown_seconds=self._escalated_cooldown(count, cooldown_seconds),
                )

    def _escalated_cooldown(self, strikes: int, override: int | None) -> int:
        # 2nd strike costs the base cooldown, each further one doubles, so a
        # persistently refusing engine drops out of rotation for good.
        base = override if override is not None else self.cooldown_seconds
        seconds = min(base * 2 ** (strikes - 2), 3600)
        jittered = seconds * random.uniform(0.8, 1.2)
        return max(1, int(jittered))


# Process-wide health state shared by every run_engine_search call: a refused
# engine cools down for the lifetime of the process, not just one call.
_REGISTRY = EngineRegistry()


def resolve_engine_order(config: Any, registry: EngineRegistry) -> list[str]:
    """Return engine names in the order they should be tried.

    ``priority``: the configured order, cooling engines skipped. If every
    engine is cooling, return only the single engine that recovers soonest
    (half-open probe): paying the full list's cost for known-bad engines
    re-arms cooldowns on every search.
    ``random_healthy``: shuffle only the healthy engines; fall back to the
    same single soonest-recovering probe when none are healthy.
    """
    selected = list(config.backends)
    probe = [min(selected, key=registry.recover_at)]
    if config.selection == "random_healthy":
        healthy = [name for name in selected if registry.is_healthy(name)]
        if healthy:
            random.shuffle(healthy)
            return healthy
        return probe
    return [name for name in selected if registry.is_healthy(name)] or probe


class _BingResultParser(html.parser.HTMLParser):
    """Extract (title, url, snippet) rows from a Bing SERP.

    Bing markup (verified live 2026-11-16): each organic result is a
    ``<li class="b_algo">`` containing an ``<h2><a href=".../ck/a?
    &u=a1a<base64url>">title</a></h2>`` and a snippet in
    ``<p class="b_lineclampN">``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._item: dict[str, str] | None = None
        self._capture: str | None = None
        self._in_h2 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d: dict[str, str | None] = dict(attrs)
        cls = d.get("class") or ""
        if tag == "h2":
            self._in_h2 = True
        elif tag == "li" and "b_algo" in cls:
            self._item = {"url": "", "title": "", "desc": ""}
        elif tag == "a" and self._item is not None and self._in_h2:
            href = d.get("href") or ""
            if href and not self._item["url"]:
                self._item["url"] = _decode_bing_url(href)
                self._capture = "title"
        elif tag == "p" and "b_lineclamp" in cls and self._item is not None:
            self._capture = "desc"

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_h2 = False
        if self._item is None:
            return
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag == "p" and self._capture == "desc":
            self._capture = None
        elif tag == "li":
            item = self._item
            self._item = None
            self._capture = None
            if item["url"] and item["title"]:
                self.results.append(
                    (
                        html.unescape(item["title"]),
                        item["url"],
                        html.unescape(item["desc"]),
                    )
                )

    def handle_data(self, data: str) -> None:
        if self._item is None or self._capture is None:
            return
        if self._capture == "title":
            self._item["title"] += data
        elif self._capture == "desc":
            self._item["desc"] += data


def _decode_bing_url(href: str) -> str:
    """Decode a Bing redirect href to its real destination.

    Bing wraps result URLs in ``/ck/a`` redirects whose ``u`` param carries
    the destination as base64url, prefixed by a short marker. Live probes
    show the marker is ``a1`` followed by the base64 (whose own leading
    ``a`` makes the literal prefix look like ``a1a``). Try both split
    points and keep only a decoded http(s) URL; anything else falls back to
    the raw href, which the fetch stage can still follow via redirects.
    """
    href = html.unescape(href)
    if _BING_REDIRECT_PATH not in href:
        return href
    query = urllib.parse.urlsplit(href).query
    u = urllib.parse.parse_qs(query).get("u", [""])[0]
    if not u:
        return href
    candidates = [u]
    if u.startswith("a1"):
        candidates.append(u[2:])
        if u.startswith("a1a"):
            candidates.append(u[3:])
    for candidate in candidates:
        try:
            padding = "=" * (-len(candidate) % 4)
            decoded = base64.urlsafe_b64decode(candidate + padding).decode("utf-8")
        except Exception:
            continue
        if decoded.startswith(("http://", "https://")):
            return decoded
    return href


def _parse_bing_results(html_text: str) -> list[tuple[str, str, str]]:
    parser = _BingResultParser()
    parser.feed(html_text)
    return parser.results


class DuckDuckGoBackend:
    """DuckDuckGo HTML endpoint, via the shared curl_cffi transport."""

    name = "duckduckgo"

    def search(
        self,
        query: str,
        limit: int,
        progress_callback: Callable[[str], None] | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> EngineResponse:
        from chunkhound.utils.websearch_core import search as ddg_search

        return ddg_search(query, limit, progress_callback, cancel_event=cancel_event)


class BingBackend:
    """Bing HTML SERP via the shared curl_cffi transport."""

    name = "bing"

    def search(
        self,
        query: str,
        limit: int,
        progress_callback: Callable[[str], None] | None = None,
        *,
        cancel_event: threading.Event | None = None,
    ) -> EngineResponse:
        """Fetch and classify Bing SERP pages until limit or pagination end.

        Verdict rules, per page and in order: parsed rows -> SERVED;
        non-200 -> ``verdict_from_status``; 0 rows with the soft-empty
        sentinel (``b_no`` class or "There are no results for") -> EMPTY;
        0 rows with the ``b_results`` container missing -> DRIFT; 0 rows
        with the container but no sentinel -> DRIFT. An empty page after
        page 1 is the normal end of pagination, not a signal. The poison
        strings in Bing's own JS config (challenge URLs etc.) are never
        consulted — only the results container and sentinel decide.
        """
        results: list[tuple[str, str, str]] = []
        first = 1
        page_num = 0
        status = 200
        while cancel_event is None or not cancel_event.is_set():
            page_num += 1
            if progress_callback:
                progress_callback(f"Fetching page {page_num}...")
            status, body = _http_request(
                "get",
                BING_SEARCH_URL,
                params={
                    "q": query,
                    "count": str(_BING_PAGE_SIZE),
                    "first": str(first),
                    # Bias toward English results; class-based parsing is
                    # locale-independent either way.
                    "setlang": "en",
                    "cc": "US",
                },
                headers={
                    "User-Agent": _UA,
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            rows = _parse_bing_results(body.decode())
            if rows:
                results.extend(rows)
                if len(results) >= limit or len(rows) < _BING_PAGE_SIZE:
                    break
                if page_num >= _BING_MAX_PAGES:
                    break
                first += _BING_PAGE_SIZE
                continue
            if status != 200:
                return EngineResponse(
                    (),
                    verdict_from_status(status),
                    status,
                    status_evidence(status),
                )
            if page_num > 1:
                break  # empty later page = end of pagination
            html_text = body.decode()
            if _BING_NO_CLASS.search(html_text) or (
                _BING_NO_RESULTS_TEXT in html_text.lower()
            ):
                return EngineResponse(
                    (), Verdict.EMPTY, status, "no results sentinel (b_no)"
                )
            if _BING_RESULTS_CONTAINER not in html_text:
                return EngineResponse(
                    (), Verdict.DRIFT, status, "b_results container missing"
                )
            return EngineResponse(
                (),
                Verdict.DRIFT,
                status,
                "b_results present, no rows, no sentinel",
            )
        return EngineResponse(tuple(results[:limit]), Verdict.SERVED, status)


ALL_BACKENDS: dict[str, Any] = {
    "duckduckgo": DuckDuckGoBackend(),
    "bing": BingBackend(),
}


def _merge_rank_major(
    batches: list[list[tuple[str, str, str]]],
) -> list[tuple[str, str, str]]:
    """Rank-major interleave + dedupe by normalized URL (first wins)."""
    seen: dict[str, tuple[str, str, str]] = {}
    for rank_row in itertools.zip_longest(*batches):
        for row in rank_row:
            if row is None:
                continue
            key = _normalize_url(row[1])
            if key not in seen:
                seen[key] = row
    return list(seen.values())


# Floor on one engine's attempt budget so a large engine count can never
# starve the first engine into instant timeouts.
_MIN_ENGINE_BUDGET_S = 5.0


async def _run_engine(
    backend: Any,
    queries: list[str],
    limit: int,
    *,
    progress_callback: Callable[[str], None] | None,
    failure_callback: Callable[[str, urllib.error.URLError], None] | None,
    deadline: float | None = None,
) -> EngineRun:
    """Run one engine's full multi-query scan and aggregate one verdict.

    Each response is evidence only; aggregation decides the run's verdict:
    rows -> SERVED (with a strike when any refusal was seen), refusals ->
    REFUSED, drift -> DRIFT, transient/URLError -> TRANSIENT, all empty ->
    EMPTY. Early-break after 2 refusals with no rows yet. URLError is a
    transient signal, never a genuine-empty one, and never aborts the scan.
    """
    n = len(queries)
    batches: list[list[tuple[str, str, str]]] = []
    evidences: list[str] = []
    refusals = 0
    drifts = 0
    transients = 0
    urlerrors = 0
    cancel_event = threading.Event()
    try:
        for i, q in enumerate(queries, start=1):
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    evidences.append("engine budget exhausted")
                    transients += 1
                    break
            per_query_progress: Callable[[str], None] | None = None
            if progress_callback is not None:

                def per_query_progress(
                    msg: str, *, _prefix: str = f"[{i}/{n}] "
                ) -> None:
                    progress_callback(f"{_prefix}{msg}")

            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        backend.search,
                        q,
                        limit,
                        per_query_progress,
                        cancel_event=cancel_event,
                    ),
                    timeout=remaining,
                )
            except (asyncio.TimeoutError, TimeoutError):
                evidences.append(f"timed out after {remaining:.0f}s")
                transients += 1
                break  # budget spent; more queries would only overrun too
            except urllib.error.URLError as e:
                # The query text is sensitive: keep it out of warnings.
                if failure_callback is not None:
                    failure_callback(q, e)
                else:
                    logger.debug(f"failed query: {q!r}")
                    logger.warning(
                        f"Search query failed: {e.reason}; "
                        "continuing with remaining queries"
                    )
                evidences.append(f"connection error: {e.reason}")
                urlerrors += 1
                continue
            # Verdict and rows are independent: a refusal that arrives after
            # some rows is still evidence of an unhealthy engine (a "strike"),
            # even though the partial rows are kept and served.
            if response.rows:
                batches.append(list(response.rows))
            if response.verdict is Verdict.REFUSED:
                refusals += 1
            elif response.verdict is Verdict.DRIFT:
                drifts += 1
            elif response.verdict is Verdict.TRANSIENT:
                transients += 1
            evidences.append(response.evidence)
            if refusals >= 2 and not batches:
                break  # two refusals and no rows: this engine is done
    finally:
        cancel_event.set()
    evidence = "; ".join(dict.fromkeys(evidences))
    rows = _merge_rank_major(batches)[:limit]
    # Transport failures are transient, never a genuine empty — partial or
    # total, so an unreachable engine fails over instead of reporting [].
    had_transport_error = urlerrors > 0
    if rows:
        return EngineRun(rows, Verdict.SERVED, evidence, strike=refusals > 0)
    if refusals and (drifts or transients or had_transport_error):
        return EngineRun([], Verdict.TRANSIENT, evidence, strike=True)
    if refusals:
        return EngineRun([], Verdict.REFUSED, evidence, strike=True)
    if drifts:
        return EngineRun([], Verdict.DRIFT, evidence)
    if transients or had_transport_error:
        return EngineRun([], Verdict.TRANSIENT, evidence)
    return EngineRun([], Verdict.EMPTY, evidence)


async def run_engine_search(
    queries: list[str],
    limit: int,
    *,
    progress_callback: Callable[[str], None] | None = None,
    failure_callback: Callable[[str, urllib.error.URLError], None] | None = None,
    websearch_config: Any,
    registry: EngineRegistry | None = None,
    timeout_s: float | None = None,
) -> list[tuple[str, str, str]]:
    """Try configured engines in order until one serves.

    The total ``timeout_s`` budget is partitioned per engine
    (remaining / engines remaining, floored): an engine that overruns its
    slice yields TRANSIENT and fails over, so one hanging engine cannot
    starve the fallbacks. EMPTY/DRIFT/TRANSIENT continue to the next engine;
    only rows end the search. When no engine served: any EMPTY verdict
    means genuine no-results and returns []; every attempted engine REFUSED
    raises ``AllEnginesBlockedError``; anything else (transient/drift/
    mixed) raises ``AllEnginesUnavailableError``.
    """
    queries = list(dict.fromkeys(queries))  # dedupe identical strings; preserves order
    if not queries:
        return []
    registry = registry or _REGISTRY
    order = resolve_engine_order(websearch_config, registry)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s if timeout_s is not None else None
    reasons: list[tuple[str, str]] = []
    all_refused = True
    saw_empty = False
    for idx, engine_name in enumerate(order):
        engines_remaining = len(order) - idx
        engine_deadline: float | None = None
        if deadline is not None:
            remaining = max(0.0, deadline - loop.time())
            budget = max(_MIN_ENGINE_BUDGET_S, remaining / engines_remaining)
            engine_deadline = loop.time() + budget
        backend = ALL_BACKENDS[engine_name]
        if progress_callback is not None and idx > 0:
            progress_callback(f"Trying {engine_name}...")
        run = await _run_engine(
            backend,
            queries,
            limit,
            progress_callback=progress_callback,
            failure_callback=failure_callback,
            deadline=engine_deadline,
        )
        registry.record(
            engine_name,
            run,
            cooldown_seconds=websearch_config.cooldown_seconds,
        )
        if run.rows:
            return run.rows
        if run.verdict is Verdict.EMPTY:
            saw_empty = True
        if run.verdict is not Verdict.REFUSED:
            all_refused = False
        reasons.append((engine_name, run.evidence))
        logger.warning(
            f"websearch: engine {engine_name} {run.verdict.value} ({run.evidence})"
        )
        if progress_callback is not None:
            remaining_engines = order[idx + 1 :]
            hint = f"; trying {remaining_engines[0]}" if remaining_engines else ""
            if run.verdict is Verdict.REFUSED:
                progress_callback(f"Blocked by {engine_name} ({run.evidence}){hint}")
            elif run.verdict is Verdict.EMPTY:
                progress_callback(f"{engine_name} returned no results{hint}")
            elif run.verdict is Verdict.DRIFT:
                progress_callback(
                    f"{engine_name} layout drifted ({run.evidence}){hint}"
                )
            else:
                progress_callback(f"{engine_name} transient ({run.evidence}){hint}")
    if saw_empty:
        return []
    if all_refused:
        raise AllEnginesBlockedError(reasons)
    raise AllEnginesUnavailableError(reasons)
