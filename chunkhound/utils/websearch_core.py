"""Protocol-neutral websearch helpers shared by the CLI and MCP entry points.

Both ``chunkhound/api/cli/commands/websearch.py`` (CLI) and
``chunkhound/mcp_server/tools.py`` (``websearch_impl``) consume these helpers.
Keeping them here breaks the prior MCP→CLI import direction so both call
sites depend on this neutral module instead of each other.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import html.parser
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Request, Route

from chunkhound.core.config.config import Config

_MAX_FETCH_CONCURRENCY = 5


def websearch_timeout() -> float:
    """Overall wall-clock timeout (seconds) for the websearch subprocess.

    Reads CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS; falls back to 600.0 on
    unset or malformed values.
    """
    raw = os.environ.get("CHUNKHOUND_WEBSEARCH_TIMEOUT_SECONDS")
    if raw is None:
        return 600.0
    try:
        return float(raw)
    except ValueError:
        return 600.0


class _NextFormParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._form: list[dict[str, str | None]] | None = None
        self._forms: list[list[dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d: dict[str, str | None] = dict(attrs)
        if tag == "form":
            self._form = []
        elif tag == "input" and self._form is not None:
            self._form.append(d)

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._form is not None:
            self._forms.append(self._form)
            self._form = None

    def next_params(self) -> dict[str, str] | None:
        for form in self._forms:
            if any(
                a.get("type") == "submit" and a.get("value") == "Next" for a in form
            ):
                return {
                    name: a.get("value") or ""
                    for a in form
                    if a.get("type") == "hidden" and (name := a.get("name")) is not None
                }
        return None


class _ResultParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._capture: str | None = None
        self._title = ""
        self._url = ""
        self._desc = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d: dict[str, str | None] = dict(attrs)
        cls = d.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._url = d.get("href") or ""
            self._title = ""
            self._capture = "title"
        elif tag == "a" and "result__snippet" in cls:
            self._desc = ""
            self._capture = "desc"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag == "a" and self._capture == "desc":
            self._capture = None
            if self._title and self._url:
                self.results.append(
                    (
                        html.unescape(self._title),
                        self._url,
                        html.unescape(self._desc),
                    )
                )
            self._title = self._url = self._desc = ""

    def handle_data(self, data: str) -> None:
        if self._capture == "title":
            self._title += data
        elif self._capture == "desc":
            self._desc += data


def _fetch(params: dict[str, str]) -> str:
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def _url_to_filename(url: str, max_length: int = 100) -> str:
    # Append a short stable hash of the full URL so distinct URLs cannot
    # collide via the lossy [^\w.-]→_ substitution or via truncation when
    # two URLs share a long common prefix.
    name = re.sub(r"^https?://", "", url)
    name = re.sub(r"[^\w.-]", "_", name)
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{name[: max(0, max_length - 9)]}_{digest}"[:max_length]


def _html_to_markdown(html_text: str) -> str:
    # Lazy import: defers loading markdownify + deps (beautifulsoup4, soupsieve, six)
    from markdownify import MarkdownConverter

    class _Converter(MarkdownConverter):
        # strip=... only drops the wrapper tag, not its children, so raw
        # JS/CSS/SVG source still leaks. Override to discard the body.
        def convert_script(self, el, text, parent_tags):
            return ""

        def convert_style(self, el, text, parent_tags):
            return ""

        def convert_svg(self, el, text, parent_tags):
            return ""

    return _Converter(
        strip=[
            "head",
            "nav", "footer", "header", "aside",
            "form", "button", "iframe", "noscript",
        ],
        heading_style="ATX",
    ).convert(html_text)


def _fetch_url(url: str) -> tuple[str, str | bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        ct = resp.headers.get_content_type() or ""
        raw = resp.read()
        if ct == "application/pdf":
            return ".pdf", raw
        if ct == "text/html" or not ct:
            charset = resp.headers.get_content_charset() or "utf-8"
            return ".md", _html_to_markdown(raw.decode(charset, errors="replace"))
        raise ValueError(f"Unsupported content-type: {ct!r}")


async def _fetch_page(context: BrowserContext, url: str) -> tuple[str, str | bytes]:
    """Fetch a single URL using an existing Playwright browser context."""
    page = await context.new_page()
    try:
        pdf_body: bytes | None = None
        fetch_error: Exception | None = None

        async def handle(route: Route, request: Request) -> None:
            nonlocal pdf_body, fetch_error
            if request.resource_type != "document":
                await route.continue_()
                return
            try:
                resp = await route.fetch()
            except Exception as e:
                fetch_error = e
                await route.abort()
                return
            ct = resp.headers.get("content-type", "").split(";")[0].strip()
            if ct == "application/pdf":
                pdf_body = await resp.body()
                await resp.dispose()
                # Fulfill with empty HTML so page.goto completes without error
                await route.fulfill(body=b"", content_type="text/html")
            else:
                try:
                    await route.fulfill(response=resp)
                finally:
                    await resp.dispose()

        await page.route("**/*", handle)
        try:
            response = await page.goto(url, timeout=30000)
        except Exception:
            if fetch_error is not None:
                raise fetch_error
            raise
        if fetch_error is not None:
            raise fetch_error

        if pdf_body is not None:
            if pdf_body[:4] == b"%PDF":
                return ".pdf", pdf_body
            # Chromium's PDF viewer can consume the response before the route
            # handler receives the full body — re-fetch directly to bypass it.
            direct = await context.request.get(url)
            try:
                if not direct.ok:
                    raise ValueError(
                        f"PDF fallback fetch failed: HTTP {direct.status}"
                    )
                direct_ct = direct.headers.get("content-type", "").split(";")[0].strip()
                if direct_ct != "application/pdf":
                    raise ValueError(
                        f"PDF magic bytes missing and direct fetch returned {direct_ct!r}"
                    )
                return ".pdf", await direct.body()
            finally:
                await direct.dispose()

        if response is None:
            raise ValueError("Navigation failed: no response")
        ct = response.headers.get("content-type", "").split(";")[0].strip()
        if ct and ct != "text/html":
            raise ValueError(f"Unsupported content-type: {ct!r}")
        return ".md", _html_to_markdown(await page.content())
    finally:
        await page.close()


async def _fetch_one(
    url: str,
    tmpdir: Path,
    context: BrowserContext | None,
    progress_callback: Callable[[str], None] | None,
    warning_callback: Callable[[str], None] | None,
    semaphore: asyncio.Semaphore,
    mapping: dict[str, str] | None,
) -> None:
    async with semaphore:
        if progress_callback:
            progress_callback(f"Fetching {url}...")
        try:
            if context is not None:
                ext, content = await _fetch_page(context, url)
            else:
                ext, content = await asyncio.to_thread(_fetch_url, url)
            path = tmpdir / (_url_to_filename(url) + ext)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
            if mapping is not None:
                mapping[path.name] = url
        except Exception as e:
            if warning_callback:
                warning_callback(f"Failed to fetch {url}: {type(e).__name__}: {e}")


async def fetch_and_save(
    urls: list[str],
    tmpdir: Path,
    progress_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
    mapping: dict[str, str] | None = None,
) -> None:
    """Fetch each URL concurrently (bounded) and save content to tmpdir."""
    semaphore = asyncio.Semaphore(_MAX_FETCH_CONCURRENCY)

    async def _run(context: BrowserContext | None) -> None:
        tasks = [
            _fetch_one(
                url, tmpdir, context, progress_callback, warning_callback,
                semaphore, mapping,
            )
            for url in urls
        ]
        await asyncio.gather(*tasks)

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        await _run(None)
        return

    async with async_playwright() as pw:
        exe = pw.chromium.executable_path
        if not (exe and Path(exe).exists()):
            if warning_callback:
                warning_callback(
                    "Chromium not installed — run: playwright install chromium."
                    " Falling back to urllib."
                )
            await _run(None)
            return
        browser = await pw.chromium.launch()
        try:
            context = await browser.new_context()
            try:
                await _run(context)
            finally:
                await context.close()
        finally:
            await browser.close()


def search(
    query: str,
    limit: int = 30,
    progress_callback: Callable[[str], None] | None = None,
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    params: dict[str, str] = {"q": query, "b": ""}  # b = submit button name
    page_num = 0
    while True:
        page_num += 1
        if progress_callback:
            progress_callback(f"Fetching page {page_num}...")
        page = _fetch(params)
        rp = _ResultParser()
        rp.feed(page)
        if not rp.results:
            break
        results.extend(rp.results)
        if len(results) >= limit:
            break
        nfp = _NextFormParser()
        nfp.feed(page)
        next_params = nfp.next_params()
        if not next_params:
            break
        params = next_params
    return results[:limit]


def build_quickresearch_argv_core(
    query: str,
    tmpdir: Path,
    path_filter: str | None,
    config: Config,
) -> list[str]:
    """Build argv to invoke _quickresearch as a subprocess.

    Forwards the config source file as an absolute path so the child process
    does not need to re-run config discovery (which would otherwise fall back
    to env vars / defaults under the MCP server's working directory).
    """
    cmd: list[str] = [
        sys.executable,
        "-m", "chunkhound.api.cli.main",
        "_quickresearch",
        query,
        str(tmpdir),
    ]
    if path_filter is not None:
        cmd.extend(["--path-filter", path_filter])
    source = config.config_file or config.local_config_file
    if source is not None:
        cmd.extend(["--config", str(Path(source).resolve())])
    return cmd
