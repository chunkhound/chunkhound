from __future__ import annotations

import html as html_module
from pathlib import Path

from tests.site.html_helpers import (
    canonical_href,
    meta_tag_content,
    visible_text,
)

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
EXPLANATION_ROUTE = DIST / "docs" / "architecture" / "index.html"
CANONICAL_URL = "https://chunkhound.ai/docs/architecture/"


def _read(route: Path) -> str:
    return html_module.unescape(route.read_text(encoding="utf-8"))


def _text(route: Path) -> str:
    return visible_text(_read(route))


def test_architecture_page_renders_at_canonical_route() -> None:
    document = _read(EXPLANATION_ROUTE)

    assert canonical_href(document) == CANONICAL_URL
    assert meta_tag_content(document, "property", "og:type") == "website"
    og_image = meta_tag_content(document, "property", "og:image")
    assert og_image is not None
    assert og_image.endswith("/og-image-dark.png")


def test_architecture_page_states_the_adaptive_cutoff_contract() -> None:
    """The differentiator (a data-driven cutoff) is stated, not just implied."""
    text = _text(EXPLANATION_ROUTE)

    assert "No fixed top-k. The score curve sets the cutoff." in text

    # No elbow means the result set is kept whole; the median is only the
    # fallback for exploration's intermediate follow-up thresholds.
    assert "elbow" in text
    assert "kept whole" in text
    assert "median" in text


def test_architecture_page_claims_one_pipeline_for_every_source() -> None:
    """The context diagram names every source; the pipeline claims one path."""
    text = _text(EXPLANATION_ROUTE)

    for source in ("Repository files", "Git history", "Web pages"):
        assert source in text
    assert "every source enters the same path" in text
    assert "One question" in text
    assert "one cited answer" in text


def test_architecture_page_states_how_the_index_is_built_accurately() -> None:
    """The index build runs on a Rust engine; Python keeps parsing and the embedding fallback.

    Guards the bug class the redesign removed: a page claiming the indexing
    pipeline is Python-orchestrated, or crediting Rust with parsing/embedding it
    does not do. v6.0.0 moved the engine (scan, diff, scheduling, storage,
    compaction) to Rust; native embeddings are Rust too, leaving tree-sitter
    parsing and the provider fallback in Python.
    """
    text = _text(EXPLANATION_ROUTE)

    # The phase sequence is intact.
    assert "Parse ∥ store" in text
    # The engine is Rust.
    assert "written in Rust" in text
    assert "The indexing engine" in text
    # Parsing is still called back into Python; native embeddings are Rust.
    assert "tree-sitter" in text
    assert "embedding fallback" in text
    assert "OpenAI-compatible" in text

    # The old, now-false framing must never come back.
    for false_claim in (
        "Indexing is Python-orchestrated",
        "Rust accelerates file discovery only",
        "identical chunks and embeddings",
    ):
        assert false_claim not in text


def test_architecture_is_reachable_from_docs_navigation() -> None:
    document = _read(EXPLANATION_ROUTE)

    assert 'href="/docs/architecture/"' in document


def test_retired_enterprise_architecture_route_redirects_here() -> None:
    redirect = DIST / "enterprise" / "architecture" / "index.html"

    assert redirect.exists()
    assert "/docs/architecture/" in redirect.read_text(encoding="utf-8")
