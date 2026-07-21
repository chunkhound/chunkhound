from __future__ import annotations

import html as html_module
import re
from pathlib import Path

from tests.site.html_helpers import attributes, canonical_href, visible_text

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "site" / "dist"
ROUTE = DIST / "enterprise" / "architecture" / "index.html"
CANONICAL_URL = "https://chunkhound.ai/enterprise/architecture/"


def _document() -> str:
    return html_module.unescape(ROUTE.read_text(encoding="utf-8"))


def test_enterprise_architecture_states_current_boundary_and_decision() -> None:
    text = visible_text(_document())

    for phrase in (
        "ChunkHound is local-only by design",
        "The product is local-first",
        "local-first at the workload boundary",
        "not a ChunkHound server",
        "Our thesis: fix the DB itself, and the server isn't needed",
        "The cure is a better DB.",
        "Focus on local-first",
        "local indexing",
    ):
        assert phrase in text

    assert "External embedding, reranking, or LLM providers" in text
    assert "central query service" in text
    assert "central server" in text


def test_enterprise_architecture_comparison_is_incremental_and_balanced() -> None:
    """Comparison table has correct semantic structure and all 9 dimensions."""
    document = _document()
    text = visible_text(document)

    for dimension in (
        "Focus and polish",
        "Developer scale",
        "Non-developer access",
        "Deployment shape",
        "Cross-repository views",
        "Index maintenance",
        "Upgrades and monitoring",
        "Permissions and auth",
        "Data and freshness",
    ):
        assert dimension in text

    # Structural table contracts
    assert (
        "Incremental trade-offs after the current local-first product baseline."
        in text
    )
    assert "Focus on local-first" in text
    assert "Add mixed server path" in text
    assert re.search(r'<table(?:\s|>)', document)
    assert re.search(r"<caption(?:\s|>)", document)
    assert 'scope="col"' in document
    assert 'scope="row"' in document
    assert 'class="winner"' in document
    assert 'class="loser"' in document
    row_count = document.count('scope="row"')
    assert row_count >= 9, f"Expected ≥9 dimension rows, got {row_count}"


def test_enterprise_architecture_diagram_is_semantic_html() -> None:
    """Architecture diagram is a semantic figure with labeled nodes."""
    document = _document()
    figure = re.search(r'<figure\s+[^>]*class="architecture-figure"[^>]*>', document)

    assert figure is not None
    assert attributes(figure.group(0))["aria-labelledby"] == "architecture-map-title"
    assert "Editor or AI client" in document
    assert "ChunkHound process" in document
    assert "Project database" in document
    assert "provider dependencies" in visible_text(document)
    assert "not a central ChunkHound query service" in visible_text(document)


def test_enterprise_metadata_uses_canonical_route() -> None:
    document = _document()

    assert canonical_href(document) == CANONICAL_URL
    assert 'property="og:url"' in document
    assert 'property="og:type"' in document
    assert 'property="og:image:type" content="image/png"' in document
    assert 'property="og:image"' in document
    assert 'name="twitter:image"' in document


def test_enterprise_route_is_hidden_from_global_navigation_and_footer() -> None:
    homepage = (DIST / "index.html").read_text(encoding="utf-8")

    assert 'href="/enterprise/architecture/"' not in homepage
