#!/usr/bin/env python3
"""Unit tests for snapshot chunk-systems visualization HTML renderer."""

from __future__ import annotations

import json
import re

from chunkhound.snapshot.chunk_systems_viz import render_chunk_systems_viz_html


def _extract_embedded_json(html: str) -> str:
    m = re.search(
        r'<script type="application/json" id="chunkhound-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert m is not None
    return str(m.group(1))


def test_render_chunk_systems_viz_html_embeds_parseable_json_and_escapes_lt() -> None:
    adjacency_payload = {
        "schema_version": "snapshot.chunk_systems.system_adjacency.v1",
        "schema_revision": "2026-02-17",
        "systems": [
            {
                "cluster_id": 1,
                "size": 2,
                "label": "<danger>",
                "top_files": ["a.py"],
                "top_symbols": ["A.foo"],
            }
        ],
        "links": [],
        "truncation": {"links_before": 0, "links_after": 0},
    }

    html = render_chunk_systems_viz_html(adjacency_payload=adjacency_payload, system_metrics={})
    assert "<!doctype html>" in html.lower()

    embedded = _extract_embedded_json(html)
    # The JSON inside the <script> tag must not contain raw "<".
    assert "<" not in embedded
    assert "\\u003c" in embedded

    parsed = json.loads(embedded)
    assert parsed["adjacency"]["systems"][0]["label"] == "<danger>"

    # Operators should be able to switch presentation modes in the viz.
    assert 'id="layoutSel"' in html
    assert 'value="force"' in html
    assert 'value="kcore"' in html
    assert 'value="hub"' in html
    assert 'value="hub_grav"' in html
