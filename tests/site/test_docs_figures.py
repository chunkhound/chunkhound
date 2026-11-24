"""Built-docs figure contracts: accessibility, reuse, fit, and row alignment.

`DiagramFigure.astro` holds the accessible contract once: a labelled group, a
visible caption, and a screen-reader description that states the relationships.
The enterprise page writes that markup as raw HTML, because a `.md` page cannot
import components, so these guards read the shipped markup and stylesheet.

A row is one rail (name + qualifier) beside one lane (the parts that answer it).
Subgrid keeps both halves on the same two lines however the text wraps; centering
the rail block instead drifted by half the height difference (7.6px at 1440 and
1024). Arrow rows must share the band lanes' box, and the docs column caps at
72ch, so no figure may outgrow it. happy-dom has no layout engine, so fit and
alignment are measured in Chromium against the built site.
"""

from __future__ import annotations

import re
from functools import cache

from tests.site.browser_helpers import page_probe
from tests.site.dom_helpers import DIST
from tests.site.tsx_runner import ROOT

FIGURES = re.compile(r'<figure class="d-figure".*?</figure>', re.DOTALL)
ROW_TAG = re.compile(r'<(?:section|div) class="([^"]*)"[^>]*>')
LANE_MARKER = 'class="d-row-lane"'
# The stylesheet owns the diagram vocabulary: a class with no rule is a
# building block nobody can reuse, i.e. a one-off.
ANATOMY_RULES = ROOT / "site" / "src" / "styles" / "diagrams.css"

_TOLERANCE = 0.6
# Desktop layouts the row figure ships. Below 640 the rail stacks above the lane.
_ROW_WIDTHS = [1440, 1280, 1024, 710]
# Below the stack point the rail sits above the lane; the two-line contract
# becomes the stacking contract.
_STACK_WIDTHS = (640, 390)
# Both callers of the row primitive: the component figure and the raw copy.
_ROW_FIGURES = ("/docs/pi-schematic/", "/docs/enterprise/")
# Desktop, the docs shell's three tracks, the figure's stack point, and phones.
_FIT_WIDTHS = [1440, 1280, 1024, 710, 640, 390]

_ROW_MEASURE = """
return page.evaluate(() => ({
    overflow: [...document.querySelectorAll('.d-figure')]
      .reduce((sum, figure) => sum + figure.scrollWidth - figure.clientWidth, 0),
    rows: [...document.querySelectorAll('.d-figure .d-row')].map(row => {
      const label = row.querySelector('.d-band-label');
      const caption = row.querySelector('.d-row-caption');
      const rail = row.querySelector('.d-row-rail');
      const mark = row.querySelector('.d-row-icon, .d-row-mark');
      return {
        label: label?.textContent.trim() ?? null,
        rail: rail?.getBoundingClientRect().toJSON() ?? null,
        mark: mark?.getBoundingClientRect().toJSON() ?? null,
        lane: row.querySelector('.d-row-lane').getBoundingClientRect().toJSON(),
        labelBox: label?.getBoundingClientRect().toJSON() ?? null,
        caption: caption?.getBoundingClientRect().toJSON() ?? null,
        nodes: [...row.querySelectorAll('.d-node')].map(node => ({
          title: node.querySelector('.d-node-title').getBoundingClientRect().toJSON(),
          detail: node.querySelector('.d-node-detail').getBoundingClientRect().toJSON(),
        })),
        arrows: [...row.querySelectorAll('.d-row-arrow')].map(arrow =>
          arrow.getBoundingClientRect().toJSON()),
      };
    }),
}));
"""

_FIT_MEASURE = """
return page.evaluate(() => ({
  figures: [...document.querySelectorAll('.d-figure')].map(figure => ({
    caption: figure.querySelector('figcaption')?.textContent.trim() ?? '',
    overflow: figure.scrollWidth - figure.clientWidth,
  })),
}));
"""


@cache
def _docs_html() -> list[tuple[str, str]]:
    return [
        (
            "/" + str(page.relative_to(DIST).parent).replace("\\", "/") + "/",
            page.read_text(encoding="utf-8"),
        )
        for page in sorted((DIST / "docs").rglob("index.html"))
    ]


def _docs_pages() -> list[str]:
    return [route for route, _html in _docs_html()]


def _docs_figures() -> list[str]:
    figures = [
        match for _route, html in _docs_html() for match in FIGURES.findall(html)
    ]
    assert figures, "no d-figure found in the built docs"
    return figures


def _figure_rows(figure: str) -> list[str]:
    """Each row's markup, cut at the next sibling row."""
    starts = [
        match.start()
        for match in ROW_TAG.finditer(figure)
        if "d-row" in match.group(1).split()
    ]
    return [figure[start:end] for start, end in zip(starts, starts[1:] + [len(figure)])]


@cache
def _rows(path: str) -> dict[str, dict]:
    """One measurement per figure and width set: every contract reads the same
    layout, so tests cannot disagree about what the browser rendered."""
    return page_probe(path, _ROW_WIDTHS, _ROW_MEASURE)


@cache
def _stack_rows(path: str) -> dict[str, dict]:
    return page_probe(path, list(_STACK_WIDTHS), _ROW_MEASURE)


@cache
def _fit(path: str) -> dict[str, dict]:
    return page_probe(path, _FIT_WIDTHS, _FIT_MEASURE)


def _assert_on_one_line(value: float, targets: list[float], what: str) -> None:
    for target in targets:
        assert abs(value - target) <= _TOLERANCE, (
            f"{what} is off by {value - target:+.1f}px (got {value}, expected {target})"
        )


def _centre(box: dict) -> float:
    return box["left"] + box["width"] / 2


def _middle(box: dict) -> float:
    return box["top"] + box["height"] / 2


def _column(box: dict) -> tuple[float, float]:
    return round(box["left"], 1), round(box["width"], 1)


def test_every_docs_figure_is_a_labelled_group() -> None:
    for figure in _docs_figures():
        assert 'role="group"' in figure and "<figcaption>" in figure
        label = re.search(r'aria-label="([^"]*)"', figure)
        assert label and len(label.group(1).split()) >= 3, figure[:200]
        description = re.search(r'class="sr-only">([^<]+)<', figure)
        # Labels alone do not explain a diagram; the description must carry
        # the edges, so a few words per node is the floor, not a style rule.
        assert description and len(description.group(1).split()) >= 10, figure[:200]


def test_every_docs_row_keeps_the_shared_anatomy() -> None:
    """A row is one rail (mark, name, qualifier) beside one lane of parts."""
    for figure in _docs_figures():
        for row in _figure_rows(figure):
            tag = row[: row.index(">") + 1]
            assert row.count(LANE_MARKER) == 1, f"row must have one lane: {tag}"
            split = row.index(LANE_MARKER)
            head, lane = row[:split], row[split:]
            assert 'class="d-node"' not in head, f"node outside the lane: {tag}"
            assert 'class="d-row-rail"' not in lane, f"rail inside the lane: {tag}"
            if 'class="d-row-rail"' in row:
                assert row.count('class="d-row-rail"') == 1, tag
                assert row.count('class="d-band-label') == 1, f"one name: {tag}"
                assert row.count('class="d-band-caption') == 1, f"one qualifier: {tag}"
                marks = row.count('class="d-row-icon"') + row.count(
                    'class="d-row-mark"'
                )
                assert marks == 1, f"one mark per rail, got {marks}: {tag}"
                assert 'class="d-band-label' not in lane, f"name inside the lane: {tag}"
            if 'class="d-row-arrow"' in row:
                assert 'data-cols="2"' in tag, (
                    f"a row with arrows must pin the columns it points at: {tag}"
                )


def test_docs_figures_reuse_the_published_anatomy() -> None:
    """No figure may invent a class the stylesheet does not define."""
    rules = ANATOMY_RULES.read_text(encoding="utf-8")
    for figure in _docs_figures():
        for attributes in re.findall(r'class="([^"]*)"', figure):
            for name in attributes.split():
                if name.startswith("d-"):
                    assert re.search(rf"\.{re.escape(name)}\b", rules), (
                        f".{name} has no rule in diagrams.css: the figure invented a "
                        "building block instead of reusing one"
                    )


def test_every_docs_figure_fits_its_column(built_site) -> None:
    """Enumerate after the build: a clean checkout has no dist at collection."""
    assert (pages := _docs_pages()), f"no docs pages found under {DIST / 'docs'}"
    for path in pages:
        for width, measured in _fit(path).items():
            for figure in measured["figures"]:
                assert figure["overflow"] <= 0, (
                    f"{path} at {width}px: figure {figure['caption']!r} "
                    f"overflows its column by {figure['overflow']}px"
                )


def test_every_rail_names_the_line_its_lane_starts_on(built_site) -> None:
    """Names sit on the titles' line and qualifiers on the details' line."""
    for path in _ROW_FIGURES:
        for width, measured in _rows(path).items():
            assert measured["overflow"] <= 0, f"{width}px {path}: figure overflows"
            rows = measured["rows"]
            band_rows = [row for row in rows if row["rail"]]
            assert band_rows, f"{width}px {path}: no labelled rows"

            rails = {_column(row["rail"]) for row in band_rows}
            lanes = {_column(row["lane"]) for row in rows}
            assert len(rails) == 1, f"{width}px {path}: rows disagree on the rail"
            assert len(lanes) == 1, f"{width}px {path}: rows disagree on the lane"

            for row in band_rows:
                where = f"{width}px {path} {row['label']}"
                assert row["caption"], f"{where}: no qualifier"
                titles = [node["title"]["top"] for node in row["nodes"]]
                details = [node["detail"]["top"] for node in row["nodes"]]
                assert titles and len(titles) == len(details), f"{where}: unpaired node"
                _assert_on_one_line(row["labelBox"]["top"], titles, f"{where}: name")
                _assert_on_one_line(
                    row["caption"]["top"], details, f"{where}: qualifier"
                )
                _assert_on_one_line(
                    row["caption"]["left"],
                    [row["labelBox"]["left"]],
                    f"{where}: qualifier left edge",
                )
                # The mark belongs to the whole rail, so it centres on its height.
                assert row["mark"], f"{where}: no mark"
                _assert_on_one_line(
                    _middle(row["mark"]), [_middle(row["rail"])], f"{where}: mark"
                )


def test_every_arrow_points_at_the_column_it_connects(built_site) -> None:
    """Arrows are centred on the data columns of the bands they sit between."""
    for path in _ROW_FIGURES:
        for width, measured in _rows(path).items():
            rows = measured["rows"]
            columns = [_centre(node["title"]) for row in rows for node in row["nodes"]]
            for row in rows:
                for arrow in row["arrows"]:
                    centre = _centre(arrow)
                    nearest = min(columns, key=lambda column: abs(column - centre))
                    _assert_on_one_line(
                        centre, [nearest], f"{width}px {path}: arrow between bands"
                    )


def test_narrow_rows_stack_the_rail_above_one_full_width_column(built_site) -> None:
    """Below the stack point the rail sits above a single column of nodes."""
    for path in _ROW_FIGURES:
        for width, measured in _stack_rows(path).items():
            assert measured["overflow"] <= 0, f"{width}px {path}: figure overflows"
            for row in measured["rows"]:
                if row["rail"]:
                    assert row["lane"]["top"] > row["rail"]["top"], (
                        f"{width}px {path} {row['label']}: lane follows the rail"
                    )
                    assert (
                        abs(row["lane"]["left"] - row["rail"]["left"]) <= _TOLERANCE
                    ), f"{width}px {path} {row['label']}: lane is not the full width"
                for node in row["nodes"]:
                    assert node["detail"]["top"] > node["title"]["top"], (
                        f"{width}px {path}: detail does not follow its title"
                    )
