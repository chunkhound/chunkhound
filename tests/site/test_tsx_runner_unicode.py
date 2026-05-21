from __future__ import annotations

from tests.site.tsx_runner import run_tsx_json


def test_run_tsx_json_preserves_unicode_literals() -> None:
    rendered = run_tsx_json(r'''
console.log(JSON.stringify({
  bullet: "·",
  nbsp: "\u00a0",
  combined: "Generated report · 14 files\u00a0",
}));
''')

    assert rendered == {
        "bullet": "·",
        "nbsp": "\u00a0",
        "combined": "Generated report · 14 files\u00a0",
    }
