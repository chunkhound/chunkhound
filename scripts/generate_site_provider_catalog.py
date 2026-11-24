"""Regenerate the site configurator's provider catalog from the backend SSOT.

WHY: the configurator must offer exactly the providers the backend dispatches.
Those facts already exist in the backend, so the site file is derived, never
hand-typed. Human judgment — which model to recommend, what to call a provider —
stays in ``providers.ts``. This script also emits ``requiresModel``: the backend
knows a model is mandatory, never which one.

    uv run python scripts/generate_site_provider_catalog.py          # write
    uv run python scripts/generate_site_provider_catalog.py --check  # verify
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, get_args

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.llm_config import (
    OPENAI_COMPATIBLE_LLM_PROVIDERS,
    REMOVED_PROVIDERS,
    LLMConfig,
    LLMProviderLiteral,
)
from chunkhound.core.config.provider_registry import OPENAI_COMPATIBLE_PROVIDERS

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / (
    "site/src/components/configurator/provider-catalog.gen.json"
)

REGENERATE = "uv run python scripts/generate_site_provider_catalog.py"


def _display_path(path: pathlib.Path) -> str:
    """Repo-relative inside the repo; plain path for temp paths in tests."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _llm_facts() -> list[dict[str, Any]]:
    """``authUrl`` is empty for native providers: the backend holds no such
    fact, so the site owns that URL itself.

    ``customEndpointRequiresModel`` mirrors the provider set consulted by
    ``LLMConfig._require_explicit_model_for_custom_openai``: any of these
    providers pointed at a non-official ``base_url`` must pin a model.
    """
    facts: list[dict[str, Any]] = []
    for name in sorted(set(get_args(LLMProviderLiteral)) - set(REMOVED_PROVIDERS)):
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(name)
        facts.append(
            {
                "id": name,
                "requiresModel": LLMConfig.requires_explicit_model(name),
                "customEndpointRequiresModel": (
                    name in OPENAI_COMPATIBLE_LLM_PROVIDERS
                ),
                "authUrl": spec.auth_url if spec else "",
            }
        )
    return facts


def _catalog() -> dict[str, Any]:
    return {
        "llm": _llm_facts(),
        "embedding": sorted(
            get_args(EmbeddingConfig.model_fields["provider"].annotation)
        ),
    }


def render_catalog() -> str:
    """Deterministic: sorted ids, fixed field order, LF, no timestamps."""
    return json.dumps(_catalog(), indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the site provider catalog."
    )
    parser.add_argument("--check", action="store_true", help="Fail if stale.")
    args = parser.parse_args(argv)

    expected = render_catalog()
    if not args.check:
        OUTPUT_PATH.write_text(expected, encoding="utf-8", newline="\n")
        return 0

    actual = OUTPUT_PATH.read_text(encoding="utf-8") if OUTPUT_PATH.exists() else ""
    if actual == expected:
        return 0
    sys.stderr.write(
        f"{_display_path(OUTPUT_PATH)} is stale after a backend "
        "provider change.\n"
        f"Regenerate and commit it:\n  {REGENERATE}\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
