from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from chunkhound.autodoc.generator import generate_docsite
from chunkhound.autodoc.models import CleanupConfig
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse
from tests.autodoc.site_tree_manifest import build_tree_manifest


class _FrozenDatetime:
    @classmethod
    def now(cls, tz=None):  # noqa: ANN001
        return dt.datetime(2025, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc)


class _Provider(LLMProvider):
    def __init__(self) -> None:
        self._model = "fake"

    @property
    def name(self) -> str:
        return "fake"

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> LLMResponse:
        return LLMResponse(
            content="## Overview\nEnd-user overview.\n\n- Use case 1\n- Use case 2",
            tokens_used=0,
            model=self._model,
            finish_reason="stop",
        )

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, object],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> dict[str, object]:
        return {
            "nav": {"groups": [{"title": "Group", "slugs": ["01-topic-one"]}]},
            "glossary": [
                {"term": "Term", "definition": "Definition.", "pages": ["01-topic-one"]}
            ],
        }

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        return [
            LLMResponse(
                content="## Overview\nCleaned.\n\n## Details\nMore.",
                tokens_used=0,
                model=self._model,
                finish_reason="stop",
            )
            for _ in prompts
        ]

    def estimate_tokens(self, text: str) -> int:
        return 0

    async def health_check(self) -> dict[str, object]:
        return {"ok": True}

    def get_usage_stats(self) -> dict[str, object]:
        return {}


class _LLMManager:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def get_synthesis_provider(self) -> LLMProvider:
        return self._provider


def _write_minimal_input_dir(input_dir: Path) -> None:
    (input_dir / "scope_code_mapper_index.md").write_text(
        "\n".join(
            [
                "# AutoDoc Topics (/repo)",
                "",
                "1. [Topic One](topic_one.md)",
            ]
        ),
        encoding="utf-8",
    )

    (input_dir / "topic_one.md").write_text(
        "\n".join(
            [
                "# Topic One",
                "",
                "Overview body.",
                "",
                "## Sources",
                "",
                "└── repo/",
                "\t└── [1] x.py (1 chunks: L1-2)",
            ]
        ),
        encoding="utf-8",
    )


_LLM_EXPECTED_MANIFEST: dict[str, str] = {
    "README.md": "cbf6809284131ae9896d8c424e845ceb27fa8fe019fd3e7d38e01938737c07d8",
    "astro.config.mjs": (
        "7fcafc68489f3a8262965976a470c0d86da51979cbc64edb7efde601c5de4c32"
    ),
    "package.json": "02c248cf0220edef9e20ab988cb3e39db533cc00a0940cdfaac121ddae22f71d",
    "public/favicon.ico": (
        "d014edc031656dd8a5cb7740ed900d658ba3108ff6fcb977fc3ff4f758c10f0b"
    ),
    "src/data/nav.json": (
        "e6b5c6c9ebfd335199f501c1c88c11afde89671f89a7bd672c543b318c323af1"
    ),
    "src/data/search.json": (
        "934d302ac5555b6e87a1c9c2c79c8dfce3d7d39a81d2766c982d4359d8072deb"
    ),
    "src/data/site.json": (
        "33e943d1f4e8512ccd202ad44dc60e917c0e6a6642fee3cd399f9b30a936a182"
    ),
    "src/layouts/DocLayout.astro": (
        "55ab6e3eff2a4c4fa20dec670c77899acd237499098805bf786bf115e36e0994"
    ),
    "src/pages/glossary.md": (
        "b060fc17e9ce790346481ab7e967a38659a39e850f9f9cb4a1d908eee06235bb"
    ),
    "src/pages/index.md": (
        "c711195af8be20151b8a0e9b64c35efbd5d3dc5b6da01afdbe700a7b37ed9087"
    ),
    "src/pages/topics/01-topic-one.md": (
        "354dc58d49946889c1c4fd8d3e631671173ba60cbe04ab494de8cb0a1553cef9"
    ),
    "src/styles/global.css": (
        "cc160bacd9f2702fa7b305fe798f380f25135639426451bbd886e276220bc565"
    ),
    "tsconfig.json": "ec0d7fbe45c5b2efb1c617eec3a7ee249d72974163c5309843420904703ee0a4",
}


@pytest.mark.asyncio
async def test_generate_docsite_requires_llm_manager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("chunkhound.autodoc.generator.datetime", _FrozenDatetime)

    input_dir = Path("input")
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    with pytest.raises(RuntimeError, match="LLM"):
        await generate_docsite(
            input_dir=input_dir,
            output_dir=Path("out"),
            llm_manager=None,
            cleanup_config=CleanupConfig(
                mode="llm",
                batch_size=1,
                max_completion_tokens=512,
                audience="end-user",
            ),
            site_title=None,
            site_tagline=None,
        )

    assert not Path("out").exists()


@pytest.mark.asyncio
async def test_generate_docsite_llm_mode_emits_byte_stable_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("chunkhound.autodoc.generator.datetime", _FrozenDatetime)

    input_dir = Path("input")
    input_dir.mkdir(parents=True)
    _write_minimal_input_dir(input_dir)

    await generate_docsite(
        input_dir=input_dir,
        output_dir=Path("out"),
        llm_manager=_LLMManager(_Provider()),  # type: ignore[arg-type]
        cleanup_config=CleanupConfig(
            mode="llm",
            batch_size=1,
            max_completion_tokens=512,
            audience="end-user",
        ),
        site_title=None,
        site_tagline=None,
    )

    assert build_tree_manifest(Path("out")) == _LLM_EXPECTED_MANIFEST
