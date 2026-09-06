"""User-facing contracts for multi-hop materialization-limit failures."""

import argparse
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import search as search_command_module
from chunkhound.core.exceptions import MaterializationLimitError
from chunkhound.mcp_server import tools as tools_module

LIMIT_MESSAGE = (
    "Requested result window ends at 501, beyond the multi-hop materialization "
    "limit of 500"
)


def _search_args() -> argparse.Namespace:
    return argparse.Namespace(
        verbose=False,
        single_hop=False,
        multi_hop=False,
        commit_range=None,
        commit_hash=None,
        last_n=None,
        regex=False,
        query="query",
        page_size=2,
        offset=499,
        path_filter=None,
        vector_source="diff",
    )


def _patch_cli_dependencies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    errors: list[str] = []
    monkeypatch.setattr(
        search_command_module, "verify_database_exists", lambda _: ":memory:"
    )
    monkeypatch.setattr(search_command_module, "configure_registry", lambda _: None)
    monkeypatch.setattr(search_command_module, "create_services", lambda **_: object())
    monkeypatch.setattr(
        search_command_module.RichOutputFormatter,
        "error",
        lambda _, message: errors.append(message),
    )
    return errors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            MaterializationLimitError(LIMIT_MESSAGE),
            f"Search failed: {LIMIT_MESSAGE}\nReduce --page-size or --offset; "
            "multi-hop search results must fit within their configured result limit.",
        ),
        (
            MaterializationLimitError(
                f"{LIMIT_MESSAGE}. Reduce page_size or offset; multi-hop search "
                "must fit within its configured result limit.",
                guidance_present=True,
            ),
            f"Search failed: {LIMIT_MESSAGE}. Reduce page_size or offset; multi-hop "
            "search must fit within its configured result limit.",
        ),
    ],
)
async def test_cli_explains_materialization_limit(
    monkeypatch: pytest.MonkeyPatch,
    error: MaterializationLimitError,
    expected_message: str,
) -> None:
    errors = _patch_cli_dependencies(monkeypatch)

    async def raise_limit(**_: object) -> dict[str, object]:
        raise error

    monkeypatch.setattr(search_command_module, "search_impl", raise_limit)

    with pytest.raises(SystemExit, match="1"):
        await search_command_module.search_command(
            _search_args(), SimpleNamespace(embedding=None)
        )

    assert errors == [expected_message]


@pytest.mark.asyncio
async def test_cli_does_not_misclassify_other_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = _patch_cli_dependencies(monkeypatch)

    async def raise_value_error(**_: object) -> dict[str, object]:
        raise ValueError("invalid semantic request")

    monkeypatch.setattr(search_command_module, "search_impl", raise_value_error)

    with pytest.raises(SystemExit, match="1"):
        await search_command_module.search_command(
            _search_args(), SimpleNamespace(embedding=None)
        )

    assert errors == ["Search failed: invalid semantic request"]


@pytest.mark.asyncio
async def test_mcp_explains_materialization_limit() -> None:
    async def raise_limit(
        **_: object,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        raise MaterializationLimitError(LIMIT_MESSAGE)

    services = SimpleNamespace(
        search_service=SimpleNamespace(search_semantic=raise_limit)
    )
    embedding_manager = SimpleNamespace(
        list_providers=lambda: [object()],
        get_provider=lambda: SimpleNamespace(name="provider", model="model"),
    )

    with pytest.raises(MaterializationLimitError) as exc_info:
        await tools_module.search_impl(
            services=services,
            embedding_manager=embedding_manager,
            type="semantic",
            query="query",
        )

    assert exc_info.value.guidance_present is True
    assert str(exc_info.value) == (
        f"{LIMIT_MESSAGE}. Reduce page_size or offset; multi-hop search must fit "
        "within its configured result limit."
    )
