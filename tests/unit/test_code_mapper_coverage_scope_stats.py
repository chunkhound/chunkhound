from typing import Any

from chunkhound.code_mapper.coverage import compute_db_scope_stats


class _ProviderWithScopeStats:
    def get_scope_stats(self, scope_prefix: str | None) -> tuple[int, int]:
        if scope_prefix == "scope/":
            return 2, 10
        if scope_prefix is None:
            return 5, 25
        return 0, 0

    def get_all_chunks_with_metadata(self) -> list[dict[str, Any]]:
        raise AssertionError("compute_db_scope_stats should not call legacy chunk scan")


class _Services:
    def __init__(self) -> None:
        self.provider = _ProviderWithScopeStats()


def test_compute_db_scope_stats_prefers_provider_aggregation() -> None:
    services = _Services()

    files_total, chunks_total, scoped_files = compute_db_scope_stats(services, "scope")
    assert files_total == 2
    assert chunks_total == 10
    assert scoped_files == set()


def test_compute_db_scope_stats_root_scope_uses_none_prefix() -> None:
    services = _Services()

    files_total, chunks_total, scoped_files = compute_db_scope_stats(services, "/")
    assert files_total == 5
    assert chunks_total == 25
    assert scoped_files == set()
