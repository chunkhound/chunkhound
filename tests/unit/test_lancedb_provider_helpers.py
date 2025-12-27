"""Unit tests for LanceDB provider helpers."""

from chunkhound.providers.database.lancedb_provider import (
    LanceDBProvider,
    _escape_like_pattern,
)
from chunkhound.providers.database.like_utils import escape_like_pattern


class _FakeSearch:
    def __init__(self, table) -> None:
        self._table = table

    def where(self, clause: str):  # noqa: ANN001 - test stub
        self._table.where_clauses.append(clause)
        return self

    def to_list(self) -> list[dict[str, object]]:
        return self._table.rows


class _FakeFilesTable:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.where_clauses: list[str] = []

    def search(self) -> _FakeSearch:
        return _FakeSearch(self)


class _FakeTable:
    def __init__(self, count: int) -> None:
        self.num_rows = count

    def __len__(self) -> int:
        return self.num_rows


class _FakeChunksTable:
    def __init__(self) -> None:
        self.filters: list[str] = []

    def to_lance(self):  # noqa: ANN001 - test stub
        return self

    def to_table(self, filter: str):  # noqa: ANN001 - test stub
        self.filters.append(filter)
        _, _, ids = filter.partition("IN (")
        ids = ids.rstrip(")")
        count = len([item for item in ids.split(",") if item.strip()])
        return _FakeTable(count)


class _FakeFuzzySearch:
    def __init__(self, table) -> None:
        self._table = table

    def where(self, clause: str):  # noqa: ANN001 - test stub
        self._table.where_clauses.append(clause)
        return self

    def limit(self, limit: int):  # noqa: ANN001 - test stub
        self._table.limits.append(limit)
        return self

    def to_list(self) -> list[dict[str, object]]:
        return self._table.rows


class _FakeChunksTableForFuzzy:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.where_clauses: list[str] = []
        self.limits: list[int] = []

    def search(self) -> _FakeFuzzySearch:
        return _FakeFuzzySearch(self)


def test_escape_like_pattern_handles_metacharacters() -> None:
    escaped = _escape_like_pattern("scope_%[path]'\\name")
    assert "\\%" in escaped
    assert "\\_" in escaped
    assert "\\[" in escaped
    assert "''" in escaped
    assert "\\\\" in escaped


def test_build_path_like_clause_uses_escape() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    clause = provider._build_path_like_clause("scope_%")
    assert "ESCAPE '\\\\'" in clause
    assert "scope\\_\\%" in clause


def test_fetch_file_paths_by_ids_batches_queries() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._files_table = _FakeFilesTable(
        [
            {"id": 2, "path": "b.py"},
            {"id": 1, "path": "a.py"},
        ]
    )

    file_map = provider._fetch_file_paths_by_ids([2, 2, 1])

    assert file_map == {1: "a.py", 2: "b.py"}
    assert len(provider._files_table.where_clauses) == 1
    clause = provider._files_table.where_clauses[0]
    ids = clause.partition("IN (")[2].rstrip(")")
    parsed = {int(item.strip()) for item in ids.split(",") if item.strip()}
    assert parsed == {1, 2}


def test_count_chunks_for_file_ids_batches_large_sets() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._chunks_table = _FakeChunksTable()

    total = provider._count_chunks_for_file_ids(list(range(1001)))

    assert total == 1001
    assert len(provider._chunks_table.filters) == 2
    for clause in provider._chunks_table.filters:
        ids = clause.partition("IN (")[2].rstrip(")")
        parsed = [item for item in ids.split(",") if item.strip()]
        assert len(parsed) <= 1000


def test_search_fuzzy_escapes_like_and_quotes() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._chunks_table = _FakeChunksTableForFuzzy(
        [
            {
                "id": 1,
                "file_id": 1,
                "content": "hello",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "text",
                "language": "python",
                "name": "",
            }
        ]
    )
    provider._files_table = _FakeFilesTable([{"id": 1, "path": "a.py"}])

    query = "a'b%_\\["
    expected = escape_like_pattern(query, escape_quotes=True)

    formatted, _pagination = provider._executor_search_fuzzy(
        object(), {}, query, page_size=10, offset=0, path_filter=None
    )

    assert formatted, "Expected at least one formatted result"
    assert provider._chunks_table.where_clauses, "Expected a WHERE clause to be used"
    clause = provider._chunks_table.where_clauses[0]
    assert clause == f"content LIKE '%{expected}%' ESCAPE '\\\\'"
