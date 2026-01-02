"""Unit tests for LanceDB provider helpers."""

from chunkhound.providers.database.lancedb_provider import (
    LanceDBProvider,
    _escape_like_pattern,
)
from chunkhound.providers.database.like_utils import escape_like_pattern


class _FakeSearch:
    def __init__(self, table) -> None:
        self._table = table
        self._where_clauses: list[str] = []

    def where(self, clause: str):  # noqa: ANN001 - test stub
        self._table.where_clauses.append(clause)
        self._where_clauses.append(clause)
        return self

    def to_list(self) -> list[dict[str, object]]:
        rows = list(self._table.rows)
        for clause in self._where_clauses:
            clause = clause.strip()
            if clause.startswith("id IN (") and clause.endswith(")"):
                ids = clause.partition("IN (")[2].rstrip(")")
                allowed = {
                    int(item.strip())
                    for item in ids.split(",")
                    if item.strip() and item.strip().lstrip("-").isdigit()
                }
                rows = [row for row in rows if int(row.get("id", -1)) in allowed]
                continue

            if clause.startswith("path LIKE '") and "ESCAPE" in clause:
                # Format: path LIKE 'prefix%' ESCAPE '\\'
                pattern = clause.partition("path LIKE '")[2].partition("'")[0]
                if pattern.startswith("%") and pattern.endswith("%") and len(pattern) >= 2:
                    needle = pattern.strip("%")
                    rows = [
                        row
                        for row in rows
                        if needle in str(row.get("path", ""))
                    ]
                elif pattern.startswith("%"):
                    needle = pattern[1:]
                    rows = [
                        row
                        for row in rows
                        if str(row.get("path", "")).endswith(needle)
                    ]
                elif pattern.endswith("%"):
                    prefix = pattern[:-1]
                    rows = [
                        row
                        for row in rows
                        if str(row.get("path", "")).startswith(prefix)
                    ]
                else:
                    rows = [
                        row
                        for row in rows
                        if str(row.get("path", "")) == pattern
                    ]
                continue

        return rows


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
        self._where_clauses: list[str] = []

    def where(self, clause: str):  # noqa: ANN001 - test stub
        self._table.where_clauses.append(clause)
        self._where_clauses.append(clause)
        return self

    def limit(self, limit: int):  # noqa: ANN001 - test stub
        self._table.limits.append(limit)
        return self

    def to_list(self) -> list[dict[str, object]]:
        rows = list(self._table.rows)
        last_limit = self._table.limits[-1] if self._table.limits else None
        if isinstance(last_limit, int) and last_limit >= 0:
            rows = rows[:last_limit]
        return rows


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


def test_search_fuzzy_empty_returns_next_offset() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._chunks_table = None

    _results, pagination = provider._executor_search_fuzzy(
        object(), {}, "needle", page_size=10, offset=0, path_filter=None
    )

    assert pagination["has_more"] is False
    assert pagination["next_offset"] is None


def test_search_fuzzy_applies_path_filter() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._chunks_table = _FakeChunksTableForFuzzy(
        [
            {
                "id": 1,
                "file_id": 1,
                "content": "needle",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "text",
                "language": "python",
                "name": "",
            },
            {
                "id": 2,
                "file_id": 2,
                "content": "needle",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "text",
                "language": "python",
                "name": "",
            },
        ]
    )
    provider._files_table = _FakeFilesTable(
        [
            {"id": 1, "path": "src/a.py"},
            {"id": 2, "path": "tests/b.py"},
        ]
    )

    results, _pagination = provider._executor_search_fuzzy(
        object(), {}, "needle", page_size=10, offset=0, path_filter="src/"
    )

    assert results
    assert all("src/" in row["file_path"] for row in results)
    assert provider._files_table.where_clauses, "Expected files-table filtering"


def test_search_fuzzy_has_more_works_with_offset() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    provider._chunks_table = _FakeChunksTableForFuzzy(
        [
            {
                "id": idx,
                "file_id": 1,
                "content": "needle",
                "start_line": 1,
                "end_line": 1,
                "chunk_type": "text",
                "language": "python",
                "name": "",
            }
            for idx in range(1, 6)
        ]
    )
    provider._files_table = _FakeFilesTable([{"id": 1, "path": "src/a.py"}])

    results, pagination = provider._executor_search_fuzzy(
        object(), {}, "needle", page_size=2, offset=0, path_filter=None
    )

    assert len(results) == 2
    assert pagination["offset"] == 0
    assert pagination["page_size"] == 2
    assert pagination["has_more"] is True
    assert pagination["total_is_estimate"] is True
    assert pagination["next_offset"] == 2
    # total is estimated; it should at least cover the returned page.
    assert pagination["total"] >= 2


class _FakeHead:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def to_list(self) -> list[dict[str, object]]:
        return list(self._rows)


class _FakeVectorSearch:
    def __init__(self, table) -> None:
        self._table = table
        self._limit: int | None = None
        self._where_clauses: list[str] = []

    def where(self, clause: str):  # noqa: ANN001 - test stub
        self._where_clauses.append(clause)
        return self

    def limit(self, limit: int):  # noqa: ANN001 - test stub
        self._limit = limit
        return self

    def to_list(self) -> list[dict[str, object]]:
        rows = list(self._table.rows)
        last_clause = self._where_clauses[-1] if self._where_clauses else ""
        if "file_id IN (" in last_clause:
            ids_tail = last_clause.partition("file_id IN (")[2]
            ids = ids_tail.split(")")[0]
            allowed = {
                int(item.strip())
                for item in ids.split(",")
                if item.strip() and item.strip().lstrip("-").isdigit()
            }
            rows = [row for row in rows if int(row.get("file_id", -1)) in allowed]

        if "_distance <=" in last_clause:
            raw = last_clause.partition("_distance <=")[2].strip().split()[0]
            try:
                max_distance = float(raw)
            except ValueError:
                max_distance = None
            if max_distance is not None:
                rows = [
                    row
                    for row in rows
                    if float(row.get("_distance", 0.0)) <= max_distance
                ]

        if isinstance(self._limit, int) and self._limit >= 0:
            rows = rows[: self._limit]
        return rows


class _FakeChunksTableForSemantic:
    def __init__(self, rows: list[dict[str, object]], sample_rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self._sample_rows = sample_rows

    def count_rows(self) -> int:
        return len(self.rows)

    def head(self, n: int):  # noqa: ANN001 - test stub
        return _FakeHead(self._sample_rows[:n])

    def search(self, *args, **kwargs) -> _FakeVectorSearch:  # noqa: ANN001 - test stub
        return _FakeVectorSearch(self)


def test_search_semantic_threshold_is_similarity_and_has_more_works() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    rows = [
        {
            "id": 1,
            "file_id": 1,
            "_distance": 0.05,  # sim 0.95
            "content": "a",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
        {
            "id": 2,
            "file_id": 1,
            "_distance": 0.25,  # sim 0.75
            "content": "b",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
        {
            "id": 3,
            "file_id": 1,
            "_distance": 0.10,  # sim 0.90
            "content": "c",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
        {
            "id": 4,
            "file_id": 1,
            "_distance": 0.15,  # sim 0.85
            "content": "d",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
        {
            "id": 5,
            "file_id": 1,
            "_distance": 0.20,  # sim 0.80
            "content": "e",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
    ]
    provider._chunks_table = _FakeChunksTableForSemantic(rows=rows, sample_rows=rows)
    provider._files_table = _FakeFilesTable([{"id": 1, "path": "src/a.py"}])

    results, pagination = provider._executor_search_semantic(
        object(),
        {},
        [0.1],
        "p",
        "m",
        page_size=2,
        offset=0,
        threshold=0.8,
        path_filter=None,
    )

    assert len(results) == 2
    assert pagination["has_more"] is True
    assert pagination["total_is_estimate"] is True
    assert pagination["next_offset"] == 2
    # Similarity threshold >= 0.8 should exclude id=2 (sim 0.75).
    returned_ids = {r["chunk_id"] for r in results}
    assert 2 not in returned_ids


def test_search_semantic_applies_path_filter_like_duckdb() -> None:
    provider = LanceDBProvider.__new__(LanceDBProvider)
    rows = [
        {
            "id": 1,
            "file_id": 1,
            "_distance": 0.05,
            "content": "a",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
        {
            "id": 2,
            "file_id": 2,
            "_distance": 0.05,
            "content": "b",
            "start_line": 1,
            "end_line": 1,
            "chunk_type": "text",
            "language": "python",
            "name": "",
            "provider": "p",
            "model": "m",
            "embedding": [0.1],
        },
    ]
    provider._chunks_table = _FakeChunksTableForSemantic(rows=rows, sample_rows=rows)
    provider._files_table = _FakeFilesTable(
        [
            {"id": 1, "path": "src/a.py"},
            {"id": 2, "path": "tests/b.py"},
        ]
    )

    results, _pagination = provider._executor_search_semantic(
        object(),
        {},
        [0.1],
        "p",
        "m",
        page_size=10,
        offset=0,
        threshold=None,
        path_filter="src/",
    )

    assert results
    assert all("src/" in r["file_path"] for r in results)
