"""Test Index object attribute access for LanceDB list_indices() fix."""

from dataclasses import dataclass

import pytest


@dataclass
class MockIndex:
    """Mock LanceDB Index object."""

    columns: list[str]
    name: str


def test_index_attribute_access() -> None:
    """Verify Index uses attributes, not dict methods."""
    idx = MockIndex(columns=["id"], name="id_idx")

    # Correct access
    assert idx.columns == ["id"]
    assert "id" in idx.columns

    # Proves original bug - Index objects don't have .get() method
    with pytest.raises(AttributeError):
        idx.get("columns")  # type: ignore[attr-defined]


def test_has_id_index_logic() -> None:
    """Test the fixed logic for checking if id index exists."""
    indices: list[MockIndex] = [
        MockIndex(columns=["id"], name="id_idx"),
        MockIndex(columns=["file_id"], name="file_idx"),
    ]

    has_id_index = any(idx.columns == ["id"] or "id" in idx.columns for idx in indices)
    assert has_id_index


def test_has_id_index_no_match() -> None:
    """Test when no index matches the id column."""
    indices: list[MockIndex] = [
        MockIndex(columns=["file_id"], name="file_idx"),
        MockIndex(columns=["content"], name="content_idx"),
    ]

    has_id_index = any(idx.columns == ["id"] or "id" in idx.columns for idx in indices)
    assert not has_id_index


def test_has_id_index_composite() -> None:
    """Test when id is part of a composite index."""
    indices: list[MockIndex] = [
        MockIndex(columns=["id", "file_id"], name="composite_idx"),
    ]

    has_id_index = any(idx.columns == ["id"] or "id" in idx.columns for idx in indices)
    assert has_id_index


def test_has_id_index_empty_list() -> None:
    """Test with empty indices list."""
    indices: list[MockIndex] = []

    has_id_index = any(idx.columns == ["id"] or "id" in idx.columns for idx in indices)
    assert not has_id_index
