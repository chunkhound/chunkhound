"""Unit tests for IndexProfile helpers."""

from chunkhound.core.diagnostics.index_profile import DbOpStats, IndexProfile


def test_db_op_stats_accumulates() -> None:
    s = DbOpStats()
    s.record_merge_insert(10, 0.5)
    s.record_merge_insert(5, 0.25)
    s.record_optimize(1.0)
    d = s.as_dict()
    assert d["merge_insert_calls"] == 2
    assert d["merge_insert_rows"] == 15
    assert d["merge_insert_s"] == 0.75
    assert d["optimize_calls"] == 1
    assert d["optimize_s"] == 1.0


def test_index_profile_phases() -> None:
    p = IndexProfile()
    with p.phase("a"):
        pass
    with p.phase("a"):
        pass
    with p.phase("b"):
        pass
    d = p.as_dict()
    assert "a" in d["phases_s"]
    assert "b" in d["phases_s"]
    assert d["phases_s"]["a"] >= 0.0
    report = p.format_report()
    assert "index profile" in report


def test_index_profile_nested_phases_accumulate_independently() -> None:
    """Outer and inner phase timers both accumulate (Instr soak sub-phases)."""
    p = IndexProfile()
    with p.phase("outer"):
        with p.phase("inner"):
            pass
        with p.phase("inner"):
            pass
    d = p.as_dict()
    assert "outer" in d["phases_s"]
    assert "inner" in d["phases_s"]
    assert d["phases_s"]["outer"] >= d["phases_s"]["inner"]
