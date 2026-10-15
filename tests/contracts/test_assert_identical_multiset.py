"""assert_identical must catch duplicate rows, not just set differences.

`assert_identical` (`tests/contracts/pipeline_harness.py`) is the shared
contract-enforcement helper every Python-vs-Rust parity test in this
directory relies on. Comparing `chunk_tuples`/`embedding_tuples` via Python
`set()` discards duplicate multiplicity: if one pipeline wrote a chunk twice
and the other wrote it once, both sides produce the same *set*, and the
comparison passes silently even though the two runs are not byte-identical.
`chunks_written` is a separately reported counter (not `len(chunk_tuples)`),
so it doesn't backstop this gap either. These tests pin the multiset
contract this helper must enforce.
"""

import pytest

from tests.contracts.pipeline_harness import IndexResult, assert_identical


def _base_result(**overrides: object) -> IndexResult:
    defaults: dict = dict(
        files_processed=1,
        chunks_written=2,
        embeddings_generated=0,
        chunk_tuples=[
            ("main.py", "function", "foo", "def foo(): pass", 1, 1),
            ("main.py", "function", "foo", "def foo(): pass", 1, 1),
        ],
        embedding_tuples=[],
    )
    defaults.update(overrides)
    return IndexResult(**defaults)


class TestAssertIdenticalCatchesDuplicates:
    def test_duplicated_chunk_tuple_is_a_mismatch(self):
        """A has the same chunk twice, B has it once -- must not pass."""
        result_a = _base_result()
        result_b = _base_result(
            chunk_tuples=[
                ("main.py", "function", "foo", "def foo(): pass", 1, 1),
            ]
        )

        with pytest.raises(AssertionError):
            assert_identical(result_a, result_b)

    def test_duplicated_embedding_tuple_is_a_mismatch(self):
        """Same embedding row twice on one side, once on the other."""
        embedding = (
            "main.py", "function", "foo", "openai", "text-embedding-3", 8, (0.1,) * 8
        )
        result_a = _base_result(
            chunk_tuples=[("main.py", "function", "foo", "code", 1, 1)],
            embedding_tuples=[embedding, embedding],
        )
        result_b = _base_result(
            chunk_tuples=[("main.py", "function", "foo", "code", 1, 1)],
            embedding_tuples=[embedding],
        )

        with pytest.raises(AssertionError):
            assert_identical(result_a, result_b)

    def test_identical_multisets_still_pass(self):
        """Sanity: equal duplicate counts on both sides is a real match."""
        result_a = _base_result()
        result_b = _base_result()

        assert_identical(result_a, result_b)  # must not raise
