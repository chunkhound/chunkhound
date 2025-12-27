"""Tests for ConstantsLedger."""

import pytest

from chunkhound.services.research.shared.constants_ledger import (
    CONSTANTS_INSTRUCTION_FULL,
    CONSTANTS_INSTRUCTION_SHORT,
    ConstantEntry,
    ConstantsLedger,
)


class TestConstantEntry:
    """Tests for ConstantEntry dataclass."""

    def test_creation_with_all_fields(self):
        entry = ConstantEntry(name="MAX_SIZE", file_path="config.py", type="int")
        assert entry.name == "MAX_SIZE"
        assert entry.file_path == "config.py"
        assert entry.type == "int"

    def test_creation_without_type(self):
        entry = ConstantEntry(name="TIMEOUT", file_path="settings.py")
        assert entry.name == "TIMEOUT"
        assert entry.file_path == "settings.py"
        assert entry.type is None

    def test_frozen_immutability(self):
        entry = ConstantEntry(name="X", file_path="a.py")
        with pytest.raises(AttributeError):
            entry.name = "Y"  # type: ignore[misc]


class TestConstantsLedger:
    """Tests for ConstantsLedger."""

    def test_from_chunks_extracts_constants(self):
        chunks = [
            {
                "file_path": "config.py",
                "metadata": {
                    "constants": [
                        {"name": "MAX_SIZE", "type": "int"},
                        {"name": "TIMEOUT"},
                    ]
                },
            },
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 2
        assert "config.py:MAX_SIZE" in ledger.entries
        assert ledger.entries["config.py:MAX_SIZE"].type == "int"
        assert "config.py:TIMEOUT" in ledger.entries
        assert ledger.entries["config.py:TIMEOUT"].type is None

    def test_from_chunks_handles_missing_metadata(self):
        chunks = [{"file_path": "foo.py"}]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 0

    def test_from_chunks_handles_none_metadata(self):
        chunks = [{"file_path": "foo.py", "metadata": None}]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 0

    def test_from_chunks_handles_none_constants(self):
        chunks = [{"file_path": "foo.py", "metadata": {"constants": None}}]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 0

    def test_from_chunks_handles_empty_constants(self):
        chunks = [{"file_path": "foo.py", "metadata": {"constants": []}}]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 0

    def test_from_chunks_skips_constants_without_name(self):
        chunks = [
            {
                "file_path": "a.py",
                "metadata": {
                    "constants": [
                        {"type": "str"},  # No name
                        {"name": "VALID"},
                    ]
                },
            }
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 1
        assert "a.py:VALID" in ledger.entries

    def test_from_chunks_deduplicates_by_file_and_name(self):
        chunks = [
            {"file_path": "a.py", "metadata": {"constants": [{"name": "X"}]}},
            {"file_path": "a.py", "metadata": {"constants": [{"name": "X"}]}},
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 1

    def test_from_chunks_different_files_same_name(self):
        chunks = [
            {"file_path": "a.py", "metadata": {"constants": [{"name": "X"}]}},
            {"file_path": "b.py", "metadata": {"constants": [{"name": "X"}]}},
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        assert len(ledger) == 2
        assert "a.py:X" in ledger.entries
        assert "b.py:X" in ledger.entries

    def test_merge_combines_ledgers(self):
        ledger1 = ConstantsLedger(
            entries={
                "a.py:X": ConstantEntry(name="X", file_path="a.py"),
            }
        )
        ledger2 = ConstantsLedger(
            entries={
                "b.py:Y": ConstantEntry(name="Y", file_path="b.py"),
            }
        )
        merged = ledger1.merge(ledger2)
        assert len(merged) == 2
        assert "a.py:X" in merged.entries
        assert "b.py:Y" in merged.entries

    def test_merge_is_immutable(self):
        ledger1 = ConstantsLedger(
            entries={
                "a.py:X": ConstantEntry(name="X", file_path="a.py"),
            }
        )
        ledger2 = ConstantsLedger(
            entries={
                "b.py:Y": ConstantEntry(name="Y", file_path="b.py"),
            }
        )
        _ = ledger1.merge(ledger2)
        assert len(ledger1) == 1  # Original unchanged
        assert len(ledger2) == 1  # Original unchanged

    def test_merge_overwrites_duplicates(self):
        ledger1 = ConstantsLedger(
            entries={
                "a.py:X": ConstantEntry(name="X", file_path="a.py", type="int"),
            }
        )
        ledger2 = ConstantsLedger(
            entries={
                "a.py:X": ConstantEntry(name="X", file_path="a.py", type="str"),
            }
        )
        merged = ledger1.merge(ledger2)
        assert len(merged) == 1
        assert merged.entries["a.py:X"].type == "str"  # Second wins


class TestConstantsLedgerPromptContext:
    """Tests for get_prompt_context method."""

    def test_empty_ledger_returns_empty_string(self):
        ledger = ConstantsLedger()
        assert ledger.get_prompt_context() == ""

    def test_prompt_context_includes_header(self):
        chunks = [
            {"file_path": "c.py", "metadata": {"constants": [{"name": "X"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "## Global Constants" in context

    def test_prompt_context_includes_file_path(self):
        chunks = [
            {"file_path": "config.py", "metadata": {"constants": [{"name": "X"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "**config.py**" in context

    def test_prompt_context_includes_constant_name(self):
        chunks = [
            {"file_path": "c.py", "metadata": {"constants": [{"name": "MAX_SIZE"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "MAX_SIZE" in context

    def test_prompt_context_includes_type_suffix(self):
        chunks = [
            {
                "file_path": "c.py",
                "metadata": {"constants": [{"name": "MAX_SIZE", "type": "int"}]},
            }
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "(int)" in context

    def test_prompt_context_includes_values(self):
        chunks = [
            {
                "file_path": "c.py",
                "metadata": {
                    "constants": [{"name": "MAX_SIZE", "value": "100", "type": "int"}]
                },
            }
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "MAX_SIZE = 100 (int)" in context

    def test_prompt_context_respects_max_entries(self):
        # Use zero-padded names for predictable sorting
        chunks = [
            {
                "file_path": "c.py",
                "metadata": {
                    "constants": [{"name": f"CONST_{i:03d}"} for i in range(100)]
                },
            }
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context(max_entries=10)
        assert "CONST_000" in context
        assert "CONST_009" in context
        assert "CONST_010" not in context  # Should be cut off
        assert "... and 90 more constants" in context

    def test_prompt_context_groups_by_file(self):
        chunks = [
            {"file_path": "a.py", "metadata": {"constants": [{"name": "X"}]}},
            {"file_path": "b.py", "metadata": {"constants": [{"name": "Y"}]}},
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        context = ledger.get_prompt_context()
        assert "**a.py**" in context
        assert "**b.py**" in context
        # X should appear after a.py header
        a_pos = context.find("**a.py**")
        x_pos = context.find("  - X")
        assert a_pos < x_pos


class TestConstantsLedgerReportSuffix:
    """Tests for get_report_suffix method."""

    def test_empty_ledger_returns_empty_string(self):
        ledger = ConstantsLedger()
        assert ledger.get_report_suffix() == ""

    def test_report_suffix_has_header(self):
        chunks = [
            {"file_path": "x.py", "metadata": {"constants": [{"name": "A"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        suffix = ledger.get_report_suffix()
        assert "## Constants Referenced" in suffix

    def test_report_suffix_includes_file_path(self):
        chunks = [
            {"file_path": "config.py", "metadata": {"constants": [{"name": "A"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        suffix = ledger.get_report_suffix()
        assert "**config.py**" in suffix

    def test_report_suffix_includes_constant_with_value_and_type(self):
        chunks = [
            {
                "file_path": "x.py",
                "metadata": {
                    "constants": [{"name": "A", "value": "42", "type": "int"}]
                },
            }
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        suffix = ledger.get_report_suffix()
        assert "A = 42 (int)" in suffix

    def test_report_suffix_constant_without_value(self):
        chunks = [
            {"file_path": "x.py", "metadata": {"constants": [{"name": "B"}]}}
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        suffix = ledger.get_report_suffix()
        assert "  - B" in suffix

    def test_report_suffix_sorted_by_file_then_name(self):
        chunks = [
            {"file_path": "z.py", "metadata": {"constants": [{"name": "B"}]}},
            {"file_path": "a.py", "metadata": {"constants": [{"name": "X"}]}},
            {"file_path": "a.py", "metadata": {"constants": [{"name": "A"}]}},
        ]
        ledger = ConstantsLedger.from_chunks(chunks)
        suffix = ledger.get_report_suffix()
        # a.py should come before z.py
        a_pos = suffix.find("a.py")
        z_pos = suffix.find("z.py")
        assert a_pos < z_pos
        # Within a.py, A should come before X
        a_const_pos = suffix.find("  - A")
        x_const_pos = suffix.find("  - X")
        assert a_const_pos < x_const_pos


class TestConstantsLedgerLen:
    """Tests for __len__ method."""

    def test_empty_ledger_len_zero(self):
        ledger = ConstantsLedger()
        assert len(ledger) == 0

    def test_len_matches_entries_count(self):
        ledger = ConstantsLedger(
            entries={
                "a.py:X": ConstantEntry(name="X", file_path="a.py"),
                "b.py:Y": ConstantEntry(name="Y", file_path="b.py"),
            }
        )
        assert len(ledger) == 2


class TestConstantsInstructionConstants:
    """Test constant instruction text helpers."""

    def test_instruction_constants_exist(self):
        """Test that instruction constants are defined."""
        assert CONSTANTS_INSTRUCTION_FULL
        assert CONSTANTS_INSTRUCTION_SHORT
        assert isinstance(CONSTANTS_INSTRUCTION_FULL, str)
        assert isinstance(CONSTANTS_INSTRUCTION_SHORT, str)

    def test_instruction_full_has_important_prefix(self):
        """Test that full instruction starts with IMPORTANT."""
        assert CONSTANTS_INSTRUCTION_FULL.startswith("IMPORTANT:")

    def test_instruction_short_no_important_prefix(self):
        """Test that short instruction does not have IMPORTANT prefix."""
        assert not CONSTANTS_INSTRUCTION_SHORT.startswith("IMPORTANT:")

    def test_instruction_full_mentions_constant_names(self):
        """Test that full instruction mentions constant names."""
        assert "constant names" in CONSTANTS_INSTRUCTION_FULL
        assert "MAX_RETRIES" in CONSTANTS_INSTRUCTION_FULL

    def test_instruction_short_mentions_constant_names(self):
        """Test that short instruction mentions constant names."""
        assert "constant names" in CONSTANTS_INSTRUCTION_SHORT


class TestConstantsLedgerPromptInstruction:
    """Test ConstantsLedger.get_prompt_instruction() method."""

    def test_empty_ledger_returns_empty_string(self):
        """Test that empty ledger returns empty string."""
        ledger = ConstantsLedger()
        result = ledger.get_prompt_instruction()
        assert result == ""

    def test_ledger_with_constants_returns_instruction(self):
        """Test that ledger with constants returns formatted instruction."""
        ledger = ConstantsLedger()
        ledger.entries["file.py:MAX"] = ConstantEntry(
            name="MAX", file_path="file.py", value="100", type="int"
        )

        result = ledger.get_prompt_instruction()

        # Should contain the constants context
        assert "## Global Constants" in result
        assert "MAX" in result
        assert "file.py" in result

        # Should contain the full instruction
        assert CONSTANTS_INSTRUCTION_FULL in result

    def test_instruction_full_form_by_default(self):
        """Test that full form instruction is used by default."""
        ledger = ConstantsLedger()
        ledger.entries["file.py:MAX"] = ConstantEntry(
            name="MAX", file_path="file.py", value="100"
        )

        result = ledger.get_prompt_instruction()
        assert CONSTANTS_INSTRUCTION_FULL in result
        assert CONSTANTS_INSTRUCTION_SHORT not in result

    def test_instruction_short_form_when_requested(self):
        """Test that short form instruction is used when requested."""
        ledger = ConstantsLedger()
        ledger.entries["file.py:MAX"] = ConstantEntry(
            name="MAX", file_path="file.py", value="100"
        )

        result = ledger.get_prompt_instruction(use_short_form=True)
        assert CONSTANTS_INSTRUCTION_SHORT in result
        assert CONSTANTS_INSTRUCTION_FULL not in result

    def test_instruction_respects_max_entries(self):
        """Test that max_entries parameter is passed through."""
        ledger = ConstantsLedger()
        for i in range(100):
            ledger.entries[f"file.py:CONST_{i:03d}"] = ConstantEntry(
                name=f"CONST_{i:03d}", file_path="file.py", value=str(i)
            )

        result = ledger.get_prompt_instruction(max_entries=10)

        # Should have truncation message
        assert "... and" in result
        # Should still have instruction
        assert CONSTANTS_INSTRUCTION_FULL in result

    def test_instruction_format_matches_expected_pattern(self):
        """Test that instruction follows expected format."""
        ledger = ConstantsLedger()
        ledger.entries["config.py:MAX_RETRIES"] = ConstantEntry(
            name="MAX_RETRIES", file_path="config.py", value="3"
        )

        result = ledger.get_prompt_instruction()

        # Expected format: \n\n{context}\n\n{instruction}
        lines = result.split("\n")
        assert lines[0] == ""  # Leading newline
        assert lines[1] == ""  # Second newline
        assert "## Global Constants" in result
        # Should have newlines before instruction
        assert "\n\n" + CONSTANTS_INSTRUCTION_FULL in result

    def test_multiple_constants_from_different_files(self):
        """Test instruction with constants from multiple files."""
        ledger = ConstantsLedger()
        ledger.entries["config.py:MAX"] = ConstantEntry(
            name="MAX", file_path="config.py", value="100"
        )
        ledger.entries["settings.py:API_KEY"] = ConstantEntry(
            name="API_KEY", file_path="settings.py", value='"secret"'
        )

        result = ledger.get_prompt_instruction()

        # Should contain both files and constants
        assert "config.py" in result
        assert "settings.py" in result
        assert "MAX" in result
        assert "API_KEY" in result
        # Should have instruction at the end
        assert CONSTANTS_INSTRUCTION_FULL in result
