"""Tests for the unified EvidenceLedger module.

Comprehensive unit tests covering:
1. EvidenceType and ConfidenceLevel enums
2. ConstantEntry and FactEntry dataclasses
3. EvidenceLedger collection and operations
4. FactExtractor LLM-based extraction
5. EntityLink and FactConflict
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.services.research.shared.evidence_ledger import (
    ConfidenceLevel,
    ConstantEntry,
    EntityLink,
    EvidenceLedger,
    EvidenceType,
    FactConflict,
    FactEntry,
    FactExtractor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_constant():
    """Sample constant for testing."""
    return ConstantEntry(
        name="MAX_RETRIES",
        file_path="config.py",
        value="3",
        type="int",
    )


@pytest.fixture
def sample_fact():
    """Sample fact for testing."""
    return FactEntry(
        fact_id="abc123def456",
        statement="SearchService retries up to MAX_RETRIES times",
        file_path="services/search.py",
        start_line=45,
        end_line=52,
        category="behavior",
        confidence=ConfidenceLevel.DEFINITE,
        entities=("SearchService", "MAX_RETRIES"),
        cluster_id=0,
    )


@pytest.fixture
def sample_ledger(sample_constant, sample_fact):
    """Ledger with a sample constant and fact."""
    ledger = EvidenceLedger()
    ledger.add_constant(sample_constant)
    ledger.add_fact(sample_fact)
    return ledger


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing extractor."""
    provider = MagicMock()
    provider.complete = AsyncMock()
    return provider


# =============================================================================
# EvidenceType Tests
# =============================================================================


class TestEvidenceType:
    """Tests for the EvidenceType enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert EvidenceType.CONSTANT.value == "constant"
        assert EvidenceType.FACT.value == "fact"

    def test_all_types_present(self):
        """Test all expected evidence types exist."""
        types = list(EvidenceType)
        assert len(types) == 2


# =============================================================================
# ConfidenceLevel Tests
# =============================================================================


class TestConfidenceLevel:
    """Tests for the ConfidenceLevel enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert ConfidenceLevel.DEFINITE.value == "definite"
        assert ConfidenceLevel.LIKELY.value == "likely"
        assert ConfidenceLevel.INFERRED.value == "inferred"
        assert ConfidenceLevel.UNCERTAIN.value == "uncertain"

    def test_string_conversion(self):
        """Test conversion from string to enum."""
        assert ConfidenceLevel("definite") == ConfidenceLevel.DEFINITE
        assert ConfidenceLevel("likely") == ConfidenceLevel.LIKELY
        assert ConfidenceLevel("inferred") == ConfidenceLevel.INFERRED
        assert ConfidenceLevel("uncertain") == ConfidenceLevel.UNCERTAIN

    def test_invalid_string_raises(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            ConfidenceLevel("invalid")

    def test_all_levels_present(self):
        """Test all expected confidence levels exist."""
        levels = list(ConfidenceLevel)
        assert len(levels) == 4


# =============================================================================
# ConstantEntry Tests
# =============================================================================


class TestConstantEntry:
    """Tests for the ConstantEntry dataclass."""

    def test_creation_with_all_fields(self, sample_constant):
        """Test ConstantEntry can be created with all fields."""
        assert sample_constant.name == "MAX_RETRIES"
        assert sample_constant.file_path == "config.py"
        assert sample_constant.value == "3"
        assert sample_constant.type == "int"

    def test_creation_with_minimal_fields(self):
        """Test ConstantEntry can be created with minimal fields."""
        entry = ConstantEntry(name="DEBUG", file_path="settings.py")
        assert entry.name == "DEBUG"
        assert entry.file_path == "settings.py"
        assert entry.value is None
        assert entry.type is None

    def test_frozen_immutability(self, sample_constant):
        """Test ConstantEntry is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_constant.name = "modified"


# =============================================================================
# FactEntry Tests
# =============================================================================


class TestFactEntry:
    """Tests for the FactEntry dataclass."""

    def test_creation_with_all_fields(self, sample_fact):
        """Test FactEntry can be created with all fields."""
        assert sample_fact.fact_id == "abc123def456"
        assert sample_fact.statement == "SearchService retries up to MAX_RETRIES times"
        assert sample_fact.file_path == "services/search.py"
        assert sample_fact.start_line == 45
        assert sample_fact.end_line == 52
        assert sample_fact.category == "behavior"
        assert sample_fact.confidence == ConfidenceLevel.DEFINITE
        assert sample_fact.entities == ("SearchService", "MAX_RETRIES")
        assert sample_fact.cluster_id == 0

    def test_frozen_immutability(self, sample_fact):
        """Test FactEntry is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_fact.statement = "modified"

    def test_generate_id_consistent(self):
        """Test generate_id produces consistent hashes for same input."""
        id1 = FactEntry.generate_id("statement", "file.py", 10, 20)
        id2 = FactEntry.generate_id("statement", "file.py", 10, 20)
        assert id1 == id2
        assert len(id1) == 12

    def test_generate_id_different_for_different_inputs(self):
        """Test generate_id produces different hashes for different inputs."""
        id1 = FactEntry.generate_id("statement A", "file.py", 10, 20)
        id2 = FactEntry.generate_id("statement B", "file.py", 10, 20)
        id3 = FactEntry.generate_id("statement A", "other.py", 10, 20)
        id4 = FactEntry.generate_id("statement A", "file.py", 11, 20)
        id5 = FactEntry.generate_id("statement A", "file.py", 10, 21)

        # All IDs should be different
        all_ids = {id1, id2, id3, id4, id5}
        assert len(all_ids) == 5

    def test_generate_id_is_hex(self):
        """Test generate_id produces valid hex string."""
        id_result = FactEntry.generate_id("test", "file.py", 1, 10)
        # Should be valid hex
        int(id_result, 16)

    def test_entities_is_tuple(self, sample_fact):
        """Test entities field is a tuple (immutable)."""
        assert isinstance(sample_fact.entities, tuple)


# =============================================================================
# EvidenceLedger Basic Tests
# =============================================================================


class TestEvidenceLedgerBasic:
    """Basic EvidenceLedger tests."""

    def test_empty_ledger_creation(self):
        """Test creating an empty ledger."""
        ledger = EvidenceLedger()
        assert len(ledger) == 0
        assert ledger.constants_count == 0
        assert ledger.facts_count == 0
        assert len(ledger.constants) == 0
        assert len(ledger.facts) == 0
        assert len(ledger.entity_index) == 0
        assert len(ledger.conflicts) == 0

    def test_add_constant_with_dedup(self, sample_constant):
        """Test add_constant deduplicates by file_path:name."""
        ledger = EvidenceLedger()
        ledger.add_constant(sample_constant)

        assert ledger.constants_count == 1

        # Adding same constant again should not increase count
        ledger.add_constant(sample_constant)
        assert ledger.constants_count == 1

        # Different constant should add
        other = ConstantEntry(name="OTHER", file_path="config.py", value="x")
        ledger.add_constant(other)
        assert ledger.constants_count == 2

    def test_add_fact_adds_to_facts(self, sample_fact):
        """Test add_fact adds fact to facts dict."""
        ledger = EvidenceLedger()
        ledger.add_fact(sample_fact)

        assert ledger.facts_count == 1
        assert sample_fact.fact_id in ledger.facts
        assert ledger.facts[sample_fact.fact_id] == sample_fact

    def test_add_fact_updates_entity_index(self, sample_fact):
        """Test add_fact updates entity index."""
        ledger = EvidenceLedger()
        ledger.add_fact(sample_fact)

        # Both entities should be indexed (normalized to lowercase)
        assert "searchservice" in ledger.entity_index
        assert "max_retries" in ledger.entity_index

        # Each should point to the fact
        assert sample_fact.fact_id in ledger.entity_index["searchservice"].fact_ids
        assert sample_fact.fact_id in ledger.entity_index["max_retries"].fact_ids

    def test_add_fact_multiple_facts_same_entity(self):
        """Test multiple facts referencing same entity accumulate in index."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="Statement 1",
            file_path="file.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SharedEntity",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Statement 2",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=("SharedEntity",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        link = ledger.entity_index["sharedentity"]
        assert len(link.fact_ids) == 2
        assert "fact1" in link.fact_ids
        assert "fact2" in link.fact_ids

    def test_constants_count_and_facts_count_properties(self, sample_ledger):
        """Test constants_count and facts_count properties."""
        assert sample_ledger.constants_count == 1
        assert sample_ledger.facts_count == 1

    def test_len_returns_total(self, sample_ledger):
        """Test __len__ returns total evidence count."""
        assert len(sample_ledger) == 2  # 1 constant + 1 fact


# =============================================================================
# EvidenceLedger Merge Tests
# =============================================================================


class TestEvidenceLedgerMerge:
    """Tests for ledger merge operation."""

    def test_merge_combines_constants_and_facts(self):
        """Test merge combines entries from both ledgers."""
        ledger1 = EvidenceLedger()
        ledger2 = EvidenceLedger()

        const1 = ConstantEntry(name="A", file_path="file1.py", value="1")
        const2 = ConstantEntry(name="B", file_path="file2.py", value="2")
        fact1 = FactEntry(
            fact_id="fact1",
            statement="Statement 1",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("Entity1",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Statement 2",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=("Entity2",),
            cluster_id=1,
        )

        ledger1.add_constant(const1)
        ledger1.add_fact(fact1)
        ledger2.add_constant(const2)
        ledger2.add_fact(fact2)

        merged = ledger1.merge(ledger2)

        assert merged.constants_count == 2
        assert merged.facts_count == 2
        assert "fact1" in merged.facts
        assert "fact2" in merged.facts

    def test_merge_is_immutable(self, sample_ledger):
        """Test merge does not modify original ledgers."""
        ledger1 = sample_ledger
        ledger2 = EvidenceLedger()

        original_const_count = ledger1.constants_count
        original_fact_count = ledger1.facts_count

        fact2 = FactEntry(
            fact_id="fact2",
            statement="Another statement",
            file_path="file2.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.LIKELY,
            entities=("OtherEntity",),
            cluster_id=1,
        )
        ledger2.add_fact(fact2)
        ledger2.add_constant(ConstantEntry(name="X", file_path="x.py", value="y"))

        _ = ledger1.merge(ledger2)

        # Original ledgers unchanged
        assert ledger1.constants_count == original_const_count
        assert ledger1.facts_count == original_fact_count
        assert ledger2.constants_count == 1
        assert ledger2.facts_count == 1

    def test_merge_combines_conflicts(self):
        """Test merge combines conflict lists."""
        ledger1 = EvidenceLedger()
        ledger2 = EvidenceLedger()

        ledger1.conflicts.append(FactConflict("a", "b", "reason1"))
        ledger2.conflicts.append(FactConflict("c", "d", "reason2"))

        merged = ledger1.merge(ledger2)

        assert len(merged.conflicts) == 2


# =============================================================================
# EvidenceLedger Query Tests (Facts Only)
# =============================================================================


class TestEvidenceLedgerQueries:
    """Tests for ledger query methods."""

    def test_get_facts_for_files(self):
        """Test get_facts_for_files returns correct facts."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="Statement 1",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=(),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Statement 2",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=(),
            cluster_id=0,
        )
        fact3 = FactEntry(
            fact_id="fact3",
            statement="Statement 3",
            file_path="file1.py",
            start_line=20,
            end_line=25,
            category="constraint",
            confidence=ConfidenceLevel.INFERRED,
            entities=(),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)
        ledger.add_fact(fact3)

        # Query for file1.py
        results = ledger.get_facts_for_files({"file1.py"})
        assert len(results) == 2
        fact_ids = {f.fact_id for f in results}
        assert "fact1" in fact_ids
        assert "fact3" in fact_ids
        assert "fact2" not in fact_ids

    def test_get_facts_for_entity(self):
        """Test get_facts_for_entity returns correct facts."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="Statement about SearchService",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Statement about DatabaseService",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=("DatabaseService",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        results = ledger.get_facts_for_entity("SearchService")
        assert len(results) == 1
        assert results[0].fact_id == "fact1"

    def test_get_facts_for_entity_case_insensitive(self):
        """Test get_facts_for_entity is case insensitive."""
        ledger = EvidenceLedger()

        fact = FactEntry(
            fact_id="fact1",
            statement="Statement",
            file_path="file.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )
        ledger.add_fact(fact)

        # Query with different case
        results = ledger.get_facts_for_entity("searchservice")
        assert len(results) == 1

        results = ledger.get_facts_for_entity("SEARCHSERVICE")
        assert len(results) == 1

    def test_get_facts_for_entity_not_found(self, sample_ledger):
        """Test get_facts_for_entity returns empty list for unknown entity."""
        results = sample_ledger.get_facts_for_entity("NonExistentEntity")
        assert results == []

    def test_get_related_facts(self):
        """Test get_related_facts uses entity links."""
        ledger = EvidenceLedger()

        # Fact in file1 mentions Entity1
        fact1 = FactEntry(
            fact_id="fact1",
            statement="Statement in file1 about Entity1",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("Entity1",),
            cluster_id=0,
        )
        # Fact in file2 also mentions Entity1
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Statement in file2 about Entity1",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=("Entity1",),
            cluster_id=0,
        )
        # Unrelated fact in file3
        fact3 = FactEntry(
            fact_id="fact3",
            statement="Unrelated statement",
            file_path="file3.py",
            start_line=20,
            end_line=25,
            category="constraint",
            confidence=ConfidenceLevel.INFERRED,
            entities=("OtherEntity",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)
        ledger.add_fact(fact3)

        # Query for file1.py - should get fact1 and related fact2 via Entity1
        results = ledger.get_related_facts({"file1.py"})
        fact_ids = {f.fact_id for f in results}

        assert "fact1" in fact_ids  # Direct from file1
        assert "fact2" in fact_ids  # Related via Entity1
        assert "fact3" not in fact_ids  # Unrelated


# =============================================================================
# EvidenceLedger Conflict Detection Tests
# =============================================================================


class TestEvidenceLedgerConflictDetection:
    """Tests for conflict detection."""

    def test_detect_conflicts_negation(self):
        """Test detect_conflicts finds negation-based conflicts.

        The conflict detection uses XOR - exactly one fact should have
        negation pattern (like "never", "cannot", "does not") to trigger.
        """
        ledger = EvidenceLedger()

        # Fact 1: Positive statement (no negation pattern)
        fact1 = FactEntry(
            fact_id="fact1",
            statement="SearchService retries failed queries automatically",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )
        # Fact 2: Negative statement (has "never" negation pattern)
        fact2 = FactEntry(
            fact_id="fact2",
            statement="SearchService never retries failed queries",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        conflicts = ledger.detect_conflicts()

        assert len(conflicts) == 1
        assert "negation" in conflicts[0].reason.lower()

    def test_detect_conflicts_numeric_difference(self):
        """Test detect_conflicts finds numeric value conflicts."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="MAX_RETRIES is set to 3",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="constraint",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("MAX_RETRIES",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="MAX_RETRIES is set to 5",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="constraint",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("MAX_RETRIES",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        conflicts = ledger.detect_conflicts()

        assert len(conflicts) == 1
        assert "numeric" in conflicts[0].reason.lower()

    def test_detect_conflicts_no_conflict(self):
        """Test detect_conflicts returns empty for non-conflicting facts."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="SearchService handles HTTP requests",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="SearchService logs all errors",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        conflicts = ledger.detect_conflicts()
        assert len(conflicts) == 0


# =============================================================================
# Prompt Generation Tests
# =============================================================================


class TestEvidenceLedgerConstantsPromptGeneration:
    """Tests for constants prompt context generation."""

    def test_get_constants_prompt_context_formats_correctly(self):
        """Test get_constants_prompt_context formats constants properly."""
        ledger = EvidenceLedger()
        ledger.add_constant(ConstantEntry(name="DEBUG", file_path="config.py", value="True", type="bool"))
        ledger.add_constant(ConstantEntry(name="TIMEOUT", file_path="config.py", value="30", type="int"))

        context = ledger.get_constants_prompt_context()

        assert "Global Constants" in context
        assert "config.py" in context
        assert "DEBUG" in context
        assert "True" in context
        assert "TIMEOUT" in context

    def test_get_constants_prompt_context_empty_for_no_constants(self):
        """Test get_constants_prompt_context returns empty for no constants."""
        ledger = EvidenceLedger()
        context = ledger.get_constants_prompt_context()
        assert context == ""


class TestEvidenceLedgerFactsPromptGeneration:
    """Tests for facts prompt context generation."""

    def test_get_facts_map_prompt_context_formats_correctly(self, sample_ledger):
        """Test get_facts_map_prompt_context formats facts for map phase."""
        context = sample_ledger.get_facts_map_prompt_context({"services/search.py"})

        # Should contain the fact in simple format
        assert "SearchService retries up to MAX_RETRIES times" in context
        assert "search.py" in context  # Basename only in simple format
        assert "[DEF]" in context  # Simple format uses abbreviated confidence
        assert "Verified Facts" in context

    def test_get_facts_map_prompt_context_empty_for_no_matches(self, sample_ledger):
        """Test get_facts_map_prompt_context returns empty for non-matching files."""
        context = sample_ledger.get_facts_map_prompt_context({"other/file.py"})
        assert context == ""

    def test_get_facts_reduce_prompt_context_includes_all_facts(self, sample_ledger):
        """Test get_facts_reduce_prompt_context includes all facts."""
        context = sample_ledger.get_facts_reduce_prompt_context()

        assert "SearchService retries up to MAX_RETRIES times" in context
        assert "Facts Ledger" in context

    def test_get_facts_reduce_prompt_context_includes_conflicts(self):
        """Test get_facts_reduce_prompt_context includes conflict warnings."""
        ledger = EvidenceLedger()

        fact1 = FactEntry(
            fact_id="fact1",
            statement="Value is 3",
            file_path="file1.py",
            start_line=1,
            end_line=5,
            category="constraint",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("Value",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Value is 5",
            file_path="file2.py",
            start_line=10,
            end_line=15,
            category="constraint",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("Value",),
            cluster_id=0,
        )

        ledger.add_fact(fact1)
        ledger.add_fact(fact2)
        ledger.conflicts = ledger.detect_conflicts()

        context = ledger.get_facts_reduce_prompt_context()

        assert "Conflicts" in context
        assert "fact1" in context
        assert "fact2" in context

    def test_get_facts_reduce_prompt_context_empty_ledger(self):
        """Test get_facts_reduce_prompt_context returns empty for empty ledger."""
        ledger = EvidenceLedger()
        context = ledger.get_facts_reduce_prompt_context()
        assert context == ""


# =============================================================================
# Report Generation Tests
# =============================================================================


class TestEvidenceLedgerReportGeneration:
    """Tests for report formatting."""

    def test_get_report_suffix_includes_both_types(self):
        """Test get_report_suffix includes both constants and facts."""
        ledger = EvidenceLedger()
        ledger.add_constant(ConstantEntry(name="DEBUG", file_path="config.py", value="True"))
        ledger.add_fact(
            FactEntry(
                fact_id="fact1",
                statement="Some behavior",
                file_path="file.py",
                start_line=1,
                end_line=5,
                category="behavior",
                confidence=ConfidenceLevel.DEFINITE,
                entities=(),
                cluster_id=0,
            )
        )

        suffix = ledger.get_report_suffix()

        assert "## Evidence" in suffix
        assert "### Constants Referenced" in suffix
        assert "DEBUG" in suffix
        assert "### Verified Facts" in suffix
        assert "Some behavior" in suffix

    def test_get_report_suffix_formats_facts_by_confidence(self, sample_ledger):
        """Test get_report_suffix groups facts by confidence."""
        suffix = sample_ledger.get_report_suffix()

        assert "SearchService retries up to MAX_RETRIES times" in suffix
        assert "Definite Facts" in suffix

    def test_get_report_suffix_empty_ledger(self):
        """Test get_report_suffix returns empty for empty ledger."""
        ledger = EvidenceLedger()
        suffix = ledger.get_report_suffix()
        assert suffix == ""

    def test_insert_into_report_before_sources(self, sample_ledger):
        """Test insert_into_report inserts before Sources section."""
        report = """# Research Report

Some findings here.

## Sources

- Source 1
- Source 2
"""
        result = sample_ledger.insert_into_report(report)

        # Evidence section should be before Sources
        evidence_pos = result.find("## Evidence")
        sources_pos = result.find("## Sources")

        assert evidence_pos < sources_pos
        assert "SearchService" in result

    def test_insert_into_report_no_sources_section(self, sample_ledger):
        """Test insert_into_report appends when no Sources section."""
        report = """# Research Report

Some findings here.
"""
        result = sample_ledger.insert_into_report(report)

        assert "## Evidence" in result
        assert result.startswith("# Research Report")

    def test_insert_into_report_empty_ledger(self):
        """Test insert_into_report returns unchanged for empty ledger."""
        ledger = EvidenceLedger()
        report = "# Report\n\n## Sources"

        result = ledger.insert_into_report(report)
        assert result == report


# =============================================================================
# Progress Table Tests
# =============================================================================


class TestEvidenceLedgerProgressTable:
    """Tests for progress table formatting."""

    def test_format_progress_table_uses_evidence_context_header(self):
        """Test format_progress_table uses Evidence Context header."""
        ledger = EvidenceLedger()
        ledger.add_constant(ConstantEntry(name="DEBUG", file_path="config.py", value="True"))

        table = ledger.format_progress_table()

        assert "## Evidence Context" in table
        assert "## Global Constants" not in table  # Old header should be gone
        assert "### Constants" in table
        assert "**config.py**:" in table
        assert "- DEBUG = True" in table

    def test_format_progress_table_includes_constants(self):
        """Test format_progress_table includes constants in markdown format."""
        ledger = EvidenceLedger()
        ledger.add_constant(ConstantEntry(name="MY_CONST", file_path="config.py", value="42"))

        table = ledger.format_progress_table()

        assert "MY_CONST" in table
        assert "42" in table
        assert "config.py" in table

    def test_format_progress_table_includes_facts(self):
        """Test format_progress_table includes facts."""
        ledger = EvidenceLedger()
        ledger.add_fact(
            FactEntry(
                fact_id="fact1",
                statement="Some behavior",
                file_path="file.py",
                start_line=1,
                end_line=5,
                category="behavior",
                confidence=ConfidenceLevel.DEFINITE,
                entities=(),
                cluster_id=0,
            )
        )

        table = ledger.format_progress_table()

        assert "Some behavior" in table
        assert "Facts: 1" in table

    def test_format_progress_table_has_summary_footer(self):
        """Test format_progress_table has summary footer."""
        ledger = EvidenceLedger()
        ledger.add_constant(ConstantEntry(name="A", file_path="a.py", value="1"))
        ledger.add_constant(ConstantEntry(name="B", file_path="b.py", value="2"))

        table = ledger.format_progress_table()

        assert "Constants: 2" in table
        assert "Facts: 0" in table

    def test_format_progress_table_empty_ledger(self):
        """Test format_progress_table handles empty ledger."""
        ledger = EvidenceLedger()

        table = ledger.format_progress_table()

        assert "Constants: 0" in table
        assert "Facts: 0" in table


# =============================================================================
# Serialization Tests
# =============================================================================


class TestEvidenceLedgerSerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict_and_from_dict_roundtrip(self, sample_ledger):
        """Test to_dict and from_dict produce equivalent ledger."""
        data = sample_ledger.to_dict()
        restored = EvidenceLedger.from_dict(data)

        assert len(restored) == len(sample_ledger)
        assert restored.constants_count == sample_ledger.constants_count
        assert restored.facts_count == sample_ledger.facts_count

    def test_to_dict_and_from_dict_roundtrip_constants(self, sample_constant):
        """Test roundtrip preserves constant data."""
        ledger = EvidenceLedger()
        ledger.add_constant(sample_constant)

        data = ledger.to_dict()
        restored = EvidenceLedger.from_dict(data)

        assert restored.constants_count == 1
        restored_const = list(restored.constants.values())[0]
        assert restored_const.name == sample_constant.name
        assert restored_const.file_path == sample_constant.file_path
        assert restored_const.value == sample_constant.value
        assert restored_const.type == sample_constant.type

    def test_to_dict_and_from_dict_roundtrip_facts(self, sample_fact):
        """Test roundtrip preserves fact data."""
        ledger = EvidenceLedger()
        ledger.add_fact(sample_fact)

        data = ledger.to_dict()
        restored = EvidenceLedger.from_dict(data)

        assert restored.facts_count == 1
        restored_fact = list(restored.facts.values())[0]

        assert restored_fact.fact_id == sample_fact.fact_id
        assert restored_fact.statement == sample_fact.statement
        assert restored_fact.file_path == sample_fact.file_path
        assert restored_fact.start_line == sample_fact.start_line
        assert restored_fact.end_line == sample_fact.end_line
        assert restored_fact.category == sample_fact.category
        assert restored_fact.confidence == sample_fact.confidence
        assert restored_fact.entities == sample_fact.entities
        assert restored_fact.cluster_id == sample_fact.cluster_id

    def test_to_dict_includes_conflicts(self):
        """Test to_dict includes conflicts."""
        ledger = EvidenceLedger()
        ledger.conflicts.append(FactConflict("a", "b", "reason"))

        data = ledger.to_dict()

        assert "conflicts" in data
        assert len(data["conflicts"]) == 1
        assert data["conflicts"][0]["fact_id_a"] == "a"
        assert data["conflicts"][0]["fact_id_b"] == "b"
        assert data["conflicts"][0]["reason"] == "reason"

    def test_from_dict_restores_conflicts(self):
        """Test from_dict restores conflicts."""
        data = {
            "constants": {},
            "facts": {},
            "conflicts": [{"fact_id_a": "x", "fact_id_b": "y", "reason": "test reason"}],
        }

        ledger = EvidenceLedger.from_dict(data)

        assert len(ledger.conflicts) == 1
        assert ledger.conflicts[0].fact_id_a == "x"
        assert ledger.conflicts[0].fact_id_b == "y"
        assert ledger.conflicts[0].reason == "test reason"

    def test_from_dict_rebuilds_entity_index(self):
        """Test from_dict rebuilds entity index."""
        data = {
            "constants": {},
            "facts": {
                "fact1": {
                    "fact_id": "fact1",
                    "statement": "Test statement",
                    "file_path": "file.py",
                    "start_line": 1,
                    "end_line": 5,
                    "category": "behavior",
                    "confidence": "definite",
                    "entities": ["TestEntity", "OtherEntity"],
                    "cluster_id": 0,
                }
            },
            "conflicts": [],
        }

        ledger = EvidenceLedger.from_dict(data)

        assert "testentity" in ledger.entity_index
        assert "otherentity" in ledger.entity_index
        assert "fact1" in ledger.entity_index["testentity"].fact_ids


# =============================================================================
# FactExtractor Tests
# =============================================================================


class TestFactExtractor:
    """Tests for LLM-based fact extraction."""

    @pytest.mark.asyncio
    async def test_extract_from_cluster_parses_json_response(self, mock_llm_provider):
        """Test extract_from_cluster parses JSON response correctly."""
        # Mock LLM response with valid JSON
        mock_response = MagicMock()
        mock_response.content = """```json
[
  {
    "statement": "SearchService handles retries",
    "file_path": "search.py",
    "start_line": 10,
    "end_line": 20,
    "category": "behavior",
    "confidence": "definite",
    "entities": ["SearchService"]
  }
]
```"""
        mock_llm_provider.complete.return_value = mock_response

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"search.py": "class SearchService: pass"},
            root_query="How does search work?",
        )

        assert ledger.facts_count == 1
        fact = list(ledger.facts.values())[0]
        assert fact.statement == "SearchService handles retries"
        assert fact.file_path == "search.py"
        assert fact.confidence == ConfidenceLevel.DEFINITE

    @pytest.mark.asyncio
    async def test_extract_from_cluster_handles_malformed_json(self, mock_llm_provider):
        """Test extract_from_cluster handles malformed JSON gracefully."""
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON at all {"

        mock_llm_provider.complete.return_value = mock_response

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"file.py": "code"},
            root_query="query",
        )

        # Should return empty ledger, not raise
        assert ledger.facts_count == 0

    @pytest.mark.asyncio
    async def test_extract_from_cluster_handles_partial_json(self, mock_llm_provider):
        """Test extract_from_cluster handles partially valid JSON."""
        mock_response = MagicMock()
        mock_response.content = """[
  {
    "statement": "Valid fact",
    "file_path": "file.py",
    "start_line": 1,
    "end_line": 5,
    "category": "behavior",
    "confidence": "definite",
    "entities": []
  },
  {
    "missing": "required fields"
  }
]"""
        mock_llm_provider.complete.return_value = mock_response

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"file.py": "code"},
            root_query="query",
        )

        # Should extract the valid fact, skip the invalid one
        assert ledger.facts_count == 1
        assert list(ledger.facts.values())[0].statement == "Valid fact"

    @pytest.mark.asyncio
    async def test_extract_from_cluster_empty_content(self, mock_llm_provider):
        """Test extract_from_cluster returns empty ledger for empty content."""
        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={},
            root_query="query",
        )

        assert ledger.facts_count == 0
        mock_llm_provider.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_from_cluster_handles_llm_error(self, mock_llm_provider):
        """Test extract_from_cluster handles LLM errors gracefully."""
        mock_llm_provider.complete.side_effect = Exception("API Error")

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"file.py": "code"},
            root_query="query",
        )

        # Should return empty ledger, not raise
        assert ledger.facts_count == 0

    @pytest.mark.asyncio
    async def test_extract_from_clusters_merges_results(self, mock_llm_provider):
        """Test extract_from_clusters merges results from multiple clusters."""
        # Create responses for two clusters
        response1 = MagicMock()
        response1.content = """[
  {
    "statement": "Fact from cluster 1",
    "file_path": "file1.py",
    "start_line": 1,
    "end_line": 5,
    "category": "behavior",
    "confidence": "definite",
    "entities": []
  }
]"""
        response2 = MagicMock()
        response2.content = """[
  {
    "statement": "Fact from cluster 2",
    "file_path": "file2.py",
    "start_line": 10,
    "end_line": 15,
    "category": "architecture",
    "confidence": "likely",
    "entities": []
  }
]"""

        mock_llm_provider.complete.side_effect = [response1, response2]

        extractor = FactExtractor(mock_llm_provider)
        # clusters format: (cluster_id, {file_path: content}, max_facts)
        clusters = [
            (0, {"file1.py": "code1"}, 30),
            (1, {"file2.py": "code2"}, 30),
        ]

        ledger = await extractor.extract_from_clusters(
            clusters=clusters,
            root_query="query",
            max_concurrency=2,
        )

        assert ledger.facts_count == 2
        statements = {f.statement for f in ledger.facts.values()}
        assert "Fact from cluster 1" in statements
        assert "Fact from cluster 2" in statements

    @pytest.mark.asyncio
    async def test_extract_from_clusters_empty_list(self, mock_llm_provider):
        """Test extract_from_clusters handles empty cluster list."""
        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_clusters(
            clusters=[],
            root_query="query",
        )

        assert ledger.facts_count == 0

    @pytest.mark.asyncio
    async def test_extract_from_cluster_unknown_confidence_defaults(
        self, mock_llm_provider
    ):
        """Test unknown confidence levels default to UNCERTAIN."""
        mock_response = MagicMock()
        mock_response.content = """[
  {
    "statement": "Test fact",
    "file_path": "file.py",
    "start_line": 1,
    "end_line": 5,
    "category": "behavior",
    "confidence": "invalid_level",
    "entities": []
  }
]"""
        mock_llm_provider.complete.return_value = mock_response

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"file.py": "code"},
            root_query="query",
        )

        assert ledger.facts_count == 1
        fact = list(ledger.facts.values())[0]
        assert fact.confidence == ConfidenceLevel.UNCERTAIN


# =============================================================================
# EntityLink and FactConflict Tests
# =============================================================================


class TestEntityLink:
    """Tests for EntityLink dataclass."""

    def test_entity_link_creation(self):
        """Test EntityLink can be created."""
        link = EntityLink(
            entity_name="SearchService",
            fact_ids=("fact1", "fact2"),
        )

        assert link.entity_name == "SearchService"
        assert link.fact_ids == ("fact1", "fact2")

    def test_entity_link_frozen(self):
        """Test EntityLink is immutable."""
        link = EntityLink(entity_name="Test", fact_ids=("a",))
        with pytest.raises(AttributeError):
            link.entity_name = "Modified"


class TestFactConflict:
    """Tests for FactConflict dataclass."""

    def test_fact_conflict_creation(self):
        """Test FactConflict can be created."""
        conflict = FactConflict(
            fact_id_a="fact1",
            fact_id_b="fact2",
            reason="Contradictory statements",
        )

        assert conflict.fact_id_a == "fact1"
        assert conflict.fact_id_b == "fact2"
        assert conflict.reason == "Contradictory statements"

    def test_fact_conflict_frozen(self):
        """Test FactConflict is immutable."""
        conflict = FactConflict("a", "b", "reason")
        with pytest.raises(AttributeError):
            conflict.reason = "Modified"


# =============================================================================
# Simple Format Tests
# =============================================================================


class TestSimpleFormatting:
    """Tests for simple markdown fact formatting."""

    def test_format_facts_simple_basic(self):
        """Test simple format produces expected markdown structure."""
        ledger = EvidenceLedger()
        fact = FactEntry(
            fact_id="abc123def456",
            statement="Uses exponential backoff",
            file_path="services/search.py",
            start_line=45,
            end_line=52,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService",),
            cluster_id=0,
        )
        ledger.add_fact(fact)

        context = ledger.get_facts_reduce_prompt_context()

        # Check format: - [CONF] statement (file:lines)
        assert "- [DEF] Uses exponential backoff (search.py:45-52)" in context

    def test_format_facts_simple_sorts_by_confidence(self):
        """Test facts are sorted by confidence (definite first)."""
        ledger = EvidenceLedger()

        # Add facts with different confidence levels (in reverse order)
        for i, conf in enumerate(
            [ConfidenceLevel.UNCERTAIN, ConfidenceLevel.DEFINITE, ConfidenceLevel.LIKELY]
        ):
            fact = FactEntry(
                fact_id=f"fact{i}",
                statement=f"Fact {conf.value}",
                file_path="file.py",
                start_line=i,
                end_line=i + 5,
                category="behavior",
                confidence=conf,
                entities=(),
                cluster_id=0,
            )
            ledger.add_fact(fact)

        context = ledger.get_facts_reduce_prompt_context()

        # Definite should come before Likely before Uncertain
        def_pos = context.find("[DEF]")
        lik_pos = context.find("[LIK]")
        unc_pos = context.find("[UNC]")

        assert def_pos < lik_pos < unc_pos

    def test_format_facts_simple_truncates_at_500(self):
        """Test facts are truncated at 500 to stay under 15k tokens."""
        ledger = EvidenceLedger()

        # Add 600 facts
        for i in range(600):
            fact = FactEntry(
                fact_id=f"fact{i:04d}",
                statement=f"Statement {i}",
                file_path=f"file{i % 5}.py",
                start_line=i,
                end_line=i + 5,
                category="behavior",
                confidence=ConfidenceLevel.DEFINITE,
                entities=(),
                cluster_id=0,
            )
            ledger.add_fact(fact)

        context = ledger.get_facts_reduce_prompt_context()

        # Should contain truncation notice
        assert "truncated to 500 facts" in context
        # Should have exactly 500 fact lines (starting with "- [")
        fact_lines = [line for line in context.split("\n") if line.startswith("- [")]
        assert len(fact_lines) == 500


class TestStatementTruncation:
    """Tests for statement truncation at extraction time."""

    @pytest.mark.asyncio
    async def test_extractor_truncates_long_statements(self, mock_llm_provider):
        """Test extractor truncates statements over MAX_STATEMENT_CHARS."""
        from chunkhound.services.research.shared.evidence_ledger.extractor import (
            MAX_STATEMENT_CHARS,
        )

        # Create a response with an overly long statement
        long_statement = "X" * 200  # Well over the limit
        mock_response = MagicMock()
        mock_response.content = f"""[
  {{
    "statement": "{long_statement}",
    "file_path": "file.py",
    "start_line": 1,
    "end_line": 5,
    "category": "behavior",
    "confidence": "definite",
    "entities": []
  }}
]"""
        mock_llm_provider.complete.return_value = mock_response

        from chunkhound.services.research.shared.evidence_ledger import FactExtractor

        extractor = FactExtractor(mock_llm_provider)
        ledger = await extractor.extract_from_cluster(
            cluster_id=0,
            cluster_content={"file.py": "code"},
            root_query="query",
        )

        assert ledger.facts_count == 1
        fact = list(ledger.facts.values())[0]
        # Statement should be truncated to MAX_STATEMENT_CHARS
        assert len(fact.statement) <= MAX_STATEMENT_CHARS
        assert fact.statement.endswith("...")


# =============================================================================
# Clustered Extraction Tests
# =============================================================================


class TestClusteredExtraction:
    """Tests for clustered fact extraction utility."""

    @pytest.fixture
    def mock_embedding_provider(self):
        """Mock embedding provider for clustering."""
        provider = MagicMock()
        # Return fixed embeddings for clustering (8 dimensions)
        provider.generate_embeddings = AsyncMock(
            return_value=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
        )
        provider.estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)
        return provider

    @pytest.mark.asyncio
    async def test_extract_facts_with_clustering_empty_files(
        self, mock_llm_provider, mock_embedding_provider
    ):
        """Test clustered extraction returns empty result for empty files."""
        from chunkhound.services.research.shared.evidence_ledger import (
            ClusteredExtractionResult,
            extract_facts_with_clustering,
        )

        result = await extract_facts_with_clustering(
            files={},
            root_query="query",
            llm_provider=mock_llm_provider,
            embedding_provider=mock_embedding_provider,
        )

        assert isinstance(result, ClusteredExtractionResult)
        assert result.evidence_ledger.facts_count == 0
        assert result.cluster_groups == []
        assert result.cluster_metadata["num_clusters"] == 0

    @pytest.mark.asyncio
    async def test_extract_facts_with_clustering_returns_result_dataclass(
        self, mock_llm_provider, mock_embedding_provider
    ):
        """Test clustered extraction returns ClusteredExtractionResult."""
        from chunkhound.services.research.shared.evidence_ledger import (
            ClusteredExtractionResult,
            extract_facts_with_clustering,
        )
        from chunkhound.services.clustering_service import ClusterGroup

        # Mock the clustering service behavior by patching it
        mock_cluster = ClusterGroup(
            cluster_id=0,
            file_paths=["file.py"],
            files_content={"file.py": "code"},
            total_tokens=1,
        )

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = """[
  {
    "statement": "Test fact",
    "file_path": "file.py",
    "start_line": 1,
    "end_line": 5,
    "category": "behavior",
    "confidence": "definite",
    "entities": []
  }
]"""
        mock_llm_provider.complete.return_value = mock_response
        mock_llm_provider.estimate_tokens = MagicMock(side_effect=lambda x: len(x) // 4)

        # Patch ClusteringService
        with patch(
            "chunkhound.services.research.shared.evidence_ledger.clustered_extractor.ClusteringService"
        ) as MockClusteringService:
            mock_service_instance = MagicMock()
            mock_service_instance.cluster_files_hdbscan_bounded = AsyncMock(
                return_value=([mock_cluster], {"num_clusters": 1})
            )
            MockClusteringService.return_value = mock_service_instance

            result = await extract_facts_with_clustering(
                files={"file.py": "code"},
                root_query="query",
                llm_provider=mock_llm_provider,
                embedding_provider=mock_embedding_provider,
            )

            assert isinstance(result, ClusteredExtractionResult)
            assert hasattr(result, "evidence_ledger")
            assert hasattr(result, "cluster_groups")
            assert hasattr(result, "cluster_metadata")
            assert len(result.cluster_groups) == 1
            assert result.cluster_metadata["num_clusters"] == 1

    @pytest.mark.asyncio
    async def test_extract_facts_with_clustering_clusters_reusable(
        self, mock_llm_provider, mock_embedding_provider
    ):
        """Test that returned clusters can be reused for synthesis."""
        from chunkhound.services.research.shared.evidence_ledger import (
            extract_facts_with_clustering,
        )
        from chunkhound.services.clustering_service import ClusterGroup

        # Create test cluster
        mock_cluster = ClusterGroup(
            cluster_id=0,
            file_paths=["file1.py", "file2.py"],
            files_content={"file1.py": "code1", "file2.py": "code2"},
            total_tokens=10,
        )

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "[]"
        mock_llm_provider.complete.return_value = mock_response
        mock_llm_provider.estimate_tokens = MagicMock(return_value=10)

        with patch(
            "chunkhound.services.research.shared.evidence_ledger.clustered_extractor.ClusteringService"
        ) as MockClusteringService:
            mock_service_instance = MagicMock()
            mock_service_instance.cluster_files_hdbscan_bounded = AsyncMock(
                return_value=(
                    [mock_cluster],
                    {"num_clusters": 1, "avg_tokens_per_cluster": 10},
                )
            )
            MockClusteringService.return_value = mock_service_instance

            result = await extract_facts_with_clustering(
                files={"file1.py": "code1", "file2.py": "code2"},
                root_query="query",
                llm_provider=mock_llm_provider,
                embedding_provider=mock_embedding_provider,
            )

            # Clusters should be reusable - check structure
            assert len(result.cluster_groups) == 1
            cluster = result.cluster_groups[0]
            assert isinstance(cluster, ClusterGroup)
            assert cluster.cluster_id == 0
            assert "file1.py" in cluster.files_content
            assert "file2.py" in cluster.files_content


# =============================================================================
# K-means Cluster Count Calculation Tests
# =============================================================================


class TestKMeansClusterCountCalculation:
    """Tests for k-means cluster count calculation formula.

    The formula: n_clusters = max(1, ceil(total_tokens / 50000))
    """

    def test_small_token_count_yields_one_cluster(self):
        """Test 10,000 tokens produces 1 cluster (below threshold)."""
        import math

        total_tokens = 10_000
        max_tokens_per_cluster = 50_000

        n_clusters = max(1, math.ceil(total_tokens / max_tokens_per_cluster))

        assert n_clusters == 1

    def test_exactly_threshold_yields_one_cluster(self):
        """Test 50,000 tokens produces exactly 1 cluster (at threshold)."""
        import math

        total_tokens = 50_000
        max_tokens_per_cluster = 50_000

        n_clusters = max(1, math.ceil(total_tokens / max_tokens_per_cluster))

        assert n_clusters == 1

    def test_just_over_threshold_yields_two_clusters(self):
        """Test 50,001 tokens produces 2 clusters (just over threshold)."""
        import math

        total_tokens = 50_001
        max_tokens_per_cluster = 50_000

        n_clusters = max(1, math.ceil(total_tokens / max_tokens_per_cluster))

        assert n_clusters == 2

    def test_large_token_count_yields_three_clusters(self):
        """Test 150,000 tokens produces 3 clusters."""
        import math

        total_tokens = 150_000
        max_tokens_per_cluster = 50_000

        n_clusters = max(1, math.ceil(total_tokens / max_tokens_per_cluster))

        assert n_clusters == 3

    def test_zero_tokens_yields_one_cluster(self):
        """Test 0 tokens still produces 1 cluster (max ensures minimum)."""
        import math

        total_tokens = 0
        max_tokens_per_cluster = 50_000

        n_clusters = max(1, math.ceil(total_tokens / max_tokens_per_cluster))

        assert n_clusters == 1


# =============================================================================
# Replace Constants from Chunks Tests
# =============================================================================


class TestReplaceConstantsFromChunks:
    """Tests for replace_constants_from_chunks method."""

    def test_replace_removes_stale_constants(self):
        """Test that constants from removed files are removed after replace.

        Creates ledger with constants from file1.py and file2.py,
        then replaces with only file1.py chunks - file2.py constants
        should be removed.
        """
        ledger = EvidenceLedger()

        # Add constants from two files
        ledger.add_constant(
            ConstantEntry(name="CONST_A", file_path="file1.py", value="1")
        )
        ledger.add_constant(
            ConstantEntry(name="CONST_B", file_path="file1.py", value="2")
        )
        ledger.add_constant(
            ConstantEntry(name="CONST_C", file_path="file2.py", value="3")
        )
        ledger.add_constant(
            ConstantEntry(name="CONST_D", file_path="file2.py", value="4")
        )

        assert ledger.constants_count == 4

        # Create chunks that only include file1.py constants
        chunks_only_file1 = [
            {
                "file_path": "file1.py",
                "metadata": {
                    "constants": [
                        {"name": "CONST_A", "value": "1"},
                        {"name": "CONST_B", "value": "2"},
                    ]
                },
            }
        ]

        # Replace constants - should remove file2.py constants
        updated_ledger = ledger.replace_constants_from_chunks(chunks_only_file1)

        # Only file1.py constants should remain
        assert updated_ledger.constants_count == 2

        constant_keys = list(updated_ledger.constants.keys())
        assert "file1.py:CONST_A" in constant_keys
        assert "file1.py:CONST_B" in constant_keys
        assert "file2.py:CONST_C" not in constant_keys
        assert "file2.py:CONST_D" not in constant_keys

    def test_replace_preserves_facts(self):
        """Test that facts are preserved through constants replacement."""
        ledger = EvidenceLedger()

        # Add constants
        ledger.add_constant(
            ConstantEntry(name="CONST_A", file_path="file1.py", value="1")
        )
        ledger.add_constant(
            ConstantEntry(name="CONST_B", file_path="file2.py", value="2")
        )

        # Add facts
        fact1 = FactEntry(
            fact_id="fact1",
            statement="Important behavior fact",
            file_path="file1.py",
            start_line=10,
            end_line=20,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("ServiceA",),
            cluster_id=0,
        )
        fact2 = FactEntry(
            fact_id="fact2",
            statement="Architecture decision",
            file_path="file2.py",
            start_line=30,
            end_line=40,
            category="architecture",
            confidence=ConfidenceLevel.LIKELY,
            entities=("ServiceB",),
            cluster_id=1,
        )
        ledger.add_fact(fact1)
        ledger.add_fact(fact2)

        assert ledger.facts_count == 2

        # Replace constants with empty chunks (removes all constants)
        updated_ledger = ledger.replace_constants_from_chunks([])

        # Constants should be 0
        assert updated_ledger.constants_count == 0

        # Facts should be preserved
        assert updated_ledger.facts_count == 2
        assert "fact1" in updated_ledger.facts
        assert "fact2" in updated_ledger.facts
        assert updated_ledger.facts["fact1"].statement == "Important behavior fact"
        assert updated_ledger.facts["fact2"].statement == "Architecture decision"

    def test_replace_preserves_entity_index(self):
        """Test that entity index is preserved through constants replacement."""
        ledger = EvidenceLedger()

        # Add a fact with entities
        fact = FactEntry(
            fact_id="fact1",
            statement="SearchService handles queries",
            file_path="search.py",
            start_line=1,
            end_line=10,
            category="behavior",
            confidence=ConfidenceLevel.DEFINITE,
            entities=("SearchService", "QueryHandler"),
            cluster_id=0,
        )
        ledger.add_fact(fact)

        # Verify entity index exists
        assert "searchservice" in ledger.entity_index
        assert "queryhandler" in ledger.entity_index

        # Replace constants with empty chunks
        updated_ledger = ledger.replace_constants_from_chunks([])

        # Entity index should be preserved
        assert "searchservice" in updated_ledger.entity_index
        assert "queryhandler" in updated_ledger.entity_index
        assert "fact1" in updated_ledger.entity_index["searchservice"].fact_ids

    def test_replace_preserves_conflicts(self):
        """Test that conflicts list is preserved through constants replacement."""
        ledger = EvidenceLedger()

        # Add a conflict
        ledger.conflicts.append(FactConflict("fact_a", "fact_b", "Numeric mismatch"))

        assert len(ledger.conflicts) == 1

        # Replace constants with empty chunks
        updated_ledger = ledger.replace_constants_from_chunks([])

        # Conflicts should be preserved
        assert len(updated_ledger.conflicts) == 1
        assert updated_ledger.conflicts[0].fact_id_a == "fact_a"
        assert updated_ledger.conflicts[0].reason == "Numeric mismatch"

    def test_replace_is_immutable(self):
        """Test that replace_constants_from_chunks does not modify original."""
        ledger = EvidenceLedger()

        # Add constant and fact
        ledger.add_constant(
            ConstantEntry(name="CONST_A", file_path="file1.py", value="1")
        )
        ledger.add_fact(
            FactEntry(
                fact_id="fact1",
                statement="Test",
                file_path="file.py",
                start_line=1,
                end_line=5,
                category="behavior",
                confidence=ConfidenceLevel.DEFINITE,
                entities=(),
                cluster_id=0,
            )
        )

        original_const_count = ledger.constants_count
        original_fact_count = ledger.facts_count

        # Replace constants
        _ = ledger.replace_constants_from_chunks([])

        # Original should be unchanged
        assert ledger.constants_count == original_const_count
        assert ledger.facts_count == original_fact_count
